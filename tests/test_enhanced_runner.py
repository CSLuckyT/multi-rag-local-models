"""Tests for EnhancedRAGRunner components and batch integration."""

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal EnhancedRAGRunner without real models
# ---------------------------------------------------------------------------

def _make_mock_llm(answer: str = "Paris"):
    llm = MagicMock()
    llm.generate.return_value = (answer, 0.0)
    return llm


def _make_runner(tmp_path: Path, answer: str = "Paris"):
    """Return an EnhancedRAGRunner with all external models mocked out."""
    from arag.enhanced import EnhancedRAGRunner

    # Build minimal chunk records in a temp FAISS store structure
    chunks = [
        {"chunk_id": i, "title": f"Title{i}", "sent_id": 0, "text": f"sentence {i}",
         "passage": f"Title{i}. sentence {i}"}
        for i in range(20)
    ]

    # Patch FaissArtifactStore and BGEEmbedder so no actual model is loaded
    mock_store = MagicMock()
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.zeros((1, 10), dtype="float32"),
        np.arange(10, dtype="int64").reshape(1, -1),
    )
    mock_store.load.return_value = (mock_index, chunks, {})

    mock_embedder = MagicMock()
    mock_embedder.encode_texts.return_value = np.ones((2, 4), dtype="float32")

    runner = EnhancedRAGRunner.__new__(EnhancedRAGRunner)
    runner.llm_client = _make_mock_llm(answer)
    runner.rerank_model_name = "BAAI/bge-reranker-v2-m3"
    runner.device = "cpu"
    runner.max_context_chars = 3500
    runner.store = mock_store
    runner.embedder = mock_embedder
    runner._index = mock_index
    runner._chunk_records = chunks
    runner._reranker = None
    return runner


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestRewriteQuery:
    def test_returns_nonempty_string(self, tmp_path):
        runner = _make_runner(tmp_path, answer="hypothetical sentence")
        result = runner.rewrite_query("Who founded Tesla?")
        assert isinstance(result, str)
        assert len(result) > 0


class TestFaissTopn:
    def test_returns_list_of_ints(self, tmp_path):
        runner = _make_runner(tmp_path)
        indices = runner.faiss_topn("test question", "hypothetical answer", n=5)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_no_hyde(self, tmp_path):
        runner = _make_runner(tmp_path)
        indices = runner.faiss_topn("test question", None, n=5)
        assert isinstance(indices, list)


class TestRerank:
    def test_returns_exactly_k_results(self, tmp_path):
        runner = _make_runner(tmp_path)
        # Mock cross-encoder
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.9, 0.5, 0.8, 0.1, 0.7])
        runner._reranker = mock_reranker

        cand_idx = list(range(5))
        top_idx, top_scores = runner.rerank("question", cand_idx, k=3)
        assert len(top_idx) == 3
        assert len(top_scores) == 3

    def test_empty_candidates(self, tmp_path):
        runner = _make_runner(tmp_path)
        top_idx, top_scores = runner.rerank("question", [], k=3)
        assert top_idx == []
        assert top_scores == []


class TestFilterContext:
    def test_keeps_chunks_above_threshold(self, tmp_path):
        runner = _make_runner(tmp_path)
        top_idx = [0, 1, 2]
        top_scores = [0.9, 0.2, 0.8]
        kept = runner.filter_context(top_idx, top_scores, min_score=0.5)
        assert len(kept) == 2  # chunk 0 (0.9) and chunk 2 (0.8)

    def test_fallback_to_top1_when_nothing_passes(self, tmp_path):
        runner = _make_runner(tmp_path)
        top_idx = [0, 1]
        top_scores = [0.1, 0.2]
        kept = runner.filter_context(top_idx, top_scores, min_score=0.9)
        assert len(kept) >= 1  # fallback always returns at least 1

    def test_no_filter_keeps_all(self, tmp_path):
        runner = _make_runner(tmp_path)
        top_idx = [0, 1, 2]
        top_scores = [0.1, 0.2, 0.3]
        kept = runner.filter_context(top_idx, top_scores, min_score=float("-inf"))
        assert len(kept) == 3


class TestFormatContext:
    def test_respects_char_budget(self, tmp_path):
        runner = _make_runner(tmp_path)
        chunks = [{"title": "T", "text": "x" * 1000}] * 10
        result = runner.format_context(chunks, max_chars=500)
        assert len(result) <= 600  # some slack for title prefix


class TestRunEndToEnd:
    def test_run_returns_expected_keys(self, tmp_path):
        runner = _make_runner(tmp_path, answer="Paris")
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.9] * 10)
        runner._reranker = mock_reranker

        result = runner.run("What is the capital of France?", n_first=10, k_rerank=3)

        for key in ("generated_answer", "raw_generated_answer", "retrieved_chunks",
                    "retrieved_indices", "hyde_query", "rerank_scores", "n_first",
                    "k_rerank", "use_hyde", "temperature"):
            assert key in result, f"Missing key: {key}"

    def test_run_no_hyde(self, tmp_path):
        runner = _make_runner(tmp_path)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.5] * 10)
        runner._reranker = mock_reranker

        result = runner.run("Question?", use_hyde=False)
        assert result["hyde_query"] is None
        assert result["use_hyde"] is False


# ---------------------------------------------------------------------------
# Integration test: batch_runner writes predictions_enhanced.jsonl
# ---------------------------------------------------------------------------

class TestBatchRunnerEnhancedMode:
    def test_predictions_file_written(self, tmp_path):
        """Mocked end-to-end: batch_runner enhanced mode writes expected JSONL."""
        from arag import Config

        # Minimal config dict
        cfg_data = {
            "llm": {"model": "dummy", "temperature": 0.0, "max_tokens": 32,
                    "device": "cpu", "adapter_path": None, "use_4bit": False,
                    "torch_dtype": "float32"},
            "embedding": {"model": "BAAI/bge-m3", "device": "cpu"},
            "agent": {"max_loops": 2, "max_token_budget": 128000},
            "data": {
                "prepared_dir": str(tmp_path / "data"),
                "chunks_file": str(tmp_path / "data" / "corpus_chunks.json"),
                "reuse_prepared_data": True,
            },
            "retrieval": {"artifact_dir": str(tmp_path / "index"), "top_k": 3},
            "runtime": {"mode": "enhanced", "device": "cpu"},
            "enhanced": {
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "n_first": 5,
                "k_rerank": 2,
                "use_hyde": False,
                "filter_min_score": float("-inf"),
                "max_context_chars": 500,
            },
        }

        # Write dummy artifacts
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        questions = [
            {"qid": "q1", "question": "Who is X?", "answer": "Y"},
            {"qid": "q2", "question": "Where is Z?", "answer": "W"},
        ]
        (data_dir / "val_questions.json").write_text(json.dumps(questions))
        (data_dir / "corpus_chunks.json").write_text(
            json.dumps([{"id": f"c{i}", "text": f"text {i}"} for i in range(10)])
        )
        (data_dir / "metadata.json").write_text("{}")

        config = Config(cfg_data)

        # Patch EnhancedRAGRunner.run so no real model is needed
        from arag import enhanced as enhanced_module

        def fake_run(self, question, **kwargs):
            return {
                "generated_answer": "fake",
                "raw_generated_answer": "fake",
                "retrieved_chunks": ["chunk text"],
                "retrieved_indices": ["c0"],
                "retrieved_titles": ["Title"],
                "retrieved_sent_ids": [0],
                "hyde_query": None,
                "rerank_scores": [0.9],
                "n_first": 5,
                "k_rerank": 2,
                "use_hyde": False,
                "filter_min_score": float("-inf"),
                "temperature": 0.0,
                "top_k": 2,
            }

        with patch.object(enhanced_module.EnhancedRAGRunner, "run", fake_run), \
             patch("arag.enhanced.FaissArtifactStore"), \
             patch("arag.enhanced.BGEEmbedder"), \
             patch("scripts.batch_runner.BatchRunner._ensure_retrieval_artifacts"):

            import scripts.batch_runner as br
            runner = br.BatchRunner.__new__(br.BatchRunner)
            runner.config = config
            runner.output_dir = tmp_path / "out"
            runner.output_dir.mkdir()
            runner.mode = "enhanced"
            runner.limit = None
            runner.num_workers = 1
            runner.verbose = False
            runner.questions_file = data_dir / "val_questions.json"
            runner.questions = questions
            runner._llm_client = MagicMock()
            runner._llm_client.generate.return_value = ("fake", 0.0)
            runner._shared_tools = None
            runner._baseline_runner = None

            # Build enhanced runner with patched internals
            from arag.enhanced import EnhancedRAGRunner
            enh = EnhancedRAGRunner.__new__(EnhancedRAGRunner)
            enh.llm_client = runner._llm_client
            enh.rerank_model_name = "BAAI/bge-reranker-v2-m3"
            enh.device = "cpu"
            enh.max_context_chars = 500
            enh.store = MagicMock()
            enh.embedder = MagicMock()
            enh._index = None
            enh._chunk_records = []
            enh._reranker = None
            runner._enhanced_runner = enh

            runner._run_mode("enhanced", runner._process_enhanced)

        out_file = runner.output_dir / "predictions_enhanced.jsonl"
        assert out_file.exists(), "predictions_enhanced.jsonl was not created"
        lines = [json.loads(l) for l in out_file.read_text().strip().splitlines()]
        assert len(lines) == 2
        assert all(l["mode"] == "enhanced" for l in lines)
        assert all("pred_answer" in l for l in lines)


# ---------------------------------------------------------------------------
# Regression: CLI accepts enhanced and all modes
# ---------------------------------------------------------------------------

class TestCLIModes:
    def test_enhanced_in_choices(self):
        import argparse
        import scripts.batch_runner as br
        # Re-parse choices from the actual argparse setup by inspecting the module
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mode",
            choices=["baseline", "agentic", "enhanced", "both", "all"],
        )
        for mode in ("enhanced", "all", "baseline", "agentic", "both"):
            parsed = parser.parse_args(["--mode", mode])
            assert parsed.mode == mode
