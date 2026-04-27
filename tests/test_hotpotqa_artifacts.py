"""Tests for HotPotQA artifact generation behavior."""

import json

import arag.data.hotpotqa as hotpotqa


def _example(example_id: str, title_prefix: str):
    return {
        "id": example_id,
        "question": f"question-{example_id}",
        "answer": f"answer-{example_id}",
        "context": {
            "title": [f"{title_prefix}-A", f"{title_prefix}-B"],
            "sentences": [["Sentence one."], ["Sentence two."]],
        },
        "supporting_facts": {
            "title": [f"{title_prefix}-A"],
            "sentences": [0],
        },
    }


def test_build_hotpotqa_artifacts_uses_broader_corpus_chunks(tmp_path, monkeypatch):
    fake_splits = {
        "train": [_example("t1", "train")],
        "validation": [_example("v1", "val")],
        "test": [_example("x1", "test")],
        "full": [],
    }

    monkeypatch.setattr(hotpotqa, "load_hotpotqa_splits", lambda seed=42: fake_splits)

    artifacts = hotpotqa.build_hotpotqa_artifacts(output_dir=str(tmp_path), seed=42)

    assert "corpus_chunks" in artifacts
    corpus_chunks = json.loads((tmp_path / "corpus_chunks.json").read_text(encoding="utf-8"))

    chunk_example_ids = {chunk["example_id"] for chunk in corpus_chunks}
    assert {"t1", "v1", "x1"}.issubset(chunk_example_ids)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["corpus_chunks"] == len(corpus_chunks)
    assert metadata["validation_chunks"] > 0
    assert metadata["test_chunks"] > 0
