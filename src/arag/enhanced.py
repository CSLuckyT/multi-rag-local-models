"""Enhanced RAG pipeline: HyDE query rewrite + FAISS + cross-encoder reranking."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arag.evaluation.metrics import clean_prediction
from arag.retrieval.faiss_store import BGEEmbedder, FaissArtifactStore

REWRITE_SYS = (
    "You are a retrieval assistant. "
    "Given a question, write one short hypothetical answer sentence "
    "that would appear in a relevant Wikipedia passage. "
    "Output only the sentence, no explanation."
)

QA_SYS = (
    "You are a question-answering assistant. "
    "Answer using only the provided context. "
    "Return only the exact short answer phrase. "
    "Do not explain. Do not add dates, years, or extra words."
)


class EnhancedRAGRunner:
    """HyDE + FAISS + cross-encoder reranker pipeline for HotPotQA.

    Mirrors the interface of BaselineRAGRunner so it can be dropped in
    alongside baseline and agentic modes in batch_runner.py.
    """

    def __init__(
        self,
        llm_client,
        artifact_dir: str,
        embedding_model: str = "BAAI/bge-m3",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_context_chars: int = 3500,
    ):
        self.llm_client = llm_client
        self.rerank_model_name = rerank_model
        self.device = device
        self.max_context_chars = max_context_chars

        self.store = FaissArtifactStore(
            artifact_dir=artifact_dir,
            embedding_model=embedding_model,
            device=device,
        )
        self.embedder = BGEEmbedder(model_name=embedding_model, device=device)

        # FAISS index + chunk records loaded lazily on first run
        self._index = None
        self._chunk_records: Optional[List[Dict[str, Any]]] = None

        # Cross-encoder loaded lazily to avoid hard dependency
        self._reranker = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _ensure_index(self):
        if self._index is None:
            self._index, self._chunk_records, _ = self.store.load()

    def _ensure_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for enhanced mode. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._reranker = CrossEncoder(self.rerank_model_name, max_length=512)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def rewrite_query(self, question: str) -> str:
        """HyDE: generate a hypothetical answer sentence for dense retrieval."""
        msgs = [
            {"role": "system", "content": REWRITE_SYS},
            {"role": "user", "content": f"Question: {question}\nHypothetical answer sentence:"},
        ]
        raw, _ = self.llm_client.generate(msgs, temperature=0.0)
        return raw.strip()

    def faiss_topn(
        self,
        question: str,
        hyde: Optional[str],
        n: int = 30,
    ) -> List[int]:
        """Dense retrieval: embed question (+ HyDE), average, search FAISS top-n."""
        self._ensure_index()
        queries = [question] + ([hyde] if hyde else [])
        qv = self.embedder.encode_texts(queries)
        qv_mean = qv.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(qv_mean, axis=1, keepdims=True) + 1e-9
        qv_mean = qv_mean / norm
        distances, indices = self._index.search(qv_mean.astype("float32"), n)
        valid = [int(i) for i in indices[0] if 0 <= i < len(self._chunk_records)]
        return valid

    def rerank(
        self,
        question: str,
        cand_idx: List[int],
        k: int,
    ) -> Tuple[List[int], List[float]]:
        """Cross-encoder reranking: score candidates, return top-k."""
        if not cand_idx:
            return [], []
        self._ensure_reranker()
        passages = [
            self._chunk_records[i].get("passage")
            or f"{self._chunk_records[i].get('title', '')}. {self._chunk_records[i]['text']}"
            for i in cand_idx
        ]
        pairs = [(question, p) for p in passages]
        scores = self._reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        scores = list(scores)
        order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:k]
        top_idx = [cand_idx[i] for i in order]
        top_scores = [float(scores[i]) for i in order]
        return top_idx, top_scores

    def filter_context(
        self,
        top_idx: List[int],
        top_scores: List[float],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """Keep chunks whose rerank score >= min_score; fallback to top-1."""
        kept = [
            self._chunk_records[i]
            for i, s in zip(top_idx, top_scores)
            if s >= min_score
        ]
        if not kept and top_idx:
            kept = [self._chunk_records[top_idx[0]]]
        return kept

    def format_context(self, chunks: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
        """Join chunks as '[title] text', respecting optional char budget."""
        limit = max_chars or self.max_context_chars
        out: List[str] = []
        total = 0
        for chunk in chunks:
            line = f"[{chunk.get('title', '')}] {chunk['text']}"
            if total + len(line) > limit:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out)

    def generate_answer(
        self,
        question: str,
        ctx_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> str:
        """Generate a short answer from context chunks."""
        ctx_text = self.format_context(ctx_chunks)
        msgs = [
            {"role": "system", "content": QA_SYS},
            {
                "role": "user",
                "content": f"Context:\n{ctx_text}\n\nQuestion: {question}\nAnswer:",
            },
        ]
        raw, _ = self.llm_client.generate(msgs, temperature=temperature)
        return raw.strip()

    # ------------------------------------------------------------------
    # End-to-end run
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        n_first: int = 30,
        k_rerank: int = 5,
        use_hyde: bool = True,
        filter_min_score: float = float("-inf"),
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """Run the full Enhanced RAG pipeline for a single question.

        Returns a dict with fields parallel to BaselineRAGRunner.run() plus
        enhanced-specific metadata (hyde_query, rerank_scores, etc.).
        """
        hyde = self.rewrite_query(question) if use_hyde else None
        cand_idx = self.faiss_topn(question, hyde, n=n_first)
        top_idx, top_scores = self.rerank(question, cand_idx, k=k_rerank)
        ctx_chunks = self.filter_context(top_idx, top_scores, min_score=filter_min_score)
        raw_answer = self.generate_answer(
            question, ctx_chunks, temperature=temperature, max_new_tokens=max_new_tokens
        )

        return {
            "question": question,
            "generated_answer": clean_prediction(raw_answer),
            "raw_generated_answer": raw_answer,
            "retrieved_chunks": [c["text"] for c in ctx_chunks],
            "retrieved_indices": [c.get("chunk_id", i) for i, c in enumerate(ctx_chunks)],
            "retrieved_titles": [c.get("title", "") for c in ctx_chunks],
            "retrieved_sent_ids": [c.get("sent_id") for c in ctx_chunks],
            "hyde_query": hyde,
            "rerank_scores": top_scores,
            "n_first": n_first,
            "k_rerank": k_rerank,
            "use_hyde": use_hyde,
            "filter_min_score": filter_min_score,
            "temperature": temperature,
            "top_k": k_rerank,
        }
