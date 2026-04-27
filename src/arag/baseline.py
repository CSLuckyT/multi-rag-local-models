"""Baseline RAG execution flow for side-by-side comparison with agentic RAG."""

from __future__ import annotations

from typing import Any, Dict, Optional

from arag.evaluation.metrics import clean_prediction
from arag.retrieval.faiss_store import BGEEmbedder, FaissArtifactStore


class BaselineRAGRunner:
    """Notebook-style retrieve-then-generate baseline RAG runner."""

    def __init__(
        self,
        llm_client,
        artifact_dir: str,
        embedding_model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.store = FaissArtifactStore(
            artifact_dir=artifact_dir,
            embedding_model=embedding_model,
            device=device,
        )
        self.embedder = BGEEmbedder(model_name=embedding_model, device=device)

    def run(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        results = self.store.search(question, top_k=top_k, embedder=self.embedder)
        retrieved_chunks = [item["text"] for item in results]
        retrieved_context = "\n\n".join(retrieved_chunks)

        prompt = f"""Answer the question using only the retrieved context.

Return only the exact short answer phrase.
Do not explain.
Do not add dates, years, or extra words.

Context:
{retrieved_context}

Question:
{question}

Short answer:
"""

        generated_answer, _ = self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_indices": [item["id"] for item in results],
            "generated_answer": clean_prediction(generated_answer),
            "raw_generated_answer": generated_answer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }