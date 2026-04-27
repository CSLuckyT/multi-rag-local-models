"""FAISS-backed retrieval artifacts for baseline and agentic RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:  # pragma: no cover - import-only environments
    faiss = None
    HAS_FAISS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

from arag.utils.device import resolve_device


class BGEEmbedder:
    """Notebook-style BGE-M3 encoder with mean pooling and normalization."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("torch and transformers are required for embeddings.")

        self.model_name = model_name
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_texts(self, texts: Sequence[str], max_length: int = 512) -> np.ndarray:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype("float32")


class FaissArtifactStore:
    """Persistent retrieval artifact manager with reuse checks."""

    def __init__(
        self,
        artifact_dir: str,
        embedding_model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.embedding_model = embedding_model
        self.device = resolve_device(device)
        self.index_path = self.artifact_dir / "index.faiss"
        self.embeddings_path = self.artifact_dir / "embeddings.npy"
        self.chunk_records_path = self.artifact_dir / "chunk_records.json"
        self.metadata_path = self.artifact_dir / "index_metadata.json"

    def exists(self) -> bool:
        return all(
            path.exists()
            for path in [self.index_path, self.embeddings_path, self.chunk_records_path, self.metadata_path]
        )

    def load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def matches(self, expected: Dict[str, Any]) -> bool:
        metadata = self.load_metadata()
        return all(metadata.get(key) == value for key, value in expected.items())

    def build(
        self,
        chunk_records: List[Dict[str, Any]],
        batch_size: int = 32,
        expected_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if not HAS_FAISS:
            raise ImportError("faiss is required to build retrieval artifacts.")

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        embedder = BGEEmbedder(model_name=self.embedding_model, device=self.device)
        chunk_texts = [record["text"] for record in chunk_records]
        embeddings_batches: List[np.ndarray] = []

        for start in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[start:start + batch_size]
            embeddings_batches.append(embedder.encode_texts(batch))

        embeddings = np.vstack(embeddings_batches) if embeddings_batches else np.empty((0, 1024), dtype="float32")
        if len(embeddings) == 0:
            raise ValueError("No embeddings were generated from the chunk records.")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        np.save(self.embeddings_path, embeddings)
        self.chunk_records_path.write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = {
            "embedding_model": self.embedding_model,
            "device": self.device,
            "chunk_count": len(chunk_records),
        }
        if expected_metadata:
            metadata.update(expected_metadata)
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "index": str(self.index_path),
            "embeddings": str(self.embeddings_path),
            "chunk_records": str(self.chunk_records_path),
            "metadata": str(self.metadata_path),
        }

    def load(self) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
        if not self.exists():
            raise FileNotFoundError(f"Retrieval artifacts not found in {self.artifact_dir}")
        index = faiss.read_index(str(self.index_path))
        chunk_records = json.loads(self.chunk_records_path.read_text(encoding="utf-8"))
        metadata = self.load_metadata()
        return index, chunk_records, metadata

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedder: Optional[BGEEmbedder] = None,
    ) -> List[Dict[str, Any]]:
        index, chunk_records, _ = self.load()
        active_embedder = embedder or BGEEmbedder(model_name=self.embedding_model, device=self.device)
        query_embedding = active_embedder.encode_texts([query])
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunk_records):
                continue
            record = dict(chunk_records[idx])
            record["distance"] = float(distances[0][rank])
            record["rank"] = rank + 1
            results.append(record)
        return results