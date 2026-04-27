"""Semantic search tool - FAISS-backed similarity matching."""

from typing import Dict, Any, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool
from arag.retrieval.faiss_store import BGEEmbedder, FaissArtifactStore

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class SemanticSearchTool(BaseTool):
    """Semantic search using BGE-M3 embeddings and FAISS retrieval."""
    
    def __init__(
        self,
        chunks_file: str,
        index_dir: str = "index",
        model_name: str = "BAAI/bge-m3",
        device: str = None
    ):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")

        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = model_name
        self.device = device

        self.store = FaissArtifactStore(
            artifact_dir=index_dir,
            embedding_model=model_name,
            device=device,
        )
        self.embedding_model = BGEEmbedder(model_name=model_name, device=device)
        self._load_index()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    @property
    def name(self) -> str:
        return "semantic_search"
    
    def _load_index(self):
        self.index, self.chunk_records, self.metadata = self.store.load()
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, context: 'AgentContext', query: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)
        
        query_embedding = self.embedding_model.encode_texts([query])
        distances, indices = self.index.search(query_embedding, top_k)

        top_chunks = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunk_records):
                continue
            record = self.chunk_records[idx]
            similarity = float(1.0 / (1.0 + distances[0][rank]))
            top_chunks.append((record, similarity))

        if not top_chunks:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts = []
        matched_texts = []
        for record, similarity in top_chunks:
            snippet = record['text'][:400]
            matched_texts.append(snippet)
            result_parts.append(
                f"Chunk ID: {record['id']} (Similarity: {similarity:.3f})\nMatched: ... {snippet} ..."
            )

        tool_result = "\n\n".join(result_parts)

        retrieved_tokens = len(self.tokenizer.encode("\n".join(matched_texts))) if matched_texts else 0

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={"query": query, "chunks_found": len(top_chunks)}
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}
