"""Minimal smoke tests - verify package imports work correctly."""


def test_import_arag():
    import arag
    assert hasattr(arag, "__version__")
    assert hasattr(arag, "BaselineRAGRunner")


def test_import_core():
    from arag.core.config import Config
    from arag.core.context import AgentContext
    from arag.core.llm import LLMClient

    assert Config is not None
    assert AgentContext is not None
    assert LLMClient is not None


def test_import_new_modules():
    from arag.baseline import BaselineRAGRunner
    from arag.data.hotpotqa import build_document_chunks, normalize_sample_limit
    from arag.evaluation.metrics import clean_prediction, compute_em, compute_f1
    from arag.retrieval.faiss_store import FaissArtifactStore
    from arag.utils.device import format_device_message, resolve_device

    assert BaselineRAGRunner is not None
    assert build_document_chunks is not None
    assert normalize_sample_limit("All") is None
    assert clean_prediction("answer\nmore") == "answer"
    assert compute_em("The Beatles", "beatles") == 1
    assert compute_f1("new york city", "new york") > 0
    assert FaissArtifactStore is not None
    assert format_device_message("cpu") == "Using device: cpu"
    assert resolve_device("cpu") == "cpu"


def test_import_tools():
    from arag.tools.base import BaseTool
    from arag.tools.registry import ToolRegistry

    assert BaseTool is not None
    assert ToolRegistry is not None


def test_import_agent():
    from arag.agent.base import BaseAgent
    assert BaseAgent is not None


def test_context_basic():
    from arag.core.context import AgentContext

    ctx = AgentContext()
    assert ctx.total_retrieved_tokens == 0

    ctx.mark_chunk_as_read("42")
    assert ctx.is_chunk_read("42")
    assert not ctx.is_chunk_read("99")

    ctx.add_retrieval_log("test_tool", tokens=100)
    assert ctx.total_retrieved_tokens == 100

    summary = ctx.get_summary()
    assert summary["chunks_read_count"] == 1


def test_tool_registry():
    from arag.tools.registry import ToolRegistry

    registry = ToolRegistry()
    assert registry.list_tools() == []

    result, log = registry.execute("nonexistent", None)
    assert "Error" in result


def test_config_basic():
    from arag.core.config import Config

    cfg = Config({"llm": {"model": "test", "temperature": 0.5}})
    assert cfg.get("llm.model") == "test"
    assert cfg.get("llm.temperature") == 0.5
    assert cfg.get("llm.missing", "default") == "default"


def test_build_document_chunks_shape():
    from arag.data.hotpotqa import build_document_chunks

    example = {
        "id": "sample-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "context": {
            "title": ["Hamlet"],
            "sentences": [["Hamlet is a tragedy", "It was written by William Shakespeare"]],
        },
        "supporting_facts": {"title": ["Hamlet"], "sent_id": [1]},
    }

    processed = build_document_chunks(example)
    assert processed["id"] == "sample-1"
    assert len(processed["chunks"]) == 1
    assert processed["chunks"][0]["id"] == "sample-1::0"
    assert "William Shakespeare" in processed["chunks"][0]["text"]


def test_local_llm_tool_parsing_without_model_load():
    from arag.core.llm import LLMClient

    client = LLMClient(model="Qwen/Qwen3-4B-Instruct-2507", device="cpu")
    parsed = client._extract_json_payload('{"action":"tool","tool_name":"keyword_search","arguments":{"keywords":["beatles"]}}')
    normalized = client._normalize_tool_response('{"action":"tool","tool_name":"keyword_search","arguments":{"keywords":["beatles"]}}')

    assert parsed["action"] == "tool"
    assert normalized["tool_calls"][0]["function"]["name"] == "keyword_search"
