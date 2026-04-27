"""Regression tests for agentic answer extraction and sanitization."""

import importlib.util
from pathlib import Path

from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient


class _MockToolRegistry:
    def get_all_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "keyword_search",
                    "description": "Search chunks",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def execute(self, func_name, context, **kwargs):
        del context
        del kwargs
        return f"tool_result_for_{func_name}", {"retrieved_tokens": 3}


class _MockLLMMixedTranscript(LLMClient):
    def __init__(self):
        super().__init__(model="mock", device="cpu")
        self._calls = 0

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        del messages
        del temperature
        del max_tokens

        self._calls += 1
        if tools and self._calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        'Assistant: {"action":"tool","tool_name":"keyword_search","arguments":{}}\n'
                        '{"action":"final","answer":"William Shakespeare"}'
                    ),
                },
                "cost": 0.0,
            }

        return {
            "message": {
                "role": "assistant",
                "content": '{"action":"final","answer":"William Shakespeare"}',
            },
            "cost": 0.0,
        }


class _MockLLMNumericFinalAnswer(LLMClient):
    def __init__(self):
        super().__init__(model="mock", device="cpu")

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        del messages
        del tools
        del temperature
        del max_tokens
        return {
            "message": {
                "role": "assistant",
                "content": '{"action":"final","answer":1972}',
            },
            "cost": 0.0,
        }


class _MockLLMMalformedToolArgs(LLMClient):
    def __init__(self):
        super().__init__(model="mock", device="cpu")
        self._calls = 0

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        del messages
        del temperature
        del max_tokens
        self._calls += 1

        if tools and self._calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"action":"tool","tool_name":"read_chunk",'
                        '"arguments":"{\\"chunk_ids\\":\\"abc::1\\"}"}'
                    ),
                },
                "cost": 0.0,
            }

        return {
            "message": {
                "role": "assistant",
                "content": '{"action":"final","answer":"ok"}',
            },
            "cost": 0.0,
        }


class _MockLLMToolOnlyThenForcedFinal(LLMClient):
    def __init__(self):
        super().__init__(model="mock", device="cpu")
        self._calls = 0

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        del messages
        del temperature
        del max_tokens

        self._calls += 1
        if tools and self._calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"action":"tool","tool_name":"keyword_search","arguments":{}}',
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "keyword_search", "arguments": "{}"},
                        }
                    ],
                },
                "cost": 0.0,
            }

        if tools:
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"action":"tool","tool_name":"keyword_search","arguments":{}}',
                },
                "cost": 0.0,
            }

        return {
            "message": {
                "role": "assistant",
                "content": '{"action":"final","answer":"forced final answer"}',
            },
            "cost": 0.0,
        }


def test_llm_extract_agent_action_payload_single_tool_object():
    client = LLMClient(model="mock", device="cpu")
    payload = client.extract_agent_action_payload(
        '{"action":"tool","tool_name":"keyword_search","arguments":{"k":"v"}}'
    )
    assert payload["action"] == "tool"
    assert payload["tool_name"] == "keyword_search"


def test_llm_extract_agent_action_payload_direct_tool_action_object():
    client = LLMClient(model="mock", device="cpu")
    payload = client.extract_agent_action_payload(
        '{"action":"read_chunk","arguments":{"chunk_ids":["abc::1"]}}'
    )
    assert payload["action"] == "tool"
    assert payload["tool_name"] == "read_chunk"
    assert payload["arguments"] == {"chunk_ids": ["abc::1"]}


def test_llm_extract_agent_action_payload_single_final_object():
    client = LLMClient(model="mock", device="cpu")
    payload = client.extract_agent_action_payload('{"action":"final","answer":"Paris"}')
    assert payload["action"] == "final"
    assert payload["answer"] == "Paris"


def test_llm_mixed_transcript_agent_and_final_extractors_diverge_as_expected():
    client = LLMClient(model="mock", device="cpu")
    text = (
        "Human: question\n"
        "Assistant: {\"action\":\"tool\",\"tool_name\":\"keyword_search\",\"arguments\":{}}\n"
        "Assistant: {\"action\":\"final\",\"answer\":\"Marie Curie\"}\n"
    )

    agent_payload = client.extract_agent_action_payload(text)
    assert agent_payload["action"] == "tool"
    assert agent_payload["tool_name"] == "keyword_search"

    assert client.extract_final_answer_text(text) == "Marie Curie"


def test_llm_extract_agent_action_payload_bad_json_fallbacks_to_final_plain_text():
    client = LLMClient(model="mock", device="cpu")
    payload = client.extract_agent_action_payload("{\"action\": \"tool\" this is broken json")
    assert payload["action"] == "final"
    assert "broken json" in payload["answer"]


def test_llm_extract_final_answer_text_handles_non_string_final_answer():
    client = LLMClient(model="mock", device="cpu")
    assert client.extract_final_answer_text('{"action":"final","answer":1972}') == "1972"


def test_llm_extract_final_answer_text_salvages_forced_final_malformed_blob():
    client = LLMClient(model="mock", device="cpu")
    raw = (
        '{"action":"tool","tool_name":"read_chunk","arguments":{"chunk_ids":["x"]}}\n\n'
        "[forced_final]\n"
        '{"action":"final","answer":"Danish nationality"}'
    )
    assert client.extract_final_answer_text(raw) == "Danish nationality"


def test_agent_run_sanitizes_final_answer_from_mixed_transcript():
    agent = BaseAgent(
        llm_client=_MockLLMMixedTranscript(),
        tools=_MockToolRegistry(),
        max_loops=3,
        verbose=False,
    )

    result = agent.run("Who wrote Hamlet?")

    assert result["answer"] == "William Shakespeare"
    assert len(result["trajectory"]) > 0
    assert "tool_name" not in result["answer"]
    assert "action" in result["raw_answer"]


def test_agent_run_forces_final_when_tool_json_has_no_final():
    agent = BaseAgent(
        llm_client=_MockLLMToolOnlyThenForcedFinal(),
        tools=_MockToolRegistry(),
        max_loops=3,
        verbose=False,
    )

    result = agent.run("Who wrote Hamlet?")

    assert result["answer"] == "forced final answer"
    assert "forced final answer" in result["raw_answer"]


def test_agent_run_normalizes_malformed_tool_arguments_and_completes():
    class _ReadChunkRegistry(_MockToolRegistry):
        def get_all_schemas(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_chunk",
                        "description": "Read chunk",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def execute(self, func_name, context, **kwargs):
            del context
            assert func_name == "read_chunk"
            assert kwargs.get("chunk_ids") == ["abc::1"]
            return "read ok", {"retrieved_tokens": 1}

    agent = BaseAgent(
        llm_client=_MockLLMMalformedToolArgs(),
        tools=_ReadChunkRegistry(),
        max_loops=3,
        verbose=False,
    )

    result = agent.run("test")
    assert result["answer"] == "ok"
    assert len(result["trajectory"]) > 0


def test_batch_runner_agentic_prediction_keeps_clean_answer_and_raw_trace():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "batch_runner.py"
    spec = importlib.util.spec_from_file_location("batch_runner_script", script_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BatchRunner = module.BatchRunner

    class _FakeAgent:
        def run(self, question):
            del question
            return {
                "answer": "Mount Everest",
                "raw_answer": '{"action":"final","answer":"Mount Everest"}',
                "trajectory": [],
                "total_cost": 0.0,
                "loops": 1,
            }

    runner = BatchRunner.__new__(BatchRunner)
    runner._create_agent = lambda: _FakeAgent()
    runner._llm_client = LLMClient(model="mock", device="cpu")

    item = {"qid": "q1", "question": "Highest mountain?", "answer": "Mount Everest"}
    prediction = runner._process_agentic(item)

    assert prediction["pred_answer"] == "Mount Everest"
    assert "action" not in prediction["pred_answer"]
    assert "action" in prediction["pred_answer_raw"]


def test_batch_runner_prefers_non_structured_candidate_for_pred_answer():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "batch_runner.py"
    spec = importlib.util.spec_from_file_location("batch_runner_script", script_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BatchRunner = module.BatchRunner

    class _FakeAgent:
        def run(self, question):
            del question
            return {
                "answer": '{"action":"final","answer":"Long sentence answer"}',
                "raw_answer": '{"action":"tool","tool_name":"read_chunk"}\n\n[forced_final]\n{"action":"final","answer":"Short answer"}',
                "trajectory": [],
                "total_cost": 0.0,
                "loops": 1,
            }

    runner = BatchRunner.__new__(BatchRunner)
    runner._create_agent = lambda: _FakeAgent()
    runner._llm_client = LLMClient(model="mock", device="cpu")

    item = {"qid": "q2", "question": "test?", "answer": "Short answer"}
    prediction = runner._process_agentic(item)
    assert prediction["pred_answer"] == "Short answer"


def test_eval_fallback_handles_numeric_answer_without_strip_errors():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_script", script_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    extracted = module.extract_final_answer_fallback('{"action":"final","answer":1972}')
    assert extracted == "1972"
