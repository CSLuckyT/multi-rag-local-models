"""Local HuggingFace-backed LLM client for ARAG."""

from __future__ import annotations

import json
import os
import re
import threading
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - import-only environments
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - import-only environments
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None
    HAS_PEFT = False

from arag.utils.device import resolve_device


TOOL_CALL_SYSTEM_PROMPT = """You are an agentic retrieval model.

You must respond with exactly one JSON object and no extra text.

If you need to use a tool, output:
{"action": "tool", "tool_name": "<tool name>", "arguments": { ... }}

If you can answer the question, output:
{"action": "final", "answer": "<final answer>"}

Rules:
- Use only tool names that appear in the tool list.
- Arguments must be valid JSON objects.
- Do not wrap the JSON in markdown code fences.
- If prior tool results are sufficient, return a final answer.
- For action=final, answer must be a concise span only (entity/date/number/yes-no), with no explanation.
"""


class LLMClient:
    """Unified local LLM interface for agentic and baseline RAG."""

    _cache_lock = threading.Lock()
    _model_cache: Dict[Tuple[str, str, Optional[str], bool], Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    _known_tool_actions = {"keyword_search", "semantic_search", "read_chunk"}

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        reasoning_effort: str = None,
        device: Optional[str] = None,
        adapter_path: Optional[str] = None,
        use_4bit: bool = False,
        torch_dtype: Optional[str] = None,
    ):
        del api_key
        del base_url
        del reasoning_effort

        self.model = model or os.getenv("ARAG_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = resolve_device(device or os.getenv("ARAG_DEVICE"))
        self.adapter_path = adapter_path or os.getenv("ARAG_ADAPTER_PATH")
        self.use_4bit = use_4bit or os.getenv("ARAG_USE_4BIT", "false").lower() == "true"
        self.torch_dtype = torch_dtype or os.getenv("ARAG_TORCH_DTYPE", "auto")

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer, self._model = self._load_model_bundle()
        return self._tokenizer

    @property
    def model_instance(self):
        if self._model is None:
            self._tokenizer, self._model = self._load_model_bundle()
        return self._model

    def _resolve_torch_dtype(self):
        if not HAS_TORCH:
            return None
        if self.torch_dtype == "auto":
            return torch.float16 if self.device.startswith("cuda") else torch.float32
        return getattr(torch, self.torch_dtype, torch.float16)

    def _load_model_bundle(self):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for local model execution. Install project dependencies first."
            )

        resolved_adapter_path = None
        if self.adapter_path:
            resolved_adapter_path = os.path.abspath(os.path.expanduser(self.adapter_path))
            if not os.path.isdir(resolved_adapter_path):
                raise FileNotFoundError(
                    f"Adapter path does not exist: {self.adapter_path}"
                )

        cache_key = (self.model, self.device, resolved_adapter_path, self.use_4bit)
        with self._cache_lock:
            if cache_key in self._model_cache and self.model in self._tokenizer_cache:
                return self._tokenizer_cache[self.model], self._model_cache[cache_key]

            tokenizer = AutoTokenizer.from_pretrained(self.model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs: Dict[str, Any] = {
                "torch_dtype": self._resolve_torch_dtype(),
            }

            if self.use_4bit and BitsAndBytesConfig is not None:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs["device_map"] = "auto"
            elif self.device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)

            if resolved_adapter_path:
                if not HAS_PEFT:
                    raise ImportError("peft is required to load adapter checkpoints.")
                model = PeftModel.from_pretrained(model, resolved_adapter_path)

            if HAS_TORCH and not self.use_4bit and not self.device.startswith("cuda"):
                model = model.to(self.device)

            model.eval()

            self._tokenizer_cache[self.model] = tokenizer
            self._model_cache[cache_key] = model
            return tokenizer, model

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            if "tool_calls" in msg:
                total += self.count_tokens(json.dumps(msg["tool_calls"]))
        return total

    def calculate_cost(self, usage: dict) -> float:
        del usage
        return 0.0

    def _build_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
        parts: List[str] = []
        if tools:
            parts.append(TOOL_CALL_SYSTEM_PROMPT)
            parts.append("Available tools:")
            for tool in tools:
                function = tool.get("function", {})
                parts.append(
                    f"- {function.get('name')}: {function.get('description', '').strip()}"
                )
                parameters = function.get("parameters")
                if parameters:
                    parts.append(json.dumps(parameters, ensure_ascii=False))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                parts.append(f"TOOL RESULT:\n{content}")
            else:
                parts.append(f"{role.upper()}:\n{content}")

        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    def _extract_json_payload(self, text: str) -> Dict[str, Any]:
        """Backward-compatible final-answer-oriented payload extraction."""
        return self.extract_final_answer_payload(text)

    def _strip_for_json_parsing(self, text: Any) -> str:
        stripped = "" if text is None else str(text)
        stripped = stripped.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        # Drop chat transcript labels that commonly appear in model output.
        stripped = re.sub(r"(?im)^\s*(human|assistant|user|system)\s*:\s*", "", stripped)
        return stripped.strip()

    def extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON objects from free-form text in left-to-right order."""
        source = self._strip_for_json_parsing(text)
        decoder = JSONDecoder()
        objects: List[Dict[str, Any]] = []

        for idx, char in enumerate(source):
            if char != "{":
                continue
            try:
                decoded, _ = decoder.raw_decode(source[idx:])
            except JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                objects.append(decoded)

        return objects

    def _coerce_tool_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Coerce variant tool-call JSON formats into canonical action=tool payloads."""
        action = payload.get("action")
        tool_name: Optional[str] = None

        if isinstance(action, str):
            normalized_action = action.strip()
            if normalized_action == "tool":
                raw_tool_name = payload.get("tool_name")
                tool_name = raw_tool_name.strip() if isinstance(raw_tool_name, str) else None
            elif normalized_action in self._known_tool_actions:
                tool_name = normalized_action

        if not tool_name:
            raw_tool_name = payload.get("tool_name")
            if isinstance(raw_tool_name, str) and raw_tool_name.strip():
                tool_name = raw_tool_name.strip()

        if not tool_name:
            return None

        arguments = payload.get("arguments", payload.get("args", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        return {
            "action": "tool",
            "tool_name": tool_name,
            "arguments": arguments,
        }

    def extract_agent_action_payload(self, text: str) -> Dict[str, Any]:
        """Extract one actionable payload for agent control flow."""
        stripped = self._strip_for_json_parsing(text)

        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                tool_payload = self._coerce_tool_payload(decoded)
                if tool_payload is not None:
                    return tool_payload
                return decoded
        except JSONDecodeError:
            pass

        objects = self.extract_json_objects(stripped)
        if objects:
            for obj in objects:
                tool_payload = self._coerce_tool_payload(obj)
                if tool_payload is not None:
                    return tool_payload
            for obj in objects:
                if obj.get("action") == "final":
                    return obj

        return {"action": "final", "answer": stripped}

    def extract_final_answer_payload(self, text: str) -> Dict[str, Any]:
        """Extract payload for answer cleanup/evaluation from mixed outputs."""
        stripped = self._strip_for_json_parsing(text)

        # If a forced-final segment exists, prioritize parsing only the forced suffix.
        if "[forced_final]" in stripped:
            forced_suffix = stripped.split("[forced_final]", 1)[1].strip()
            if forced_suffix:
                forced_payload = self.extract_final_answer_payload(forced_suffix)
                if forced_payload.get("action") == "final":
                    return forced_payload

        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                return decoded
        except JSONDecodeError:
            pass

        objects = self.extract_json_objects(stripped)
        if objects:
            for obj in objects:
                if obj.get("action") == "final" and isinstance(obj.get("answer"), str):
                    answer = obj.get("answer", "").strip()
                    if answer:
                        return obj
            for obj in objects:
                if obj.get("action") == "final":
                    return obj
            return objects[-1]

        # Heuristic salvage for malformed JSON-like final action blocks.
        marker = re.search(r'"action"\s*:\s*"final"', stripped, flags=re.IGNORECASE)
        if marker:
            tail = stripped[marker.start():]
            answer_match = re.search(r'"answer"\s*:\s*(.*)', tail, flags=re.IGNORECASE | re.DOTALL)
            if answer_match:
                raw_answer = answer_match.group(1).strip()
                # Trim obvious trailing wrappers.
                raw_answer = re.sub(r'\}\s*$', "", raw_answer).strip()
                raw_answer = re.sub(r'\n\s*\[forced_final\].*$', "", raw_answer, flags=re.DOTALL).strip()
                raw_answer = raw_answer.strip()
                if raw_answer.startswith('"'):
                    raw_answer = raw_answer[1:]
                if raw_answer.endswith('"'):
                    raw_answer = raw_answer[:-1]
                raw_answer = raw_answer.replace('\\"', '"').strip()
                if raw_answer:
                    return {"action": "final", "answer": raw_answer}

        return {"action": "final", "answer": stripped}

    def extract_final_answer_text(self, text: str) -> Optional[str]:
        """Extract the final answer text from JSON-style tool transcripts."""
        payload = self.extract_final_answer_payload(text)
        if payload.get("action") == "final":
            answer = payload.get("answer")
            if isinstance(answer, str):
                return answer.strip()
            if answer is not None:
                return str(answer).strip()

        # If we can parse JSON objects but none are final, signal caller to continue reasoning.
        objects = self.extract_json_objects(text)
        has_tool_json = any(obj.get("action") == "tool" for obj in objects)
        if has_tool_json and not any(obj.get("action") == "final" for obj in objects):
            return None

        stripped = self._strip_for_json_parsing(text)
        if re.search(r'"action"\s*:\s*"tool"|"tool_name"\s*:', stripped):
            return None
        return stripped

    def _generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        tokenizer = self.tokenizer
        model = self.model_instance
        active_temperature = self.temperature if temperature is None else temperature
        active_max_tokens = max_tokens or self.max_tokens

        if not HAS_TORCH:
            raise ImportError("torch is required for local model execution.")

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": active_max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if active_temperature == 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = active_temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        input_tokens = int(inputs["input_ids"].shape[1])
        output_tokens = int(generated_ids.shape[0])
        return text, input_tokens, output_tokens

    def _normalize_tool_response(self, raw_text: str) -> Dict[str, Any]:
        payload = self.extract_agent_action_payload(raw_text)
        action = payload.get("action", "final")

        if action == "tool":
            tool_name = payload.get("tool_name", "")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            return {
                "role": "assistant",
                "content": raw_text,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }

        answer = payload.get("answer") or raw_text
        return {"role": "assistant", "content": answer}

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(messages, tools)
        raw_text, input_tokens, output_tokens = self._generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = self._normalize_tool_response(raw_text) if tools else {"role": "assistant", "content": raw_text}
        return {
            "message": message,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": 0.0,
            "raw_response": {"text": raw_text, "device": self.device},
        }

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs,
    ) -> tuple:
        """Generate response (compatibility method for eval and baseline flows)."""
        del kwargs
        if system:
            messages = [{"role": "system", "content": system}] + messages

        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        return content, result["cost"]
