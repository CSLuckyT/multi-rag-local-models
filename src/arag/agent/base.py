"""Base agent implementation for ARAG."""

import json
from typing import Any, Dict, List, Optional

import tiktoken

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry


class BaseAgent:
    """Base agent with tool calling capabilities."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total
    
    def _force_final_answer(self, messages: List[Dict[str, Any]], context: AgentContext,
                           total_cost: float, reason: str) -> tuple:
        """Force the model to give a final answer when limits are reached."""
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. "
            "Return only the shortest direct answer span (entity/date/number/yes-no), with no explanation. "
            "Do not include reasoning, uncertainty statements, or prefaces like 'The answer is'."
        )
        
        messages.append({"role": "user", "content": force_prompt})
        
        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            total_cost += response["cost"]
            raw_answer = response["message"].get("content", "")
            final_answer = self.llm.extract_final_answer_text(raw_answer)
            if not final_answer:
                final_answer = str(raw_answer).strip()
            
            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
        except Exception as e:
            if self.verbose:
                print(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."
            raw_answer = final_answer
        
        return final_answer, raw_answer, total_cost

    @staticmethod
    def _looks_like_tool_payload(text: Optional[str]) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return (
            '"action"' in lowered
            or '"tool_name"' in lowered
            or 'tool result:' in lowered
            or lowered.lstrip().startswith("{")
        )
    
    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        
        trajectory = []
        total_cost = 0.0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            print(f"{'='*60}\n")
        
        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1
            
            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                if self.verbose:
                    print(f"Token budget exceeded ({current_tokens} > {self.max_token_budget}), forcing answer...")
                
                final_answer, raw_answer, total_cost = self._force_final_answer(
                    messages, context, total_cost, "Token budget exceeded"
                )
                
                return {
                    "answer": final_answer,
                    "raw_answer": raw_answer,
                    "trajectory": trajectory,
                    "total_cost": total_cost,
                    "loops": loop_count,
                    "token_budget_exceeded": True,
                    **context.get_summary()
                }
            
            if self.verbose:
                print(f"Loop {loop_count}/{self.max_loops} (Tokens: {current_tokens}/{self.max_token_budget})")
            
            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                break
            
            total_cost += response["cost"]
            message = response["message"]
            messages.append(message)
            
            if self.verbose and message.get("content"):
                print(f"Assistant: {message['content'][:200]}...")
            
            tool_calls = message.get("tool_calls")
            if not tool_calls and message.get("content"):
                payload = self.llm.extract_agent_action_payload(message.get("content", ""))
                if payload.get("action") == "tool" and payload.get("tool_name"):
                    payload_args = payload.get("arguments", {})
                    if isinstance(payload_args, str):
                        try:
                            payload_args = json.loads(payload_args)
                        except json.JSONDecodeError:
                            payload_args = {}
                    if not isinstance(payload_args, dict):
                        payload_args = {}
                    tool_calls = [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": payload.get("tool_name"),
                                "arguments": json.dumps(payload_args, ensure_ascii=False),
                            },
                        }
                    ]

            if not tool_calls:
                # No tool calls - agent is done
                raw_answer = message.get("content", "")
                final_answer = self.llm.extract_final_answer_text(raw_answer)

                if final_answer is None or self._looks_like_tool_payload(final_answer):
                    if self.verbose:
                        print("Detected tool-only JSON without final answer, forcing one final synthesis...")
                    final_answer, forced_raw_answer, total_cost = self._force_final_answer(
                        messages, context, total_cost, "Tool-only response without final answer"
                    )
                    raw_answer = f"{raw_answer}\n\n[forced_final]\n{forced_raw_answer}".strip()

                return {
                    "answer": final_answer,
                    "raw_answer": raw_answer,
                    "trajectory": trajectory,
                    "total_cost": total_cost,
                    "loops": loop_count,
                    **context.get_summary()
                }
            
            # Execute tool calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        func_args = {}
                if not isinstance(func_args, dict):
                    func_args = {}

                chunk_ids = func_args.get("chunk_ids")
                if chunk_ids is not None and not isinstance(chunk_ids, list):
                    func_args["chunk_ids"] = [str(chunk_ids)]
                elif isinstance(chunk_ids, list):
                    func_args["chunk_ids"] = [str(cid) for cid in chunk_ids]
                
                if self.verbose:
                    print(f"Tool: {func_name}")
                    print(f"  Args: {func_args}")
                
                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}
                
                if self.verbose:
                    output_preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    print(f"  Result: {output_preview}")
                    if tool_log.get("retrieved_tokens", 0) > 0:
                        print(f"  Tokens: {tool_log['retrieved_tokens']}")
                    print()
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
                
                # Record trajectory (same format as original A-RAG)
                traj_entry = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log  # includes retrieved_tokens, chunks_found, etc.
                }
                trajectory.append(traj_entry)
        
        # Max loops reached - force final answer
        if self.verbose:
            print(f"Max loops reached ({self.max_loops}), forcing answer...")
        
        final_answer, raw_answer, total_cost = self._force_final_answer(
            messages, context, total_cost, "Maximum loops exceeded"
        )
        
        return {
            "answer": final_answer,
            "raw_answer": raw_answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loop_count,
            "max_loops_exceeded": True,
            **context.get_summary()
        }
