#!/usr/bin/env python3
"""Batch runner for local baseline and agentic HotPotQA experiments."""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from arag import LLMClient, BaseAgent, ToolRegistry, Config, BaselineRAGRunner, EnhancedRAGRunner
from arag.data.hotpotqa import build_hotpotqa_artifacts
from arag.retrieval.faiss_store import FaissArtifactStore
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.utils.device import format_device_message

logging.basicConfig(level=logging.ERROR)


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""
    
    def __init__(
        self,
        config: Config,
        questions_file: Optional[str],
        output_dir: str,
        limit: int = None,
        num_workers: int = 1,
        mode: Optional[str] = None,
        verbose: bool = False
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.mode = mode or self.config.get("runtime.mode", "agentic")
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.data_paths = self._ensure_data_ready()
        self.questions_file = Path(questions_file) if questions_file else Path(self.data_paths["val_questions"])
        self.questions = self._load_questions()
        self._shared_tools = self._init_shared_tools() if self._run_agentic() else None
        self._llm_client = self._create_llm_client()
        self._baseline_runner = self._init_baseline_runner() if self._run_baseline() else None
        self._enhanced_runner = self._init_enhanced_runner() if self._run_enhanced() else None

        prompt_file = Path(__file__).parent.parent / "src/arag/agent/prompts/default.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        else:
            self._system_prompt = "You are a helpful assistant."

    def _run_agentic(self) -> bool:
        return self.mode in {"agentic", "both", "all"}

    def _run_baseline(self) -> bool:
        return self.mode in {"baseline", "both", "all"}

    def _run_enhanced(self) -> bool:
        return self.mode in {"enhanced", "all"}

    def _ensure_data_ready(self) -> Dict[str, str]:
        data_dir = self.config.get("data.prepared_dir", "data/hotpotqa")
        expected_paths = {
            "corpus_chunks": str(Path(data_dir) / "corpus_chunks.json"),
            "val_questions": str(Path(data_dir) / "val_questions.json"),
            "test_questions": str(Path(data_dir) / "test_questions.json"),
            "sft_train": str(Path(data_dir) / "sft_train.jsonl"),
            "sft_val": str(Path(data_dir) / "sft_val.jsonl"),
            "metadata": str(Path(data_dir) / "metadata.json"),
        }
        expected_paths["train_chunks"] = expected_paths["corpus_chunks"]
        if self.config.get("data.reuse_prepared_data", True) and all(Path(path).exists() for path in expected_paths.values()):
            return expected_paths

        return build_hotpotqa_artifacts(
            output_dir=data_dir,
            seed=self.config.get("data.split_seed", 42),
            embed_sample_limit=self.config.get("data.embed_sample_limit"),
            question_sample_limit=self.config.get("data.question_sample_limit"),
        )

    def _ensure_retrieval_artifacts(self):
        store = FaissArtifactStore(
            artifact_dir=self.config.get("retrieval.artifact_dir", "data/hotpotqa/index"),
            embedding_model=self.config.get("embedding.model", "BAAI/bge-m3"),
            device=self.config.get("embedding.device") or self.config.get("runtime.device"),
        )
        if not store.exists():
            raise FileNotFoundError(
                f"Retrieval artifacts not found in {store.artifact_dir}. Run scripts/build_index.py first."
            )
        return store
    
    def _init_shared_tools(self) -> ToolRegistry:
        """Initialize shared tools for the agentic mode."""
        chunks_file = self.config.get(
            "data.chunks_file",
            self.data_paths.get("corpus_chunks", self.data_paths["train_chunks"]),
        )
        index_dir = self.config.get("retrieval.artifact_dir", "data/hotpotqa/index")

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        index_file = Path(index_dir) / "index.faiss"
        if index_file.exists():
            tools.register(SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=self.config.get("embedding.model", "BAAI/bge-m3"),
                device=self.config.get("embedding.device") or self.config.get("runtime.device"),
            ))
        else:
            print(f"Warning: Index not found at {index_file}, semantic search disabled")

        return tools

    def _init_baseline_runner(self) -> BaselineRAGRunner:
        return BaselineRAGRunner(
            llm_client=self._llm_client,
            artifact_dir=self.config.get("retrieval.artifact_dir", "data/hotpotqa/index"),
            embedding_model=self.config.get("embedding.model", "BAAI/bge-m3"),
            device=self.config.get("runtime.device") or self.config.get("embedding.device"),
        )

    def _init_enhanced_runner(self) -> EnhancedRAGRunner:
        return EnhancedRAGRunner(
            llm_client=self._llm_client,
            artifact_dir=self.config.get("retrieval.artifact_dir", "data/hotpotqa/index"),
            embedding_model=self.config.get("embedding.model", "BAAI/bge-m3"),
            rerank_model=self.config.get("enhanced.rerank_model", "BAAI/bge-reranker-v2-m3"),
            device=self.config.get("runtime.device") or self.config.get("embedding.device"),
            max_context_chars=self.config.get("enhanced.max_context_chars", 3500),
        )
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from file."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        if self.limit:
            questions = questions[:self.limit]

        return questions

    def _create_llm_client(self) -> LLMClient:
        return LLMClient(
            model=self.config.get("llm.model"),
            temperature=self.config.get("llm.temperature", 0.0),
            max_tokens=self.config.get("llm.max_tokens", 256),
            device=self.config.get("llm.device") or self.config.get("runtime.device"),
            adapter_path=self.config.get("llm.adapter_path"),
            use_4bit=self.config.get("llm.use_4bit", False),
            torch_dtype=self.config.get("llm.torch_dtype", "auto"),
        )

    def _create_agent(self) -> BaseAgent:
        agent_config = self.config.get('agent', {})

        return BaseAgent(
            llm_client=self._llm_client,
            tools=self._shared_tools,  # Use shared tools
            system_prompt=self._system_prompt,
            max_loops=agent_config.get('max_loops', 10),
            max_token_budget=agent_config.get('max_token_budget', 128000),
            verbose=self.verbose
        )

    @staticmethod
    def _looks_structured_output(text: str) -> bool:
        if not text:
            return True
        lowered = text.lower()
        return (
            '"action"' in lowered
            or '"tool_name"' in lowered
            or 'tool result:' in lowered
            or '[forced_final]' in lowered
        )
    
    def _process_agentic(self, item: Dict[str, Any]) -> Dict[str, Any]:
        agent = self._create_agent()
        qid = item.get('qid') or item.get('id')
        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))

        try:
            result = agent.run(question)
            raw_answer = result.get('raw_answer', result.get('answer', ''))
            answer_field = result.get('answer', '')

            clean_from_raw = self._llm_client.extract_final_answer_text(raw_answer)
            clean_from_answer = self._llm_client.extract_final_answer_text(answer_field)

            candidates = [
                c.strip()
                for c in [clean_from_raw, clean_from_answer, str(answer_field), str(raw_answer)]
                if c is not None and str(c).strip()
            ]

            non_artifact = [c for c in candidates if not self._looks_structured_output(c)]
            if non_artifact:
                clean_answer = min(non_artifact, key=len)
            elif candidates:
                clean_answer = min(candidates, key=len)
            else:
                clean_answer = ""

            return {
                'mode': 'agentic',
                'qid': qid,
                'question': question,
                'trajectory': result['trajectory'],
                'gold_answer': gold_answer,
                'pred_answer': clean_answer,
                'pred_answer_raw': raw_answer,
                'total_cost': result['total_cost'],
                'loops': result['loops'],
                'total_retrieved_tokens': result.get('total_retrieved_tokens', 0),
                'retrieval_logs': result.get('retrieval_logs', []),
                'chunks_read_count': result.get('chunks_read_count', 0),
                'chunks_read_ids': result.get('chunks_read_ids', [])
            }
        except Exception as e:
            return {
                'mode': 'agentic',
                'qid': qid,
                'question': question,
                'trajectory': [],
                'gold_answer': gold_answer,
                'pred_answer': f"Error: {str(e)}",
                'pred_answer_raw': f"Error: {str(e)}",
                'total_cost': 0,
                'loops': 0,
                'total_retrieved_tokens': 0,
                'retrieval_logs': [],
                'chunks_read_count': 0,
                'chunks_read_ids': [],
                'error': str(e)
            }

    def _process_enhanced(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))

        try:
            result = self._enhanced_runner.run(
                question=question,
                n_first=self.config.get("enhanced.n_first", 30),
                k_rerank=self.config.get("enhanced.k_rerank", 5),
                use_hyde=self.config.get("enhanced.use_hyde", True),
                filter_min_score=self.config.get("enhanced.filter_min_score", float("-inf")),
                temperature=self.config.get("llm.temperature", 0.0),
                max_new_tokens=self.config.get("llm.max_tokens", 32),
            )
            return {
                "mode": "enhanced",
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": result["generated_answer"],
                "pred_answer_raw": result["raw_generated_answer"],
                "hyde_query": result.get("hyde_query"),
                "retrieved_chunks": result["retrieved_chunks"],
                "retrieved_indices": result["retrieved_indices"],
                "retrieved_titles": result.get("retrieved_titles", []),
                "retrieved_sent_ids": result.get("retrieved_sent_ids", []),
                "rerank_scores": result.get("rerank_scores", []),
                "n_first": result["n_first"],
                "k_rerank": result["k_rerank"],
                "use_hyde": result["use_hyde"],
                "temperature": result["temperature"],
            }
        except Exception as e:
            return {
                "mode": "enhanced",
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {str(e)}",
                "pred_answer_raw": f"Error: {str(e)}",
                "error": str(e),
            }

    def _process_baseline(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get('qid') or item.get('id')
        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))

        result = self._baseline_runner.run(
            question=question,
            top_k=self.config.get("retrieval.top_k", 3),
            max_new_tokens=self.config.get("llm.max_tokens", 20),
            temperature=self.config.get("llm.temperature", 0.0),
        )
        return {
            'mode': 'baseline',
            'qid': qid,
            'question': question,
            'gold_answer': gold_answer,
            'pred_answer': result['generated_answer'],
            'retrieved_chunks': result['retrieved_chunks'],
            'retrieved_indices': result['retrieved_indices'],
            'temperature': result['temperature'],
            'top_k': result['top_k'],
            'max_new_tokens': result['max_new_tokens'],
        }

    def _run_mode(self, mode_name: str, processor):
        output_file = self.output_dir / f"predictions_{mode_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as handle:
            for item in tqdm(self.questions, desc=f"Running {mode_name}"):
                prediction = processor(item)
                handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        print(f"Saved {mode_name} predictions to {output_file}")

    def run(self):
        print(format_device_message(self.config.get("runtime.device") or self.config.get("llm.device")))
        self._ensure_retrieval_artifacts()

        if self._run_baseline():
            self._run_mode("baseline", self._process_baseline)
        if self._run_enhanced():
            self._run_mode("enhanced", self._process_enhanced)
        if self._run_agentic():
            self._run_mode("agentic", self._process_agentic)


def main():
    parser = argparse.ArgumentParser(description="ARAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--questions", "-q", default=None, help="Questions file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Reserved for future use with local models")
    parser.add_argument("--mode", choices=["baseline", "agentic", "enhanced", "both", "all"], default=None, help="Execution mode: both=baseline+agentic, all=baseline+enhanced+agentic")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        mode=args.mode,
        verbose=args.verbose
    )
    
    runner.run()


if __name__ == "__main__":
    main()
