#!/usr/bin/env python3
"""Grid-search hyperparameter tuning for baseline, agentic, and enhanced HotPotQA runs."""

import argparse
from pathlib import Path

import pandas as pd

from arag import Config
import batch_runner as batch_runner_module
import eval as eval_module


def _run_baseline_agentic_grid(config: Config, tuning_dir: Path, num_samples: int) -> list:
    """Grid search over top_k / temperature / max_new_tokens for baseline and agentic modes."""
    results = []
    modes = [m for m in config.get("tuning.modes", ["baseline", "agentic"]) if m != "enhanced"]
    top_k_values = config.get("tuning.top_k_values", [1, 3, 5])
    temperature_values = config.get("tuning.temperature_values", [0.0, 0.1, 0.3])
    max_new_tokens_values = config.get("tuning.max_new_tokens_values", [20, 50, 80])

    for mode in modes:
        for top_k in top_k_values:
            for temperature in temperature_values:
                for max_new_tokens in max_new_tokens_values:
                    config.set("runtime.mode", mode)
                    config.set("retrieval.top_k", top_k)
                    config.set("llm.temperature", temperature)
                    config.set("llm.max_tokens", max_new_tokens)
                    run_dir = tuning_dir / f"{mode}_k{top_k}_t{temperature}_m{max_new_tokens}"
                    runner = batch_runner_module.BatchRunner(
                        config=config,
                        questions_file=config.get("data.validation_questions_file"),
                        output_dir=str(run_dir),
                        limit=num_samples,
                        mode=mode,
                        verbose=False,
                    )
                    runner.run()
                    prediction_file = run_dir / f"predictions_{mode}.jsonl"
                    summary_df = eval_module.evaluate_predictions(str(prediction_file), output_dir=str(run_dir))
                    summary_record = summary_df.to_dict(orient="records")[0]
                    summary_record.update(
                        {
                            "mode": mode,
                            "top_k": top_k,
                            "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                            "n_first": None,
                            "k_rerank": None,
                            "use_hyde": None,
                            "filter_min_score": None,
                        }
                    )
                    results.append(summary_record)
    return results


def _run_enhanced_grid(config: Config, tuning_dir: Path, num_samples: int) -> list:
    """Grid search over n_first / k_rerank / use_hyde / temperature for enhanced mode."""
    results = []
    n_first_values = config.get("tuning.enhanced_n_first_values", [20, 30])
    k_rerank_values = config.get("tuning.enhanced_k_rerank_values", [3, 5, 8])
    use_hyde_values = config.get("tuning.enhanced_use_hyde_values", [True, False])
    temperature_values = config.get("tuning.temperature_values", [0.0, 0.1])
    filter_min_score = config.get("enhanced.filter_min_score", float("-inf"))

    for n_first in n_first_values:
        for k_rerank in k_rerank_values:
            for use_hyde in use_hyde_values:
                for temperature in temperature_values:
                    config.set("enhanced.n_first", n_first)
                    config.set("enhanced.k_rerank", k_rerank)
                    config.set("enhanced.use_hyde", use_hyde)
                    config.set("llm.temperature", temperature)
                    tag = f"enhanced_n{n_first}_k{k_rerank}_h{int(use_hyde)}_t{temperature}"
                    run_dir = tuning_dir / tag
                    runner = batch_runner_module.BatchRunner(
                        config=config,
                        questions_file=config.get("data.validation_questions_file"),
                        output_dir=str(run_dir),
                        limit=num_samples,
                        mode="enhanced",
                        verbose=False,
                    )
                    runner.run()
                    prediction_file = run_dir / "predictions_enhanced.jsonl"
                    summary_df = eval_module.evaluate_predictions(str(prediction_file), output_dir=str(run_dir))
                    summary_record = summary_df.to_dict(orient="records")[0]
                    summary_record.update(
                        {
                            "mode": "enhanced",
                            "top_k": None,
                            "temperature": temperature,
                            "max_new_tokens": None,
                            "n_first": n_first,
                            "k_rerank": k_rerank,
                            "use_hyde": use_hyde,
                            "filter_min_score": filter_min_score,
                        }
                    )
                    results.append(summary_record)
    return results


def run_hyperparameter_tuning(config: Config, output_dir: str):
    tuning_dir = Path(output_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    num_samples = config.get("tuning.num_samples", 50)
    modes = config.get("tuning.modes", ["baseline", "agentic"])

    results = []
    if any(m in modes for m in ("baseline", "agentic")):
        results += _run_baseline_agentic_grid(config, tuning_dir, num_samples)
    if "enhanced" in modes:
        results += _run_enhanced_grid(config, tuning_dir, num_samples)

    comparison_df = pd.DataFrame(results)
    out_path = tuning_dir / "tuning_results.csv"
    comparison_df.to_csv(out_path, index=False)
    print(comparison_df.to_string())
    print(f"\nTuning results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune baseline, agentic, and enhanced RAG hyperparameters"
    )
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()
    run_hyperparameter_tuning(Config.from_yaml(args.config), args.output)


if __name__ == "__main__":
    main()