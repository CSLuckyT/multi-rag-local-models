#!/usr/bin/env python3
"""Grid-search hyperparameter tuning for baseline and agentic HotPotQA runs."""

import argparse
from pathlib import Path

import pandas as pd

from arag import Config
import batch_runner as batch_runner_module
import eval as eval_module


def run_hyperparameter_tuning(config: Config, output_dir: str):
    tuning_dir = Path(output_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    results = []

    modes = config.get("tuning.modes", ["baseline", "agentic"])
    top_k_values = config.get("tuning.top_k_values", [1, 3, 5])
    temperature_values = config.get("tuning.temperature_values", [0.0, 0.1, 0.3])
    max_new_tokens_values = config.get("tuning.max_new_tokens_values", [20, 50, 80])
    num_samples = config.get("tuning.num_samples", 50)

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
                        }
                    )
                    results.append(summary_record)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(tuning_dir / "tuning_results.csv", index=False)
    print(comparison_df)


def main():
    parser = argparse.ArgumentParser(description="Tune baseline and agentic RAG hyperparameters")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()
    run_hyperparameter_tuning(Config.from_yaml(args.config), args.output)


if __name__ == "__main__":
    main()