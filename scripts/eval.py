#!/usr/bin/env python3
"""Notebook-style EM/F1 evaluation for baseline, agentic, and enhanced predictions."""

import argparse
import json
import logging
import re
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import List, Optional

import pandas as pd

from arag.evaluation.metrics import clean_prediction, compute_em, compute_f1


def load_predictions(predictions_path: str):
    with open(predictions_path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_em_f1_bar_plot(summary_df: pd.DataFrame, output_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping EM/F1 plot generation.")
        return None

    mean_em = float(summary_df["EM"].mean())
    mean_f1 = float(summary_df["F1"].mean())
    metrics = ["EM", "F1"]
    values = [mean_em, mean_f1]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(metrics, values)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(title)

    for index, value in enumerate(values):
        axis.text(index, min(value + 0.02, 0.99), f"{value:.3f}", ha="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def extract_final_answer_fallback(pred_answer) -> str:
    """Best-effort extraction of final answer text from JSON-like agent traces."""
    text = "" if pred_answer is None else str(pred_answer)
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"(?im)^\s*(human|assistant|user|system)\s*:\s*", "", text)

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("action") == "final":
            answer = payload.get("answer")
            if isinstance(answer, str):
                return answer
            if answer is not None:
                return str(answer)
    except JSONDecodeError:
        pass

    decoder = JSONDecoder()
    objects = []
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)

    finals = [obj for obj in objects if obj.get("action") == "final" and isinstance(obj.get("answer"), str)]
    if finals:
        return finals[-1]["answer"]

    return text


def compute_support_recall(prediction: dict) -> Optional[float]:
    """Compute supporting-fact recall for enhanced predictions.

    Returns None when the prediction doesn't carry retrieved_titles/sent_ids
    (i.e. baseline/agentic rows) so it silently skips.
    """
    gold_titles = prediction.get("supporting_titles")
    gold_sent_ids = prediction.get("supporting_sent_ids")
    if not gold_titles or not gold_sent_ids:
        return None

    retrieved_titles = prediction.get("retrieved_titles", [])
    retrieved_sent_ids = prediction.get("retrieved_sent_ids", [])
    if not retrieved_titles:
        return None

    gold_set = set(zip(gold_titles, gold_sent_ids))
    got_set = set(zip(retrieved_titles, retrieved_sent_ids))
    return len(gold_set & got_set) / len(gold_set) if gold_set else float("nan")


def compute_bertscore_f1(
    predictions: List[str], references: List[str], model_type: str = "distilbert-base-uncased"
) -> List[float]:
    """Optional BERTScore-F1 computation. Returns list of per-sample scores.

    Uses distilbert-base-uncased by default (fast); caller can pass
    'microsoft/deberta-xlarge-mnli' for higher quality.
    Returns a list of NaN if bert-score is not installed.
    """
    try:
        from bert_score import score as bertscore_fn
    except ImportError:
        logging.warning("bert-score not installed; skipping BERTScore-F1. Install with: pip install bert-score")
        return [float("nan")] * len(predictions)

    _, _, f1_tensor = bertscore_fn(
        predictions, references, model_type=model_type, verbose=False
    )
    return f1_tensor.tolist()


def evaluate_predictions(predictions_path: str, output_dir: str = None, save_plot: bool = True,
                         bertscore: bool = False, bertscore_model: str = "distilbert-base-uncased"):
    predictions = load_predictions(predictions_path)
    rows = []

    for prediction in predictions:
        pred_text = extract_final_answer_fallback(prediction.get("pred_answer", ""))
        cleaned = clean_prediction(pred_text)
        gold = prediction.get("gold_answer") or prediction.get("answer", "")
        em = compute_em(cleaned, gold)
        f1 = compute_f1(cleaned, gold)
        support_rec = compute_support_recall(prediction)
        prediction["pred_answer_extracted"] = pred_text
        prediction["clean_pred_answer"] = cleaned
        prediction["em"] = em
        prediction["f1"] = f1
        if support_rec is not None:
            prediction["support_recall"] = support_rec
        rows.append(
            {
                "Mode": prediction.get("mode", "unknown"),
                "QID": prediction.get("qid"),
                "Question": prediction.get("question"),
                "Gold": gold,
                "Prediction": cleaned,
                "EM": em,
                "F1": f1,
                "SupportRecall": support_rec,
            }
        )

    results_df = pd.DataFrame(rows)

    # Optional BERTScore
    if bertscore:
        preds_list = results_df["Prediction"].tolist()
        golds_list = results_df["Gold"].tolist()
        bs_scores = compute_bertscore_f1(preds_list, golds_list, model_type=bertscore_model)
        results_df["BERTScore_F1"] = bs_scores
        for i, prediction in enumerate(predictions):
            prediction["bertscore_f1"] = bs_scores[i]

    agg_cols = ["EM", "F1"]
    if "SupportRecall" in results_df.columns and results_df["SupportRecall"].notna().any():
        agg_cols.append("SupportRecall")
    if "BERTScore_F1" in results_df.columns:
        agg_cols.append("BERTScore_F1")

    summary_df = results_df.groupby("Mode", dropna=False)[agg_cols].mean().reset_index()
    summary_df["Samples"] = results_df.groupby("Mode", dropna=False).size().values

    output_path = Path(output_dir or Path(predictions_path).parent)
    output_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(predictions_path).stem

    with open(output_path / f"{base_name}_scored.jsonl", 'w', encoding='utf-8') as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')

    results_df.to_csv(output_path / f"{base_name}_per_example.csv", index=False)
    summary_df.to_csv(output_path / f"{base_name}_summary.csv", index=False)

    if save_plot:
        plot_path = output_path / f"{base_name}_em_f1_bar.png"
        title = f"EM/F1 Summary: {base_name}"
        save_em_f1_bar_plot(summary_df, plot_path, title)

    print(summary_df)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAG predictions")
    parser.add_argument("--predictions", "-p", required=True, help="Predictions file path (.json or .jsonl)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the EM/F1 bar chart")
    parser.add_argument("--bertscore", action="store_true", help="Compute optional BERTScore-F1 (requires bert-score)")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased", help="BERTScore model type")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print(f"\n{'=' * 60}")
    print(f"Evaluating predictions from: {args.predictions}")
    evaluate_predictions(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        save_plot=not args.no_plot,
        bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
    )


if __name__ == "__main__":
    main()
