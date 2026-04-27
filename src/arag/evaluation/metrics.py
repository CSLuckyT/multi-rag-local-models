"""Notebook-style EM/F1 evaluation helpers."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def clean_prediction(text: str) -> str:
    cleaned = (text or "").strip().split("\n")[0].strip()
    return re.sub(r"\(.*?\)", "", cleaned).strip()


def compute_em(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(int(pred_tokens == gold_tokens))
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)