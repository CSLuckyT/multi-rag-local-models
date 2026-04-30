"""RAG evaluation helpers — EM/F1 (traditional) + BERTScore + RAGAS (modern)."""

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


# ── Modern metrics (added by Zachary Powell) ───────────────────────────────────────── #


def compute_bertscore(
    predictions: list[str],
    ground_truths: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """Return macro-averaged BERTScore precision, recall, and F1.

    Parameters
    ----------
    predictions : model answers, one per example.
    ground_truths : reference answers, one per example.
    model_type : HuggingFace encoder to use for contextual embeddings.
    device : ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    batch_size : mini-batch size for the encoder forward pass.

    Returns
    -------
    Dict with keys ``bertscore_precision``, ``bertscore_recall``, ``bertscore_f1``.
    """
    from bert_score import score as _bs_score  # pip install bert-score

    P, R, F = _bs_score(
        predictions,
        ground_truths,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def compute_ragas(
    questions: list[str],
    predictions: list[str],
    retrieved_contexts: list[list[str]],
    ground_truths: list[str],
    llm,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate a RAG system with RAGAS framework metrics.

    Metrics computed
    ----------------
    * **faithfulness** — answer is grounded in the retrieved passages.
    * **answer_relevancy** — answer is relevant to the question.
    * **context_recall** — retrieved passages collectively cover the ground truth.

    Parameters
    ----------
    questions : input questions.
    predictions : generated answers.
    retrieved_contexts : list of passage lists, one per question.
    ground_truths : reference answers.
    llm : RAGAS-compatible LLM (e.g. ``langchain_openai.ChatOpenAI``).
    max_samples : cap rows evaluated to control API cost; ``None`` = all rows.

    Returns
    -------
    Dict mapping each RAGAS metric name to its mean score (float).
    """
    from datasets import Dataset  # pip install datasets
    from ragas import evaluate  # pip install ragas>=0.2
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    rows: dict[str, list] = {
        "question": list(questions),
        "answer": list(predictions),
        "contexts": list(retrieved_contexts),
        "ground_truth": list(ground_truths),
    }
    if max_samples is not None:
        rows = {k: v[:max_samples] for k, v in rows.items()}

    ds = Dataset.from_dict(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
    )
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}