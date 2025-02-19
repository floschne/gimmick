import evaluate
import numpy as np
from functools import lru_cache


def compute_match_accuracy(preds: list[str], targets: list[str]) -> dict[str, float]:
    if len(preds) != len(targets):
        raise ValueError("Length of predictions and targets must match")
    correct = 0
    relaxed_correct = 0
    very_relaxed_correct = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            correct += 1
        if pred.startswith(target) or target.startswith(pred):
            relaxed_correct += 1
        if pred in target or target in pred:
            very_relaxed_correct += 1

    return {
        "exact_match_accuracy": correct / len(preds),
        "relaxed_match_accuracy": relaxed_correct / len(preds),
        "very_relaxed_match_accuracy": very_relaxed_correct / len(preds),
    }


def compute_score_accuracy(
    scores: list[int],
    correct_thresholds: list[int] = [100, 95, 90, 75, 50, 25],
) -> dict[str, str | float]:
    corrects = {f"correct_{threshold}": 0 for threshold in correct_thresholds}
    for score in scores:
        for threshold in correct_thresholds:
            if score >= threshold:
                corrects[f"correct_{threshold}"] += 1

    return {
        f"lmm_judge_accuracy_{threshold}": correct / len(scores)
        for threshold, correct in corrects.items()
    }


@lru_cache(maxsize=1)
def _get_bert_score_metric():
    metric = evaluate.load("bertscore")
    if metric is None:
        raise ValueError("BERTScore metric is not available")
    return metric


def compute_bert_score(preds: list[str], targets: list[str]) -> dict[str, float | str]:
    metric = _get_bert_score_metric()
    results = metric.compute(predictions=preds, references=targets, lang="en")

    return {
        "precision": float(np.mean(results["precision"])),  # type: ignore
        "recall": float(np.mean(results["recall"])),  # type: ignore
        "f1": float(np.mean(results["f1"])),  # type: ignore
        "hashcode": results["hashcode"],  # type: ignore
    }
