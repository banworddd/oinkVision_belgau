"""Metrics for multi-label classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from .constants import LABELS


def apply_thresholds(probs: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    return (probs >= thresholds[None, :]).astype(np.int64)


def compute_macro_f1(
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = list(labels or LABELS)
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)

    preds = apply_thresholds(probs, thresholds)
    per_class = {}
    scores = []
    supports: dict[str, int] = {}
    present_scores = []
    for idx, label in enumerate(labels):
        class_targets = targets[:, idx]
        score = f1_score(class_targets, preds[:, idx], zero_division=0)
        per_class[label] = float(score)
        supports[label] = int(np.sum(class_targets))
        scores.append(score)
        if supports[label] > 0:
            present_scores.append(score)

    return {
        "macro_f1": float(np.mean(scores)),
        "macro_f1_present_classes": float(np.mean(present_scores)) if present_scores else 0.0,
        "per_class_f1": per_class,
        "per_class_support": supports,
        "thresholds": [float(x) for x in np.asarray(thresholds, dtype=np.float32)],
    }


def optimize_thresholds(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold_grid: list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = list(labels or LABELS)
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
    threshold_grid = np.asarray(threshold_grid, dtype=np.float32)

    best_thresholds: list[float] = []
    per_class_best: dict[str, float] = {}
    for idx, label in enumerate(labels):
        class_targets = targets[:, idx]
        class_probs = probs[:, idx]

        best_threshold = 0.5
        best_score = -1.0
        for threshold in threshold_grid:
            class_preds = (class_probs >= threshold).astype(np.int64)
            score = f1_score(class_targets, class_preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        best_thresholds.append(best_threshold)
        per_class_best[label] = float(best_score)

    tuned_metrics = compute_macro_f1(targets, probs, thresholds=best_thresholds, labels=labels)
    tuned_metrics["threshold_search_scores"] = per_class_best
    return tuned_metrics
