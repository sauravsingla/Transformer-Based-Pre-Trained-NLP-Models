"""Evaluation metrics used by the paper-aligned experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    labels: np.ndarray, logits: np.ndarray, num_labels: int = 5
) -> dict[str, float]:
    """Return accuracy, macro/weighted scores and one-vs-rest macro ROC-AUC."""
    predictions = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(labels)
    macro = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
    }
    probabilities = _softmax(np.asarray(logits))
    binary_labels = label_binarize(labels, classes=np.arange(num_labels))
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(binary_labels, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def classification_report_dict(
    labels: np.ndarray, predictions: np.ndarray, target_names: list[str]
) -> dict[str, object]:
    """Create a JSON-serializable per-class classification report."""
    return classification_report(
        labels,
        predictions,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
