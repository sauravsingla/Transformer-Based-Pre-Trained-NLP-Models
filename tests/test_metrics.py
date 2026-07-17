import numpy as np

from transformer_sentiment.metrics import compute_classification_metrics


def test_perfect_predictions_produce_perfect_scores() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    logits = np.eye(5) * 10
    metrics = compute_classification_metrics(labels, logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == 1.0


def test_metrics_are_finite_when_a_prediction_class_is_missing() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    logits = np.zeros((10, 5))
    metrics = compute_classification_metrics(labels, logits)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
