import numpy as np

from signet_experiments.metrics import classification_metrics, select_threshold


def test_far_and_frr_follow_notebook_label_convention():
    metrics, confusion = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.2, 0.9]),
        threshold=0.5,
    )

    assert confusion.tolist() == [[1, 1], [1, 1]]
    assert metrics["far"] == 0.5
    assert metrics["frr"] == 0.5
    assert metrics["f1_forged"] == 0.5


def test_threshold_selection_returns_a_candidate_maximizing_validation_f1():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, metrics = select_threshold(labels, probabilities)

    assert 0.01 <= threshold <= 0.99
    assert metrics["f1_forged"] == 1.0
