"""Metrics for coarse four-class emotion predictions."""

from typing import Iterable, Sequence

from sonar.fusion import COARSE_LABELS


def accuracy(truth: Sequence[str], predicted: Sequence[str]) -> float:
    _validate_labels(truth, predicted)
    if not truth:
        raise ValueError("at least one labeled example is required")
    return sum(a == b for a, b in zip(truth, predicted)) / len(truth)


def macro_f1(truth: Sequence[str], predicted: Sequence[str]) -> float:
    _validate_labels(truth, predicted)
    if not truth:
        raise ValueError("at least one labeled example is required")
    class_scores = []
    for label in COARSE_LABELS:
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        denominator = 2 * tp + fp + fn
        class_scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(class_scores) / len(COARSE_LABELS)


def expected_calibration_error(
    truth: Sequence[str], predicted: Sequence[str], confidence: Sequence[float], bins: int = 15
) -> float:
    _validate_labels(truth, predicted)
    if len(confidence) != len(truth) or not truth:
        raise ValueError("truth, predicted, and confidence must have equal non-zero lengths")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if any(not 0.0 <= value <= 1.0 for value in confidence):
        raise ValueError("confidence values must be between 0 and 1")

    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidence)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if members:
            bin_accuracy = sum(truth[i] == predicted[i] for i in members) / len(members)
            bin_confidence = sum(confidence[i] for i in members) / len(members)
            error += len(members) / len(truth) * abs(bin_accuracy - bin_confidence)
    return error


def _validate_labels(truth: Sequence[str], predicted: Sequence[str]) -> None:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal lengths")
    invalid = (set(truth) | set(predicted)) - set(COARSE_LABELS)
    if invalid:
        raise ValueError("unknown coarse labels: " + ", ".join(sorted(invalid)))
