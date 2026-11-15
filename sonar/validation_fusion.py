"""Select a probability mixture using development labels and evaluate once on test.

Requires saved paired predictions from the MELD checkpoint study. This is an
additional experiment; its probabilities are conditional on four classes.
"""
import csv
import json
from pathlib import Path

import numpy as np

from sonar.fusion import COARSE_LABELS
from sonar.metrics import accuracy, expected_calibration_error, macro_f1


def evaluate(rows, text_weight):
    if not rows:
        raise ValueError("split has no mapped examples")
    truth = [row["true_label_4"] for row in rows]
    text = np.array([json.loads(row["text_probs_4_calibrated"]) for row in rows])
    speech = np.array([json.loads(row["speech_probs_4_calibrated"]) for row in rows])
    mixed = text_weight * text + (1 - text_weight) * speech
    predictions = [COARSE_LABELS[i] for i in np.argmax(mixed, axis=1)]
    return {"n": len(rows), "accuracy": accuracy(truth, predictions),
            "macro_f1": macro_f1(truth, predictions),
            "ece_15": expected_calibration_error(truth, predictions, mixed.max(axis=1).tolist())}


def run(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    dev = [r for r in rows if r["split"] == "dev" and r["true_label_4"]]
    test = [r for r in rows if r["split"] == "test" and r["true_label_4"]]
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    development = [{"text_weight": w, **evaluate(dev, w)} for w in weights]
    # Tie break toward the text-only model, a fixed rule specified in advance.
    selected = max(development, key=lambda item: (item["macro_f1"], item["text_weight"]))
    return {"method": "fixed five-point mixture grid selected by development macro-F1",
            "caveat": "conditional four-class saved checkpoint predictions; official splits share episodes",
            "development_grid": development, "selected_text_weight": selected["text_weight"],
            "test_selected": evaluate(test, selected["text_weight"]),
            "test_text_only": evaluate(test, 1.0), "test_equal_mix": evaluate(test, 0.5)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    result = run(root / "results/meld_checkpoint_predictions.csv")
    (root / "results/meld_validation_mixture.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
