"""Recalculate reported four-class MELD metrics from saved predictions.

This verifies downstream calculations, not provenance of the model outputs.
"""
import csv
import json
from pathlib import Path

from sonar.simulation import evaluate_fusion_rows
from sonar.meld import _probability_metrics
import numpy as np

root = Path(__file__).resolve().parent
with (root / "results/meld_checkpoint_predictions.csv").open(encoding="utf-8", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["split"] == "test" and row["true_label_4"]]
truth = [row["true_label_4"] for row in rows]
branches = {}
for name in ("text", "speech"):
    probabilities = np.array([json.loads(row[name + "_probs_4_calibrated"]) for row in rows])
    branches[name] = _probability_metrics(truth, probabilities)
fusion = evaluate_fusion_rows([
    {"true_label": row["true_label_4"], "text_probs": row["text_probs_4_calibrated"],
     "speech_probs": row["speech_probs_4_calibrated"]}
    for row in rows
])
result = {"source": "saved predictions; checkpoint provenance not reverified", "count": len(rows),
          "branches": branches, "fusion": fusion}
destination = root / "results/meld_prediction_recalculation.json"
destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
