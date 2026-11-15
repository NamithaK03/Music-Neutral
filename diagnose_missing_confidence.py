"""Diagnostic only: compare fixed and rescaled trust on saved MELD predictions."""
import csv
import json
from pathlib import Path

from sonar.fusion import COARSE_LABELS, FusionConfig, Prediction, fuse_emotions
from sonar.metrics import accuracy, macro_f1

root = Path(__file__).resolve().parent
with (root / "results/meld_checkpoint_complete_train_predictions.csv").open(encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["true_label_4"]]

result = {"scope": "post-hoc diagnostic; not the pre-specified MELD experiment",
          "threshold": 0.6, "sample_count": len(rows), "variants": {}}
for name, config in [
    ("fixed_missing_auxiliary_as_zero", FusionConfig()),
    ("rescaled_primary_confidence_diagnostic", FusionConfig(text_primary_weight=1.0,
        text_auxiliary_weight=0.0, speech_primary_weight=1.0, asr_weight=0.0)),
]:
    predictions, truth, fallbacks = [], [], 0
    for row in rows:
        probs = [json.loads(row[k]) for k in ("text_probs_4_calibrated", "speech_probs_4_calibrated")]
        indices = [max(range(len(p)), key=p.__getitem__) for p in probs]
        text, speech = [Prediction(COARSE_LABELS[i], p[i]) for p, i in zip(probs, indices)]
        fused = fuse_emotions(text, speech, config=config)
        predictions.append(fused.label)
        truth.append(row["true_label_4"])
        fallbacks += fused.fallback
    result["variants"][name] = {"accuracy": accuracy(truth, predictions),
        "macro_f1": macro_f1(truth, predictions), "fallback_rate": fallbacks / len(rows)}
(root / "results/meld_missing_confidence_diagnostic.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
