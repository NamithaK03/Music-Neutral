import csv
import json
import tempfile
import unittest
from pathlib import Path

from sonar.validation_fusion import run


class ValidationFusionTests(unittest.TestCase):
    def test_selection_uses_development_only(self):
        fields = ["split", "true_label_4", "text_probs_4_calibrated", "speech_probs_4_calibrated"]
        text = json.dumps([0.9, 0.05, 0.03, 0.02])
        speech = json.dumps([0.05, 0.9, 0.03, 0.02])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {"split": "dev", "true_label_4": "happy", "text_probs_4_calibrated": text, "speech_probs_4_calibrated": speech},
                    {"split": "test", "true_label_4": "sad", "text_probs_4_calibrated": text, "speech_probs_4_calibrated": speech},
                ])
            result = run(path)
        self.assertEqual(result["selected_text_weight"], 1.0)
        self.assertEqual(result["test_selected"]["accuracy"], 0.0)
        self.assertEqual(result["development_grid"][0]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
