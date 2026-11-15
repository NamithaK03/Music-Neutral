import csv
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sonar.fusion import COARSE_LABELS
from sonar.meld import _read_audio_features, evaluate_meld


_SOURCE_LABELS = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "neutral": "neutral",
}
_TEXTS = {
    "happy": "bright joyful music and cheerful smiles",
    "sad": "gloomy sorrow and a lonely rainy day",
    "angry": "furious rage and an irritating argument",
    "neutral": "ordinary facts and a plain simple statement",
}
_CSV_COLUMNS = (
    "Sr No.",
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
)


def _write_split(path, split_index, counts_per_class):
    rows = []
    feature_map = {}
    serial = 1
    for label_index, label in enumerate(COARSE_LABELS):
        source_label = _SOURCE_LABELS[label]
        for item_index in range(counts_per_class):
            dialogue_id = str(label_index * 100 + item_index)
            utterance_id = "0"
            sample_id = dialogue_id + "_" + utterance_id
            features = np.zeros(8, dtype=np.float64)
            features[label_index] = 2.0
            features += item_index * 0.001
            feature_map[sample_id] = features
            rows.append(
                {
                    "Sr No.": str(serial),
                    "Utterance": _TEXTS[label],
                    "Speaker": "Speaker",
                    "Emotion": source_label,
                    "Sentiment": "neutral",
                    "Dialogue_ID": dialogue_id,
                    "Utterance_ID": utterance_id,
                    "Season": "1",
                    "Episode": str(label_index + 1),
                    "StartTime": "00:00:00,000",
                    "EndTime": "00:00:01,000",
                }
            )
            serial += 1
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return feature_map


class MeldEvaluationTests(unittest.TestCase):
    def test_runs_train_dev_test_text_audio_and_fusion_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_csv = root / "train.csv"
            dev_csv = root / "dev.csv"
            test_csv = root / "test.csv"
            train_features = _write_split(train_csv, 0, 6)
            dev_features = _write_split(dev_csv, 1, 2)
            test_features = _write_split(test_csv, 2, 2)
            features_path = root / "audio.pkl"
            with features_path.open("wb") as output:
                pickle.dump(
                    [train_features, dev_features, test_features], output
                )

            result, predictions = evaluate_meld(
                train_csv,
                dev_csv,
                test_csv,
                features_path,
            )

        self.assertEqual(result["dataset"], "MELD")
        self.assertEqual(result["splits"]["counts"], {"train": 24, "dev": 8, "test": 8})
        self.assertEqual(len(predictions), 16)
        self.assertIn("confidence_aware_fallback", result["metrics"]["test"]["fusion_and_conflicts"]["fusion"])
        self.assertIn("0.6", result["metrics"]["test"]["fallback_threshold_sensitivity"])
        self.assertEqual(result["models"]["speech"]["feature_dimensions"], 8)
        self.assertNotIn("text", predictions[0])

    def test_rejects_pickle_with_executable_global(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pkl"
            with path.open("wb") as output:
                pickle.dump(os.system, output)

            with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported global"):
                _read_audio_features(path)


if __name__ == "__main__":
    unittest.main()
