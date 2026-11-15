import csv
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from sonar.meld import write_predictions_csv
from sonar.meld_checkpoints import evaluate_meld_checkpoints


_LABELS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")
_COLUMNS = (
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


def _write_split(path):
    rows = []
    text_map = {}
    speech_map = {}
    for label_index, label in enumerate(_LABELS):
        dialogue_id = str(label_index)
        rows.append(
            {
                "Sr No.": str(label_index + 1),
                "Utterance": "example " + label,
                "Speaker": "Speaker",
                "Emotion": label,
                "Sentiment": "neutral",
                "Dialogue_ID": dialogue_id,
                "Utterance_ID": "0",
                "Season": "1",
                "Episode": str(label_index + 1),
                "StartTime": "00:00:00,000",
                "EndTime": "00:00:01,000",
            }
        )
        text = np.zeros((1, 600), dtype=np.float32)
        text[0, label_index] = 4.0
        speech = np.zeros((1, 300), dtype=np.float32)
        speech[0, label_index] = 4.0
        text_map[dialogue_id] = text
        speech_map[dialogue_id] = speech

    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return text_map, speech_map


def _write_checkpoint(path, *, layers, weights):
    model_config = {
        "config": {
            "layers": [
                {
                    "class_name": "TimeDistributed",
                    "config": {
                        "name": name,
                        "layer": {
                            "class_name": "Dense",
                            "config": {"activation": activation, "units": units},
                        },
                    },
                }
                for name, activation, units in layers
            ]
        }
    }
    with h5py.File(path, "w") as output:
        output.attrs["model_config"] = json.dumps(model_config)
        for layer_name, kernel, bias in weights:
            group = output.require_group(
                "model_weights/{0}/{0}".format(layer_name)
            )
            group.create_dataset("kernel:0", data=kernel)
            group.create_dataset("bias:0", data=bias)


class MeldCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(
        h5py is not None and importlib.util.find_spec("sklearn") is not None,
        "install the optional MELD study dependencies",
    )
    def test_reconstructs_publisher_output_heads_and_saves_split_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_data = [
                _write_split(root / (name + ".csv"))
                for name in ("train", "dev", "test")
            ]
            label_pickle = root / "labels.p"
            with label_pickle.open("wb") as output:
                pickle.dump([None, None, None, None, None, {
                    label: index for index, label in enumerate(_LABELS)
                }], output)

            text_features = root / "text.pkl"
            speech_features = root / "speech.pkl"
            with text_features.open("wb") as output:
                pickle.dump([split[0] for split in split_data], output)
            with speech_features.open("wb") as output:
                pickle.dump([split[1] for split in split_data], output)

            text_kernel = np.zeros((600, 7), dtype=np.float32)
            text_kernel[:7, :7] = np.eye(7, dtype=np.float32)
            text_weights = root / "text.h5"
            _write_checkpoint(
                text_weights,
                layers=(("time_distributed_1", "softmax", 7),),
                weights=(("time_distributed_1", text_kernel, np.zeros(7)),),
            )

            projection = np.zeros((300, 150), dtype=np.float32)
            projection[:7, :7] = np.eye(7, dtype=np.float32)
            output_kernel = np.zeros((150, 7), dtype=np.float32)
            output_kernel[:7, :7] = np.eye(7, dtype=np.float32)
            speech_weights = root / "speech.h5"
            _write_checkpoint(
                speech_weights,
                layers=(
                    ("time_distributed_1", "relu", 150),
                    ("time_distributed_2", "softmax", 7),
                ),
                weights=(
                    ("time_distributed_1", projection, np.zeros(150)),
                    ("time_distributed_2", output_kernel, np.zeros(7)),
                ),
            )

            result, predictions = evaluate_meld_checkpoints(
                root / "train.csv",
                root / "dev.csv",
                root / "test.csv",
                label_pickle,
                text_features,
                speech_features,
                text_weights,
                speech_weights,
            )
            predictions_path = root / "predictions.csv"
            write_predictions_csv(predictions_path, predictions)
            with predictions_path.open("r", encoding="utf-8", newline="") as source:
                saved_rows = list(csv.DictReader(source))

        self.assertEqual(result["split_counts"], {"train": 7, "dev": 7, "test": 7})
        self.assertEqual(len(predictions), 14)
        self.assertEqual(len(saved_rows), 14)
        self.assertEqual(
            result["metrics"]["test"]["full_seven_class"]["text"]["accuracy"],
            1.0,
        )
        self.assertEqual(
            result["metrics"]["test"]["mapped_four_class_subset"]["text"]["raw"][
                "accuracy"
            ],
            1.0,
        )
        self.assertEqual(
            result["metrics"]["test"]["mapped_four_class_subset"]["sample_count"],
            4,
        )
        self.assertIn("text_probs_7", saved_rows[0])

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None,
        "install the optional MELD study dependencies",
    )
    def test_episode_disjoint_filter_excludes_train_and_dev_episodes(self):
        from sonar.meld_checkpoints import evaluate_meld_episode_disjoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_annotations(path, rows):
                with path.open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)

            def annotation(dialogue, utterance, episode, emotion):
                return {
                    "Sr No.": str(dialogue + 1),
                    "Utterance": "example " + emotion,
                    "Speaker": "Speaker",
                    "Emotion": emotion,
                    "Sentiment": "neutral",
                    "Dialogue_ID": str(dialogue),
                    "Utterance_ID": str(utterance),
                    "Season": "1",
                    "Episode": str(episode),
                    "StartTime": "00:00:00,000",
                    "EndTime": "00:00:01,000",
                }

            train = [annotation(1, 0, 1, "joy")]
            dev = [annotation(2, 0, 2, "neutral")]
            test = [
                annotation(3, 0, 1, "sadness"),
                annotation(4, 0, 3, "joy"),
                annotation(4, 1, 3, "surprise"),
            ]
            write_annotations(root / "train.csv", train)
            write_annotations(root / "dev.csv", dev)
            write_annotations(root / "test.csv", test)

            predictions = root / "predictions.csv"
            columns = (
                "split",
                "dialogue_id",
                "episode_id",
                "utterance_id",
                "true_label_7",
                "true_label_4",
                "text_probs_7",
                "speech_probs_7",
                "text_probs_4_calibrated",
                "speech_probs_4_calibrated",
            )
            text_joy = [1.0, 0.0, 0.0, 0.0]
            text_surprise = [0.0, 1.0, 0.0, 0.0]
            full_joy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            full_surprise = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            prediction_rows = [
                {
                    "split": "test",
                    "dialogue_id": "3",
                    "episode_id": "1_1",
                    "utterance_id": "0",
                    "true_label_7": "sadness",
                    "true_label_4": "sad",
                    "text_probs_7": json.dumps(full_joy),
                    "speech_probs_7": json.dumps(full_joy),
                    "text_probs_4_calibrated": json.dumps(text_joy),
                    "speech_probs_4_calibrated": json.dumps(text_joy),
                },
                {
                    "split": "test",
                    "dialogue_id": "4",
                    "episode_id": "1_3",
                    "utterance_id": "0",
                    "true_label_7": "joy",
                    "true_label_4": "happy",
                    "text_probs_7": json.dumps(full_joy),
                    "speech_probs_7": json.dumps(full_joy),
                    "text_probs_4_calibrated": json.dumps(text_joy),
                    "speech_probs_4_calibrated": json.dumps(text_joy),
                },
                {
                    "split": "test",
                    "dialogue_id": "4",
                    "episode_id": "1_3",
                    "utterance_id": "1",
                    "true_label_7": "surprise",
                    "true_label_4": "",
                    "text_probs_7": json.dumps(full_surprise),
                    "speech_probs_7": json.dumps(full_surprise),
                    "text_probs_4_calibrated": "",
                    "speech_probs_4_calibrated": "",
                },
            ]
            with predictions.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(prediction_rows)

            result = evaluate_meld_episode_disjoint(
                root / "train.csv",
                root / "dev.csv",
                root / "test.csv",
                predictions,
            )

        self.assertEqual(result["sample_counts"]["all_seven_class"], 2)
        self.assertEqual(result["sample_counts"]["mapped_four_class"], 1)
        self.assertEqual(result["episodes"]["selected_test_episode_ids"], ["1_3"])
        self.assertIsNone(result["metrics"])
        self.assertIn("fewer than 10", result["metrics_status"])


if __name__ == "__main__":
    unittest.main()
