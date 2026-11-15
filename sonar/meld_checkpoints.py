"""Reproduce MELD publisher baseline predictions from released hidden states and weights."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sonar.fusion import COARSE_LABELS
from sonar.meld import (
    _MELD_LABELS,
    _NumpyFeatureUnpickler,
    _REQUIRED_COLUMNS,
    _conflict_summaries,
    _fit_temperature,
    _probability_metrics,
    _read_source_label_counts,
    _sha256,
    _temperature_scale,
    _threshold_metrics,
)


_PUBLISHER_LABELS = {
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
}
_PUBLISHER_LABEL_ORDER = (
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
)


def _read_all_rows(path: Path, split_name: str) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("unexpected MELD annotation columns in " + str(path))
        rows = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                emotion = row["Emotion"].strip().lower()
                dialogue_id = str(int(row["Dialogue_ID"]))
                utterance_id = int(row["Utterance_ID"])
                season = str(int(row["Season"]))
                episode = str(int(row["Episode"]))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    "invalid MELD annotation at {}:{}".format(path, line_number)
                ) from error
            sample_id = dialogue_id + "_" + str(utterance_id)
            if sample_id in seen_ids:
                raise ValueError("duplicate MELD utterance ID in " + str(path))
            seen_ids.add(sample_id)
            rows.append(
                {
                    "id": sample_id,
                    "dialogue_id": dialogue_id,
                    "utterance_index": utterance_id,
                    "episode_id": season + "_" + episode,
                    "source_emotion": emotion,
                    "label": _MELD_LABELS.get(emotion),
                    "split": split_name,
                }
            )
    if not rows:
        raise ValueError("MELD split is empty: " + split_name)
    return rows


def _read_publisher_label_order(data_pickle: Path) -> Tuple[str, ...]:
    with data_pickle.open("rb") as source:
        payload = _NumpyFeatureUnpickler(source).load()
    if not isinstance(payload, (list, tuple)) or len(payload) != 6:
        raise ValueError("unexpected publisher MELD emotion data pickle")
    label_index = payload[5]
    if not isinstance(label_index, dict) or set(label_index) != _PUBLISHER_LABELS:
        raise ValueError("unexpected emotion label map in publisher MELD data")
    indices = sorted(label_index.values())
    if indices != list(range(len(_PUBLISHER_LABELS))):
        raise ValueError("publisher MELD emotion labels are not contiguous")
    ordered = [None] * len(indices)
    for label, index in label_index.items():
        ordered[index] = label
    return tuple(ordered)


def _read_contextual_features(
    path: Path, *, expected_splits: int = 3
) -> Tuple[Dict[str, np.ndarray], ...]:
    with path.open("rb") as source:
        feature_splits = _NumpyFeatureUnpickler(source).load()
    if not isinstance(feature_splits, (list, tuple)) or len(feature_splits) != expected_splits:
        raise ValueError("publisher contextual features must contain train, dev, and test")

    validated = []
    expected_width = None
    for split_name, feature_map in zip(("train", "dev", "test"), feature_splits):
        if not isinstance(feature_map, dict) or not feature_map:
            raise ValueError("publisher " + split_name + " contextual features are empty")
        current = {}
        for dialogue_id, raw_features in feature_map.items():
            if not isinstance(dialogue_id, str):
                raise ValueError("publisher dialogue IDs must be strings")
            try:
                features = np.asarray(raw_features, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "invalid publisher contextual features for dialogue "
                    + dialogue_id
                ) from error
            if (
                features.ndim != 2
                or features.shape[0] == 0
                or features.shape[1] == 0
                or not np.all(np.isfinite(features))
            ):
                raise ValueError(
                    "invalid publisher contextual features for dialogue "
                    + dialogue_id
                )
            if expected_width is None:
                expected_width = features.shape[1]
            elif features.shape[1] != expected_width:
                raise ValueError("inconsistent publisher contextual feature dimensions")
            current[dialogue_id] = features
        validated.append(current)
    return tuple(validated)


def _aligned_context(rows: Sequence[dict], feature_map: Dict[str, np.ndarray]) -> np.ndarray:
    vectors = []
    for row in rows:
        dialogue_id = row["dialogue_id"]
        if dialogue_id not in feature_map:
            raise ValueError("missing publisher features for dialogue " + dialogue_id)
        features = feature_map[dialogue_id]
        index = row["utterance_index"]
        if index >= features.shape[0]:
            raise ValueError(
                "publisher feature sequence is too short for utterance " + row["id"]
            )
        vectors.append(features[index])
    return np.vstack(vectors)


def _read_dense_head(path: Path, layer_name: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "install the optional MELD study dependencies with "
            'python -m pip install ".[meld-study]"'
        ) from error

    kernel_path = (
        "model_weights/{0}/{0}/kernel:0".format(layer_name)
    )
    bias_path = "model_weights/{0}/{0}/bias:0".format(layer_name)
    with h5py.File(path, "r") as source:
        try:
            kernel = np.asarray(source[kernel_path], dtype=np.float64)
            bias = np.asarray(source[bias_path], dtype=np.float64)
            model_config = json.loads(source.attrs["model_config"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unexpected MELD checkpoint format in " + str(path)) from error
    if kernel.ndim != 2 or bias.shape != (kernel.shape[1],):
        raise ValueError("invalid dense output head in " + str(path))
    return kernel, bias, model_config


def _validate_time_distributed_head(
    model_config: dict,
    layer_name: str,
    *,
    activation: str,
    units: int,
) -> None:
    layers = model_config.get("config", {}).get("layers", [])
    for layer in layers:
        config = layer.get("config", {})
        if config.get("name") != layer_name:
            continue
        if layer.get("class_name") != "TimeDistributed":
            break
        dense = config.get("layer", {})
        dense_config = dense.get("config", {})
        if (
            dense.get("class_name") == "Dense"
            and dense_config.get("activation") == activation
            and dense_config.get("units") == units
        ):
            return
        break
    raise ValueError("unexpected " + layer_name + " architecture in MELD checkpoint")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _predict_text(hidden: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if hidden.shape[1] != kernel.shape[0] or kernel.shape[1] != len(_PUBLISHER_LABELS):
        raise ValueError("text hidden states do not match the released checkpoint head")
    return _softmax(hidden @ kernel + bias)


def _predict_speech(
    hidden: np.ndarray,
    projection_kernel: np.ndarray,
    projection_bias: np.ndarray,
    output_kernel: np.ndarray,
    output_bias: np.ndarray,
) -> np.ndarray:
    if (
        hidden.shape[1] != projection_kernel.shape[0]
        or projection_kernel.shape[1] != len(projection_bias)
        or len(output_bias) != len(_PUBLISHER_LABELS)
        or output_kernel.shape != (len(projection_bias), len(output_bias))
    ):
        raise ValueError("speech hidden states do not match the released checkpoint heads")
    projected = np.maximum(hidden @ projection_kernel + projection_bias, 0.0)
    return _softmax(projected @ output_kernel + output_bias)


def _all_class_metrics(
    labels: Sequence[str], probabilities: np.ndarray, class_order: Sequence[str]
) -> dict:
    from sklearn.metrics import f1_score

    indices = np.asarray([class_order.index(label) for label in labels])
    predictions = [class_order[index] for index in np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    correct = np.asarray(predictions) == np.asarray(labels)
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        members = (confidence >= lower) & (
            (confidence < upper) | ((bin_index == 14) & (confidence == 1.0))
        )
        if np.any(members):
            ece += float(np.mean(members)) * abs(
                float(np.mean(correct[members]))
                - float(np.mean(confidence[members]))
            )
    selected = np.clip(probabilities[np.arange(len(indices)), indices], 1e-12, 1.0)
    return {
        "accuracy": float(np.mean(correct)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(class_order), average="macro", zero_division=0)
        ),
        "ece_15": ece,
        "nll": float(-np.mean(np.log(selected))),
    }


def _four_class_probabilities(
    probabilities: np.ndarray, class_order: Sequence[str]
) -> np.ndarray:
    source_for_target = {target: source for source, target in _MELD_LABELS.items()}
    selected = probabilities[
        :,
        [class_order.index(source_for_target[label]) for label in COARSE_LABELS],
    ]
    totals = np.sum(selected, axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("publisher checkpoint assigns zero probability to mapped labels")
    return selected / totals


def evaluate_meld_checkpoints(
    train_csv: Path,
    dev_csv: Path,
    test_csv: Path,
    label_pickle: Path,
    text_features_pickle: Path,
    speech_features_pickle: Path,
    text_weights_hdf5: Path,
    speech_weights_hdf5: Path,
) -> Tuple[dict, List[dict]]:
    """Generate predictions from released MELD text and speech checkpoints."""
    class_order = _read_publisher_label_order(label_pickle)
    train_rows = _read_all_rows(train_csv, "train")
    dev_rows = _read_all_rows(dev_csv, "dev")
    test_rows = _read_all_rows(test_csv, "test")
    text_splits = _read_contextual_features(text_features_pickle)
    speech_splits = _read_contextual_features(speech_features_pickle)
    text_kernel, text_bias, text_config = _read_dense_head(
        text_weights_hdf5, "time_distributed_1"
    )
    _validate_time_distributed_head(
        text_config, "time_distributed_1", activation="softmax", units=7
    )
    speech_projection_kernel, speech_projection_bias, speech_config = _read_dense_head(
        speech_weights_hdf5, "time_distributed_1"
    )
    speech_kernel, speech_bias, speech_output_config = _read_dense_head(
        speech_weights_hdf5, "time_distributed_2"
    )
    _validate_time_distributed_head(
        speech_config, "time_distributed_1", activation="relu", units=150
    )
    _validate_time_distributed_head(
        speech_output_config, "time_distributed_2", activation="softmax", units=7
    )

    rows_by_split = {"train": train_rows, "dev": dev_rows, "test": test_rows}
    text_probabilities = {}
    speech_probabilities = {}
    for index, split in enumerate(("train", "dev", "test")):
        text_hidden = _aligned_context(rows_by_split[split], text_splits[index])
        speech_hidden = _aligned_context(rows_by_split[split], speech_splits[index])
        text_probabilities[split] = _predict_text(
            text_hidden, text_kernel, text_bias
        )
        speech_probabilities[split] = _predict_speech(
            speech_hidden,
            speech_projection_kernel,
            speech_projection_bias,
            speech_kernel,
            speech_bias,
        )

    dev_core_rows = [row for row in dev_rows if row["label"] is not None]
    test_core_rows = [row for row in test_rows if row["label"] is not None]
    temperatures = {}
    calibrated_core = {"text": {}, "speech": {}}
    for modality, all_probabilities in (
        ("text", text_probabilities),
        ("speech", speech_probabilities),
    ):
        dev_indices = [dev_rows.index(row) for row in dev_core_rows]
        dev_core_probs = _four_class_probabilities(
            all_probabilities["dev"][dev_indices], class_order
        )
        dev_core_labels = [row["label"] for row in dev_core_rows]
        temperature, dev_nll = _fit_temperature(dev_core_probs, dev_core_labels)
        temperatures[modality] = {
            "temperature": temperature,
            "validation_nll_after_scaling": dev_nll,
        }
        for split, split_rows in (("dev", dev_core_rows), ("test", test_core_rows)):
            indices = [rows_by_split[split].index(row) for row in split_rows]
            four_probs = _four_class_probabilities(
                all_probabilities[split][indices], class_order
            )
            calibrated_core[modality][split] = _temperature_scale(
                four_probs, temperature
            )

    metrics_by_split = {}
    prediction_rows = []
    selected_threshold = None
    for split in ("dev", "test"):
        all_rows = rows_by_split[split]
        full_truth = [row["source_emotion"] for row in all_rows]
        core_rows = [row for row in all_rows if row["label"] is not None]
        core_truth = [row["label"] for row in core_rows]
        core_indices = [all_rows.index(row) for row in core_rows]
        text_core_raw = _four_class_probabilities(
            text_probabilities[split][core_indices], class_order
        )
        speech_core_raw = _four_class_probabilities(
            speech_probabilities[split][core_indices], class_order
        )
        text_core_calibrated = calibrated_core["text"][split]
        speech_core_calibrated = calibrated_core["speech"][split]
        fusion_rows = [
            {
                "true_label": row["label"],
                "text_probs": json.dumps(
                    [float(value) for value in text_core_calibrated[index]]
                ),
                "speech_probs": json.dumps(
                    [float(value) for value in speech_core_calibrated[index]]
                ),
            }
            for index, row in enumerate(core_rows)
        ]
        metrics_by_split[split] = {
            "full_seven_class": {
                "sample_count": len(all_rows),
                "text": _all_class_metrics(
                    full_truth, text_probabilities[split], class_order
                ),
                "speech": _all_class_metrics(
                    full_truth, speech_probabilities[split], class_order
                ),
            },
            "mapped_four_class_subset": {
                "sample_count": len(core_rows),
                "text": {
                    "raw": _probability_metrics(core_truth, text_core_raw),
                    "temperature_scaled": _probability_metrics(
                        core_truth, text_core_calibrated
                    ),
                },
                "speech": {
                    "raw": _probability_metrics(core_truth, speech_core_raw),
                    "temperature_scaled": _probability_metrics(
                        core_truth, speech_core_calibrated
                    ),
                },
                "fusion_and_conflicts": _conflict_summaries(fusion_rows),
                "fallback_threshold_sensitivity": {
                    str(threshold): _threshold_metrics(fusion_rows, threshold)
                    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8)
                },
            },
        }
        if split == "dev":
            threshold_results = metrics_by_split[split][
                "mapped_four_class_subset"
            ]["fallback_threshold_sensitivity"]
            selected_threshold = max(
                (0.4, 0.5, 0.6, 0.7, 0.8),
                key=lambda value: (
                    threshold_results[str(value)]["macro_f1"],
                    -abs(value - 0.6),
                ),
            )
        for row_index, row in enumerate(all_rows):
            core_label = row["label"]
            if core_label is None:
                text_core_json = ""
                speech_core_json = ""
            else:
                core_index = core_rows.index(row)
                text_core_json = json.dumps(
                    [
                        float(value)
                        for value in calibrated_core["text"][split][core_index]
                    ]
                )
                speech_core_json = json.dumps(
                    [
                        float(value)
                        for value in calibrated_core["speech"][split][core_index]
                    ]
                )
            prediction_rows.append(
                {
                    "split": split,
                    "dialogue_id": row["dialogue_id"],
                    "episode_id": row["episode_id"],
                    "utterance_id": str(row["utterance_index"]),
                    "true_label_7": row["source_emotion"],
                    "true_label_4": core_label or "",
                    "text_probs_7": json.dumps(
                        [
                            float(value)
                            for value in text_probabilities[split][row_index]
                        ]
                    ),
                    "speech_probs_7": json.dumps(
                        [
                            float(value)
                            for value in speech_probabilities[split][row_index]
                        ]
                    ),
                    "text_probs_4_calibrated": text_core_json,
                    "speech_probs_4_calibrated": speech_core_json,
                }
            )

    episode_sets = {
        split: {row["episode_id"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    test_selected_threshold = _threshold_metrics(
        [
            {
                "true_label": row["label"],
                "text_probs": json.dumps(
                    [
                        float(value)
                        for value in calibrated_core["text"]["test"][index]
                    ]
                ),
                "speech_probs": json.dumps(
                    [
                        float(value)
                        for value in calibrated_core["speech"]["test"][index]
                    ]
                ),
            }
            for index, row in enumerate(test_core_rows)
        ],
        selected_threshold,
    )
    result = {
        "evidence_type": "measured_using_released_meld_branch_checkpoints",
        "dataset": "MELD",
        "dataset_source": "https://github.com/declare-lab/MELD",
        "dataset_source_revision": "e8cedf27b5d2877e198332c957127e16eb214afe",
        "publisher_class_order": list(class_order),
        "label_mapping": {
            "joy": "happy",
            "sadness": "sad",
            "anger": "angry",
            "neutral": "neutral",
            "excluded_from_four_class_fusion": ["disgust", "fear", "surprise"],
        },
        "split_method": "official MELD train/dev/test dialogue splits",
        "split_counts": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "source_emotion_counts": {
            "train": _read_source_label_counts(train_csv),
            "dev": _read_source_label_counts(dev_csv),
            "test": _read_source_label_counts(test_csv),
        },
        "source_episode_overlap": {
            "train_dev": len(episode_sets["train"] & episode_sets["dev"]),
            "train_test": len(episode_sets["train"] & episode_sets["test"]),
            "dev_test": len(episode_sets["dev"] & episode_sets["test"]),
        },
        "models": {
            "text": {
                "checkpoint": "MELD publisher text_weights_emotion.hdf5",
                "hidden_features": "MELD publisher text_emotion.pkl",
                "output_head": "600-dimensional contextual state to 7-class softmax",
                "checkpoint_sha256": _sha256(text_weights_hdf5),
                "hidden_features_sha256": _sha256(text_features_pickle),
            },
            "speech": {
                "checkpoint": "MELD publisher audio_weights_emotion.hdf5",
                "hidden_features": "MELD publisher audio_emotion.pkl",
                "output_head": "300-to-150 ReLU-to-7-class softmax",
                "checkpoint_sha256": _sha256(speech_weights_hdf5),
                "hidden_features_sha256": _sha256(speech_features_pickle),
            },
            "source_label_map_sha256": _sha256(label_pickle),
            "prediction_method": (
                "Apply the released Keras dense output head(s) to the publisher's "
                "saved utterance-level hidden states; no model is retrained."
            ),
        },
        "calibration": {
            "method": "scalar temperature scaling over a fixed 201-value geometric grid",
            "grid": {"min": 0.25, "max": 4.0, "points": 201},
            "fit_split": "dev, restricted to the mapped four-class subset",
            "temperatures": temperatures,
        },
        "fusion": {
            "fixed_paper_rule": {
                "text_primary_weight": 0.7,
                "speech_primary_weight": 0.7,
                "asr_confidence_available": False,
                "fallback_threshold": 0.6,
            },
            "validation_selected_fallback_threshold": selected_threshold,
            "test_metrics_at_validation_selected_threshold": test_selected_threshold,
        },
        "metrics": metrics_by_split,
        "limitations": [
            "MELD is conversational emotion recognition, not music-track or listener-mood recognition.",
            "Four-class fusion excludes disgust, fear, and surprise; full seven-class branch metrics are also reported.",
            "Official splits share source TV episodes, so episode-level leakage may inflate held-out scores.",
            "The released contextual hidden states use dialogue-level context and may include future utterances.",
            "These are MELD publisher baseline models and do not reproduce the original manuscript's RoBERTa, DeBERTa, WavLM, or Whisper claims.",
            "No music recommendation or listener outcomes are evaluated.",
        ],
    }
    return result, prediction_rows


def evaluate_meld_episode_disjoint(
    train_csv: Path,
    dev_csv: Path,
    test_csv: Path,
    predictions_csv: Path,
) -> dict:
    """Score official-test episodes absent from both model-training and calibration splits."""
    train_rows = _read_all_rows(train_csv, "train")
    dev_rows = _read_all_rows(dev_csv, "dev")
    test_rows = _read_all_rows(test_csv, "test")
    seen_episodes = {
        row["episode_id"] for row in train_rows + dev_rows
    }
    selected_rows = [
        row for row in test_rows if row["episode_id"] not in seen_episodes
    ]
    if not selected_rows:
        raise ValueError("no test episodes are disjoint from train and dev")
    episode_ids = sorted({row["episode_id"] for row in selected_rows})
    if len(episode_ids) < 10:
        return {
            "dataset": "MELD",
            "evidence_type": "episode_disjoint_audit",
            "subset_rule": (
                "official test episodes absent from both official train and dev episode sets"
            ),
            "episodes": {
                "selected_test_episode_count": len(episode_ids),
                "selected_test_episode_ids": episode_ids,
                "train_episode_count": len({row["episode_id"] for row in train_rows}),
                "dev_episode_count": len({row["episode_id"] for row in dev_rows}),
                "official_test_episode_count": len(
                    {row["episode_id"] for row in test_rows}
                ),
            },
            "sample_counts": {
                "all_seven_class": len(selected_rows),
                "mapped_four_class": sum(
                    row["label"] is not None for row in selected_rows
                ),
                "official_test_total": len(test_rows),
            },
            "metrics": None,
            "metrics_status": (
                "omitted: fewer than 10 independent test episodes; this subset is "
                "too small for a meaningful episode-disjoint performance estimate"
            ),
            "minimum_independent_episodes_for_metrics": 10,
        }

    selected_ids = {row["id"]: row for row in selected_rows}
    predictions = {}
    with Path(predictions_csv).open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        reader = csv.DictReader(source)
        required = {
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
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("unexpected MELD predictions CSV columns")
        for row in reader:
            if row["split"] != "test":
                continue
            sample_id = row["dialogue_id"] + "_" + row["utterance_id"]
            if sample_id not in selected_ids:
                continue
            if sample_id in predictions:
                raise ValueError("duplicate MELD prediction for " + sample_id)
            annotation = selected_ids[sample_id]
            if (
                row["episode_id"] != annotation["episode_id"]
                or row["true_label_7"] != annotation["source_emotion"]
                or row["true_label_4"] != (annotation["label"] or "")
            ):
                raise ValueError("MELD prediction metadata mismatch for " + sample_id)
            predictions[sample_id] = row
    missing = set(selected_ids) - set(predictions)
    if missing:
        raise ValueError(
            "missing episode-disjoint MELD predictions, first missing ID: "
            + sorted(missing)[0]
        )

    selected_predictions = [predictions[row["id"]] for row in selected_rows]

    def probabilities(field: str, selected: Sequence[dict]) -> np.ndarray:
        parsed = []
        for row in selected:
            try:
                values = np.asarray(json.loads(row[field]), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid MELD probability vector in " + field) from error
            if (
                values.ndim != 1
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or not np.isclose(np.sum(values), 1.0, atol=1e-5)
            ):
                raise ValueError("invalid MELD probability vector in " + field)
            parsed.append(values)
        if len({len(values) for values in parsed}) != 1:
            raise ValueError("inconsistent MELD probability dimensions in " + field)
        return np.vstack(parsed)

    full_truth = [row["source_emotion"] for row in selected_rows]
    text_full = probabilities("text_probs_7", selected_predictions)
    speech_full = probabilities("speech_probs_7", selected_predictions)
    core_indices = [
        index for index, row in enumerate(selected_rows) if row["label"] is not None
    ]
    core_rows = [selected_rows[index] for index in core_indices]
    core_predictions = [selected_predictions[index] for index in core_indices]
    core_truth = [row["label"] for row in core_rows]
    text_core = probabilities("text_probs_4_calibrated", core_predictions)
    speech_core = probabilities("speech_probs_4_calibrated", core_predictions)
    fusion_rows = [
        {
            "true_label": row["label"],
            "text_probs": json.dumps([float(value) for value in text_core[index]]),
            "speech_probs": json.dumps(
                [float(value) for value in speech_core[index]]
            ),
        }
        for index, row in enumerate(core_rows)
    ]
    return {
        "dataset": "MELD",
        "evidence_type": "measured_on_episode_disjoint_official_test_subset",
        "subset_rule": (
            "official test episodes absent from both official train and dev episode sets"
        ),
        "episodes": {
            "selected_test_episode_count": len(episode_ids),
            "selected_test_episode_ids": episode_ids,
            "train_episode_count": len({row["episode_id"] for row in train_rows}),
            "dev_episode_count": len({row["episode_id"] for row in dev_rows}),
            "official_test_episode_count": len(
                {row["episode_id"] for row in test_rows}
            ),
        },
        "sample_counts": {
            "all_seven_class": len(selected_rows),
            "mapped_four_class": len(core_rows),
            "official_test_total": len(test_rows),
        },
        "metrics_status": "reported on the selected episode-disjoint subset",
        "minimum_independent_episodes_for_metrics": 10,
        "calibration": (
            "Uses the same dev-fitted branch temperatures recorded in the supplied "
            "checkpoint prediction CSV; no parameters are fit on this subset."
        ),
        "metrics": {
            "full_seven_class": {
                "text": _all_class_metrics(
                    full_truth, text_full, _PUBLISHER_LABEL_ORDER
                ),
                "speech": _all_class_metrics(
                    full_truth, speech_full, _PUBLISHER_LABEL_ORDER
                ),
            },
            "mapped_four_class": {
                "text": _probability_metrics(core_truth, text_core),
                "speech": _probability_metrics(core_truth, speech_core),
                "fusion_and_conflicts": _conflict_summaries(fusion_rows),
                "fixed_fallback_threshold_0_6": _threshold_metrics(
                    fusion_rows, 0.6
                ),
            },
        },
        "limitations": [
            "This is a post-hoc subset of the released checkpoint's official test predictions, not a model retrained with episode-grouped splits.",
            "Contextual hidden states may use future turns within an episode.",
            "MELD is conversational emotion recognition and does not evaluate music recommendation or listener outcomes.",
        ],
    }
