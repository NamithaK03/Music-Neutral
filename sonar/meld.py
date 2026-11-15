"""Paired text/audio emotion evaluation using the public MELD feature release."""

import collections
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sonar.fusion import COARSE_LABELS, FusionConfig, Prediction, fuse_emotions
from sonar.metrics import accuracy, expected_calibration_error, macro_f1
from sonar.simulation import evaluate_fusion_rows


_MELD_LABELS = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "neutral": "neutral",
}
_REQUIRED_COLUMNS = {
    "Utterance",
    "Emotion",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
}
_SOURCE_REVISION = "e8cedf27b5d2877e198332c957127e16eb214afe"
_AUDIO_FEATURE_GLOBALS = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("collections", "defaultdict"): collections.defaultdict,
    ("builtins", "float"): float,
}


class _NumpyFeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _AUDIO_FEATURE_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                "unsupported global in MELD feature pickle: " + module + "." + name
            ) from error


def _read_split(path: Path, split_name: str) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("unexpected MELD annotation columns in " + str(path))
        rows = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                emotion = row["Emotion"].strip().lower()
                label = _MELD_LABELS.get(emotion)
                dialogue_id = str(int(row["Dialogue_ID"]))
                utterance_id = str(int(row["Utterance_ID"]))
                season = str(int(row["Season"]))
                episode = str(int(row["Episode"]))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    "invalid MELD annotation at {}:{}"
                    .format(path, line_number)
                ) from error
            if label is None:
                continue
            text = row["Utterance"].strip()
            if not text:
                raise ValueError(
                    "empty MELD utterance at {}:{}".format(path, line_number)
                )
            sample_id = dialogue_id + "_" + utterance_id
            if sample_id in seen_ids:
                raise ValueError("duplicate MELD utterance ID in " + str(path))
            seen_ids.add(sample_id)
            rows.append(
                {
                    "id": sample_id,
                    "dialogue_id": dialogue_id,
                    "episode_id": season + "_" + episode,
                    "text": text,
                    "label": label,
                    "source_emotion": emotion,
                    "split": split_name,
                }
            )
    if not rows:
        raise ValueError("no four-class MELD examples found in " + str(path))
    return rows


def _read_source_label_counts(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "Emotion" not in reader.fieldnames:
            raise ValueError("MELD annotations are missing the Emotion column")
        counts = Counter(row["Emotion"].strip().lower() for row in reader)
        return dict(sorted(counts.items()))


def _read_audio_features(path: Path) -> Tuple[Dict[str, np.ndarray], ...]:
    with path.open("rb") as source:
        feature_splits = _NumpyFeatureUnpickler(source).load()
    if not isinstance(feature_splits, (list, tuple)) or len(feature_splits) != 3:
        raise ValueError("MELD audio features must contain train, dev, and test maps")

    validated = []
    for split_name, feature_map in zip(("train", "dev", "test"), feature_splits):
        if not isinstance(feature_map, dict) or not feature_map:
            raise ValueError("MELD " + split_name + " audio feature map is empty")
        current = {}
        expected_width = None
        for key, raw_features in feature_map.items():
            if not isinstance(key, str):
                raise ValueError("MELD audio feature IDs must be strings")
            try:
                features = np.asarray(raw_features, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid MELD audio features for " + key) from error
            if (
                features.ndim != 1
                or features.size == 0
                or not np.all(np.isfinite(features))
            ):
                raise ValueError("invalid MELD audio feature vector for " + key)
            if expected_width is None:
                expected_width = features.size
            elif features.size != expected_width:
                raise ValueError("inconsistent MELD audio feature dimensions")
            current[key] = features
        validated.append(current)
    return tuple(validated)


def _aligned_audio(
    rows: Sequence[dict], feature_map: Dict[str, np.ndarray], split_name: str
) -> np.ndarray:
    missing = [row["id"] for row in rows if row["id"] not in feature_map]
    if missing:
        raise ValueError(
            "missing {} MELD audio features, first missing ID: {}"
            .format(split_name, missing[0])
        )
    return np.vstack([feature_map[row["id"]] for row in rows])


def _class_probabilities(model, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    model_labels = list(model.classes_)
    if set(model_labels) != set(COARSE_LABELS):
        raise ValueError("MELD training data must contain all four emotion classes")
    return raw[:, [model_labels.index(label) for label in COARSE_LABELS]]


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _fit_temperature(
    probabilities: np.ndarray, labels: Sequence[str]
) -> Tuple[float, float]:
    label_indices = np.asarray([COARSE_LABELS.index(label) for label in labels])
    candidates = np.geomspace(0.25, 4.0, 201)
    losses = []
    for temperature in candidates:
        scaled = _temperature_scale(probabilities, float(temperature))
        losses.append(
            float(
                -np.mean(
                    np.log(
                        np.clip(
                            scaled[np.arange(len(label_indices)), label_indices],
                            1e-12,
                            1.0,
                        )
                    )
                )
            )
        )
    best = min(range(len(candidates)), key=losses.__getitem__)
    return float(candidates[best]), losses[best]


def _probability_metrics(labels: Sequence[str], probabilities: np.ndarray) -> dict:
    predicted = [
        COARSE_LABELS[index]
        for index in np.argmax(probabilities, axis=1).tolist()
    ]
    confidence = np.max(probabilities, axis=1).tolist()
    indices = np.asarray([COARSE_LABELS.index(label) for label in labels])
    chosen = np.clip(probabilities[np.arange(len(indices)), indices], 1e-12, 1.0)
    return {
        "accuracy": accuracy(labels, predicted),
        "macro_f1": macro_f1(labels, predicted),
        "ece_15": expected_calibration_error(labels, predicted, confidence, bins=15),
        "nll": float(-np.mean(np.log(chosen))),
    }


def _row_dict(row: dict, text_probs: np.ndarray, speech_probs: np.ndarray) -> dict:
    return {
        "true_label": row["label"],
        "text_probs": json.dumps([float(value) for value in text_probs]),
        "speech_probs": json.dumps([float(value) for value in speech_probs]),
    }


def _conflict_summaries(rows: Sequence[dict]) -> dict:
    truth = [row["true_label"] for row in rows]
    text_probabilities = [json.loads(row["text_probs"]) for row in rows]
    speech_probabilities = [json.loads(row["speech_probs"]) for row in rows]
    conflicts = [
        index
        for index, (text_probs, speech_probs) in enumerate(
            zip(text_probabilities, speech_probabilities)
        )
        if int(np.argmax(text_probs)) != int(np.argmax(speech_probs))
    ]

    variants = evaluate_fusion_rows(rows)
    if not conflicts:
        conflict_metrics = {"count": 0}
    else:
        conflict_truth = [truth[index] for index in conflicts]
        conflict_metrics = {
            "count": len(conflicts),
            "text_accuracy": accuracy(
                conflict_truth,
                [
                    COARSE_LABELS[int(np.argmax(text_probabilities[index]))]
                    for index in conflicts
                ],
            ),
            "speech_accuracy": accuracy(
                conflict_truth,
                [
                    COARSE_LABELS[int(np.argmax(speech_probabilities[index]))]
                    for index in conflicts
                ],
            ),
        }
    return {"fusion": variants, "conflict_subset": conflict_metrics}


def _threshold_metrics(rows: Sequence[dict], threshold: float) -> dict:
    truth = []
    predicted = []
    confidence = []
    fallback_count = 0
    conflict_truth = []
    conflict_predictions = []
    for row in rows:
        text_probs = np.asarray(json.loads(row["text_probs"]), dtype=np.float64)
        speech_probs = np.asarray(json.loads(row["speech_probs"]), dtype=np.float64)
        text_index = int(np.argmax(text_probs))
        speech_index = int(np.argmax(speech_probs))
        text = Prediction(
            COARSE_LABELS[text_index], float(text_probs[text_index])
        )
        speech = Prediction(
            COARSE_LABELS[speech_index], float(speech_probs[speech_index])
        )
        result = fuse_emotions(
            text,
            speech,
            config=FusionConfig(coarse_threshold=threshold),
            apply_fallback=True,
        )
        truth.append(row["true_label"])
        predicted.append(result.label)
        confidence.append(result.confidence)
        fallback_count += int(result.fallback)
        if text_index != speech_index:
            conflict_truth.append(row["true_label"])
            conflict_predictions.append(result.label)

    output = {
        "accuracy": accuracy(truth, predicted),
        "macro_f1": macro_f1(truth, predicted),
        "ece_trust_15": expected_calibration_error(
            truth, predicted, confidence, bins=15
        ),
        "fallback_rate": fallback_count / len(rows),
    }
    if conflict_truth:
        output["conflict_count"] = len(conflict_truth)
        output["conflict_accuracy"] = accuracy(
            conflict_truth, conflict_predictions
        )
    else:
        output["conflict_count"] = 0
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_meld(
    train_csv: Path,
    dev_csv: Path,
    test_csv: Path,
    audio_features_pickle: Path,
) -> Tuple[dict, List[dict]]:
    """Train modality-specific MELD classifiers and evaluate on the official test split."""
    try:
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "install the optional MELD study dependencies with "
            'python -m pip install ".[meld-study]"'
        ) from error

    train_rows = _read_split(train_csv, "train")
    dev_rows = _read_split(dev_csv, "dev")
    test_rows = _read_split(test_csv, "test")
    audio_splits = _read_audio_features(audio_features_pickle)
    if len(audio_splits) != 3:
        raise ValueError("MELD requires exactly three audio feature splits")

    train_y = [row["label"] for row in train_rows]
    dev_y = [row["label"] for row in dev_rows]
    if set(train_y) != set(COARSE_LABELS):
        raise ValueError("MELD training split must contain all four mapped labels")

    train_audio = _aligned_audio(train_rows, audio_splits[0], "train")
    dev_audio = _aligned_audio(dev_rows, audio_splits[1], "dev")
    test_audio = _aligned_audio(test_rows, audio_splits[2], "test")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    train_text = vectorizer.fit_transform([row["text"] for row in train_rows])
    dev_text = vectorizer.transform([row["text"] for row in dev_rows])
    test_text = vectorizer.transform([row["text"] for row in test_rows])

    text_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ).fit(train_text, train_y)

    audio_scaler = StandardScaler()
    scaled_train_audio = audio_scaler.fit_transform(train_audio)
    scaled_dev_audio = audio_scaler.transform(dev_audio)
    scaled_test_audio = audio_scaler.transform(test_audio)
    audio_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ).fit(scaled_train_audio, train_y)

    raw_probabilities = {
        "text": {
            "dev": _class_probabilities(text_model, dev_text),
            "test": _class_probabilities(text_model, test_text),
        },
        "speech": {
            "dev": _class_probabilities(audio_model, scaled_dev_audio),
            "test": _class_probabilities(audio_model, scaled_test_audio),
        },
    }
    temperatures = {}
    calibrated_probabilities = {}
    for modality in ("text", "speech"):
        temperature, dev_nll = _fit_temperature(
            raw_probabilities[modality]["dev"], dev_y
        )
        temperatures[modality] = {
            "temperature": temperature,
            "validation_nll_after_scaling": dev_nll,
        }
        calibrated_probabilities[modality] = {
            split: _temperature_scale(values, temperature)
            for split, values in raw_probabilities[modality].items()
        }

    rows_by_split = {"dev": dev_rows, "test": test_rows}
    result_by_split = {}
    prediction_rows = []
    for split, annotation_rows in rows_by_split.items():
        truth = [row["label"] for row in annotation_rows]
        text_calibrated = calibrated_probabilities["text"][split]
        speech_calibrated = calibrated_probabilities["speech"][split]
        rows_for_fusion = [
            _row_dict(row, text_calibrated[index], speech_calibrated[index])
            for index, row in enumerate(annotation_rows)
        ]
        result_by_split[split] = {
            "sample_count": len(annotation_rows),
            "text": {
                "raw": _probability_metrics(
                    truth, raw_probabilities["text"][split]
                ),
                "temperature_scaled": _probability_metrics(
                    truth, text_calibrated
                ),
            },
            "speech": {
                "raw": _probability_metrics(
                    truth, raw_probabilities["speech"][split]
                ),
                "temperature_scaled": _probability_metrics(
                    truth, speech_calibrated
                ),
            },
            "fusion_and_conflicts": _conflict_summaries(rows_for_fusion),
            "fallback_threshold_sensitivity": {
                str(threshold): _threshold_metrics(rows_for_fusion, threshold)
                for threshold in (0.4, 0.5, 0.6, 0.7, 0.8)
            },
        }
        if split == "dev":
            selected_threshold = max(
                (0.4, 0.5, 0.6, 0.7, 0.8),
                key=lambda value: (
                    result_by_split[split]["fallback_threshold_sensitivity"][
                        str(value)
                    ]["macro_f1"],
                    -abs(value - 0.6),
                ),
            )
        for index, (annotation, row) in enumerate(
            zip(annotation_rows, rows_for_fusion)
        ):
            prediction_rows.append(
                {
                    "split": split,
                    "dialogue_id": annotation["dialogue_id"],
                    "episode_id": annotation["episode_id"],
                    "utterance_id": annotation["id"].split("_", 1)[1],
                    "true_label": annotation["label"],
                    "text_probs_raw": json.dumps(
                        [
                            float(value)
                            for value in raw_probabilities["text"][split][index]
                        ]
                    ),
                    "speech_probs_raw": json.dumps(
                        [
                            float(value)
                            for value in raw_probabilities["speech"][split][index]
                        ]
                    ),
                    "text_probs": row["text_probs"],
                    "speech_probs": row["speech_probs"],
                }
            )

    episode_sets = {
        "train": {row["episode_id"] for row in train_rows},
        "dev": {row["episode_id"] for row in dev_rows},
        "test": {row["episode_id"] for row in test_rows},
    }
    result = {
        "evidence_type": "measured_on_public_meld_labels_and_publisher_audio_features",
        "dataset": "MELD",
        "dataset_source": "https://github.com/declare-lab/MELD",
        "dataset_source_revision": _SOURCE_REVISION,
        "feature_archive_source": (
            "https://web.eecs.umich.edu/~mihalcea/downloads/"
            "MELD.Features.Models.tar.gz"
        ),
        "audio_feature_file_sha256": _sha256(audio_features_pickle),
        "label_mapping": {
            "joy": "happy",
            "sadness": "sad",
            "anger": "angry",
            "neutral": "neutral",
            "excluded_source_labels": ["disgust", "fear", "surprise"],
        },
        "splits": {
            "method": "official MELD train/dev/test dialogue splits",
            "counts": {
                "train": len(train_rows),
                "dev": len(dev_rows),
                "test": len(test_rows),
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
        },
        "models": {
            "text": {
                "algorithm": "TF-IDF word unigram/bigram + multinomial logistic regression",
                "vectorizer": {
                    "min_df": 2,
                    "max_features": 50000,
                    "sublinear_tf": True,
                    "fitted_on": "train split only",
                },
            },
            "speech": {
                "algorithm": (
                    "standardized publisher-supplied openSMILE/SVM-selected features "
                    "+ multinomial logistic regression"
                ),
                "feature_dimensions": int(train_audio.shape[1]),
                "scaler_fitted_on": "train split only",
            },
            "classifier": {
                "library": "scikit-learn",
                "version": sklearn.__version__,
                "logistic_regression": {
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "random_state": 42,
                },
            },
        },
        "calibration": {
            "method": "scalar temperature scaling over a fixed 201-value geometric grid",
            "grid": {"min": 0.25, "max": 4.0, "points": 201},
            "fit_split": "dev",
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
            "test_metrics_at_validation_selected_threshold": _threshold_metrics(
                [
                    _row_dict(
                        row,
                        calibrated_probabilities["text"]["test"][index],
                        calibrated_probabilities["speech"]["test"][index],
                    )
                    for index, row in enumerate(test_rows)
                ],
                selected_threshold,
            ),
        },
        "metrics": result_by_split,
        "limitations": [
            "This is conversational emotion recognition, not music-track or listener-mood recognition.",
            "Only joy, sadness, anger, and neutral are included; disgust, fear, and surprise are excluded.",
            "The official splits share source TV episodes, so episode-level leakage may inflate held-out scores.",
            "The publisher-supplied audio vectors use SVM-based feature selection; its exact fit-split provenance is not documented in the release.",
            "The classifiers are new baselines trained on MELD train; they do not reproduce the original manuscript's model results.",
            "The study does not evaluate music recommendation or listener outcomes.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    return result, prediction_rows


def write_predictions_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("at least one MELD prediction row is required")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
