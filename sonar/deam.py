"""Track-level music-emotion regression evaluation on the DEAM dataset."""

import csv
import io
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_ANNOTATION_MEMBERS = (
    "annotations/annotations averaged per song/song_level/"
    "static_annotations_averaged_songs_1_2000.csv",
    "annotations/annotations averaged per song/song_level/"
    "static_annotations_averaged_songs_2000_2058.csv",
)


def _read_annotations(archive: zipfile.ZipFile) -> Dict[int, Tuple[float, float]]:
    annotations = {}
    for member in _ANNOTATION_MEMBERS:
        with io.TextIOWrapper(
            archive.open(member), encoding="utf-8-sig", newline=""
        ) as source:
            reader = csv.DictReader(source, skipinitialspace=True)
            required = {"song_id", "valence_mean", "arousal_mean"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("unexpected DEAM annotation columns in " + member)
            for row in reader:
                try:
                    track_id = int(row["song_id"])
                    valence = float(row["valence_mean"])
                    arousal = float(row["arousal_mean"])
                except (TypeError, ValueError) as error:
                    raise ValueError("invalid DEAM annotation row in " + member) from error
                if not math.isfinite(valence) or not math.isfinite(arousal):
                    raise ValueError("non-finite DEAM annotation for track " + str(track_id))
                if track_id in annotations:
                    raise ValueError("duplicate DEAM annotation for track " + str(track_id))
                annotations[track_id] = (valence, arousal)
    if not annotations:
        raise ValueError("DEAM annotation archive contains no song-level annotations")
    return annotations


def _read_track_features(
    archive: zipfile.ZipFile,
    track_id: int,
    expected_columns: Tuple[str, ...],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    member = "features/" + str(track_id) + ".csv"
    try:
        source = io.TextIOWrapper(
            archive.open(member), encoding="utf-8-sig", newline=""
        )
    except KeyError as error:
        raise ValueError("missing DEAM features for track " + str(track_id)) from error

    with source:
        reader = csv.reader(source, delimiter=";")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("empty DEAM feature file for track " + str(track_id)) from error
        if not header or header[0] != "frameTime" or len(header) < 2:
            raise ValueError("unexpected DEAM feature columns for track " + str(track_id))
        columns = tuple(header[1:])
        if len(set(columns)) != len(columns):
            raise ValueError("duplicate DEAM feature columns for track " + str(track_id))
        if expected_columns and columns != expected_columns:
            raise ValueError("inconsistent DEAM feature columns for track " + str(track_id))

        mean = np.zeros(len(columns), dtype=np.float64)
        second_moment = np.zeros(len(columns), dtype=np.float64)
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError("invalid DEAM feature row for track " + str(track_id))
            try:
                values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
            except ValueError as error:
                raise ValueError("non-numeric DEAM feature for track " + str(track_id)) from error
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite DEAM feature for track " + str(track_id))
            count += 1
            delta = values - mean
            mean += delta / count
            second_moment += delta * (values - mean)

    if count == 0:
        raise ValueError("DEAM feature file has no frames for track " + str(track_id))
    standard_deviation = np.sqrt(second_moment / count)
    return np.concatenate((mean, standard_deviation)), columns


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    difference = prediction - truth
    mean_truth = float(np.mean(truth))
    mean_prediction = float(np.mean(prediction))
    covariance = float(np.mean((truth - mean_truth) * (prediction - mean_prediction)))
    denominator = (
        float(np.var(truth))
        + float(np.var(prediction))
        + (mean_truth - mean_prediction) ** 2
    )
    concordance = 1.0 if denominator == 0.0 else 2.0 * covariance / denominator
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "ccc": concordance,
    }


def _fit_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_mean = np.mean(train_features, axis=0)
    feature_scale = np.std(train_features, axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    train_x = (train_features - feature_mean) / feature_scale
    test_x = (test_features - feature_mean) / feature_scale

    target_mean = np.mean(train_targets, axis=0)
    target_scale = np.std(train_targets, axis=0)
    target_scale[target_scale == 0.0] = 1.0
    train_y = (train_targets - target_mean) / target_scale

    weights = np.linalg.solve(
        train_x.T @ train_x + alpha * np.eye(train_x.shape[1]),
        train_x.T @ train_y,
    )
    prediction = (test_x @ weights) * target_scale + target_mean
    baseline = np.broadcast_to(target_mean, prediction.shape).copy()
    return prediction, baseline


def evaluate_deam(
    annotations_zip: Path,
    features_zip: Path,
    *,
    folds: int = 5,
    seed: int = 42,
    alpha: float = 1.0,
) -> Dict[str, object]:
    """Evaluate fixed-alpha ridge regression with song-disjoint cross-validation."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a finite positive number")

    with zipfile.ZipFile(annotations_zip) as annotation_archive:
        annotations = _read_annotations(annotation_archive)

    feature_rows: List[np.ndarray] = []
    target_rows: List[Tuple[float, float]] = []
    track_ids = sorted(annotations)
    expected_columns = ()
    with zipfile.ZipFile(features_zip) as feature_archive:
        for track_id in track_ids:
            features, expected_columns = _read_track_features(
                feature_archive, track_id, expected_columns
            )
            feature_rows.append(features)
            target_rows.append(annotations[track_id])

    if len(track_ids) < folds:
        raise ValueError("number of annotated tracks must be at least the fold count")
    features = np.vstack(feature_rows)
    targets = np.asarray(target_rows, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("DEAM features contain non-finite aggregated values")

    permutation = np.random.default_rng(seed).permutation(len(track_ids))
    test_folds = np.array_split(permutation, folds)
    predictions = np.empty_like(targets)
    baseline_predictions = np.empty_like(targets)
    for test_indices in test_folds:
        train_indices = np.setdiff1d(permutation, test_indices, assume_unique=True)
        predictions[test_indices], baseline_predictions[test_indices] = _fit_ridge(
            features[train_indices],
            targets[train_indices],
            features[test_indices],
            alpha,
        )

    names = ("valence", "arousal")
    ridge_metrics = {
        name: _regression_metrics(targets[:, index], predictions[:, index])
        for index, name in enumerate(names)
    }
    baseline_metrics = {
        name: _regression_metrics(targets[:, index], baseline_predictions[:, index])
        for index, name in enumerate(names)
    }
    return {
        "evidence_type": "measured_on_public_deam_annotations_and_features",
        "dataset": "DEAM",
        "source": "https://cvml.unige.ch/databases/DEAM/",
        "tracks": len(track_ids),
        "track_id_range": [track_ids[0], track_ids[-1]],
        "feature_columns": len(expected_columns),
        "track_feature_aggregation": "mean_and_standard_deviation_over_frames",
        "targets": "song-level mean valence and arousal annotations",
        "cross_validation": {
            "method": "shuffled song-disjoint K-fold",
            "folds": folds,
            "seed": seed,
            "ridge_alpha": alpha,
            "preprocessing": "feature and target scaling fit within each training fold",
        },
        "metrics": {
            "ridge": ridge_metrics,
            "training_mean_baseline": baseline_metrics,
        },
    }
