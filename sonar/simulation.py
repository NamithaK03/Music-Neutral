"""Synthetic-user evaluation for the hybrid recommender and LinUCB policy."""

import math
import json
import random
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sonar.fusion import COARSE_LABELS, FusionConfig, Prediction, fuse_emotions
from sonar.metrics import accuracy, expected_calibration_error, macro_f1
from sonar.recommender import (
    GENRES,
    LinUCB,
    Track,
    UserHistory,
    make_context,
    rank_hybrid,
)


def generate_catalog(seed: int, tracks_per_genre: int = 16) -> List[Track]:
    if tracks_per_genre < 10:
        raise ValueError("at least 10 tracks per genre are required for top-10 ranking")
    rng = np.random.default_rng(seed)
    tracks = []
    for genre_index, genre in enumerate(GENRES):
        center = np.zeros(4)
        center[genre_index % 4] = 0.65
        for item_index in range(tracks_per_genre):
            affinity = np.clip(rng.normal(center, 0.25), 0.0, 1.0)
            tracks.append(
                Track(
                    track_id=genre + "-" + str(item_index),
                    genre=genre,
                    affinities=tuple(float(value) for value in affinity),
                    popularity=float(rng.random()),
                    artist_id="artist-" + str(item_index % 8),
                )
            )
    return tracks


def run_seed(
    seed: int,
    interactions: int = 300,
    with_bandit: bool = True,
    *,
    exploration: float = 0.1,
    decay: float = 0.01,
    rule_weight: float = 0.1,
    content_weight: float = 0.8,
    collaborative_weight: float = 0.1,
) -> Dict[str, float]:
    if interactions <= 0:
        raise ValueError("interactions must be positive")
    rng = random.Random(seed)
    catalog = generate_catalog(seed)
    history = UserHistory()
    policy = LinUCB(exploration=exploration, decay=decay)
    user_preferences = {
        genre: rng.uniform(-0.2, 0.2) for genre in GENRES
    }
    repair_probability = rng.uniform(0.2, 0.8)
    observed_skips = 0
    total_reward = 0.0
    ndcg_values = []
    recommended_ids = set()

    for step in range(interactions):
        true_emotion = rng.choice(COARSE_LABELS)
        emotion = true_emotion
        confidence = rng.uniform(0.65, 0.98)
        if rng.random() < 0.12:
            alternatives = [label for label in COARSE_LABELS if label != true_emotion]
            emotion = rng.choice(alternatives)
            confidence = rng.uniform(0.35, 0.7)

        text_prediction = Prediction(emotion, confidence, auxiliary_confidence=confidence)
        speech_prediction = Prediction(emotion, confidence)
        fused = fuse_emotions(
            text_prediction,
            speech_prediction,
            asr_confidence=confidence,
            config=FusionConfig(),
        )
        context = make_context(
            fused.label,
            fused.confidence,
            sentiment=(1.0 if fused.label == "happy" else -1.0 if fused.label in ("sad", "angry") else 0.0),
            time_of_day=(step % 24) / 24.0,
            day_of_week=(step % 7) / 7.0,
            engagement=history.engagement,
            fallback=fused.fallback,
        )

        candidates, genre_scores, tracks_by_genre = rank_hybrid(
            catalog,
            fused.label,
            history,
            rule_weight=rule_weight,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
        )
        if with_bandit:
            selected_genre = policy.select(context, candidates)
        else:
            selected_genre = max(candidates, key=lambda genre: (genre_scores[genre], genre))
        playlist = tracks_by_genre[selected_genre][:10]
        if not playlist:
            raise RuntimeError("candidate genre has no ranked tracks")

        true_target = _repair_target(true_emotion, rng.random() < repair_probability)
        relevant_scores = {
            track.track_id: _utility(
                track,
                true_target,
                user_preferences,
            )
            for track in catalog
        }
        relevance_threshold = 0.55
        relevant = {
            track_id for track_id, value in relevant_scores.items() if value >= relevance_threshold
        }
        dcg = sum(
            (1.0 if track.track_id in relevant else 0.0) / math.log2(rank + 2)
            for rank, track in enumerate(playlist)
        )
        ideal = sorted(relevant_scores.values(), reverse=True)[:10]
        idcg = sum(
            (1.0 if value >= relevance_threshold else 0.0) / math.log2(rank + 2)
            for rank, value in enumerate(ideal)
        )
        ndcg_values.append(dcg / idcg if idcg else 0.0)

        recommended_ids.update(track.track_id for track in playlist)
        top_track = playlist[0]
        satisfaction = _sigmoid(
            4.0
            * (
                _utility(
                    top_track,
                    true_target,
                    user_preferences,
                )
                - 0.55
            )
        )
        if satisfaction < 0.35:
            reward = -1.0
            observed_skips += 1
        elif satisfaction < 0.8:
            reward = 0.0
        elif rng.random() < 0.12:
            reward = 1.0
        else:
            reward = 0.5
        history.record(selected_genre, reward)
        if with_bandit:
            policy.update(selected_genre, context, reward)
        total_reward += reward

    return {
        "mean_reward": total_reward / interactions,
        "skip_rate": observed_skips / interactions,
        "ndcg_at_10": mean(ndcg_values),
        "catalog_coverage": len(recommended_ids) / len(catalog),
    }


def run_experiment(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    interactions: int = 300,
    *,
    exploration: float = 0.1,
    decay: float = 0.01,
    rule_weight: float = 0.1,
    content_weight: float = 0.8,
    collaborative_weight: float = 0.1,
):
    if len(seeds) == 0:
        raise ValueError("at least one seed is required")
    variants = {}
    for name, with_bandit in (("hybrid_no_bandit", False), ("hybrid_linucb", True)):
        runs = [
            run_seed(
                seed,
                interactions,
                with_bandit,
                exploration=exploration,
                decay=decay,
                rule_weight=rule_weight,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
            )
            for seed in seeds
        ]
        variants[name] = {
            "runs": runs,
            "mean_reward": _summarize(runs, "mean_reward"),
            "skip_rate": _summarize(runs, "skip_rate"),
            "ndcg_at_10": _summarize(runs, "ndcg_at_10"),
            "catalog_coverage": _summarize(runs, "catalog_coverage"),
        }
    return {
        "evidence_type": "new_synthetic_simulation",
        "warning": (
            "Synthetic results from this implementation are not the manuscript's "
            "reported results and are not evidence about real listeners."
        ),
        "seed_count": len(seeds),
        "interactions_per_seed": interactions,
        "seeds": list(seeds),
        "variants": variants,
    }


def tune_experiment(
    validation_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    test_seeds: Sequence[int] = (142, 143, 144, 145, 146),
    interactions: int = 300,
):
    """Tune on validation seeds and report the selected setting on disjoint test seeds."""
    if not validation_seeds or not test_seeds:
        raise ValueError("validation and test seeds must both be non-empty")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("validation and test seeds must be disjoint")
    if interactions <= 0:
        raise ValueError("interactions must be positive")

    weight_sets = (
        {"rule_weight": 0.3, "content_weight": 0.4, "collaborative_weight": 0.3},
        {"rule_weight": 0.15, "content_weight": 0.65, "collaborative_weight": 0.2},
        {"rule_weight": 0.1, "content_weight": 0.8, "collaborative_weight": 0.1},
    )
    parameter_grid = [
        {
            "exploration": exploration,
            "decay": decay,
            **weights,
        }
        for weights in weight_sets
        for exploration in (0.1, 0.5, 1.0)
        for decay in (0.001, 0.01, 0.03)
    ]

    validation = []
    for parameters in parameter_grid:
        runs = [
            run_seed(seed, interactions, True, **parameters)
            for seed in validation_seeds
        ]
        validation.append(
            {
                "parameters": parameters,
                "ndcg_at_10": _summarize(runs, "ndcg_at_10"),
                "skip_rate": _summarize(runs, "skip_rate"),
            }
        )
    selected = min(
        validation,
        key=lambda item: (
            -item["ndcg_at_10"]["mean"],
            item["skip_rate"]["mean"],
            item["parameters"]["exploration"],
            item["parameters"]["decay"],
        ),
    )
    parameters = selected["parameters"]
    selected_runs = [
        run_seed(seed, interactions, True, **parameters) for seed in test_seeds
    ]
    selected_no_bandit = [
        run_seed(seed, interactions, False, **parameters) for seed in test_seeds
    ]
    paper_default = {
        "exploration": 1.0,
        "decay": 0.01,
        "rule_weight": 0.3,
        "content_weight": 0.4,
        "collaborative_weight": 0.3,
    }
    default_runs = [
        run_seed(seed, interactions, True, **paper_default) for seed in test_seeds
    ]
    return {
        "evidence_type": "new_synthetic_simulation_with_validation_tuning",
        "selection_rule": (
            "Highest validation mean NDCG@10; ties broken by lower validation skip rate."
        ),
        "warning": (
            "This is a synthetic simulator result, not a reproduction of the manuscript's "
            "results or evidence about actual listeners. Test seeds are disjoint from tuning "
            "seeds, but share the same simulator assumptions."
        ),
        "interactions_per_seed": interactions,
        "validation_seeds": list(validation_seeds),
        "test_seeds": list(test_seeds),
        "grid_size": len(parameter_grid),
        "selected_parameters": parameters,
        "validation_selected": selected,
        "held_out_test": {
            "hybrid_linucb_tuned": _summarize_runs(selected_runs),
            "hybrid_no_bandit_same_weights": _summarize_runs(selected_no_bandit),
            "hybrid_linucb_paper_defaults": _summarize_runs(default_runs),
        },
        "validation_grid": validation,
    }


def _summarize_runs(runs):
    return {
        metric: _summarize(runs, metric)
        for metric in ("mean_reward", "skip_rate", "ndcg_at_10", "catalog_coverage")
    }


def evaluate_fusion_rows(rows: Sequence[dict], bins: int = 15) -> Dict[str, dict]:
    """Evaluate four-class predictions; probability arrays follow COARSE_LABELS order."""
    if not rows:
        raise ValueError("at least one prediction row is required")
    truth = [row["true_label"] for row in rows]
    if any(label not in COARSE_LABELS for label in truth):
        raise ValueError("true_label values must use the four coarse labels")
    prepared = []
    for row in rows:
        text, text_probs = _prediction_from_row(row, "text")
        speech, speech_probs = _prediction_from_row(row, "speech")
        asr_confidence = _optional_float(row.get("asr_confidence"))
        if _is_true(row.get("text_is_asr")) and (
            asr_confidence is None or asr_confidence < FusionConfig().asr_threshold
        ):
            text, text_probs = None, None
        if text is None and speech is None:
            raise ValueError("each row needs at least one usable modality prediction")
        prepared.append((row, text, text_probs, speech, speech_probs, asr_confidence))

    outputs = {}
    for variant in ("naive", "confidence_aware", "confidence_aware_fallback"):
        predicted, confidence = [], []
        for row, text, text_probs, speech, speech_probs, asr_confidence in prepared:
            if variant == "naive":
                if text_probs is not None and speech_probs is not None:
                    average = [
                        (text_probs[i] + speech_probs[i]) / 2.0
                        for i in range(len(COARSE_LABELS))
                    ]
                    class_index = max(range(len(average)), key=average.__getitem__)
                    label, score = COARSE_LABELS[class_index], average[class_index]
                else:
                    available_probs = text_probs if text_probs is not None else speech_probs
                    class_index = max(
                        range(len(available_probs)), key=available_probs.__getitem__
                    )
                    label, score = COARSE_LABELS[class_index], available_probs[class_index]
            else:
                fused = fuse_emotions(
                    text,
                    speech,
                    asr_confidence=asr_confidence,
                    sentiment=row.get("sentiment"),
                    apply_fallback=variant == "confidence_aware_fallback",
                )
                label, score = fused.label, fused.confidence
            predicted.append(label)
            confidence.append(score)
        outputs[variant] = {
            "accuracy": accuracy(truth, predicted),
            "macro_f1": macro_f1(truth, predicted),
            "ece_15": expected_calibration_error(truth, predicted, confidence, bins),
        }

    conflict_rows = [
        item
        for item in prepared
        if item[1] is not None and item[3] is not None and item[1].label != item[3].label
    ]
    if conflict_rows:
        conflict_truth = [item[0]["true_label"] for item in conflict_rows]
        for variant in outputs:
            if variant == "naive":
                conflict_predictions = []
                for _, _, text_probs, _, speech_probs, _ in conflict_rows:
                    average = [
                        (text_probs[i] + speech_probs[i]) / 2.0
                        for i in range(len(COARSE_LABELS))
                    ]
                    conflict_predictions.append(
                        COARSE_LABELS[max(range(len(average)), key=average.__getitem__)]
                    )
            else:
                conflict_predictions = [
                    fuse_emotions(
                        text,
                        speech,
                        asr_confidence=asr_confidence,
                        apply_fallback=variant == "confidence_aware_fallback",
                    ).label
                    for _, text, _, speech, _, asr_confidence in conflict_rows
                ]
            outputs[variant]["conflict_accuracy"] = accuracy(
                conflict_truth, conflict_predictions
            )
        outputs["meta"] = {"conflict_sample_count": len(conflict_rows)}
    else:
        outputs["meta"] = {"conflict_sample_count": 0}
    return outputs


def _prediction_from_row(row, prefix):
    raw = row.get(prefix + "_probs")
    if raw in (None, ""):
        return None, None
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
        probabilities = [float(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(prefix + "_probs must be a JSON array of four probabilities") from error
    if (
        len(probabilities) != len(COARSE_LABELS)
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-4)
    ):
        raise ValueError(prefix + "_probs must contain four probabilities summing to 1")
    class_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    prediction = Prediction(
        label=COARSE_LABELS[class_index],
        confidence=probabilities[class_index],
        auxiliary_confidence=float(row.get(prefix + "_aux_confidence") or 0.0),
        fine_label=row.get(prefix + "_fine_label") or None,
        fine_confidence=float(row.get(prefix + "_fine_confidence") or 0.0),
    )
    return prediction, probabilities


def _optional_float(value):
    return None if value in (None, "") else float(value)


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def _summarize(runs, key):
    values = [run[key] for run in runs]
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def _repair_target(emotion, repair):
    if not repair:
        return emotion
    if emotion == "sad":
        return "happy"
    if emotion == "angry":
        return "neutral"
    return emotion


def _utility(track, target, preferences):
    affinity = track.affinities[COARSE_LABELS.index(target)]
    return (
        0.5 * affinity
        + 0.3 * ((preferences[track.genre] + 0.2) / 0.4)
        + 0.2 * track.popularity
    )


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
