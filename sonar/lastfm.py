"""Offline artist-ranking benchmark for the HetRec 2011 Last.fm data."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags


def _read_listening(path: Path) -> Dict[int, Dict[int, float]]:
    listening: Dict[int, Dict[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"userID", "artistID", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Last.fm listening file must contain userID, artistID, and weight")
        for line_number, row in enumerate(reader, start=2):
            try:
                user_id = int(row["userID"])
                artist_id = int(row["artistID"])
                weight = float(row["weight"])
            except (TypeError, ValueError) as error:
                raise ValueError("invalid listening row at line " + str(line_number)) from error
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("listening weights must be finite and positive")
            user_items = listening.setdefault(user_id, {})
            if artist_id in user_items:
                raise ValueError("duplicate user-artist listening row")
            user_items[artist_id] = weight
    if not listening:
        raise ValueError("Last.fm listening file is empty")
    return listening


def _read_training_tags(
    path: Path,
    train_pairs: set,
    user_indices: Mapping[int, int],
    artist_indices: Mapping[int, int],
) -> Tuple[List[int], List[int]]:
    tag_indices: Dict[int, int] = {}
    tag_rows: List[int] = []
    tag_columns: List[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"userID", "artistID", "tagID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Last.fm tag file must contain userID, artistID, and tagID"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                user_id = int(row["userID"])
                artist_id = int(row["artistID"])
                tag_id = int(row["tagID"])
            except (TypeError, ValueError) as error:
                raise ValueError("invalid tag row at line " + str(line_number)) from error
            if (user_id, artist_id) not in train_pairs:
                continue
            tag_column = tag_indices.setdefault(tag_id, len(tag_indices))
            tag_rows.append(artist_indices[artist_id])
            tag_columns.append(tag_column)
    return tag_rows, tag_columns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_scores(
    method: str,
    user_index: int,
    listening_matrix: csr_matrix,
    binary_matrix: csr_matrix,
    content_matrix: csr_matrix,
    item_similarity: csr_matrix,
    popularity: np.ndarray,
) -> np.ndarray:
    if method == "popularity":
        return popularity.copy()
    if method == "tag_content":
        profile = listening_matrix.getrow(user_index) @ content_matrix
        scores = content_matrix @ profile.T
        return np.asarray(scores.toarray()).reshape(-1)
    scores = binary_matrix.getrow(user_index) @ item_similarity
    return np.asarray(scores.toarray()).reshape(-1)


def _rank(scores: np.ndarray, history: np.ndarray, cutoff: int) -> np.ndarray:
    scores = scores.copy()
    scores[history] = -np.inf
    available = np.flatnonzero(np.isfinite(scores))
    count = min(cutoff, len(available))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    if len(available) <= count:
        selected = available
    else:
        values = scores[available]
        threshold = np.partition(values, len(values) - count)[len(values) - count]
        above = available[values > threshold]
        tied = available[values == threshold]
        tied = np.sort(tied)[: count - len(above)]
        selected = np.concatenate((above, tied))
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def _ranking_metrics(
    recommendations: Sequence[Sequence[int]],
    relevant_items: Sequence[set],
    cutoff: int,
    catalog_size: int,
) -> dict:
    recalls = []
    precisions = []
    ndcgs = []
    average_precisions = []
    recommended_items = set()
    for ranking, relevant in zip(recommendations, relevant_items):
        hits = 0
        dcg = 0.0
        precision_sum = 0.0
        for rank, item in enumerate(ranking[:cutoff], start=1):
            recommended_items.add(int(item))
            if item in relevant:
                hits += 1
                precision_sum += hits / rank
                dcg += 1.0 / np.log2(rank + 1.0)
        ideal_hits = min(len(relevant), cutoff)
        ideal_dcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / cutoff)
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        average_precisions.append(
            precision_sum / min(len(relevant), cutoff) if relevant else 0.0
        )
    return {
        "recall_at_10": float(np.mean(recalls)),
        "precision_at_10": float(np.mean(precisions)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "map_at_10": float(np.mean(average_precisions)),
        "catalog_coverage_at_10": len(recommended_items) / catalog_size,
    }


def evaluate_lastfm(
    listening_path: Path,
    tags_path: Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    minimum_user_interactions: int = 5,
    cutoff: int = 10,
) -> dict:
    """Compare popularity, artist-tag content, and item-item CF on held-out listens."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if minimum_user_interactions < 2:
        raise ValueError("minimum_user_interactions must be at least 2")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    listening_path = Path(listening_path)
    tags_path = Path(tags_path)
    all_listening = _read_listening(listening_path)
    eligible = {
        user_id: items
        for user_id, items in all_listening.items()
        if len(items) >= minimum_user_interactions
    }
    if not eligible:
        raise ValueError("no users meet minimum_user_interactions")

    random = np.random.default_rng(seed)
    train_by_user: Dict[int, Dict[int, float]] = {}
    test_by_user: Dict[int, set] = {}
    for user_id in sorted(eligible):
        items = eligible[user_id]
        artist_ids = np.asarray(sorted(items), dtype=np.int64)
        test_count = max(1, int(round(len(artist_ids) * test_fraction)))
        test_count = min(test_count, len(artist_ids) - 1)
        permutation = random.permutation(len(artist_ids))
        test_ids = set(int(value) for value in artist_ids[permutation[:test_count]])
        test_by_user[user_id] = test_ids
        train_by_user[user_id] = {
            artist_id: items[artist_id]
            for artist_id in items
            if artist_id not in test_ids
        }

    user_ids = sorted(train_by_user)
    artist_ids = sorted(
        {artist_id for items in train_by_user.values() for artist_id in items}
    )
    user_indices = {user_id: index for index, user_id in enumerate(user_ids)}
    artist_indices = {artist_id: index for index, artist_id in enumerate(artist_ids)}
    train_pairs = {
        (user_id, artist_id)
        for user_id, items in train_by_user.items()
        for artist_id in items
    }

    train_rows = []
    train_columns = []
    train_weights = []
    raw_train_weights = []
    for user_id, items in train_by_user.items():
        for artist_id, weight in items.items():
            train_rows.append(user_indices[user_id])
            train_columns.append(artist_indices[artist_id])
            train_weights.append(np.log1p(weight))
            raw_train_weights.append(weight)
    shape = (len(user_ids), len(artist_ids))
    listening_matrix = csr_matrix(
        (train_weights, (train_rows, train_columns)), shape=shape
    )
    binary_matrix = csr_matrix(
        (np.ones(len(train_rows)), (train_rows, train_columns)), shape=shape
    )
    popularity = np.log1p(
        np.bincount(
            train_columns,
            weights=raw_train_weights,
            minlength=len(artist_ids),
        )
    )

    tag_rows, tag_columns = _read_training_tags(
        tags_path, train_pairs, user_indices, artist_indices
    )
    if tag_columns:
        content_matrix = csr_matrix(
            (
                np.ones(len(tag_rows)),
                (tag_rows, tag_columns),
            ),
            shape=(len(artist_ids), max(tag_columns) + 1),
        )
        content_matrix.sum_duplicates()
        content_matrix.data[:] = 1.0
        document_frequency = np.asarray(
            content_matrix.getnnz(axis=0)
        ).reshape(-1)
        inverse_document_frequency = np.log(
            (1.0 + len(artist_ids)) / (1.0 + document_frequency)
        ) + 1.0
        content_matrix = content_matrix @ diags(inverse_document_frequency)
        row_norms = np.sqrt(
            np.asarray(content_matrix.multiply(content_matrix).sum(axis=1)).reshape(-1)
        )
        row_norms[row_norms == 0] = 1.0
        content_matrix = diags(1.0 / row_norms) @ content_matrix
        content_matrix = content_matrix.tocsr()
    else:
        content_matrix = csr_matrix((len(artist_ids), 0), dtype=float)

    cooccurrence = (binary_matrix.T @ binary_matrix).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    artist_frequency = np.asarray(binary_matrix.sum(axis=0)).reshape(-1)
    inverse_sqrt_frequency = np.zeros_like(artist_frequency, dtype=float)
    observed = artist_frequency > 0
    inverse_sqrt_frequency[observed] = 1.0 / np.sqrt(artist_frequency[observed])
    item_similarity = (
        diags(inverse_sqrt_frequency)
        @ cooccurrence
        @ diags(inverse_sqrt_frequency)
    ).tocsr()

    relevant_items = []
    user_histories = []
    test_catalog_items = 0
    total_test_items = 0
    for user_id in user_ids:
        test_ids = test_by_user[user_id]
        total_test_items += len(test_ids)
        relevant = {
            artist_indices[artist_id]
            for artist_id in test_ids
            if artist_id in artist_indices
        }
        test_catalog_items += len(relevant)
        relevant_items.append(relevant)
        user_histories.append(binary_matrix.getrow(user_indices[user_id]).indices)

    methods = ("popularity", "tag_content", "item_item")
    rankings = {method: [] for method in methods}
    for user_index, history in enumerate(user_histories):
        for method in methods:
            scores = _ranking_scores(
                method,
                user_index,
                listening_matrix,
                binary_matrix,
                content_matrix,
                item_similarity,
                popularity,
            )
            rankings[method].append(_rank(scores, history, cutoff))

    results = {
        method: _ranking_metrics(
            rankings[method], relevant_items, cutoff, len(artist_ids)
        )
        for method in methods
    }
    return {
        "dataset": "HetRec 2011 Last.fm 2K",
        "evidence_type": "observational_offline_artist_ranking",
        "warning": (
            "Randomly held-out listening events are not randomized exposures. "
            "The dataset has no mood labels or exposure propensities; these metrics "
            "do not establish emotion-aware or causal listener benefit."
        ),
        "license": "non-commercial use only, as stated in the dataset readme",
        "protocol": {
            "split": "independent random per-user interaction holdout",
            "train_fraction": 1.0 - test_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "minimum_user_interactions": minimum_user_interactions,
            "cutoff": cutoff,
            "relevance": "binary held-out artist listening relations",
            "candidate_catalog": "artists with at least one training interaction",
            "out_of_catalog_test_items_count_as_misses": True,
            "tag_provenance": (
                "artist tags from training users, restricted to user-artist pairs "
                "present in the listening training split"
            ),
            "methods": {
                "popularity": "descending log of total training listening counts",
                "tag_content": (
                    "cosine similarity between a log-count-weighted user history "
                    "profile and L2-normalized artist TF-IDF tag vectors"
                ),
                "item_item": (
                    "sum of cosine-normalized artist co-listening similarities "
                    "from the binary training user-artist matrix"
                ),
            },
        },
        "counts": {
            "users_in_source": len(all_listening),
            "users_evaluated": len(user_ids),
            "users_excluded_below_minimum": len(all_listening) - len(user_ids),
            "training_interactions": len(train_rows),
            "test_interactions": total_test_items,
            "test_interactions_in_candidate_catalog": test_catalog_items,
            "test_interactions_outside_candidate_catalog": (
                total_test_items - test_catalog_items
            ),
            "candidate_artists": len(artist_ids),
            "training_tag_assignments": len(tag_rows),
            "tag_vocabulary_size": content_matrix.shape[1],
        },
        "source_files": {
            str(listening_path.name): _sha256(listening_path),
            str(tags_path.name): _sha256(tags_path),
        },
        "metrics_macro_averaged_by_user": results,
    }


def write_lastfm_result(path: Path, result: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def evaluate_lastfm_runs(
    listening_path: Path,
    tags_path: Path,
    *,
    seed: int = 42,
    seeds: int = 5,
) -> dict:
    """Summarize repeated random interaction holdouts without tuning on test outcomes."""
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    runs = [
        evaluate_lastfm(listening_path, tags_path, seed=seed + offset)
        for offset in range(seeds)
    ]
    methods = tuple(runs[0]["metrics_macro_averaged_by_user"])
    metric_names = tuple(
        runs[0]["metrics_macro_averaged_by_user"][methods[0]]
    )
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for metric in metric_names:
            values = np.asarray(
                [
                    run["metrics_macro_averaged_by_user"][method][metric]
                    for run in runs
                ],
                dtype=np.float64,
            )
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return {
        "dataset": runs[0]["dataset"],
        "evidence_type": runs[0]["evidence_type"],
        "warning": runs[0]["warning"],
        "license": runs[0]["license"],
        "protocol": runs[0]["protocol"],
        "seed_range": list(range(seed, seed + seeds)),
        "reported_seed": seed,
        "reported_seed_counts": runs[0]["counts"],
        "source_files": runs[0]["source_files"],
        "aggregate_metrics_macro_averaged_by_user": aggregate,
        "per_seed_metrics": [
            {
                "seed": run["protocol"]["seed"],
                "counts": run["counts"],
                "metrics_macro_averaged_by_user": run[
                    "metrics_macro_averaged_by_user"
                ],
            }
            for run in runs
        ],
        "uncertainty_note": (
            "Mean and sample standard deviation describe variation across random "
            "interaction splits, not uncertainty over independent users or causal effects."
        ),
    }
