"""Hybrid genre ranking and a 32-feature LinUCB contextual bandit."""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sonar.fusion import COARSE_LABELS


GENRES = (
    "pop",
    "dance",
    "rock",
    "acoustic",
    "ambient",
    "classical",
    "jazz",
    "hiphop",
    "metal",
    "folk",
)

EMOTION_GENRES = {
    "happy": ("pop", "dance"),
    "sad": ("acoustic", "classical", "ambient"),
    "angry": ("rock", "metal", "hiphop"),
    "neutral": ("jazz", "folk", "pop"),
}


@dataclass(frozen=True)
class Track:
    track_id: str
    genre: str
    affinities: Tuple[float, float, float, float]
    popularity: float
    artist_id: str


@dataclass
class UserHistory:
    genre_rewards: Dict[str, List[float]] = field(default_factory=dict)
    engagement: List[float] = field(default_factory=list)

    def record(self, genre: str, reward: float) -> None:
        self.genre_rewards.setdefault(genre, []).append(reward)
        self.engagement.append(reward)
        self.engagement = self.engagement[-8:]

    def collaborative_score(self, genre: str) -> float:
        values = self.genre_rewards.get(genre, [])
        if not values:
            return 0.5
        return (sum(values) / len(values) + 1.0) / 2.0


class LinUCB:
    def __init__(
        self,
        actions: Sequence[str] = GENRES,
        dimension: int = 32,
        exploration: float = 0.1,
        decay: float = 0.01,
    ):
        if not actions or dimension <= 0 or exploration < 0 or decay < 0:
            raise ValueError("invalid LinUCB dimensions or parameters")
        self.actions = tuple(actions)
        self.dimension = dimension
        self.exploration = exploration
        self.decay = decay
        self.design = {action: np.eye(dimension) for action in self.actions}
        self.inverse_design = {action: np.eye(dimension) for action in self.actions}
        self.response = {action: np.zeros(dimension) for action in self.actions}
        self.step = 0

    def select(self, context: Sequence[float], candidates: Sequence[str]) -> str:
        vector = _context_vector(context, self.dimension)
        if not candidates:
            raise ValueError("candidate actions must not be empty")
        unknown = set(candidates) - set(self.actions)
        if unknown:
            raise ValueError("unknown actions: " + ", ".join(sorted(unknown)))
        eta = self.exploration * np.exp(-self.decay * self.step)
        scored = []
        for action in candidates:
            inverse = self.inverse_design[action]
            theta = inverse @ self.response[action]
            uncertainty = np.sqrt(max(0.0, vector @ inverse @ vector))
            scored.append((float(theta @ vector + eta * uncertainty), action))
        return max(scored)[1]

    def update(self, action: str, context: Sequence[float], reward: float) -> None:
        if action not in self.design:
            raise ValueError("unknown action: " + action)
        vector = _context_vector(context, self.dimension)
        inverse = self.inverse_design[action]
        projected = inverse @ vector
        self.inverse_design[action] = inverse - np.outer(projected, projected) / (
            1.0 + vector @ projected
        )
        self.design[action] += np.outer(vector, vector)
        self.response[action] += reward * vector
        self.step += 1


def make_context(
    emotion: str,
    confidence: float,
    *,
    sentiment: float = 0.0,
    time_of_day: float = 0.5,
    day_of_week: float = 0.0,
    engagement: Sequence[float] = (),
    fallback: bool = False,
) -> np.ndarray:
    if emotion not in COARSE_LABELS:
        raise ValueError("emotion must be one of: " + ", ".join(COARSE_LABELS))
    for name, value in (
        ("confidence", confidence),
        ("time_of_day", time_of_day),
        ("day_of_week", day_of_week),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(name + " must be between 0 and 1")
    if not -1.0 <= sentiment <= 1.0:
        raise ValueError("sentiment must be between -1 and 1")
    if len(engagement) > 8:
        raise ValueError("at most eight engagement values are supported")

    coarse = [float(emotion == label) for label in COARSE_LABELS]
    fine_embedding = [0.0] * 12
    fine_embedding[COARSE_LABELS.index(emotion)] = 1.0
    time_features = [
        float(np.sin(2.0 * np.pi * time_of_day)),
        float(np.cos(2.0 * np.pi * time_of_day)),
    ]
    day_features = [
        float(np.sin(2.0 * np.pi * day_of_week)),
        float(np.cos(2.0 * np.pi * day_of_week)),
    ]
    engagement_features = [0.0] * 8
    padded = [max(-1.0, min(1.0, float(value))) for value in engagement][-8:]
    if padded:
        engagement_features[-len(padded) :] = padded
    vector = np.asarray(
        coarse
        + fine_embedding
        + [sentiment, confidence]
        + time_features
        + day_features
        + engagement_features
        + [1.0, float(fallback)],
        dtype=float,
    )
    return _context_vector(vector, 32)


def rank_hybrid(
    tracks: Sequence[Track],
    emotion: str,
    history: UserHistory,
    *,
    rule_weight: float = 0.1,
    content_weight: float = 0.8,
    collaborative_weight: float = 0.1,
    candidate_limit: int = 10,
) -> Tuple[List[str], Dict[str, float], Dict[str, List[Track]]]:
    if emotion not in COARSE_LABELS:
        raise ValueError("unsupported emotion: " + emotion)
    if not tracks:
        raise ValueError("catalog must not be empty")
    if min(rule_weight, content_weight, collaborative_weight) < 0:
        raise ValueError("component weights must be non-negative")
    total_weight = rule_weight + content_weight + collaborative_weight
    if total_weight <= 0:
        raise ValueError("at least one component weight must be positive")

    emotion_index = COARSE_LABELS.index(emotion)
    desired_genres = EMOTION_GENRES[emotion]
    raw_content = [track.affinities[emotion_index] for track in tracks]
    normalized_content = _minmax(raw_content)
    raw_collab = [history.collaborative_score(track.genre) for track in tracks]
    normalized_collab = _minmax(raw_collab)
    raw_rule = [1.0 if track.genre in desired_genres else 0.25 for track in tracks]
    normalized_rule = _minmax(raw_rule)

    scores = {}
    tracks_by_genre = {}
    for i, track in enumerate(tracks):
        score = (
            rule_weight * normalized_rule[i]
            + content_weight * normalized_content[i]
            + collaborative_weight * normalized_collab[i]
        ) / total_weight
        scores[track.track_id] = float(score)
        tracks_by_genre.setdefault(track.genre, []).append(track)

    genre_scores = {}
    for genre, items in tracks_by_genre.items():
        top_items = _diverse_rank(items, scores, limit=10)
        genre_scores[genre] = sum(scores[item.track_id] for item in top_items) / len(top_items)
        tracks_by_genre[genre] = top_items
    candidates = sorted(genre_scores, key=lambda genre: (-genre_scores[genre], genre))
    candidates = candidates[:candidate_limit]
    return candidates, genre_scores, tracks_by_genre


def _diverse_rank(tracks: Sequence[Track], scores: Mapping[str, float], limit: int) -> List[Track]:
    remaining = list(tracks)
    ranked = []
    artist_counts: Dict[str, int] = {}
    while remaining and len(ranked) < limit:
        chosen = max(
            remaining,
            key=lambda track: (
                scores[track.track_id] - 0.05 * artist_counts.get(track.artist_id, 0),
                track.track_id,
            ),
        )
        ranked.append(chosen)
        artist_counts[chosen.artist_id] = artist_counts.get(chosen.artist_id, 0) + 1
        remaining.remove(chosen)
    return ranked


def _minmax(values: Sequence[float]) -> List[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _context_vector(context: Sequence[float], dimension: int) -> np.ndarray:
    vector = np.asarray(context, dtype=float)
    if vector.shape != (dimension,):
        raise ValueError("context must have exactly " + str(dimension) + " features")
    if not np.all(np.isfinite(vector)):
        raise ValueError("context features must be finite")
    return vector
