"""Confidence-aware emotion fusion and hierarchical label mapping."""

from dataclasses import dataclass
from typing import Optional


COARSE_LABELS = ("happy", "sad", "angry", "neutral")

FINE_TO_COARSE = {
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    "disappointment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "sadness": "sad",
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",
    "neutral": "neutral",
}


@dataclass(frozen=True)
class Prediction:
    """A model prediction with confidence values in the closed interval [0, 1]."""

    label: str
    confidence: float
    auxiliary_confidence: float = 0.0
    fine_label: Optional[str] = None
    fine_confidence: float = 0.0

    def __post_init__(self):
        if self.label not in COARSE_LABELS:
            raise ValueError("label must be one of: " + ", ".join(COARSE_LABELS))
        for name, value in (
            ("confidence", self.confidence),
            ("auxiliary_confidence", self.auxiliary_confidence),
            ("fine_confidence", self.fine_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(name + " must be between 0 and 1")


@dataclass(frozen=True)
class FusionConfig:
    text_primary_weight: float = 0.7
    text_auxiliary_weight: float = 0.3
    speech_primary_weight: float = 0.7
    asr_weight: float = 0.3
    coarse_threshold: float = 0.6
    fine_threshold: float = 0.75
    asr_threshold: float = 0.7

    def __post_init__(self):
        for name in (
            "text_primary_weight",
            "text_auxiliary_weight",
            "speech_primary_weight",
            "asr_weight",
            "coarse_threshold",
            "fine_threshold",
            "asr_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(name + " must be between 0 and 1")


@dataclass(frozen=True)
class FusionResult:
    label: str
    confidence: float
    fine_label: Optional[str]
    sentiment: Optional[str]
    modality: str
    fallback: bool
    text_trust: Optional[float]
    speech_trust: Optional[float]


def fuse_emotions(
    text: Optional[Prediction],
    speech: Optional[Prediction],
    asr_confidence: Optional[float] = None,
    *,
    text_is_asr: bool = False,
    sentiment: Optional[str] = None,
    config: FusionConfig = FusionConfig(),
    apply_fallback: bool = True,
) -> FusionResult:
    """Fuse available branch predictions using the paper's fixed trust rule."""
    if asr_confidence is not None and not 0.0 <= asr_confidence <= 1.0:
        raise ValueError("asr_confidence must be between 0 and 1")

    if text_is_asr and (
        asr_confidence is None or asr_confidence < config.asr_threshold
    ):
        text = None

    text_trust = None
    speech_trust = None
    if text is not None:
        text_trust = (
            config.text_primary_weight * text.confidence
            + config.text_auxiliary_weight * text.auxiliary_confidence
        )
    if speech is not None:
        speech_trust = config.speech_primary_weight * speech.confidence
        speech_trust += config.asr_weight * (asr_confidence or 0.0)

    if text is None and speech is None:
        raise ValueError("at least one usable text or speech prediction is required")

    if text is None:
        label = speech.label
        confidence = speech_trust
        modality = "speech-only"
        fine_label = None
    elif speech is None:
        label = text.label
        confidence = text_trust
        modality = "text-only"
        fine_label = text.fine_label
    elif text.label == speech.label:
        label = text.label
        confidence = max(text_trust, speech_trust)
        modality = "agreement"
        fine_label = text.fine_label
    elif text_trust > speech_trust:
        label = text.label
        confidence = (text_trust + speech_trust) / 2.0
        modality = "text-dominant"
        fine_label = text.fine_label
    else:
        label = speech.label
        confidence = (text_trust + speech_trust) / 2.0
        modality = "speech-dominant"
        fine_label = None

    fallback = apply_fallback and confidence < config.coarse_threshold
    if fallback:
        label = "neutral"
        fine_label = None

    if (
        text is None
        or text.fine_label is None
        or text.fine_confidence < config.fine_threshold
        or confidence < config.coarse_threshold
        or FINE_TO_COARSE.get(text.fine_label.lower()) != label
    ):
        fine_label = None

    return FusionResult(
        label=label,
        confidence=confidence,
        fine_label=fine_label,
        sentiment=sentiment if text is not None else None,
        modality=modality,
        fallback=fallback,
        text_trust=text_trust,
        speech_trust=speech_trust,
    )
