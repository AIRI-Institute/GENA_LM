"""Process- and context-local runtime configuration for :mod:`gena_expression`."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Literal


OutputRetention = Literal["full", "scalars"]
TrackRetention = Literal["all", "delta", "none"]
FeatureRetention = Literal["full", "scores", "none"]


class RetainedDataError(RuntimeError):
    """Raised when an operation needs data omitted by the retention policy."""


class RetentionMode(str, Enum):
    """Named retention presets exposed by the public package API."""

    FULL = "full"
    SCALARS = "scalars"


@dataclass(frozen=True, slots=True)
class PredictionRetention:
    """Choose which fields remain on returned prediction objects."""

    sequence: bool = True
    logits: bool = True
    outputs: OutputRetention = "full"
    tokens: bool = True
    description_tokens: bool = True
    provenance: bool = True

    def __post_init__(self) -> None:
        """Validate the prediction-output retention choice."""

        if self.outputs not in {"full", "scalars"}:
            raise ValueError("outputs must be 'full' or 'scalars'.")


@dataclass(frozen=True, slots=True)
class ScoringRetention:
    """Choose which fields remain on scoring results and reports."""

    prediction: bool = True
    score_window: bool = True
    tracks: TrackRetention = "all"
    features: FeatureRetention = "full"
    warnings: bool = True
    provenance: bool = True

    def __post_init__(self) -> None:
        """Validate track and feature retention choices."""

        if self.tracks not in {"all", "delta", "none"}:
            raise ValueError("tracks must be 'all', 'delta', or 'none'.")
        if self.features not in {"full", "scores", "none"}:
            raise ValueError("features must be 'full', 'scores', or 'none'.")


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Granular retention choices for predictions and scoring results."""

    prediction: PredictionRetention = PredictionRetention()
    scoring: ScoringRetention = ScoringRetention()


FULL_RETENTION = RetentionPolicy()
SCALAR_RETENTION = RetentionPolicy(
    prediction=PredictionRetention(
        sequence=False,
        logits=False,
        outputs="scalars",
        tokens=False,
        description_tokens=False,
        provenance=False,
    ),
    scoring=ScoringRetention(
        prediction=False,
        score_window=True,
        tracks="none",
        features="scores",
        warnings=True,
        provenance=False,
    ),
)

RetentionLike = RetentionPolicy | RetentionMode | str | None
_RETENTION_POLICY: ContextVar[RetentionPolicy] = ContextVar(
    "gena_expression_retention_policy",
    default=FULL_RETENTION,
)


def retention_policy_for(value: RetentionLike) -> RetentionPolicy:
    """Normalize a preset name or policy object to a retention policy."""

    if value is None:
        return _RETENTION_POLICY.get()
    if isinstance(value, RetentionPolicy):
        return value
    try:
        mode = RetentionMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in RetentionMode)
        raise ValueError(f"Unknown retention mode {value!r}; choose {choices}.") from exc
    return FULL_RETENTION if mode is RetentionMode.FULL else SCALAR_RETENTION


def get_retention_policy() -> RetentionPolicy:
    """Return the active retention policy for the current execution context."""

    return _RETENTION_POLICY.get()


def set_retention_mode(mode: RetentionMode | str) -> RetentionPolicy:
    """Set a named package-level retention preset and return the new policy."""

    policy = retention_policy_for(mode)
    _RETENTION_POLICY.set(policy)
    return policy


def set_retention_policy(policy: RetentionPolicy) -> RetentionPolicy:
    """Set a granular package-level retention policy and return it."""

    if not isinstance(policy, RetentionPolicy):
        raise TypeError("policy must be a RetentionPolicy instance.")
    _RETENTION_POLICY.set(policy)
    return policy


@contextmanager
def retention_mode(mode: RetentionLike) -> Iterator[RetentionPolicy]:
    """Temporarily use a retention mode or policy inside a ``with`` block."""

    policy = retention_policy_for(mode)
    token = _RETENTION_POLICY.set(policy)
    try:
        yield policy
    finally:
        _RETENTION_POLICY.reset(token)
