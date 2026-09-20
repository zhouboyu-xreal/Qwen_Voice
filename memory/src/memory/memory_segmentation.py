from dataclasses import dataclass
from datetime import datetime
from collections import deque
import re
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_client import EmbeddingClient
from .utils import _as_embedding_vector, _cal_embedding_cosine_similarity, _sigmoid, _centroid, _cohesion
from .memory_manager import (
    _compact_whitespace,
    _to_timestamp_text,
)

INTERACTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


@dataclass
class SegmentDecision:
    unit_index: int
    reason: str
    cut_probability: Optional[float] = None
    score: Optional[float] = None
    semantic_surprise: Optional[float] = None
    robust_surprise: Optional[float] = None
    absolute_surprise: Optional[float] = None
    cohesion_before: Optional[float] = None
    cohesion_after: Optional[float] = None
    cohesion_drop: Optional[float] = None
    length_signal: Optional[float] = None
    turn_signal: Optional[float] = None
    centroid_similarity: Optional[float] = None
    recent_similarity: Optional[float] = None
    prospective_tokens: Optional[int] = None
    prospective_units: Optional[int] = None
    time_gap_seconds: Optional[float] = None
    scoring_mode: str = "single_unit"
    rolling_window_tail_units: int = 0
    rolling_window_tail_tokens: int = 0


@dataclass
class OnlineSegmentUnit:
    index: int
    text: str
    token_count: int
    timestamp: str = ""
    ended_at: str = ""
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ActiveUnit:
    unit: Any
    embedding: Optional[np.ndarray]


@dataclass
class OnlineSegmentationConfig:
    threshold: float = 0.60
    bias: float = -1.10
    surprise_history_window: int = 64
    min_surprise_history: int = 5
    robust_surprise_weight: float = 0.8
    absolute_surprise_weight: float = 0.8
    cohesion_drop_weight: float = 1.0
    length_weight: float = 0.40
    turn_count_weight: float = 0.40
    max_pending_turns: int = 0
    max_pending_tokens: int = 500
    max_pending_chars: int = 0
    min_pending_tokens: int = 100
    min_pending_turns: int = 2
    min_segment_override_probability: float = 0.90
    max_time_gap_seconds: float = -1.0
    enforce_min_pending_tokens: bool = False
    rolling_window_enabled: bool = False
    rolling_window_tail_units: int = 0
    min_boundary_scoring_incoming_tokens: int = 0


@dataclass
class TranscriptAggregationConfig:
    """Rules for assembling VAD-sized transcript fragments into units."""

    max_gap_seconds: float = 1.0
    min_transcript_unit_tokens: int = 20
    max_transcript_unit_tokens: int = 120
    max_transcript_unit_duration_seconds: float = 20.0
    short_fragment_max_tokens: int = 8


def _estimate_interaction_token_count(text: str) -> int:
    return max(1, len(INTERACTION_TOKEN_RE.findall(str(text or ""))))


def _unit_text_from_turn(turn: Dict[str, Any]) -> str:
    user_text = _compact_whitespace(turn.get("user_message") or "")
    assistant_text = _compact_whitespace(turn.get("assistant_response") or "")
    return f"用户：{user_text}\n助手：{assistant_text}".strip()

def _clipped(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def robust_surprise_signal(
    surprise: float,
    history: Sequence[float],
    min_history: int,
) -> float:
    required_history = max(1, int(min_history))
    if len(history) < required_history:
        return 0.0
    values = np.asarray(list(history), dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, mad)
    return _clipped((surprise - median) / scale, -2.0, 4.0)


def absolute_surprise_signal(surprise: float) -> float:
    return _clipped((surprise - 0.20) / 0.14, -1.0, 2.5)


def _linear(value: float, start: float, end: float, low: float, high: float) -> float:
    if end <= start:
        return high
    ratio = _clipped((value - start) / (end - start), 0.0, 1.0)
    return low + ratio * (high - low)


def length_pressure(token_count: int, config: OnlineSegmentationConfig) -> float:
    min_tokens = max(1, int(config.min_pending_tokens))
    max_tokens = max(min_tokens + 1, int(config.max_pending_tokens))
    length = float(max(0, token_count))
    early_boundary = 0.7 * min_tokens

    if length < early_boundary:
        return -1.30
    if length < min_tokens:
        return _linear(length, early_boundary, min_tokens, -0.80, 0.0)
    if length < max_tokens:
        return _linear(length, min_tokens, max_tokens, 0.45, 2.80)
    return 3.00


def turn_count_pressure(unit_count: int) -> float:
    count = max(1, int(unit_count))
    if count == 1:
        return -0.85
    if count == 2:
        return -0.15
    if count == 3:
        return 0.15
    return min(1.0, 0.30 + 0.15 * (count - 4))

def _build_online_segmentation_config(
    segmentation_config: Dict[str, Any],
    *,
    max_pending_turns_key: str,
    max_pending_turns_default: int,
    max_pending_tokens_key: str,
    max_pending_tokens_default: int,
    min_pending_tokens_key: str,
    min_pending_tokens_default: int,
    min_pending_turns_key: str,
    min_pending_turns_default: int,
    enforce_min_pending_tokens: bool = False,
) -> OnlineSegmentationConfig:
    """Build one semantic segmenter configuration from a source-specific mapping."""
    def config_int(key: str, default: int) -> int:
        value = segmentation_config.get(key)
        if value in (None, ""):
            return default
        return int(value)

    max_gap = segmentation_config.get("max_time_gap_seconds")
    return OnlineSegmentationConfig(
        threshold=float(segmentation_config.get("threshold", 0.60)),
        bias=float(segmentation_config.get("bias", -1.10)),
        surprise_history_window=max(
            1,
            config_int("surprise_history_window", 64),
        ),
        min_surprise_history=max(
            0,
            config_int("min_surprise_history", 5),
        ),
        robust_surprise_weight=float(
            segmentation_config.get("robust_surprise_weight", 0.8),
        ),
        absolute_surprise_weight=float(
            segmentation_config.get("absolute_surprise_weight", 0.8),
        ),
        cohesion_drop_weight=float(
            segmentation_config.get("cohesion_drop_weight", 1.0),
        ),
        length_weight=float(segmentation_config.get("length_weight", 0.40)),
        turn_count_weight=float(
            segmentation_config.get("turn_count_weight", 0.40),
        ),
        max_pending_turns=max(
            1,
            config_int(max_pending_turns_key, max_pending_turns_default),
        ),
        max_pending_tokens=max(
            1,
            config_int(max_pending_tokens_key, max_pending_tokens_default),
        ),
        max_pending_chars=max(
            0,
            config_int("max_pending_transcript_chars", 0),
        ),
        min_pending_tokens=max(
            1,
            config_int(min_pending_tokens_key, min_pending_tokens_default),
        ),
        min_pending_turns=max(
            1,
            config_int(min_pending_turns_key, min_pending_turns_default),
        ),
        min_segment_override_probability=float(
            segmentation_config.get("min_segment_override_probability", 0.90),
        ),
        max_time_gap_seconds=float(-1.0 if max_gap in (None, "") else max_gap),
        enforce_min_pending_tokens=bool(enforce_min_pending_tokens),
        rolling_window_enabled=bool(
            segmentation_config.get("rolling_window_enabled", False),
        ),
        rolling_window_tail_units=max(
            0,
            config_int("rolling_window_tail_units", 0),
        ),
        min_boundary_scoring_incoming_tokens=max(
            0,
            config_int("min_boundary_scoring_incoming_tokens", 0),
        ),
    )


def build_online_segmentation_config(
    runtime_config: Dict[str, Any],
) -> OnlineSegmentationConfig:
    """Build the assistant-wakeup semantic segmentation configuration."""
    segmentation_config = runtime_config.get("assistant_wakeup_segmentation")
    if not isinstance(segmentation_config, dict):
        segmentation_config = {}
    return _build_online_segmentation_config(
        segmentation_config,
        max_pending_turns_key="max_pending_interaction_turns",
        max_pending_turns_default=5,
        max_pending_tokens_key="max_pending_interaction_tokens",
        max_pending_tokens_default=500,
        min_pending_tokens_key="min_pending_interaction_tokens",
        min_pending_tokens_default=100,
        min_pending_turns_key="min_pending_interaction_turns",
        min_pending_turns_default=2,
    )


def build_transcript_segmentation_config(
    runtime_config: Dict[str, Any],
) -> OnlineSegmentationConfig:
    """Build the all-day-recording semantic segmentation configuration."""
    segmentation_config = runtime_config.get("allday_recording_segmentation")
    if not isinstance(segmentation_config, dict):
        segmentation_config = {}
    return _build_online_segmentation_config(
        segmentation_config,
        max_pending_turns_key="max_pending_transcript_units",
        max_pending_turns_default=80,
        max_pending_tokens_key="max_pending_transcript_tokens",
        max_pending_tokens_default=2000,
        min_pending_tokens_key="min_pending_transcript_tokens",
        min_pending_tokens_default=500,
        min_pending_turns_key="min_pending_transcript_units",
        min_pending_turns_default=4,
        enforce_min_pending_tokens=True,
    )


def build_transcript_aggregation_config(
    runtime_config: Dict[str, Any],
) -> TranscriptAggregationConfig:
    """Build VAD-fragment aggregation rules for all-day transcript input."""
    segmentation_config = runtime_config.get("allday_recording_segmentation")
    if not isinstance(segmentation_config, dict):
        segmentation_config = {}
    return TranscriptAggregationConfig(
        max_gap_seconds=float(
            segmentation_config.get("segment_merge_max_gap_seconds", 1.0)
        ),
        min_transcript_unit_tokens=max(
            1,
            int(segmentation_config.get("min_transcript_unit_tokens", 20)),
        ),
        max_transcript_unit_tokens=max(
            1,
            int(segmentation_config.get("max_transcript_unit_tokens", 120)),
        ),
        max_transcript_unit_duration_seconds=max(
            0.0,
            float(
                segmentation_config.get(
                    "max_transcript_unit_duration_seconds",
                    20.0,
                )
            ),
        ),
        short_fragment_max_tokens=max(
            1,
            int(segmentation_config.get("short_fragment_max_tokens", 8)),
        ),
    )


def convert_interaction_turn_to_online_unit(
    turn: Dict[str, Any],
    index: int,
) -> OnlineSegmentUnit:
    """Normalize one interaction turn into the segmenter's unit format."""
    text = _unit_text_from_turn(turn)
    timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or ""
    return OnlineSegmentUnit(
        index=index,
        text=text,
        token_count=_estimate_interaction_token_count(text),
        timestamp=timestamp,
        raw=dict(turn),
    )


class TranscriptUnitAssembler:
    """Assemble adjacent VAD fragments into speaker-safe semantic units."""

    def __init__(
        self,
        config: Optional[TranscriptAggregationConfig] = None,
    ) -> None:
        self.config = config or TranscriptAggregationConfig()
        self._current_segments: List[Dict[str, Any]] = []
        self._current_token_count = 0
        self._next_unit_index = 1

    def append_new_segment(self, segment: Dict[str, Any]) -> Optional[OnlineSegmentUnit]:
        """Append one transcript span and return the prior completed unit."""
        normalized = dict(segment)
        if not self._segment_text(normalized):
            return None
        if not self._current_segments:
            self._start_unit(normalized)
            return None
        if self._can_append(normalized):
            self._append_to_current(normalized)
            return None
        completed = self._take_current_unit()
        self._start_unit(normalized)
        return completed

    def flush(self) -> Optional[OnlineSegmentUnit]:
        """Return the final incomplete unit at an explicit input boundary."""
        return self._take_current_unit()

    def has_pending_segments(self) -> bool:
        """Return whether a not-yet-finalized unit is being assembled."""
        return bool(self._current_segments)

    def _start_unit(self, segment: Dict[str, Any]) -> None:
        self._current_segments = [dict(segment)]
        self._current_token_count = _estimate_interaction_token_count(
            self._segment_text(segment),
        )

    def _append_to_current(self, segment: Dict[str, Any]) -> None:
        self._current_segments.append(dict(segment))
        self._current_token_count += _estimate_interaction_token_count(
            self._segment_text(segment),
        )

    def _take_current_unit(self) -> Optional[OnlineSegmentUnit]:
        if not self._current_segments:
            return None
        raw_segments = [dict(segment) for segment in self._current_segments]
        text = " ".join(
            self._segment_text(segment)
            for segment in raw_segments
            if self._segment_text(segment)
        )
        speaker_labels = list(
            dict.fromkeys(
                self._speaker_label(segment)
                for segment in raw_segments
            )
        )
        unit = OnlineSegmentUnit(
            index=self._next_unit_index,
            text=text,
            token_count=max(1, self._current_token_count),
            timestamp=self._segment_started_at(raw_segments[0]),
            ended_at=self._segment_ended_at(raw_segments[-1]),
            raw={
                "raw_segments": raw_segments,
                "speaker_labels": speaker_labels,
            },
        )
        self._next_unit_index += 1
        self._current_segments = []
        self._current_token_count = 0
        return unit

    def _can_append(self, incoming: Dict[str, Any]) -> bool:
        if not self._current_segments:
            return True
        previous = self._current_segments[-1]
        gap_seconds = OnlineSemanticSegmenter.timestamp_gap_seconds(
            self._segment_ended_at(previous),
            self._segment_started_at(incoming),
        )
        if gap_seconds is None:
            return False
        if (
            self.config.max_gap_seconds >= 0
            and gap_seconds > self.config.max_gap_seconds
        ):
            return False

        incoming_tokens = _estimate_interaction_token_count(
            self._segment_text(incoming),
        )
        if (
            self._current_token_count + incoming_tokens
            > self.config.max_transcript_unit_tokens
        ):
            return False
        duration_seconds = OnlineSemanticSegmenter.timestamp_gap_seconds(
            self._segment_started_at(self._current_segments[0]),
            self._segment_ended_at(incoming),
        )
        if (
            self.config.max_transcript_unit_duration_seconds > 0
            and duration_seconds is not None
            and duration_seconds > self.config.max_transcript_unit_duration_seconds
        ):
            return False

        incoming_speaker = self._speaker_label(incoming)
        known_current_speakers = {
            self._speaker_label(segment)
            for segment in self._current_segments
            if not self._is_unknown_speaker(self._speaker_label(segment))
        }
        if not self._is_unknown_speaker(incoming_speaker):
            if not known_current_speakers or incoming_speaker not in known_current_speakers:
                return False
        elif incoming_tokens > self.config.short_fragment_max_tokens:
            return False

        current_text = " ".join(
            self._segment_text(segment)
            for segment in self._current_segments
        )
        previous_tokens = _estimate_interaction_token_count(
            self._segment_text(previous),
        )
        return bool(
            self._current_token_count < self.config.min_transcript_unit_tokens
            or previous_tokens <= self.config.short_fragment_max_tokens
            or incoming_tokens <= self.config.short_fragment_max_tokens
            or not self._ends_sentence(current_text)
        )

    @staticmethod
    def _segment_text(segment: Dict[str, Any]) -> str:
        return _compact_whitespace(segment.get("text") or "")

    @staticmethod
    def _speaker_label(segment: Dict[str, Any]) -> str:
        return _compact_whitespace(segment.get("speaker") or "unknown_speaker")

    @staticmethod
    def _segment_started_at(segment: Dict[str, Any]) -> str:
        return _to_timestamp_text(segment.get("started_at")) or ""

    @classmethod
    def _segment_ended_at(cls, segment: Dict[str, Any]) -> str:
        return _to_timestamp_text(segment.get("ended_at")) or cls._segment_started_at(
            segment,
        )

    @staticmethod
    def _is_unknown_speaker(value: str) -> bool:
        return str(value or "").strip().lower() in {
            "",
            "unknown",
            "unknown_speaker",
        }

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return str(text or "").rstrip().endswith(("。", "！", "？", ".", "!", "?"))


class OnlineSemanticSegmenter:
    """Embedding-based online semantic boundary detector for dialogue units."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        config: Optional[OnlineSegmentationConfig] = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.config = config or OnlineSegmentationConfig()
        self.surprise_history: Deque[float] = deque(
            maxlen=max(1, self.config.surprise_history_window),
        )
        self._pending_online_units: List[OnlineSegmentUnit] = []
        self._pending_online_units_embedding: List[Optional[np.ndarray]] = []

    def append_pending_unit(
        self,
        unit: OnlineSegmentUnit,
        embedding: Optional[np.ndarray],
    ) -> None:
        """Append an unit to the segmenter's pending online buffer."""
        self._pending_online_units.append(unit)
        self._pending_online_units_embedding.append(
            _as_embedding_vector(embedding),
        )

    def pending_unit_snapshot(self) -> List[OnlineSegmentUnit]:
        """Return a shallow copy of the current pending online units."""
        return list(self._pending_online_units)

    def has_pending_units(self) -> bool:
        """Return whether the segmenter has units waiting for storage."""
        return bool(self._pending_online_units)

    def clear_pending_units(self) -> None:
        """Clear units after the corresponding memory task was queued."""
        self._pending_online_units.clear()
        self._pending_online_units_embedding.clear()

    def embed_unit(self, unit: Any) -> ActiveUnit:
        embedding = self.embedding_client.embed_text(self.unit_text(unit))
        vector = _as_embedding_vector(embedding)
        return ActiveUnit(unit=unit, embedding=vector)

    def should_finalize_pending_units(
        self,
        incoming_unit: Any,
        incoming_embedding: Optional[np.ndarray],
    ) -> Tuple[bool, SegmentDecision]:
        active_units = self._pending_online_units
        time_gap_seconds = (
            self.unit_time_gap_seconds(active_units[-1], incoming_unit)
            if active_units
            else None
        )
        if not active_units:
            decision = SegmentDecision(
                unit_index=self.unit_index(incoming_unit),
                reason="start_segment",
                prospective_tokens=self.unit_token_count(incoming_unit),
                prospective_units=1,
                time_gap_seconds=time_gap_seconds,
            )
            return False, decision
        existing_capacity_reason = self._pending_capacity_reason(active_units)
        if existing_capacity_reason:
            return True, SegmentDecision(
                unit_index=self.unit_index(incoming_unit),
                reason=existing_capacity_reason,
                prospective_tokens=sum(
                    self.unit_token_count(item) for item in active_units
                ),
                prospective_units=len(active_units),
                time_gap_seconds=time_gap_seconds,
            )
        if self.time_gap_exceeded(time_gap_seconds):
            return True, SegmentDecision(
                unit_index=self.unit_index(incoming_unit),
                reason="time_gap",
                prospective_tokens=sum(
                    self.unit_token_count(item) for item in active_units
                ),
                prospective_units=len(active_units),
                time_gap_seconds=time_gap_seconds,
            )
        incoming_token_count = self.unit_token_count(incoming_unit)
        minimum_scoring_tokens = max(
            0,
            int(self.config.min_boundary_scoring_incoming_tokens),
        )
        if (
            minimum_scoring_tokens > 0
            and incoming_token_count < minimum_scoring_tokens
        ):
            return False, SegmentDecision(
                unit_index=self.unit_index(incoming_unit),
                reason="short_incoming_append",
                prospective_tokens=(
                    sum(self.unit_token_count(item) for item in active_units)
                    + incoming_token_count
                ),
                prospective_units=len(active_units) + 1,
                time_gap_seconds=time_gap_seconds,
                scoring_mode="short_incoming_skipped",
            )
        incoming, scoring_context = self._build_boundary_scoring_incoming(
            active_units,
            incoming_unit,
            incoming_embedding,
        )
        if incoming.embedding is None or any(
            embedding is None
            for embedding in self._pending_online_units_embedding
        ):
            return False, SegmentDecision(
                unit_index=self.unit_index(incoming_unit),
                reason="embedding_unavailable",
                prospective_tokens=sum(
                    self.unit_token_count(item) for item in active_units
                ),
                prospective_units=len(active_units),
                time_gap_seconds=time_gap_seconds,
                scoring_mode="embedding_unavailable",
                rolling_window_tail_units=scoring_context["tail_units"],
                rolling_window_tail_tokens=scoring_context["tail_tokens"],
            )
        active = [
            ActiveUnit(unit=unit, embedding=embedding)
            for unit, embedding in zip(
                active_units,
                self._pending_online_units_embedding,
            )
        ]
        decision = self.score_boundary(active, incoming)
        decision.time_gap_seconds = time_gap_seconds
        decision.scoring_mode = scoring_context["scoring_mode"]
        decision.rolling_window_tail_units = scoring_context["tail_units"]
        decision.rolling_window_tail_tokens = scoring_context["tail_tokens"]
        if self.semantic_boundary_allowed(active, decision):
            decision.reason = "semantic_boundary"
            return True, decision
        decision.reason = "append"
        return False, decision

    def _build_boundary_scoring_incoming(
        self,
        active_units: Sequence[Any],
        incoming_unit: Any,
        incoming_embedding: Optional[np.ndarray],
    ) -> Tuple[ActiveUnit, Dict[str, Any]]:
        """Build the incoming embedding used only for boundary scoring."""
        tail_limit = max(0, int(self.config.rolling_window_tail_units))
        if not self.config.rolling_window_enabled or tail_limit <= 0:
            return ActiveUnit(
                unit=incoming_unit,
                embedding=_as_embedding_vector(incoming_embedding),
            ), {
                "scoring_mode": "single_unit",
                "tail_units": 0,
                "tail_tokens": 0,
            }

        tail_units = list(active_units[-tail_limit:])
        texts = [
            self.unit_text(unit)
            for unit in tail_units
            if self.unit_text(unit)
        ]
        incoming_text = self.unit_text(incoming_unit)
        if incoming_text:
            texts.append(incoming_text)
        if len(texts) <= 1:
            return ActiveUnit(
                unit=incoming_unit,
                embedding=_as_embedding_vector(incoming_embedding),
            ), {
                "scoring_mode": "single_unit",
                "tail_units": 0,
                "tail_tokens": 0,
            }

        rolling_text = "\n".join(texts)
        embedding = self.embedding_client.embed_text(rolling_text)
        vector = _as_embedding_vector(embedding)
        if vector is None:
            return ActiveUnit(
                unit=incoming_unit,
                embedding=_as_embedding_vector(incoming_embedding),
            ), {
                "scoring_mode": "single_unit",
                "tail_units": 0,
                "tail_tokens": 0,
            }
        return ActiveUnit(
            unit=incoming_unit,
            embedding=vector,
        ), {
            "scoring_mode": "rolling_window",
            "tail_units": len(tail_units),
            "tail_tokens": sum(
                self.unit_token_count(unit)
                for unit in tail_units
            ),
        }

    def _pending_capacity_reason(
        self,
        units: Sequence[Any],
    ) -> Optional[str]:
        """Return the pending semantic-unit, token, or character limit reason."""
        if not units:
            return None
        if (
            self.config.max_pending_turns > 0
            and len(units) >= self.config.max_pending_turns
        ):
            return "pending_turn_limit"
        token_count = sum(self.unit_token_count(unit) for unit in units)
        if (
            self.config.max_pending_tokens > 0
            and token_count >= self.config.max_pending_tokens
        ):
            return "pending_token_limit"
        char_count = sum(
            len(self.unit_text(unit)) for unit in units
        )
        if (
            self.config.max_pending_chars > 0
            and char_count >= self.config.max_pending_chars
        ):
            return "pending_char_limit"
        return None

    def score_boundary(
        self,
        active: Sequence[ActiveUnit],
        incoming: ActiveUnit,
    ) -> SegmentDecision:
        active_embeddings = [item.embedding for item in active]
        active_centroid = _centroid(active_embeddings)
        recent_embedding = active[-1].embedding
        centroid_sim = _cal_embedding_cosine_similarity(incoming.embedding, active_centroid)
        recent_sim = _cal_embedding_cosine_similarity(incoming.embedding, recent_embedding)
        semantic_surprise = 1.0 - max(centroid_sim, recent_sim)

        robust_surprise = robust_surprise_signal(
            semantic_surprise,
            list(self.surprise_history),
            self.config.min_surprise_history,
        )
        absolute_surprise = absolute_surprise_signal(semantic_surprise)

        cohesion_before = _cohesion(active_embeddings)
        cohesion_after = _cohesion([*active_embeddings, incoming.embedding])
        cohesion_drop = max(0.0, cohesion_before - cohesion_after)
        prospective_tokens = sum(self.unit_token_count(item.unit) for item in active) + self.unit_token_count(incoming.unit)
        prospective_units = len(active) + 1
        length_signal = length_pressure(prospective_tokens, self.config)
        turn_signal = turn_count_pressure(prospective_units)
        score = (
            self.config.robust_surprise_weight * robust_surprise
            + self.config.absolute_surprise_weight * absolute_surprise
            + self.config.cohesion_drop_weight * cohesion_drop
            + self.config.length_weight * length_signal
            + self.config.turn_count_weight * turn_signal
        )
        cut_probability = _sigmoid(self.config.bias + score)
        self.surprise_history.append(float(semantic_surprise))

        return SegmentDecision(
            unit_index=self.unit_index(incoming.unit),
            reason="score",
            cut_probability=cut_probability,
            score=score,
            semantic_surprise=semantic_surprise,
            robust_surprise=robust_surprise,
            absolute_surprise=absolute_surprise,
            cohesion_before=cohesion_before,
            cohesion_after=cohesion_after,
            cohesion_drop=cohesion_drop,
            length_signal=length_signal,
            turn_signal=turn_signal,
            centroid_similarity=centroid_sim,
            recent_similarity=recent_sim,
            prospective_tokens=prospective_tokens,
            prospective_units=prospective_units,
        )

    def semantic_boundary_allowed(
        self,
        active: Sequence[ActiveUnit],
        decision: SegmentDecision,
    ) -> bool:
        if (decision.cut_probability or 0.0) < self.config.threshold:
            return False
        min_units = max(1, int(self.config.min_pending_turns))
        active_token_count = sum(
            self.unit_token_count(item.unit) for item in active
        )
        meets_token_minimum = (
            not self.config.enforce_min_pending_tokens
            or active_token_count >= max(1, int(self.config.min_pending_tokens))
        )
        if len(active) >= min_units and meets_token_minimum:
            return True
        if self.config.enforce_min_pending_tokens:
            return False
        return (decision.cut_probability or 0.0) >= float(
            self.config.min_segment_override_probability,
        )

    def time_gap_exceeded(self, time_gap_seconds: Optional[float]) -> bool:
        return bool(
            self.config.max_time_gap_seconds >= 0
            and time_gap_seconds is not None
            and time_gap_seconds > self.config.max_time_gap_seconds
        )

    @staticmethod
    def unit_text(unit: Any) -> str:
        return str(getattr(unit, "text", "") or "")

    @staticmethod
    def unit_token_count(unit: Any) -> int:
        value = getattr(unit, "token_count", None)
        if value is not None:
            return max(1, int(value))
        return _estimate_interaction_token_count(OnlineSemanticSegmenter.unit_text(unit))

    @staticmethod
    def unit_index(unit: Any) -> int:
        return int(getattr(unit, "index", 0) or 0)

    @staticmethod
    def unit_timestamp(unit: Any) -> str:
        return _to_timestamp_text(getattr(unit, "timestamp", "")) or ""

    @classmethod
    def unit_end_timestamp(cls, unit: Any) -> str:
        return (
            _to_timestamp_text(getattr(unit, "ended_at", ""))
            or cls.unit_timestamp(unit)
        )

    @classmethod
    def unit_time_gap_seconds(
        cls,
        previous_unit: Any,
        incoming_unit: Any,
    ) -> Optional[float]:
        return cls.timestamp_gap_seconds(
            cls.unit_end_timestamp(previous_unit),
            cls.unit_timestamp(incoming_unit),
        )

    @staticmethod
    def timestamp_gap_seconds(
        previous_end: Any,
        current_start: Any,
    ) -> Optional[float]:
        """Return the gap between two timestamp values, if both are valid."""
        if not previous_end or not current_start:
            return None
        try:
            previous = datetime.fromisoformat(
                str(previous_end).replace("Z", "+00:00"),
            )
            current = datetime.fromisoformat(
                str(current_start).replace("Z", "+00:00"),
            )
        except ValueError:
            return None
        return (current - previous).total_seconds()
