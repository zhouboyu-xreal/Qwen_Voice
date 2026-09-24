from dataclasses import dataclass
from datetime import datetime
from collections import deque
import re
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_client import EmbeddingClient
from .utils import _as_embedding_vector, _cal_embedding_cosine_similarity, _sigmoid, _centroid, _cohesion
from .memory_manager import (
    _compact_whitespace,
    _to_timestamp_text,
)

INTERACTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


@dataclass
class FactExtractionBoundaryDecision:
    reason: str
    should_finalize: bool = False
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
class MemoryUnit:
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
class MemoryContextMangerConfig:
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
class EpisodeSummaryConfig:
    max_duration_seconds: float = 1800.0
    max_tokens: int = 6000
    min_tokens_for_duration: int = 1000


@dataclass
class EpisodeSummaryBoundaryDecision:
    should_trigger: bool = False
    reason: str = "append"
    accumulated_tokens: int = 0
    elapsed_seconds: Optional[float] = None


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


def length_pressure(token_count: int, config: MemoryContextMangerConfig) -> float:
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
) -> MemoryContextMangerConfig:
    """Build the shared memory-input semantic segmentation configuration."""
    def config_int(key: str, default: int) -> int:
        value = segmentation_config.get(key)
        if value in (None, ""):
            return default
        return int(value)

    max_gap = segmentation_config.get("max_time_gap_seconds")
    return MemoryContextMangerConfig(
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
            config_int("max_pending_units", 40),
        ),
        max_pending_tokens=max(
            1,
            config_int("max_pending_tokens", 1000),
        ),
        max_pending_chars=max(
            0,
            config_int("max_pending_chars", 0),
        ),
        min_pending_tokens=max(
            1,
            config_int("min_pending_tokens", 200),
        ),
        min_pending_turns=max(
            1,
            config_int("min_pending_units", 4),
        ),
        min_segment_override_probability=float(
            segmentation_config.get("min_segment_override_probability", 0.90),
        ),
        max_time_gap_seconds=float(-1.0 if max_gap in (None, "") else max_gap),
        enforce_min_pending_tokens=bool(
            segmentation_config.get("enforce_min_pending_tokens", True)
        ),
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
) -> MemoryContextMangerConfig:
    """Build the shared semantic segmentation configuration for memory input."""
    memory_input_config = runtime_config.get("memory_context_manager")
    if not isinstance(memory_input_config, dict):
        memory_input_config = {}
    segmentation_config = memory_input_config.get("fact_extraction")
    if not isinstance(segmentation_config, dict):
        segmentation_config = {}
    return _build_online_segmentation_config(segmentation_config)


def build_episode_summary_config(
    runtime_config: Dict[str, Any],
) -> EpisodeSummaryConfig:
    """Build episode-summary limits from the shared memory-input settings."""
    memory_input_config = runtime_config.get("memory_context_manager")
    if not isinstance(memory_input_config, dict):
        memory_input_config = {}
    episode_summary_config = memory_input_config.get("episode_summary")
    if not isinstance(episode_summary_config, dict):
        episode_summary_config = {}
    return EpisodeSummaryConfig(
        max_duration_seconds=max(
            0.0,
            float(episode_summary_config.get("max_duration_seconds", 1800.0)),
        ),
        max_tokens=max(
            0,
            int(episode_summary_config.get("max_tokens", 6000)),
        ),
        min_tokens_for_duration=max(
            0,
            int(episode_summary_config.get("min_tokens_for_duration", 1000)),
        ),
    )


def build_transcript_aggregation_config(
    runtime_config: Dict[str, Any],
) -> TranscriptAggregationConfig:
    """Build transcript VAD-fragment aggregation from shared input settings."""
    aggregation_config = runtime_config.get("transcript_unit_aggregation")
    if not isinstance(aggregation_config, dict):
        aggregation_config = {}
    return TranscriptAggregationConfig(
        max_gap_seconds=float(
            aggregation_config.get("segment_merge_max_gap_seconds", 1.0)
        ),
        min_transcript_unit_tokens=max(
            1,
            int(aggregation_config.get("min_transcript_unit_tokens", 20)),
        ),
        max_transcript_unit_tokens=max(
            1,
            int(aggregation_config.get("max_transcript_unit_tokens", 120)),
        ),
        max_transcript_unit_duration_seconds=max(
            0.0,
            float(
                aggregation_config.get(
                    "max_transcript_unit_duration_seconds",
                    20.0,
                )
            ),
        ),
        short_fragment_max_tokens=max(
            1,
            int(aggregation_config.get("short_fragment_max_tokens", 8)),
        ),
    )


def convert_interaction_turn_to_online_unit(
    turn: Dict[str, Any],
) -> MemoryUnit:
    """Normalize one interaction turn into one storable segmenter unit."""
    user_message = _compact_whitespace(turn.get("user_message") or "")
    assistant_response = _compact_whitespace(turn.get("assistant_response") or "")
    text = _unit_text_from_turn({
        "user_message": user_message,
        "assistant_response": assistant_response,
    })
    fallback_timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or ""
    user_started_at = _to_timestamp_text(turn.get("user_started_at")) or fallback_timestamp
    user_ended_at = _to_timestamp_text(turn.get("user_ended_at")) or user_started_at
    assistant_started_at = (
        _to_timestamp_text(turn.get("assistant_started_at")) or fallback_timestamp
    )
    assistant_ended_at = _to_timestamp_text(turn.get("assistant_ended_at")) or assistant_started_at
    tags = [
        str(tag).strip()
        for tag in turn.get("tags") or []
        if str(tag).strip()
    ]
    raw_segments: List[Dict[str, Any]] = []
    if user_message:
        raw_segments.append({
            "speaker": "用户",
            "text": user_message,
            "started_at": user_started_at,
            "ended_at": user_ended_at,
            "tags": tags,
        })
    if assistant_response:
        raw_segments.append({
            "speaker": "助手",
            "text": assistant_response,
            "started_at": assistant_started_at,
            "ended_at": assistant_ended_at,
            "tags": tags,
        })
    return MemoryUnit(
        text=text,
        token_count=_estimate_interaction_token_count(text),
        # A unified unit needs a sortable envelope while raw segments retain
        # their individual speech/playback intervals.
        timestamp=(
            user_started_at if user_message else assistant_started_at
        ) or fallback_timestamp,
        ended_at=(
            assistant_ended_at if assistant_response else user_ended_at
        ) or fallback_timestamp,
        raw={
            "raw_segments": raw_segments,
            "input_kind": "interaction",
        },
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

    def append_new_segment(self, segment: Dict[str, Any]) -> Optional[MemoryUnit]:
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

    def flush(self) -> Optional[MemoryUnit]:
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

    def _take_current_unit(self) -> Optional[MemoryUnit]:
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
        unit = MemoryUnit(
            text=text,
            token_count=max(1, self._current_token_count),
            timestamp=self._segment_started_at(raw_segments[0]),
            ended_at=self._segment_ended_at(raw_segments[-1]),
            raw={
                "raw_segments": raw_segments,
                "speaker_labels": speaker_labels,
            },
        )
        self._current_segments = []
        self._current_token_count = 0
        return unit

    def _can_append(self, incoming: Dict[str, Any]) -> bool:
        if not self._current_segments:
            return True
        previous = self._current_segments[-1]
        gap_seconds = MemoryContextManager.timestamp_gap_seconds(
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
        duration_seconds = MemoryContextManager.timestamp_gap_seconds(
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


class MemoryContextManager:
    """Embedding-based online semantic boundary detector for dialogue units."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        config: Optional[MemoryContextMangerConfig] = None,
        episode_summary_config: Optional[EpisodeSummaryConfig] = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.config = config or MemoryContextMangerConfig()
        self.surprise_history: Deque[float] = deque(
            maxlen=max(1, self.config.surprise_history_window),
        )
        self._pending_buffer: List[ActiveUnit] = []
        self._awaiting_ambient_buffer: List[MemoryUnit] = []
        self._ambient_asr_watermark: Optional[datetime] = None
        self.episode_summary_config = episode_summary_config or EpisodeSummaryConfig()
        self._episode_started_at: Optional[datetime] = None
        self._episode_latest_at: Optional[datetime] = None
        self._episode_token_count = 0

    def _insert_pending_unit(
        self,
        unit: MemoryUnit,
        embedding: Optional[np.ndarray],
    ) -> None:
        """Insert a unit into the pending buffer in chronological order."""
        self._pending_buffer.append(ActiveUnit(
            unit=unit,
            embedding=_as_embedding_vector(embedding),
        ))
        self._pending_buffer.sort(
            key=lambda item: self._unit_time_order_key(item.unit),
        )

    def _insert_awaiting_ambient_unit(self, unit: MemoryUnit) -> None:
        """Retain one input until ambient ASR has covered its event time."""
        self._awaiting_ambient_buffer.append(unit)
        self._awaiting_ambient_buffer.sort(key=self._unit_time_order_key)

    def insert_incoming_unit(
        self,
        incoming_unit: MemoryUnit,
        ambient_recording_enabled: bool = False,
    ) -> Tuple[FactExtractionBoundaryDecision, List[MemoryUnit]]:
        """Evaluate one unit and retain it in the pending context buffer.

        Ambient input whose event time has not yet been covered by ASR is
        retained separately.  Once the watermark advances,
        :meth:`iter_awaiting_ambient_units` feeds it back through this
        same method, preserving one canonical boundary path.
        """
        if (
            ambient_recording_enabled
            and not self._ambient_asr_covers_unit(incoming_unit)
        ):
            self._insert_awaiting_ambient_unit(incoming_unit)
            return FactExtractionBoundaryDecision(
                reason="awaiting_ambient_asr_watermark",
            ), []

        incoming_embedding = self.embed_unit(incoming_unit).embedding
        decision = self._evaluate_incoming_unit(
            incoming_unit,
            incoming_embedding,
            ambient_recording_enabled,
        )
        finalized_units = self.pending_unit_snapshot() if decision.should_finalize else []
        if decision.should_finalize:
            self.clear_pending_units()
        self._insert_pending_unit(incoming_unit, incoming_embedding)
        return decision, finalized_units

    def update_ambient_asr_watermark(self, value: Any) -> Optional[datetime]:
        """Advance and return the latest event time covered by ambient ASR."""
        parsed = self._parse_timestamp(value)
        if parsed is not None and (
            self._ambient_asr_watermark is None
            or parsed > self._ambient_asr_watermark
        ):
            self._ambient_asr_watermark = parsed
        return self._ambient_asr_watermark

    def awaiting_ambient_unit_count(self) -> int:
        """Return the number of input units waiting for ambient-ASR coverage."""
        return len(self._awaiting_ambient_buffer)

    def iter_awaiting_ambient_units(
        self,
        *,
        force: bool = False,
    ) -> Iterator[Tuple[MemoryUnit, FactExtractionBoundaryDecision, List[MemoryUnit]]]:
        """Yield each covered unit immediately after its boundary decision.

        ``force`` is reserved for an explicit runtime flush, when the caller
        intentionally accepts the remaining ASR-delay risk rather than
        leaving an input unit unpersisted.

        The iterator deliberately does not accumulate decisions. Its caller
        can submit a finalized prefix while later waiting units are still
        being embedded and scored.
        """
        while self._awaiting_ambient_buffer:
            unit = self._awaiting_ambient_buffer[0]
            if not force and not self._ambient_asr_covers_unit(unit):
                break
            self._awaiting_ambient_buffer.pop(0)
            decision, finalized_units = self.insert_incoming_unit(
                unit,
                ambient_recording_enabled=not force,
            )
            yield unit, decision, finalized_units

    def process_awaiting_ambient_units(
        self,
        *,
        force: bool = False,
    ) -> List[Tuple[MemoryUnit, FactExtractionBoundaryDecision, List[MemoryUnit]]]:
        """Return all released units for callers that require a materialized list.

        Runtime ingestion uses :meth:`iter_awaiting_ambient_units` so a long
        ambient-ASR batch does not delay memory-store submission until every
        later unit has been scored.
        """
        return list(self.iter_awaiting_ambient_units(force=force))

    def _ambient_asr_covers_unit(self, unit: MemoryUnit) -> bool:
        if self._ambient_asr_watermark is None:
            return False
        ended_at = self._parse_timestamp(self.unit_end_timestamp(unit))
        return ended_at is not None and ended_at <= self._ambient_asr_watermark

    def pending_unit_snapshot(self) -> List[MemoryUnit]:
        """Return a shallow copy of the current pending online units."""
        return [item.unit for item in self._pending_buffer]

    def has_pending_units(self) -> bool:
        """Return whether the segmenter has units waiting for storage."""
        return bool(self._pending_buffer or self._awaiting_ambient_buffer)

    def clear_pending_units(self) -> None:
        """Clear units after the corresponding memory task was queued."""
        self._pending_buffer.clear()

    def record_stored_units(
        self,
        units: Sequence[MemoryUnit],
    ) -> EpisodeSummaryBoundaryDecision:
        """Record successfully queued units and evaluate the episode window."""
        for unit in units:
            token_count = max(0, self.unit_token_count(unit))
            if token_count <= 0:
                continue
            started_at = self._parse_timestamp(self.unit_timestamp(unit))
            ended_at = self._parse_timestamp(self.unit_end_timestamp(unit))
            if self._episode_started_at is None and started_at is not None:
                self._episode_started_at = started_at
            latest_at = ended_at or started_at
            if latest_at is not None and (
                self._episode_latest_at is None or latest_at > self._episode_latest_at
            ):
                self._episode_latest_at = latest_at
            self._episode_token_count += token_count
        return self._evaluate_episode_summary()

    def reset_episode_summary_window(self) -> None:
        self._episode_started_at = None
        self._episode_latest_at = None
        self._episode_token_count = 0

    def _evaluate_episode_summary(self) -> EpisodeSummaryBoundaryDecision:
        token_count = self._episode_token_count
        if token_count <= 0:
            return EpisodeSummaryBoundaryDecision(accumulated_tokens=token_count)
        if (
            self.episode_summary_config.max_tokens > 0
            and token_count >= self.episode_summary_config.max_tokens
        ):
            return EpisodeSummaryBoundaryDecision(True, "max_tokens", token_count)
        if self._episode_started_at is None or self._episode_latest_at is None:
            return EpisodeSummaryBoundaryDecision(False, "append", token_count)
        elapsed_seconds = max(
            0.0,
            (self._episode_latest_at - self._episode_started_at).total_seconds(),
        )
        if (
            self.episode_summary_config.max_duration_seconds > 0
            and elapsed_seconds >= self.episode_summary_config.max_duration_seconds
            and token_count >= self.episode_summary_config.min_tokens_for_duration
        ):
            return EpisodeSummaryBoundaryDecision(
                True, "max_duration", token_count, elapsed_seconds,
            )
        return EpisodeSummaryBoundaryDecision(False, "append", token_count, elapsed_seconds)

    def embed_unit(self, unit: Any) -> ActiveUnit:
        embedding = self.embedding_client.embed_text(self.unit_text(unit))
        vector = _as_embedding_vector(embedding)
        return ActiveUnit(unit=unit, embedding=vector)

    def _evaluate_incoming_unit(
        self,
        incoming_unit: Any,
        incoming_embedding: Optional[np.ndarray],
        ambient_recording_enabled: bool = False,
    ) -> FactExtractionBoundaryDecision:
        active = list(self._pending_buffer)
        active_units = [item.unit for item in active]
        time_gap_seconds = (
            self.unit_time_gap_seconds(active_units[-1], incoming_unit)
            if active_units
            else None
        )
        if not active_units:
            decision = FactExtractionBoundaryDecision(
                reason="start_segment",
                prospective_tokens=self.unit_token_count(incoming_unit),
                prospective_units=1,
                time_gap_seconds=time_gap_seconds,
            )
            return decision
        existing_capacity_reason = self._pending_capacity_reason(active_units)
        if existing_capacity_reason:
            return self._apply_ambient_finalize_gate(
                should_finalize=True,
                decision=FactExtractionBoundaryDecision(
                    reason=existing_capacity_reason,
                    prospective_tokens=sum(
                        self.unit_token_count(item) for item in active_units
                    ),
                    prospective_units=len(active_units),
                    time_gap_seconds=time_gap_seconds,
                ),
                active_units=active_units,
                ambient_recording_enabled=ambient_recording_enabled,
            )
        if self.time_gap_exceeded(time_gap_seconds):
            return self._apply_ambient_finalize_gate(
                should_finalize=True,
                decision=FactExtractionBoundaryDecision(
                    reason="time_gap",
                    prospective_tokens=sum(
                        self.unit_token_count(item) for item in active_units
                    ),
                    prospective_units=len(active_units),
                    time_gap_seconds=time_gap_seconds,
                ),
                active_units=active_units,
                ambient_recording_enabled=ambient_recording_enabled,
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
            return FactExtractionBoundaryDecision(
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
            item.embedding is None for item in active
        ):
            return FactExtractionBoundaryDecision(
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
        decision = self.score_boundary(active, incoming)
        decision.time_gap_seconds = time_gap_seconds
        decision.scoring_mode = scoring_context["scoring_mode"]
        decision.rolling_window_tail_units = scoring_context["tail_units"]
        decision.rolling_window_tail_tokens = scoring_context["tail_tokens"]
        if self.semantic_boundary_allowed(active, decision):
            decision.reason = "semantic_boundary"
            return self._apply_ambient_finalize_gate(
                should_finalize=True,
                decision=decision,
                active_units=active_units,
                ambient_recording_enabled=ambient_recording_enabled,
            )
        decision.reason = "append"
        return decision

    def _apply_ambient_finalize_gate(
        self,
        *,
        should_finalize: bool,
        decision: FactExtractionBoundaryDecision,
        active_units: Sequence[Any],
        ambient_recording_enabled: bool,
    ) -> FactExtractionBoundaryDecision:
        """Delay a finalized prefix until ambient ASR covers its tail.

        The semantic, capacity, and time-gap decision has already been made.
        Watermark coverage is only a persistence gate, not an alternate
        boundary decision. Equality is safe: a watermark at the same instant
        as the pending tail confirms that the tail is already covered.
        """
        if not should_finalize or not ambient_recording_enabled or not active_units:
            decision.should_finalize = should_finalize
            return decision
        pending_tail_time = self._parse_timestamp(
            self.unit_end_timestamp(active_units[-1]),
        )
        if (
            self._ambient_asr_watermark is None
            or pending_tail_time is None
            or pending_tail_time > self._ambient_asr_watermark
        ):
            decision.reason = "awaiting_ambient_asr_watermark"
            decision.should_finalize = False
            return decision
        decision.should_finalize = True
        return decision

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
    ) -> FactExtractionBoundaryDecision:
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

        return FactExtractionBoundaryDecision(
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
        decision: FactExtractionBoundaryDecision,
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
        return _estimate_interaction_token_count(MemoryContextManager.unit_text(unit))

    @staticmethod
    def unit_timestamp(unit: Any) -> str:
        return _to_timestamp_text(getattr(unit, "timestamp", "")) or ""

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        text = _to_timestamp_text(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return parsed
        except (TypeError, ValueError):
            return None

    @classmethod
    def _unit_time_order_key(cls, unit: Any) -> Tuple[int, float]:
        """Sort known timestamps first; stable sorting preserves arrival order."""
        parsed = cls._parse_timestamp(cls.unit_timestamp(unit))
        return (
            0 if parsed is not None else 1,
            parsed.timestamp() if parsed is not None else float("inf"),
        )

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

    @classmethod
    def timestamp_gap_seconds(
        cls,
        previous_end: Any,
        current_start: Any,
    ) -> Optional[float]:
        """Return the gap between two timestamp values, if both are valid."""
        previous = cls._parse_timestamp(previous_end)
        current = cls._parse_timestamp(current_start)
        if previous is None or current is None:
            return None
        return (current - previous).total_seconds()
