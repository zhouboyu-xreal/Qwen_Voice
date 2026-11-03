from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .memory_types import MemoryUnit
from .memory_manager import (
    _compact_whitespace,
    _to_timestamp_text,
)

INTERACTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


@dataclass
class FactExtractionWindowConfig:
    """Capacity, time-gap and auxiliary-context limits for fact unit windows."""

    target_pending_tokens: int = 2000
    max_pending_tokens: int = 3200
    max_pending_units: int = 128
    max_pending_interaction_units: int = 6
    max_time_gap_seconds: float = -1.0
    preceding_context_max_tokens: int = 500


@dataclass
class UserActivityConfig:
    max_duration_seconds: float = 1800.0
    max_tokens: int = 6000
    min_tokens_for_duration: int = 1000


@dataclass
class MemoryContextMangerConfig:
    fact_extraction: FactExtractionWindowConfig = field(
        default_factory=FactExtractionWindowConfig,
    )
    user_activity: UserActivityConfig = field(
        default_factory=UserActivityConfig,
    )


@dataclass
class UserActivityBoundaryDecision:
    should_trigger: bool = False
    reason: str = "append"
    accumulated_tokens: int = 0
    elapsed_seconds: Optional[float] = None


@dataclass
class FactExtractionTaskDecision:
    """A submission snapshot of existing units, never the incoming unit."""

    should_trigger: bool = False
    reason: str = "append"
    batch_units: List[MemoryUnit] = field(default_factory=list, repr=False)
    time_gap_seconds: Optional[float] = None

    @property
    def batch_unit_count(self) -> int:
        return len(self.batch_units)

    @property
    def batch_token_count(self) -> int:
        return sum(unit.token_count for unit in self.batch_units)


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
    return f"speaker_用户本人：{user_text}\nspeaker_助手：{assistant_text}".strip()


def build_memory_context_manager_config(
    runtime_config: Dict[str, Any],
) -> MemoryContextMangerConfig:
    """Build fact-window capacity, context and activity settings for memory input."""
    memory_input_config = runtime_config.get("memory_context_manager")
    if not isinstance(memory_input_config, dict):
        memory_input_config = {}
    fact_extraction_trigger_config = memory_input_config.get("fact_extraction_trigger")
    if not isinstance(fact_extraction_trigger_config, dict):
        fact_extraction_trigger_config = {}
    user_activity_trigger_config = memory_input_config.get("user_activity_trigger")
    if not isinstance(user_activity_trigger_config, dict):
        user_activity_trigger_config = {}
    return MemoryContextMangerConfig(
        fact_extraction=_build_fact_extraction_window_config(fact_extraction_trigger_config or {}),
        user_activity=_build_user_activity_config(user_activity_trigger_config or {}),
    )


def _build_fact_extraction_window_config(
    extraction_config: Dict[str, Any],
) -> FactExtractionWindowConfig:
    """Read capacity and auxiliary-context limits for the fact unit window."""
    def config_int(key: str, default: int) -> int:
        value = extraction_config.get(key)
        return default if value in (None, "") else int(value)

    target_tokens = max(1, config_int("target_pending_tokens", 2000))
    max_gap = extraction_config.get("max_time_gap_seconds")
    return FactExtractionWindowConfig(
        target_pending_tokens=target_tokens,
        max_pending_tokens=max(target_tokens, config_int("max_pending_tokens", 3200)),
        max_pending_units=max(1, config_int("max_pending_units", 128)),
        max_pending_interaction_units=max(0, config_int("max_pending_interaction_units", 6)),
        max_time_gap_seconds=float(-1.0 if max_gap in (None, "") else max_gap),
        preceding_context_max_tokens=max(0, config_int("preceding_context_max_tokens", 500)),
    )


def _build_user_activity_config(
    user_activity_config: Dict[str, Any],
) -> UserActivityConfig:
    """Build limits for the longer user-activity observation window."""
    return UserActivityConfig(
        max_duration_seconds=max(
            0.0,
            float(user_activity_config.get("max_duration_seconds", 1800.0)),
        ),
        max_tokens=max(
            0,
            int(user_activity_config.get("max_tokens", 6000)),
        ),
        min_tokens_for_duration=max(
            0,
            int(user_activity_config.get("min_tokens_for_duration", 1000)),
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
    texts: List[str] = []
    speakers: List[str] = []
    started_ats: List[str] = []
    ended_ats: List[str] = []
    if user_message:
        texts.append(user_message)
        speakers.append("speaker_用户本人")
        started_ats.append(user_started_at)
        ended_ats.append(user_ended_at)
    if assistant_response:
        texts.append(assistant_response)
        speakers.append("speaker_助手")
        started_ats.append(assistant_started_at)
        ended_ats.append(assistant_ended_at)
    return MemoryUnit(
        texts=texts,
        speakers=speakers,
        token_count=_estimate_interaction_token_count(text),
        started_ats=started_ats,
        ended_ats=ended_ats,
        unit_type="interaction",
        tags=tags,
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
        tags = list(dict.fromkeys(
            str(tag).strip()
            for segment in raw_segments
            for tag in segment.get("tags") or []
            if str(tag).strip()
        ))
        unit = MemoryUnit(
            texts=[text],
            speakers=[", ".join(speaker_labels) or "unknown_speaker"],
            token_count=max(1, self._current_token_count),
            started_ats=[self._segment_started_at(raw_segments[0])],
            ended_ats=[self._segment_ended_at(raw_segments[-1])],
            unit_type="transcript",
            tags=tags,
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
    """Build fact unit windows using capacity and time-gap rules."""

    def __init__(
        self,
        config: Optional[MemoryContextMangerConfig] = None,
    ) -> None:
        self.config = config or MemoryContextMangerConfig()
        self._pending_unit_buffer: List[MemoryUnit] = []
        self._preceding_context_units: List[MemoryUnit] = []
        self._awaiting_ambient_unit_buffer: List[MemoryUnit] = []
        self._ambient_asr_watermark: Optional[datetime] = None
        self._user_activity_started_at: Optional[datetime] = None
        self._user_activity_latest_at: Optional[datetime] = None
        self._user_activity_token_count = 0

    def _insert_pending_unit(
        self,
        unit: MemoryUnit,
    ) -> None:
        """Insert a unit into the pending buffer in chronological order."""
        self._pending_unit_buffer.append(unit)
        self._pending_unit_buffer.sort(
            key=self._unit_time_order_key,
        )

    def _insert_awaiting_ambient_unit(self, unit: MemoryUnit) -> None:
        """Retain one input until ambient ASR has covered its event time."""
        self._awaiting_ambient_unit_buffer.append(unit)
        self._awaiting_ambient_unit_buffer.sort(key=self._unit_time_order_key)

    def insert_incoming_unit(
        self,
        incoming_unit: MemoryUnit,
        ambient_recording_enabled: bool = False,
    ) -> FactExtractionTaskDecision:
        """Evaluate existing pending units before inserting the incoming unit.

        A trigger detaches the previous window into decision.batch_units and
        starts pending with the incoming unit. Runtime restores the detached
        batch if submission fails; otherwise pending already is the next window.
        """
        if ambient_recording_enabled and not self._ambient_asr_covers_unit(incoming_unit):
            self._insert_awaiting_ambient_unit(incoming_unit)
            return FactExtractionTaskDecision(reason="awaiting_ambient_asr_watermark")
        decision = self._evaluate_fact_extraction_window(incoming_unit)
        if decision.should_trigger:
            self._pending_unit_buffer = [incoming_unit]
        else:
            self._insert_pending_unit(incoming_unit)
        return decision

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
        return len(self._awaiting_ambient_unit_buffer)

    def iter_awaiting_ambient_units(
        self,
        *,
        force: bool = False,
    ) -> Iterator[Tuple[MemoryUnit, FactExtractionTaskDecision]]:
        """Yield each covered unit immediately after its fact-task decision.

        ``force`` is reserved for an explicit runtime flush, when the caller
        intentionally accepts the remaining ASR-delay risk rather than
        leaving an input unit unpersisted.

        Process covered units one at a time without embedding requests. The
        caller can submit and acknowledge a window before the next unit is
        evaluated. Unconsumed units stay buffered if iteration stops early.
        """
        while self._awaiting_ambient_unit_buffer:
            unit = self._awaiting_ambient_unit_buffer[0]
            if not force and not self._ambient_asr_covers_unit(unit):
                break
            self._awaiting_ambient_unit_buffer.pop(0)
            decision = self.insert_incoming_unit(
                unit,
                ambient_recording_enabled=not force,
            )
            yield unit, decision

    def process_awaiting_ambient_units(
        self,
        *,
        force: bool = False,
    ) -> List[Tuple[MemoryUnit, FactExtractionTaskDecision]]:
        """Return all released units for callers that require a materialized list.

        Runtime ingestion uses :meth:`iter_awaiting_ambient_units` so a long
        ambient-ASR batch does not delay memory-store submission until every
        later unit has been evaluated.
        """
        return list(self.iter_awaiting_ambient_units(force=force))

    def _ambient_asr_covers_unit(self, unit: MemoryUnit) -> bool:
        if self._ambient_asr_watermark is None:
            return False
        ended_at = self._parse_timestamp(unit.end_time)
        return ended_at is not None and ended_at <= self._ambient_asr_watermark

    def pending_unit_snapshot(self) -> List[MemoryUnit]:
        """Return all unsubmitted units in chronological order."""
        return list(self._pending_unit_buffer)

    def take_pending_units(self) -> List[MemoryUnit]:
        """Detach the remaining window for explicit flush submission."""
        units = self.pending_unit_snapshot()
        self._pending_unit_buffer = []
        return units

    def restore_pending_units(self, units: Sequence[MemoryUnit]) -> None:
        """Restore a rejected batch ahead of the next window, in time order."""
        self._pending_unit_buffer = sorted(
            [*units, *self._pending_unit_buffer],
            key=self._unit_time_order_key,
        )

    def preceding_context_units_snapshot(self) -> List[MemoryUnit]:
        """Return the bounded, complete-unit tail of the last accepted batch."""
        return list(self._preceding_context_units)

    def record_preceding_context_units(self, units: Sequence[MemoryUnit]) -> None:
        """Keep a contiguous tail up to a soft token budget, without splitting units.

        The final whole unit can exceed the budget. Older units are added
        only while the accumulated tail remains below it. This context is
        replaced after acceptance, never accumulated across batches.
        """
        budget = self.config.fact_extraction.preceding_context_max_tokens
        tail: List[MemoryUnit] = []
        token_count = 0
        if budget > 0:
            for unit in reversed(units):
                tail.append(unit)
                token_count += max(0, unit.token_count)
                if token_count >= budget:
                    break
        self._preceding_context_units = list(reversed(tail))

    def record_stored_units(
        self,
        units: Sequence[MemoryUnit],
    ) -> None:
        """Record successfully queued fact units in the activity window."""
        for unit in units:
            token_count = max(0, unit.token_count)
            if token_count <= 0:
                continue
            started_at = self._parse_timestamp(unit.start_time)
            ended_at = self._parse_timestamp(unit.end_time)
            if self._user_activity_started_at is None and started_at is not None:
                self._user_activity_started_at = started_at
            latest_at = ended_at or started_at
            if latest_at is not None and (
                self._user_activity_latest_at is None or latest_at > self._user_activity_latest_at
            ):
                self._user_activity_latest_at = latest_at
            self._user_activity_token_count += token_count

    def reset_user_activity_window(self) -> None:
        self._user_activity_started_at = None
        self._user_activity_latest_at = None
        self._user_activity_token_count = 0

    def has_pending_user_activity_units(self) -> bool:
        """Whether successfully queued fact units await activity observation."""
        return self._user_activity_token_count > 0

    def evaluate_user_activity_trigger_decision(self) -> UserActivityBoundaryDecision:
        token_count = self._user_activity_token_count
        if token_count <= 0:
            return UserActivityBoundaryDecision(accumulated_tokens=token_count)
        if (
            self.config.user_activity.max_tokens > 0
            and token_count >= self.config.user_activity.max_tokens
        ):
            return UserActivityBoundaryDecision(True, "max_tokens", token_count)
        if self._user_activity_started_at is None or self._user_activity_latest_at is None:
            return UserActivityBoundaryDecision(False, "append", token_count)
        elapsed_seconds = max(
            0.0,
            (self._user_activity_latest_at - self._user_activity_started_at).total_seconds(),
        )
        if (
            self.config.user_activity.max_duration_seconds > 0
            and elapsed_seconds >= self.config.user_activity.max_duration_seconds
            and token_count >= self.config.user_activity.min_tokens_for_duration
        ):
            return UserActivityBoundaryDecision(
                True, "max_duration", token_count, elapsed_seconds,
            )
        return UserActivityBoundaryDecision(False, "append", token_count, elapsed_seconds)

    def _evaluate_fact_extraction_window(
        self,
        incoming_unit: MemoryUnit,
    ) -> FactExtractionTaskDecision:
        """Choose the existing window; all regular triggers exclude incoming."""
        pending_units = self.pending_unit_snapshot()
        decision = FactExtractionTaskDecision(batch_units=pending_units)
        if not pending_units:
            decision.reason = "start_unit"
            return decision
        config = self.config.fact_extraction
        decision.time_gap_seconds = self.unit_time_gap_seconds(
            pending_units[-1], incoming_unit,
        )
        pending_interaction_count = sum(
            unit.unit_type == "interaction" for unit in pending_units
        )
        if (config.max_time_gap_seconds >= 0
                and decision.time_gap_seconds is not None
                and decision.time_gap_seconds > config.max_time_gap_seconds):
            decision.should_trigger = True
            decision.reason = "time_gap"
        elif config.max_pending_units > 0 and len(pending_units) >= config.max_pending_units:
            decision.should_trigger = True
            decision.reason = "pending_unit_limit"
        elif (config.max_pending_interaction_units > 0
              and pending_interaction_count >= config.max_pending_interaction_units):
            decision.should_trigger = True
            decision.reason = "pending_interaction_unit_limit"
        elif (config.max_pending_tokens > 0
              and decision.batch_token_count + incoming_unit.token_count > config.max_pending_tokens):
            decision.should_trigger = True
            decision.reason = "pending_token_limit"
        elif (config.target_pending_tokens > 0
              and decision.batch_token_count >= config.target_pending_tokens):
            decision.should_trigger = True
            decision.reason = "pending_token_target"
        return decision

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
    def _unit_time_order_key(cls, unit: MemoryUnit) -> Tuple[int, float]:
        """Sort known timestamps first; stable sorting preserves arrival order."""
        parsed = cls._parse_timestamp(unit.start_time)
        return (
            0 if parsed is not None else 1,
            parsed.timestamp() if parsed is not None else float("inf"),
        )

    @classmethod
    def unit_time_gap_seconds(
        cls,
        previous_unit: MemoryUnit,
        incoming_unit: MemoryUnit,
    ) -> Optional[float]:
        return cls.timestamp_gap_seconds(
            previous_unit.end_time,
            incoming_unit.start_time,
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
