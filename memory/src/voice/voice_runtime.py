"""Voice ingestion runtime for audio-file based memory input.

This module owns the voice models and converts an audio file into normalized
transcript segments.  It deliberately does not know about the memory
database; callers can feed the returned segments into ``MemoryRuntime``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .text_utils import normalize_text


class VoiceRuntime:
    """Run VAD, speaker identification, and ASR over an audio file."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = dict(config or {})
        self.runtime_config = (
            self._section("runtime")
            if isinstance(self.config.get("runtime"), dict)
            else dict(self.config)
        )
        self.vad_config = self._section("vad")
        self.asr_config = self._resolve_asr_config()
        self.config["asr"] = dict(self.asr_config)
        self.speaker_config = self._section("speaker_identification")
        self.logger = logger or logging.getLogger(__name__)

        # Imports are kept here so memory-only MCP clients do not need to load
        # the audio model dependencies before the voice runtime is requested.
        from .vad_detector import StreamingSileroVAD

        self.vad = StreamingSileroVAD(
            sample_rate=self._sample_rate(),
            threshold=float(self.vad_config.get("threshold", 0.5)),
            min_silence_duration_ms=int(
                self.vad_config.get("min_silence_ms", 100)
            ),
            speech_pad_ms=int(self.vad_config.get("speech_pad_ms", 30)),
        )
        self.asr = self._build_asr()
        self.speaker_identifier = self._build_speaker_identifier()

    def process_audio_file(
        self,
        audio_path: Path | str,
        *,
        session_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert one audio file into timestamped transcript segments."""
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        started_at = time.monotonic()
        self.logger.info(
            "process_audio_file start path=%s session_start=%s",
            path,
            session_start or "<now>",
        )

        from .audio import load_audio_mono, slice_audio

        audio, sample_rate = load_audio_mono(path, self._sample_rate())
        limit = self.runtime_config.get("max_duration_s")
        if limit is not None:
            audio = audio[: max(0, int(float(limit) * sample_rate))]
        self.logger.info(
            "process_audio_file audio_loaded path=%s sample_rate=%s duration_s=%.3f max_duration_s=%s",
            path,
            sample_rate,
            len(audio) / sample_rate if sample_rate else 0.0,
            limit if limit is not None else "<none>",
        )

        recording_start = self._parse_session_start(session_start)
        spans = self._detect_speech_spans(audio, sample_rate)
        self.logger.info(
            "process_audio_file speech_detected path=%s speech_segments=%s",
            path,
            len(spans),
        )
        raw_chunks = [
            slice_audio(audio, sample_rate, start, end) for start, end in spans
        ]
        raw_assignments = self._identify_speakers(raw_chunks)
        spans, assignments, source_span_indexes = self._merge_speech_spans_by_speaker(
            spans,
            raw_assignments,
        )
        chunks = [
            slice_audio(audio, sample_rate, start, end) for start, end in spans
        ]
        merged_span_count = sum(
            1 for source_indexes in source_span_indexes if len(source_indexes) > 1
        )
        self.logger.info(
            "process_audio_file speech_span_merge path=%s input_spans=%s output_spans=%s "
            "merged_groups=%s enabled=%s max_gap_s=%.3f max_duration_s=%.3f",
            path,
            len(raw_chunks),
            len(chunks),
            merged_span_count,
            bool(self.vad_config.get("span_merge_enabled", True)),
            float(self.vad_config.get("span_merge_max_gap_seconds", 0.6)),
            float(self.vad_config.get("span_merge_max_duration_seconds", 20.0)),
        )
        segments: List[Dict[str, Any]] = []
        asr_texts = [""] * len(chunks)
        asr_batches = self._build_asr_batches(chunks, sample_rate)
        for batch_index, batch_indexes in enumerate(asr_batches, 1):
            batch_chunks = [chunks[index] for index in batch_indexes]
            batch_texts = self.asr.transcribe_batch_segments(batch_chunks)
            if len(batch_texts) != len(batch_chunks):
                raise RuntimeError(
                    "ASR batch result count does not match speech segment count: "
                    f"batch={batch_index} results={len(batch_texts)} "
                    f"segments={len(batch_chunks)} indexes={batch_indexes}"
                )
            for chunk_index, transcript_text in zip(batch_indexes, batch_texts):
                asr_texts[chunk_index] = transcript_text
            batch_durations = [
                len(chunks[index]) / sample_rate
                for index in batch_indexes
            ]
            self.logger.info(
                "process_audio_file asr_batch path=%s batch=%s segment_indexes=%s "
                "batch_size=%s duration_min_s=%.3f duration_max_s=%.3f returned=%s",
                path,
                batch_index,
                [index + 1 for index in batch_indexes],
                len(batch_chunks),
                min(batch_durations) if batch_durations else 0.0,
                max(batch_durations) if batch_durations else 0.0,
                len(batch_texts),
            )
        for chunk_index, transcript_text in enumerate(asr_texts):
            start, end = spans[chunk_index]
            assignment = assignments[chunk_index]
            index = chunk_index + 1
            text = normalize_text(
                transcript_text,
                bool(
                    self.asr_config.get(
                        "simplify_chinese",
                        self.config.get("output", {}).get("simplify_chinese", True),
                    )
                ),
            )
            if not text:
                self.logger.info(
                    "process_audio_file asr_empty path=%s segment=%s start=%.3f end=%.3f",
                    path,
                    index,
                    start,
                    end,
                )
                continue
            speaker, speaker_metadata = self._speaker_label(assignment)
            self.logger.info(
                "process_audio_file asr_segment path=%s segment=%s start=%.3f end=%.3f speaker=%s text=%s",
                path,
                index,
                start,
                end,
                speaker,
                text,
            )
            segments.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "started_at": (recording_start + timedelta(seconds=start)).isoformat(),
                    "ended_at": (recording_start + timedelta(seconds=end)).isoformat(),
                    "chunk_index": index,
                    "metadata": {
                        "audio_start_s": round(start, 3),
                        "audio_end_s": round(end, 3),
                        "source_vad_span_indexes": source_span_indexes[chunk_index],
                        "merged_vad_span_count": len(
                            source_span_indexes[chunk_index]
                        ),
                        "sample_rate": sample_rate,
                        **speaker_metadata,
                    },
                }
            )

        report = {
            "status": "ok",
            "audio_path": str(path),
            "sample_rate": sample_rate,
            "audio_duration_s": round(len(audio) / sample_rate, 3),
            "speech_segment_count": len(raw_chunks),
            "merged_speech_segment_count": len(spans),
            "speech_span_merge_group_count": merged_span_count,
            "asr_batch_count": len(asr_batches),
            "transcript_segment_count": len(segments),
            "segments": segments,
        }
        self.logger.info(
            "process_audio_file finish path=%s speech_segments=%s transcript_segments=%s elapsed_ms=%.2f",
            path,
            len(raw_chunks),
            len(segments),
            (time.monotonic() - started_at) * 1000.0,
        )
        return report

    def close(self) -> None:
        """Reset streaming VAD state before the host releases this runtime."""
        reset = getattr(self.vad, "reset_stream_state", None)
        if callable(reset):
            reset()

    def _section(self, name: str) -> Dict[str, Any]:
        value = self.config.get(name)
        return dict(value) if isinstance(value, dict) else {}

    def _resolve_asr_config(self) -> Dict[str, Any]:
        """Flatten the active ASR backend profile for the backend factory."""
        asr_section = self._section("asr")
        if isinstance(asr_section.get("backends"), dict):
            backend_key = str(asr_section.get("backend") or "whisper").strip().lower()
            common = asr_section.get("common")
            profiles = asr_section.get("backends")
            profile = profiles.get(backend_key) if isinstance(profiles, dict) else None
            if not isinstance(profile, dict) and isinstance(profiles, dict):
                profile = profiles.get(backend_key.replace("-", "_"))
            if not isinstance(profile, dict):
                raise ValueError(f"ASR backend profile not found: {backend_key}")
            backend = backend_key.replace("_", "-")
            resolved = {
                **(dict(common) if isinstance(common, dict) else {}),
                **profile,
                "asr_backend": backend,
            }
            if backend in {"whisper", "openai-whisper"}:
                resolved.setdefault("whisper_model_id", resolved.get("model_id"))
                resolved.setdefault("whisper_chunk_length_s", resolved.get("chunk_length_s"))
                resolved.setdefault(
                    "whisper_condition_on_prev_tokens",
                    resolved.get("condition_on_prev_tokens"),
                )
                resolved.setdefault("whisper_max_new_tokens", resolved.get("max_new_tokens"))
                resolved.setdefault(
                    "whisper_no_repeat_ngram_size",
                    resolved.get("no_repeat_ngram_size"),
                )
                resolved.setdefault(
                    "whisper_repetition_penalty",
                    resolved.get("repetition_penalty"),
                )
            elif backend in {"paraformer", "paraformer-offline", "paraformer-nonstreaming"}:
                resolved.setdefault("paraformer_model_id", resolved.get("model_id"))
            elif backend == "paraformer-streaming":
                resolved["asr_backend"] = "paraformer-streaming"
                resolved.setdefault(
                    "paraformer_streaming_model_id",
                    resolved.get("model_id"),
                )
                resolved.setdefault(
                    "funasr_streaming_chunk_size",
                    resolved.get("chunk_size"),
                )
                resolved.setdefault(
                    "funasr_encoder_chunk_look_back",
                    resolved.get("encoder_chunk_look_back"),
                )
                resolved.setdefault(
                    "funasr_decoder_chunk_look_back",
                    resolved.get("decoder_chunk_look_back"),
                )
            elif backend in {"parakeet", "parakeet-tdt", "nvidia-parakeet"}:
                resolved.setdefault("parakeet_model_id", resolved.get("model_id"))
            elif backend in {
                "sherpa-onnx-zipformer",
                "sherpa-zipformer",
                "zipformer-streaming",
            }:
                resolved.setdefault("sherpa_onnx_tokens", resolved.get("tokens"))
                resolved.setdefault("sherpa_onnx_encoder", resolved.get("encoder"))
                resolved.setdefault("sherpa_onnx_decoder", resolved.get("decoder"))
                resolved.setdefault("sherpa_onnx_joiner", resolved.get("joiner"))
                resolved.setdefault("sherpa_onnx_num_threads", resolved.get("num_threads"))
                resolved.setdefault("sherpa_onnx_provider", resolved.get("provider"))
                resolved.setdefault("sherpa_onnx_feature_dim", resolved.get("feature_dim"))
                resolved.setdefault(
                    "sherpa_onnx_decoding_method",
                    resolved.get("decoding_method"),
                )
                resolved.setdefault(
                    "sherpa_onnx_max_active_paths",
                    resolved.get("max_active_paths"),
                )
                resolved.setdefault(
                    "sherpa_onnx_hotwords_file",
                    resolved.get("hotwords_file"),
                )
                resolved.setdefault(
                    "sherpa_onnx_hotwords_score",
                    resolved.get("hotwords_score"),
                )
                resolved.setdefault(
                    "sherpa_onnx_blank_penalty",
                    resolved.get("blank_penalty"),
                )
                resolved.setdefault(
                    "sherpa_onnx_tail_padding_s",
                    resolved.get("tail_padding_s"),
                )
            return resolved

        # Keep older externally supplied configs readable until all callers migrate.
        legacy = dict(asr_section or self._section("nonstreaming_asr"))
        if "asr_backend" not in legacy:
            legacy["asr_backend"] = (
                "paraformer-offline"
                if legacy.get("paraformer_model_id")
                else "whisper"
            )
        return legacy

    def _sample_rate(self) -> int:
        return max(1, int(self.runtime_config.get("sample_rate", 16000)))

    def _build_asr(self) -> Any:
        from .asr import create_asr

        config = dict(self.asr_config)
        config.setdefault("sample_rate", self._sample_rate())
        config.setdefault("language", self.runtime_config.get("language", "zh"))
        config.setdefault("device", self.runtime_config.get("device"))
        return create_asr(SimpleNamespace(**config))

    def _build_speaker_identifier(self) -> Any:
        if not bool(self.speaker_config.get("enabled", False)):
            return None
        from .speaker_detector import SpeakerReferenceDetector

        reference_dir = self.speaker_config.get("user_reference_audio_dir")
        detector = SpeakerReferenceDetector(
            sample_rate=self._sample_rate(),
            device=self.runtime_config.get("device"),
            model_id=str(
                self.speaker_config.get(
                    "model_id",
                    "iic/speech_campplus_sv_zh-cn_16k-common",
                )
            ),
            similarity_threshold=float(
                self.speaker_config.get("similarity_threshold", 0.5)
            ),
            reference_memory_size=max(
                1,
                int(self.speaker_config.get("reference_memory_size", 20)),
            ),
            min_new_reference_segments=max(
                1,
                int(self.speaker_config.get("min_new_reference_segments", 2)),
            ),
            min_new_reference_duration_s=max(
                0.0,
                float(self.speaker_config.get("min_new_reference_duration_s", 3.0)),
            ),
            min_reference_segment_duration_s=max(
                0.0,
                float(
                    self.speaker_config.get(
                        "min_reference_segment_duration_s",
                        1.0,
                    )
                ),
            ),
            user_similarity_threshold=self.speaker_config.get(
                "user_similarity_threshold"
            ),
            user_reference_audio_dir=(
                Path(reference_dir).expanduser() if reference_dir else None
            ),
            reference_embeddings_path=(
                Path(self.speaker_config["reference_embeddings_path"]).expanduser()
                if self.speaker_config.get("reference_embeddings_path")
                else None
            ),
            speaker_assignment_method=str(
                self.speaker_config.get("speaker_assignment_method", "reference")
            ),
            embedding_batch_size=max(
                1,
                int(self.speaker_config.get("embedding_batch_size", 16)),
            ),
        )
        reference_audio = self.speaker_config.get("user_reference_audio")
        if isinstance(reference_audio, list) and reference_audio:
            detector.load_user_reference_audios(
                [Path(item).expanduser().resolve() for item in reference_audio]
            )
        return detector

    def _detect_speech_spans(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        self.vad.reset_stream_state()
        frame_samples = max(
            1,
            round(
                float(self.vad_config.get("frame_ms", 32.0))
                / 1000.0
                * sample_rate
            ),
        )
        spans: List[Tuple[float, float]] = []
        speech_start: Optional[float] = None
        total_frames = int(np.ceil(len(audio) / frame_samples)) if len(audio) else 0
        for frame_index in range(total_frames):
            frame_start_sample = frame_index * frame_samples
            frame_end_sample = min(len(audio), frame_start_sample + frame_samples)
            frame_start = frame_start_sample / sample_rate
            frame_end = frame_end_sample / sample_rate
            frame = audio[frame_start_sample:frame_end_sample]
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))
            active = self.vad.accept_frame(
                frame,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            transition = self.vad.last_transition
            if active and speech_start is None:
                speech_start = max(
                    0.0,
                    transition.time
                    if transition is not None and transition.event_type == "start"
                    else frame_start,
                )
            elif not active and speech_start is not None:
                speech_end = (
                    transition.time
                    if transition is not None and transition.event_type == "end"
                    else frame_start
                )
                if speech_end > speech_start:
                    spans.append((speech_start, min(speech_end, len(audio) / sample_rate)))
                speech_start = None
        if speech_start is not None:
            audio_end = len(audio) / sample_rate
            if audio_end > speech_start:
                spans.append((speech_start, audio_end))
        self.logger.info(
            "Voice VAD completed frames=%s speech_segments=%s duration_s=%.3f",
            total_frames,
            len(spans),
            len(audio) / sample_rate if sample_rate else 0.0,
        )
        return spans

    def _identify_speakers(self, chunks: Sequence[np.ndarray]) -> List[Any]:
        if not chunks or self.speaker_identifier is None:
            return [None] * len(chunks)
        try:
            assignments = self.speaker_identifier.identify_batch_segments_speaker(
                list(chunks)
            )
            if len(assignments) == len(chunks):
                return list(assignments)
            self.logger.warning(
                "Speaker identification returned %s assignments for %s chunks; "
                "using unknown labels",
                len(assignments),
                len(chunks),
            )
        except Exception:
            self.logger.exception("Speaker identification failed; using unknown labels")
        return [None] * len(chunks)

    def _build_asr_batches(
        self,
        chunks: Sequence[np.ndarray],
        sample_rate: int,
    ) -> List[List[int]]:
        """Group similarly sized chunks while retaining their original indexes."""
        max_batch_size = max(
            1,
            int(self.asr_config.get("max_inference_batch_size", 1) or 1),
        )
        if not bool(self.asr_config.get("dynamic_batching_enabled", True)):
            return [
                list(range(start, min(start + max_batch_size, len(chunks))))
                for start in range(0, len(chunks), max_batch_size)
            ]

        max_duration_diff_seconds = max(
            0.0,
            float(
                self.asr_config.get(
                    "max_batch_duration_diff_seconds",
                    2.0,
                )
            ),
        )
        indexed_durations = sorted(
            [
                (
                    index,
                    len(audio) / sample_rate if sample_rate else 0.0,
                )
                for index, audio in enumerate(chunks)
            ],
            key=lambda item: item[1],
        )
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_min_duration = 0.0
        for index, duration_seconds in indexed_durations:
            if not current_batch:
                current_batch = [index]
                current_min_duration = duration_seconds
                continue
            duration_within_range = (
                duration_seconds - current_min_duration
                <= max_duration_diff_seconds
            )
            if len(current_batch) >= max_batch_size or not duration_within_range:
                batches.append(current_batch)
                current_batch = [index]
                current_min_duration = duration_seconds
            else:
                current_batch.append(index)
        if current_batch:
            batches.append(current_batch)
        return batches

    def _merge_speech_spans_by_speaker(
        self,
        spans: Sequence[Tuple[float, float]],
        assignments: Sequence[Any],
    ) -> Tuple[List[Tuple[float, float]], List[Any], List[List[int]]]:
        """Merge nearby VAD spans only when their identified speaker is the same."""
        normalized_spans = [
            (float(start), float(end)) for start, end in spans if float(end) > float(start)
        ]
        if len(normalized_spans) != len(assignments):
            self.logger.warning(
                "Cannot merge speech spans: spans=%s assignments=%s",
                len(normalized_spans),
                len(assignments),
            )
            return normalized_spans, list(assignments), [
                [index] for index in range(len(normalized_spans))
            ]

        if not bool(self.vad_config.get("span_merge_enabled", True)):
            return normalized_spans, list(assignments), [
                [index] for index in range(len(normalized_spans))
            ]

        max_gap_seconds = max(
            0.0,
            float(self.vad_config.get("span_merge_max_gap_seconds", 0.6)),
        )
        max_duration_seconds = max(
            0.0,
            float(self.vad_config.get("span_merge_max_duration_seconds", 20.0)),
        )
        merged_spans: List[Tuple[float, float]] = []
        merged_assignments: List[Any] = []
        source_span_indexes: List[List[int]] = []

        for index, (span, assignment) in enumerate(zip(normalized_spans, assignments)):
            if not merged_spans:
                merged_spans.append(span)
                merged_assignments.append(assignment)
                source_span_indexes.append([index])
                continue

            previous_start, previous_end = merged_spans[-1]
            previous_assignment = merged_assignments[-1]
            gap_seconds = max(0.0, span[0] - previous_end)
            merged_duration_seconds = span[1] - previous_start
            previous_speaker_id = self._known_speaker_id(previous_assignment)
            current_speaker_id = self._known_speaker_id(assignment)
            can_merge = (
                previous_speaker_id is not None
                and previous_speaker_id == current_speaker_id
                and gap_seconds <= max_gap_seconds
                and merged_duration_seconds <= max_duration_seconds
            )
            if can_merge:
                merged_spans[-1] = (previous_start, span[1])
                source_span_indexes[-1].append(index)
                continue

            merged_spans.append(span)
            merged_assignments.append(assignment)
            source_span_indexes.append([index])

        return merged_spans, merged_assignments, source_span_indexes

    @staticmethod
    def _known_speaker_id(assignment: Any) -> Optional[int]:
        """Return a stable speaker id, excluding unknown or failed assignments."""
        if assignment is None:
            return None
        try:
            speaker_id = int(getattr(assignment, "speaker_id"))
        except (TypeError, ValueError, AttributeError):
            return None
        return speaker_id if speaker_id >= 0 else None

    @staticmethod
    def _speaker_label(assignment: Any) -> Tuple[str, Dict[str, Any]]:
        if assignment is None:
            return "unknown_speaker", {}
        speaker_id = int(getattr(assignment, "speaker_id", -1))
        if speaker_id == 0:
            label = "user"
        elif speaker_id > 0:
            label = f"speaker_{speaker_id}"
        else:
            label = "unknown_speaker"
        return label, {
            "speaker_id": speaker_id,
            "speaker_similarity": getattr(assignment, "similarity", None),
            "speaker_assignment_reason": str(
                getattr(assignment, "reason", "unknown")
            ),
        }

    @staticmethod
    def _parse_session_start(value: Optional[str]) -> datetime:
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid session_start timestamp: {value}") from exc
        return datetime.now().astimezone()


__all__ = ["VoiceRuntime"]
