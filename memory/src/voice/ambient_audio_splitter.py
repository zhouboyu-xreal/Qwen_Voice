"""Find shared quiet intervals using capture timestamps, never arrival order.

Only analysis copies are resampled. The recorder keeps every original PCM
sample, including silence. This module runs on its own serial worker so VAD
model loading/inference cannot block audio writes or load ASR/speaker models.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from math import gcd, isfinite
from typing import Callable, Optional

import numpy as np


class _SpeechFrames:
    def __init__(self, sample_rate: int, detector_factory: Callable):
        self.sample_rate = sample_rate
        self.detector = detector_factory()
        self.origin: Optional[datetime] = None
        self.pending = bytearray()
        self.processed = 0
        self.frame_samples = max(1, round(sample_rate * 0.032))

    def accept(self, pcm: bytes, started_at: datetime):
        if self.origin is None:
            self.origin = started_at
        self.pending.extend(pcm)
        frame_bytes = self.frame_samples * 2
        offset = 0
        while len(self.pending) - offset >= frame_bytes:
            samples = np.frombuffer(self.pending, dtype='<i2', count=self.frame_samples,
                                    offset=offset).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples * samples)))
            if len(samples) != 512:
                from scipy.signal import resample_poly
                divisor = gcd(512, len(samples))
                samples = resample_poly(samples, 512 // divisor, len(samples) // divisor)
            start = self.origin + timedelta(seconds=self.processed / self.sample_rate)
            self.processed += self.frame_samples
            end = self.origin + timedelta(seconds=self.processed / self.sample_rate)
            active = bool(self.detector.accept_frame(samples, frame_start=start.timestamp(),
                                                      frame_end=end.timestamp()))
            yield start, end, active, rms
            offset += frame_bytes
        del self.pending[:offset]


class JointSilenceSplitter:
    """Require an overlapping quiet span on every enabled source after the gate."""
    def __init__(self, *, sources, started_at: datetime, min_duration_seconds: float,
                 silence_seconds: float = 0.8, quiet_max_rms_dbfs: float = -35.0,
                 vad_config=None, detector_factory=None):
        if not isfinite(min_duration_seconds) or min_duration_seconds <= 0:
            raise ValueError('split minimum duration must be finite and positive')
        if not isfinite(silence_seconds) or silence_seconds <= 0:
            raise ValueError('split silence duration must be finite and positive')
        if not isfinite(quiet_max_rms_dbfs) or not -120 <= quiet_max_rms_dbfs <= 0:
            raise ValueError('quiet RMS threshold must be between -120 and 0 dBFS')
        self.quiet_max_rms = 10 ** (quiet_max_rms_dbfs / 20)
        self.min_duration = timedelta(seconds=min_duration_seconds)
        self.silence = timedelta(seconds=silence_seconds)
        self.search_after = started_at + self.min_duration
        self.sources = tuple(sources)
        self.quiet = {source: deque() for source in self.sources}
        self.coverage = {source: None for source in self.sources}
        self.tracks = {}
        self.vad_config = dict(vad_config or {})
        self.detector_factory = detector_factory or self._make_detector

    def _make_detector(self):
        # Silero loads its bundled model locally. Keep independent recurrent
        # state per source and separate from the offline ASR worker's VAD.
        from .vad_detector import StreamingSileroVAD
        return StreamingSileroVAD(sample_rate=16_000,
            threshold=float(self.vad_config.get('threshold', 0.5)),
            min_silence_duration_ms=int(self.vad_config.get('min_silence_ms', 250)),
            speech_pad_ms=int(self.vad_config.get('speech_pad_ms', 30)))

    def accept(self, source: str, pcm: bytes, sample_rate: int, started_at: datetime):
        track = self.tracks.get(source)
        if track is None:
            track = self.tracks[source] = _SpeechFrames(sample_rate, self.detector_factory)
        if track.sample_rate != sample_rate:
            raise ValueError('reset joint splitter before changing capture format')
        for start, end, active, rms in track.accept(pcm, started_at):
            self.coverage[source] = end
            spans = self.quiet[source]
            if not active and rms <= self.quiet_max_rms and end > self.search_after:
                if spans and spans[-1][1] == start:
                    spans[-1] = (spans[-1][0], end)
                else:
                    spans.append((max(start, self.search_after), end))
        cuts = []
        while all(self.quiet[source] for source in self.sources):
            spans = [self.quiet[source][0] for source in self.sources]
            start = max(self.search_after, *(span[0] for span in spans))
            end = min(span[1] for span in spans)
            if end - start >= self.silence:
                cut = start + self.silence
                cuts.append(cut)
                self.search_after = cut + self.min_duration
                for source in self.sources:
                    while self.quiet[source] and self.quiet[source][0][1] <= self.search_after:
                        self.quiet[source].popleft()
                continue
            # An interval ending before the common required end cannot ever
            # match a later interval on another source.
            if all(value is not None for value in self.coverage.values()):
                expired = [source for source in self.sources
                           if self.quiet[source][0][1] < self.coverage[source]
                           and self.quiet[source][0][1] - start < self.silence]
                if expired:
                    for source in expired:
                        self.quiet[source].popleft()
                    continue
            break
        # A source can remain busy for hours. Retain only quiet intervals
        # which the slowest source has not yet ruled out, not all history.
        if all(value is not None for value in self.coverage.values()):
            watermark = min(self.coverage.values())
            for source in self.sources:
                while self.quiet[source] and self.quiet[source][0][1] < watermark:
                    self.quiet[source].popleft()
        return cuts
