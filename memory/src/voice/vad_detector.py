from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad


@dataclass
class VADTransition:
    event_type: str
    time: float
    frame_start: float
    frame_end: float


class StreamingSileroVAD:
    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
    ) -> None:
        self.iterator = VADIterator(
            load_silero_vad(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.sample_rate = sample_rate
        self._last_transition: Optional[VADTransition] = None
        self._next_frame_start = 0.0

    def accept_frame(
        self,
        frame: np.ndarray,
        frame_start: Optional[float] = None,
        frame_end: Optional[float] = None,
    ) -> bool:
        if frame_start is None:
            frame_start = self._next_frame_start
        if frame_end is None:
            frame_end = frame_start + (len(frame) / self.sample_rate)
        self._next_frame_start = frame_end
        self._last_transition = None

        event = self.iterator(torch.tensor(frame, dtype=torch.float32), return_seconds=True)
        if event and "start" in event:
            self._last_transition = VADTransition(
                event_type="start",
                time=float(event["start"]),
                frame_start=frame_start,
                frame_end=frame_end,
            )
        elif event and "end" in event:
            self._last_transition = VADTransition(
                event_type="end",
                time=float(event["end"]),
                frame_start=frame_start,
                frame_end=frame_end,
            )
        return bool(self.iterator.triggered)

    def reset_stream_state(self) -> None:
        """Reset stream state while keeping the loaded VAD model resident."""
        self.iterator.reset_states()
        self._last_transition = None
        self._next_frame_start = 0.0

    @property
    def raw_triggered(self) -> bool:
        return bool(self.iterator.triggered)

    @property
    def stable_triggered(self) -> bool:
        return bool(self.iterator.triggered)

    @property
    def last_transition(self) -> Optional[VADTransition]:
        return self._last_transition
