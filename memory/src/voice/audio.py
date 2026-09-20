from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio_mono(audio_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    audio, original_sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if original_sample_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=original_sample_rate, target_sr=sample_rate)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sample_rate


def slice_audio(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_index = max(0, int(start * sample_rate))
    end_index = min(len(audio), int(end * sample_rate))
    if end_index <= start_index:
        return np.zeros(0, dtype=np.float32)
    return audio[start_index:end_index]

