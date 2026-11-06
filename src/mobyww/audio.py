"""Small WAV helpers on the standard library (16-bit mono PCM only).

Training audio in this project is 16 kHz, mono, signed 16-bit (configs/wakeword.yaml). Anything
else is rejected here instead of being converted silently.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Int16Array = npt.NDArray[np.int16]


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def wav_info(path: Path) -> WavInfo:
    """Header facts of a 16-bit mono WAV file; raises ValueError for anything else."""
    with wave.open(str(path), "rb") as handle:
        _check_format(path, handle.getnchannels(), handle.getsampwidth())
        return WavInfo(sample_rate=handle.getframerate(), frames=handle.getnframes())


def read_wav_int16(path: Path) -> tuple[Int16Array, int]:
    """Read a 16-bit mono WAV file. Returns (samples, sample_rate)."""
    with wave.open(str(path), "rb") as handle:
        _check_format(path, handle.getnchannels(), handle.getsampwidth())
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.int16), rate


def write_wav_int16(path: Path, samples: Int16Array, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _check_format(path: Path, channels: int, width: int) -> None:
    if channels != 1 or width != 2:
        raise ValueError(
            f"{path}: only 16-bit mono WAV is supported (got {channels} ch, {width} B)"
        )
