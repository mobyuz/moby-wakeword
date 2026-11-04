"""Audio capture as a small injectable interface, so tests need no microphone."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Protocol

SAMPLE_WIDTH_BYTES = 2  # 16 bit


class AudioSource(Protocol):
    """Records `seconds` of mono audio and returns 16-bit little-endian PCM bytes."""

    def record(self, seconds: float, sample_rate: int) -> bytes: ...


class SoundDeviceSource:
    """Real microphone through the optional `sounddevice` package (extra: `record`)."""

    def __init__(self, device: int | str | None = None) -> None:
        self._device = device

    def record(self, seconds: float, sample_rate: int) -> bytes:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "sounddevice is not installed; install the 'record' extra: pip install '.[record]'"
            ) from exc
        frames = int(seconds * sample_rate)  # pragma: no cover - needs a microphone
        data = sd.rec(  # pragma: no cover
            frames, samplerate=sample_rate, channels=1, dtype="int16", device=self._device
        )
        sd.wait()  # pragma: no cover
        return bytes(data.tobytes())  # pragma: no cover


class ScriptedSource:
    """Test double: returns prepared PCM chunks in order, without any hardware."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.calls: list[tuple[float, int]] = []

    def record(self, seconds: float, sample_rate: int) -> bytes:
        self.calls.append((seconds, sample_rate))
        if not self._chunks:
            raise RuntimeError("ScriptedSource has no more chunks")
        return self._chunks.pop(0)


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH_BYTES)
        out.setframerate(sample_rate)
        out.writeframes(pcm)


def peak_amplitude(pcm: bytes) -> int:
    """Largest absolute 16-bit sample value (0 for digital silence)."""
    peak = 0
    for i in range(0, len(pcm) - 1, 2):
        value = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        peak = max(peak, abs(value))
    return peak
