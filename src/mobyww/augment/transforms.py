"""Deterministic building blocks: SNR mixing, RIR convolution, gain and time shift.

Signals are float64 in 16-bit units (full scale +-32768) and 16 kHz mono. Nothing here clips:
saturation happens once, in the device path (`mobyww.augment.device`). Randomness always comes
from a `numpy.random.Generator` that the caller seeds, so a call is reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# RIR sets that docs/licenses.md marks "allowed". A set not listed here cannot be enabled from a
# config, even if somebody puts its name into configs/augmentation.yaml.
APPROVED_RIR_SETS = frozenset({"but_reverbdb"})


def power(x: FloatArray) -> float:
    """Mean square of the whole clip."""
    return float(np.mean(np.square(x))) if x.size else 0.0


def mix_at_snr(
    speech: FloatArray,
    noise: FloatArray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    """Add `noise` to `speech` so that 10*log10(P_speech / P_noise) == snr_db.

    Power is the mean square over the whole clip (silence in the clip counts). The noise is
    wrapped around from a random start offset (drawn from `rng`) to cover the speech length.
    Returns (mixture, offset). Raises ValueError for silent speech or noise: the SNR is undefined.
    """
    if speech.size == 0:
        raise ValueError("speech is empty")
    if noise.size == 0:
        raise ValueError("noise is empty")
    offset = int(rng.integers(0, noise.size))
    idx = (offset + np.arange(speech.size)) % noise.size
    segment = noise[idx]
    p_speech, p_noise = power(speech), power(segment)
    if p_speech == 0.0:
        raise ValueError("speech is silent: SNR is undefined")
    if p_noise == 0.0:
        raise ValueError("the chosen noise segment is silent: SNR is undefined")
    scale = float(np.sqrt(p_speech / (p_noise * 10.0 ** (snr_db / 10.0))))
    return np.asarray(speech + scale * segment, dtype=np.float64), offset


def apply_gain_db(x: FloatArray, gain_db: float) -> FloatArray:
    return np.asarray(x * 10.0 ** (gain_db / 20.0), dtype=np.float64)


def time_shift(x: FloatArray, shift_samples: int) -> FloatArray:
    """Shift by `shift_samples` (positive = later) keeping the length; zero fill, no wrap-around."""
    n = x.size
    out = np.zeros_like(x)
    if shift_samples >= n or -shift_samples >= n:
        return out
    if shift_samples >= 0:
        out[shift_samples:] = x[: n - shift_samples]
    else:
        out[: n + shift_samples] = x[-shift_samples:]
    return out


def convolve_rir(x: FloatArray, rir: FloatArray) -> FloatArray:
    """Convolve with a room impulse response, same length as `x`.

    The response is cut before its strongest tap (so the direct path is not delayed) and scaled so
    that this tap is 1: the direct path keeps gain 1 and the reverb adds on top.
    """
    if rir.size == 0:
        raise ValueError("rir is empty")
    peak = int(np.argmax(np.abs(rir)))
    h = rir[peak:]
    if h[0] == 0.0:
        raise ValueError("rir is silent")
    h = h / abs(h[0])
    n = x.size + h.size - 1
    size = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(x, size) * np.fft.rfft(h, size)
    return np.asarray(np.fft.irfft(spectrum, size)[: x.size], dtype=np.float64)


def ensure_rir_sets_allowed(names: tuple[str, ...]) -> None:
    unknown = sorted(set(names) - APPROVED_RIR_SETS)
    if unknown:
        raise ValueError(
            f"RIR sets not approved in docs/licenses.md: {', '.join(unknown)} "
            f"(approved: {', '.join(sorted(APPROVED_RIR_SETS))})"
        )
