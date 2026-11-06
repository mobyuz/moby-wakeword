"""The INMP441 device path: what the model sees on the robot.

This module reproduces the firmware conversion bit-exactly. The source of truth is
`moby-firmware/main/moby_pcm.h` and `moby_pcm.c` (read only; branch `claude/f4-ptt-audio`, not
merged, not run on hardware). The contract, in the firmware's words:

    The INMP441 sends a 24-bit two's complement sample, MSB first, in the top bits of a 32-bit
    I2S slot (bits 31..8; bits 7..0 are zero or noise). Only the left slot is used.

        pcm16 = clamp(((int32_t)slot >> 16) * gain_q8 / 256)

    an arithmetic shift right by 16 (the top 16 bits of the 24-bit value, truncating toward minus
    infinity, no rounding, no dither), then a gain of gain_q8 / 256 with saturation. The gain is
    1.0 today (256). No DC removal and no filtering.

The firmware computes the scaled value as a floor division by 256, also for negative products
(`-((-product + 255) / 256)`); here that is an arithmetic shift right by 8, which is the same
function. A gain above 4096 is clamped to 4096. The warm-up drop (the first 13 blocks of 20 ms,
260 ms) happens at capture start on the robot; here it is the separate helper `drop_warmup`.

Two things are NOT part of the firmware contract and are assumptions, marked as such:

* `to_slot` quantises a float signal to the 24-bit sample with round-to-nearest-even. That models
  the microphone's own ADC, not the firmware. For integer 16-bit input it is exact (x * 256).
* The optional low-pass (`lowpass`) is a mild anti-aliasing stand-in. It is UNMEASURED: no real
  INMP441 frame was captured in this project. See configs/augmentation.yaml for the reasoning.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SAMPLE_RATE_HZ = 16_000

# Firmware constants (moby_pcm.h). tests/test_augment_device.py compares them with the header
# when the firmware repository sits next to this one.
SLOT_BITS = 32
SAMPLE_BITS = 24
PCM_BITS = 16
PCM_SHIFT = 16
GAIN_Q8_UNITY = 256
GAIN_Q8_MAX = 4096
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE_HZ // 1000 * FRAME_MS  # 320
WARMUP_FRAMES = 13
WARMUP_SAMPLES = WARMUP_FRAMES * FRAME_SAMPLES  # 4160 samples = 260 ms
WARMUP_MS = WARMUP_FRAMES * FRAME_MS

# Taps of the optional anti-aliasing FIR (unmeasured assumption, see configs/augmentation.yaml).
DEFAULT_LOWPASS_TAPS = 33

S24_MIN = -(1 << (SAMPLE_BITS - 1))
S24_MAX = (1 << (SAMPLE_BITS - 1)) - 1
INT16_MIN = -(1 << (PCM_BITS - 1))
INT16_MAX = (1 << (PCM_BITS - 1)) - 1

FloatArray = npt.NDArray[np.float64]
SlotArray = npt.NDArray[np.uint32]
Int16Array = npt.NDArray[np.int16]


def to_slot(signal: npt.ArrayLike) -> SlotArray:
    """16-bit-range samples (int or float) -> 32-bit I2S slots, 24-bit sample in bits 31..8.

    `signal` is in 16-bit units (full scale +-32768). The 24-bit sample is `rint(x * 256)`
    clamped to [-2**23, 2**23 - 1]; bits 7..0 of the slot are zero.
    """
    x = np.asarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinity")
    s24 = np.clip(np.rint(x * 256.0), S24_MIN, S24_MAX).astype(np.int64)
    return ((s24 << 8) & 0xFFFFFFFF).astype(np.uint32)


def slot_to_pcm16(slots: SlotArray, gain_q8: int = GAIN_Q8_UNITY) -> Int16Array:
    """The firmware conversion `moby_pcm_sample` for every slot (left-slot values).

    Arithmetic shift right by 16 (floor), times gain_q8 / 256 as a floor division, saturated to
    int16. No rounding, no dither, no DC removal.
    """
    if slots.dtype != np.uint32:
        raise TypeError(f"slots must be uint32, got {slots.dtype}")
    if gain_q8 < 0:
        raise ValueError("gain_q8 must not be negative (the firmware takes an unsigned value)")
    gain = min(int(gain_q8), GAIN_Q8_MAX)
    value = (np.ascontiguousarray(slots).view(np.int32) >> PCM_SHIFT).astype(np.int64)
    if gain != GAIN_Q8_UNITY:
        value = (value * gain) >> 8  # floor division by 256, also for negative products
    return np.clip(value, INT16_MIN, INT16_MAX).astype(np.int16)


def convert_stereo(stereo: SlotArray, gain_q8: int = GAIN_Q8_UNITY) -> Int16Array:
    """`moby_pcm_convert`: interleaved stereo slots (L, R, L, R, ...) -> mono, left slot only."""
    if stereo.ndim != 1 or stereo.size % 2:
        raise ValueError("stereo must be a 1-D array of interleaved L/R slots")
    return slot_to_pcm16(stereo[0::2], gain_q8)


def interleave_stereo(left: SlotArray, right: SlotArray | None = None) -> SlotArray:
    """Left slots with a right slot of zeros (L/R tied to GND: the right slot is not used)."""
    out = np.zeros(left.size * 2, dtype=np.uint32)
    out[0::2] = left
    if right is not None:
        out[1::2] = right
    return out


def drop_warmup(
    signal: npt.NDArray[np.generic], samples: int = WARMUP_SAMPLES
) -> npt.NDArray[np.generic]:
    """Drop the first 260 ms (13 blocks of 20 ms = 4160 samples at 16 kHz).

    The INMP441 output is not valid for the first 2**18 BCLK cycles after its clock starts, so the
    firmware discards them at capture start. Use this on audio that was captured from a device
    stream start; do not use it on TTS clips, which have no startup transient.
    """
    if samples < 0:
        raise ValueError("samples must not be negative")
    return signal[samples:]


def lowpass_taps(
    cutoff_hz: float, sample_rate_hz: int = SAMPLE_RATE_HZ, taps: int = DEFAULT_LOWPASS_TAPS
) -> FloatArray:
    """Windowed-sinc (Hamming) linear-phase low-pass, DC gain 1. `cutoff_hz` is the -6 dB point."""
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps must be odd and at least 3")
    if not 0.0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError(f"cutoff_hz must be inside (0, {sample_rate_hz / 2})")
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    fc = cutoff_hz / sample_rate_hz
    h = 2.0 * fc * np.sinc(2.0 * fc * n) * np.hamming(taps)
    return np.asarray(h / h.sum(), dtype=np.float64)


def lowpass(
    signal: npt.ArrayLike,
    cutoff_hz: float,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    taps: int = DEFAULT_LOWPASS_TAPS,
) -> FloatArray:
    """Zero-phase (centered) FIR low-pass. Same length as the input; zero padding at the edges."""
    x = np.asarray(signal, dtype=np.float64)
    h = lowpass_taps(cutoff_hz, sample_rate_hz, taps)
    start = (taps - 1) // 2
    return np.asarray(np.convolve(x, h)[start : start + x.size], dtype=np.float64)


def inmp441_path(
    signal: npt.ArrayLike,
    *,
    gain_q8: int = GAIN_Q8_UNITY,
    lowpass_hz: float | None = None,
    lowpass_taps_count: int = DEFAULT_LOWPASS_TAPS,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> Int16Array:
    """Full device path: optional low-pass, then slot form, then the firmware conversion.

    `signal` is in 16-bit units. With `lowpass_hz=None` and gain 256 an int16 input comes back
    unchanged (the path is the identity on 16-bit data); a float input is quantised to the 24-bit
    grid and then floored to 16 bit, so negative fractions round toward minus infinity.
    """
    x = np.asarray(signal, dtype=np.float64)
    if lowpass_hz is not None:
        x = lowpass(x, lowpass_hz, sample_rate_hz, lowpass_taps_count)
    return slot_to_pcm16(to_slot(x), gain_q8)
