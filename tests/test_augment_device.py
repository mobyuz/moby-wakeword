"""The INMP441 device path against the firmware contract (moby-firmware main/moby_pcm.c).

The golden pairs below were derived by hand from the C code, not from this implementation:

    top   = slot >> 16 (unsigned), value = top >= 0x8000 ? top - 65536 : top
    gain == 256 -> value
    else        -> product = value * gain
                   scaled  = product >= 0 ? product / 256 : -((-product + 255) / 256)   (floor)
                   clamp to [-32768, 32767]
    a gain above 4096 is first clamped to 4096.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from mobyww.augment import device

# (slot, gain_q8, expected pcm16, why)
GOLDEN_SLOTS = [
    (0x00000000, 256, 0, "silence"),
    (0x7FFFFF00, 256, 32767, "24-bit max 0x7FFFFF -> top 0x7FFF"),
    (0x80000000, 256, -32768, "24-bit min -8388608 -> top 0x8000"),
    (0xFFFFFF00, 256, -1, "24-bit -1: floor toward minus infinity, not 0"),
    (0x0000FF00, 256, 0, "24-bit 255: truncated, not rounded up"),
    (0x00010000, 256, 1, "24-bit 256 -> 1"),
    (0xFFFF0000, 256, -1, "24-bit -256 -> -1"),
    (0xFFFEFF00, 256, -2, "24-bit -257: floor(-1.004) = -2"),
    (0xFFFF8000, 256, -1, "24-bit -128: floor(-0.5) = -1"),
    (0x12345678, 256, 0x1234, "bits 7..0 and 15..8 below the top 16 are ignored"),
    (0x00640000, 512, 200, "100 * 2.0"),
    (0x00030000, 384, 4, "3 * 1.5 = 4.5 floors to 4"),
    (0xFFFD0000, 384, -5, "-3 * 1.5 = -4.5 floors to -5, not -4"),
    (0x4E200000, 512, 32767, "20000 * 2 saturates high"),
    (0xB1E00000, 512, -32768, "-20000 * 2 saturates low"),
    (0x00010000, 5000, 16, "gain above 4096 is clamped to 4096: 1 * 16"),
    (0xFFFF0000, 1, -1, "-1 * 1/256 floors to -1 (negative stays negative)"),
    (0x00010000, 1, 0, "1 * 1/256 floors to 0"),
    (0xFFFF0000, 0, 0, "gain 0"),
    (0x80000000, 4096, -32768, "-32768 * 16 saturates low"),
    (0x7FFF0000, 4096, 32767, "32767 * 16 saturates high"),
]

# (float sample in 16-bit units, expected slot, why)
GOLDEN_TO_SLOT = [
    (0.0, 0x00000000, "zero"),
    (1.0, 0x00010000, "1 * 256 = 0x000100 in bits 31..8"),
    (-1.0, 0xFFFF0000, "-256 two's complement"),
    (0.5, 0x00008000, "128"),
    (-0.5, 0xFFFF8000, "-128"),
    (1.0 / 256.0, 0x00000100, "smallest 24-bit step"),
    (-1.0 / 256.0, 0xFFFFFF00, "-1 step"),
    (32767.0, 0x7FFF0000, "int16 max"),
    (-32768.0, 0x80000000, "int16 min"),
    (40000.0, 0x7FFFFF00, "clamped to the 24-bit max"),
    (-40000.0, 0x80000000, "clamped to the 24-bit min"),
]

# (float sample, expected pcm16 at gain 256): the floor happens after the 24-bit quantisation.
GOLDEN_END_TO_END = [
    (0.5, 0),
    (-0.5, -1),
    (1.99, 1),
    (-1.99, -2),
    (-0.01, -1),
    (123.0, 123),
    (-123.0, -123),
    (32767.9, 32767),
    (-32768.0, -32768),
]


def c_reference(slot: int, gain_q8: int) -> int:
    """A literal port of moby_pcm_sample() using Python integers (the oracle)."""
    if gain_q8 > device.GAIN_Q8_MAX:
        gain_q8 = device.GAIN_Q8_MAX
    top = slot >> 16
    value = top - 65536 if top >= 0x8000 else top
    if gain_q8 == 256:
        return value
    product = value * gain_q8
    scaled = product // 256 if product >= 0 else -((-product + 255) // 256)
    return max(-32768, min(32767, scaled))


def slots(*values: int) -> device.SlotArray:
    return np.array(values, dtype=np.uint32)


@pytest.mark.parametrize(("slot", "gain", "expected", "why"), GOLDEN_SLOTS)
def test_golden_slot_to_pcm16(slot: int, gain: int, expected: int, why: str) -> None:
    assert c_reference(slot, gain) == expected, f"hand value disagrees with the C port: {why}"
    assert int(device.slot_to_pcm16(slots(slot), gain)[0]) == expected, why


@pytest.mark.parametrize(("value", "expected", "why"), GOLDEN_TO_SLOT)
def test_golden_to_slot(value: float, expected: int, why: str) -> None:
    assert int(device.to_slot([value])[0]) == expected, why


@pytest.mark.parametrize(("value", "expected"), GOLDEN_END_TO_END)
def test_golden_end_to_end(value: float, expected: int) -> None:
    assert int(device.inmp441_path([value])[0]) == expected


def test_matches_the_c_port_on_random_slots_and_gains() -> None:
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 2**32, size=20000, dtype=np.uint64).astype(np.uint32)
    for gain in (256, 1, 100, 255, 257, 384, 512, 1000, 4095, 4096, 4097, 65535):
        got = device.slot_to_pcm16(raw, gain)
        want = [c_reference(int(s), gain) for s in raw]
        assert got.tolist() == want, f"gain {gain}"


def test_int16_input_is_unchanged_by_the_unity_path() -> None:
    every = np.arange(-32768, 32768, dtype=np.int16)
    assert np.array_equal(device.inmp441_path(every), every)
    assert np.array_equal(device.slot_to_pcm16(device.to_slot(every)), every)


def test_conversion_digest_has_not_drifted() -> None:
    """Fails if the conversion changes. The digest comes from the C port, not from device.py."""
    tops = (np.arange(65536, dtype=np.uint64) << 16).astype(np.uint32)
    digest = hashlib.sha256()
    for gain in (256, 1, 384, 512, 4096, 5000):
        digest.update(device.slot_to_pcm16(tops, gain).astype("<i2").tobytes())
    assert digest.hexdigest() == EXPECTED_DIGEST


def test_c_port_digest_matches_too() -> None:
    digest = hashlib.sha256()
    for gain in (256, 1, 384, 512, 4096, 5000):
        values = [c_reference(top << 16, gain) for top in range(65536)]
        digest.update(np.array(values, dtype="<i2").tobytes())
    assert digest.hexdigest() == EXPECTED_DIGEST


def test_no_dither_and_no_dc_removal() -> None:
    constant = np.full(
        1000, 10.25
    )  # a DC offset must survive, and the same input gives the same output
    out = device.inmp441_path(constant)
    assert set(out.tolist()) == {10}
    assert np.array_equal(out, device.inmp441_path(constant))


def test_stereo_conversion_uses_only_the_left_slot() -> None:
    stereo = slots(0x00010000, 0xDEADBEEF, 0xFFFF0000, 0x12345678)
    assert device.convert_stereo(stereo).tolist() == [1, -1]
    left = slots(0x00010000, 0xFFFF0000)
    assert np.array_equal(device.interleave_stereo(left)[0::2], left)
    assert device.convert_stereo(device.interleave_stereo(left)).tolist() == [1, -1]
    with pytest.raises(ValueError, match="interleaved"):
        device.convert_stereo(slots(1, 2, 3))


def test_wrong_input_types_are_rejected() -> None:
    with pytest.raises(TypeError):
        device.slot_to_pcm16(np.array([1], dtype=np.int32))
    with pytest.raises(ValueError, match="negative"):
        device.slot_to_pcm16(slots(0), -1)
    with pytest.raises(ValueError, match="NaN"):
        device.to_slot([float("nan")])


def test_warmup_drop_is_260_ms() -> None:
    assert device.WARMUP_SAMPLES == 4160
    assert device.WARMUP_MS == 260
    assert device.WARMUP_SAMPLES * 1000 // device.SAMPLE_RATE_HZ == 260
    signal = np.arange(5000, dtype=np.int16)
    out = device.drop_warmup(signal)
    assert out.size == 840
    assert out[0] == 4160
    assert device.drop_warmup(signal[:100]).size == 0
    with pytest.raises(ValueError, match="negative"):
        device.drop_warmup(signal, -1)


def test_lowpass_is_mild_and_keeps_length() -> None:
    rate = device.SAMPLE_RATE_HZ
    t = np.arange(4000) / rate
    low = 1000 * np.sin(2 * np.pi * 1000 * t)
    high = 1000 * np.sin(2 * np.pi * 7800 * t)
    out_low = device.lowpass(low, 7200)
    out_high = device.lowpass(high, 7200)
    assert out_low.size == low.size
    mid = slice(500, 3500)  # away from the zero-padded edges
    assert np.max(np.abs(out_low[mid] - low[mid])) < 1.0  # 1 kHz passes within 0.1 %
    assert np.max(np.abs(out_high[mid])) < 0.05 * 1000  # 7.8 kHz is down by more than 26 dB
    assert np.allclose(device.lowpass(np.full(300, 5.0), 7200)[100:200], 5.0)  # DC gain 1
    assert device.lowpass(np.ones(3), 7200).size == 3  # shorter than the filter
    with pytest.raises(ValueError, match="cutoff_hz"):
        device.lowpass(low, 8000)
    with pytest.raises(ValueError, match="odd"):
        device.lowpass_taps(7200, taps=32)


def test_constants_match_the_firmware_header_when_it_is_next_to_this_repo() -> None:
    header = Path(__file__).resolve().parents[2] / "moby-firmware" / "main" / "moby_pcm.h"
    if not header.is_file():
        pytest.skip("moby-firmware is not checked out next to this repository")
    text = header.read_text(encoding="utf-8")

    def number(name: str) -> int:
        match = re.search(rf"\b{name}\s*=\s*(\d+)\s*,", text)
        assert match, f"{name} not found as a plain number in {header.name}"
        return int(match.group(1))

    assert number("MOBY_PCM_SAMPLE_RATE") == device.SAMPLE_RATE_HZ
    assert number("MOBY_PCM_SHIFT") == device.PCM_SHIFT
    assert number("MOBY_PCM_GAIN_Q8") == device.GAIN_Q8_UNITY
    assert number("MOBY_PCM_GAIN_Q8_MAX") == device.GAIN_Q8_MAX
    assert number("MOBY_PCM_FRAME_MS") == device.FRAME_MS
    # MOBY_PCM_FRAME_SAMPLES = MOBY_PCM_SAMPLE_RATE / 1000 * MOBY_PCM_FRAME_MS
    assert "MOBY_PCM_SAMPLE_RATE / 1000 * MOBY_PCM_FRAME_MS" in text
    assert re.search(r"MOBY_PCM_WARMUP_SAMPLES\s*=\s*13\s*\*\s*MOBY_PCM_FRAME_SAMPLES", text)


# sha256 of the concatenated little-endian int16 outputs for every top-16 value, gains
# 256, 1, 384, 512, 4096, 5000, generated by the literal C port above.
EXPECTED_DIGEST = "0af0bc63161def7d6a9eb2ba2158eb4120138c5e947898f7011f1b1249f8b608"
