"""The configs/augmentation.yaml plan as a deterministic, logged function.

Order of stages for one clip: time shift, room impulse response (only an approved set, none by
default), noise at a random SNR, gain, then the INMP441 device path (mild low-pass, slot form,
firmware conversion). Every random draw is made in a fixed order from a generator seeded by
(config seed, clip id), so the same inputs give the same output and adding or removing a stage
does not move the other draws. Each call returns a record with the seeds and parameters used; it is
also written to the `mobyww.augment` logger as one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from mobyww.augment import device, transforms

LOGGER = logging.getLogger("mobyww.augment")

Int16Array = npt.NDArray[np.int16]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Range:
    min: float
    max: float

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"range min {self.min} is above max {self.max}")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.min, self.max))


@dataclass(frozen=True)
class LowpassConfig:
    kind: str
    cutoff_hz: float
    taps: int = device.DEFAULT_LOWPASS_TAPS
    measured: bool = False


@dataclass(frozen=True)
class DevicePathConfig:
    gain_q8: int = device.GAIN_Q8_UNITY
    lowpass: LowpassConfig | None = None


@dataclass(frozen=True)
class AugmentConfig:
    seed: int
    snr_db: Range
    gain_db: Range
    time_shift_ms: Range
    rir_sets: tuple[str, ...]
    device_path: DevicePathConfig


@dataclass(frozen=True)
class AugmentResult:
    audio: Int16Array
    record: dict[str, Any]


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing '{key}'")
    return mapping[key]


def _range(mapping: dict[str, Any], key: str) -> Range:
    entry = _require(mapping, key, "augmentation config")
    return Range(float(_require(entry, "min", key)), float(_require(entry, "max", key)))


def parse_config(data: dict[str, Any]) -> AugmentConfig:
    """Validate a parsed configs/augmentation.yaml. The device path must match the firmware."""
    snr = _range(data, "snr_db")
    if snr.min < 0:
        raise ValueError("snr_db.min must not be negative")
    rir_sets = tuple(str(s) for s in (data.get("rir") or {}).get("sets") or [])
    transforms.ensure_rir_sets_allowed(rir_sets)

    dev = _require(data, "device_path", "augmentation config")
    bits = _require(dev, "bit_reduction", "device_path")
    expected = {
        "slot_bits": device.SLOT_BITS,
        "from_bits": device.SAMPLE_BITS,
        "to_bits": device.PCM_BITS,
        "shift": device.PCM_SHIFT,
    }
    for key, value in expected.items():
        if bits.get(key) != value:
            raise ValueError(f"device_path.bit_reduction.{key} must be {value} (firmware contract)")
    if bits.get("channel") != "left":
        raise ValueError("device_path.bit_reduction.channel must be 'left' (firmware contract)")
    gain_q8 = int(bits.get("gain_q8", device.GAIN_Q8_UNITY))
    if not 0 <= gain_q8 <= device.GAIN_Q8_MAX:
        raise ValueError(f"gain_q8 must be in [0, {device.GAIN_Q8_MAX}]")

    lowpass: LowpassConfig | None = None
    raw_lowpass = dev.get("lowpass")
    if raw_lowpass is not None:
        cutoff = raw_lowpass.get("cutoff_hz")
        if not isinstance(cutoff, (int, float)):
            raise ValueError("device_path.lowpass.cutoff_hz must be a number (it is still TBD?)")
        device.lowpass_taps(
            float(cutoff), taps=int(raw_lowpass.get("taps", device.DEFAULT_LOWPASS_TAPS))
        )  # validates
        lowpass = LowpassConfig(
            kind=str(raw_lowpass.get("kind", "mild")),
            cutoff_hz=float(cutoff),
            taps=int(raw_lowpass.get("taps", device.DEFAULT_LOWPASS_TAPS)),
            measured=bool(raw_lowpass.get("measured", False)),
        )
    return AugmentConfig(
        seed=int(_require(data, "seed", "augmentation config")),
        snr_db=snr,
        gain_db=_range(data, "gain_db"),
        time_shift_ms=_range(data, "time_shift_ms"),
        rir_sets=rir_sets,
        device_path=DevicePathConfig(gain_q8=gain_q8, lowpass=lowpass),
    )


def load_config(path: Path) -> AugmentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a mapping")
    return parse_config(data)


def clip_seed(seed: int, clip_id: str) -> int:
    """Stable 64-bit seed for one clip (independent of call order and of Python's hash seed)."""
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _as_float(signal: npt.ArrayLike) -> FloatArray:
    return np.asarray(signal, dtype=np.float64)


class Augmenter:
    """Applies the plan. Holds only the config; per-clip state is derived from the clip id."""

    def __init__(self, config: AugmentConfig) -> None:
        self.config = config

    def augment(
        self,
        clip_id: str,
        speech: npt.ArrayLike,
        *,
        noise: npt.ArrayLike | None = None,
        noise_id: str = "",
        rir: npt.ArrayLike | None = None,
        rir_set: str = "",
        rir_id: str = "",
    ) -> AugmentResult:
        cfg = self.config
        x = _as_float(speech)
        derived = clip_seed(cfg.seed, clip_id)
        rng = np.random.default_rng(derived)
        # Fixed draw order, always drawn, so skipped stages do not shift the others.
        snr_db = cfg.snr_db.draw(rng)
        gain_db = cfg.gain_db.draw(rng)
        shift_ms = cfg.time_shift_ms.draw(rng)
        shift_samples = round(shift_ms * device.SAMPLE_RATE_HZ / 1000.0)
        noise_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

        y = transforms.time_shift(x, shift_samples)

        rir_used = ""
        if rir is not None:
            if rir_set not in cfg.rir_sets:
                raise ValueError(
                    f"RIR set {rir_set!r} is not enabled in the config (enabled: {cfg.rir_sets})"
                )
            y = transforms.convolve_rir(y, _as_float(rir))
            rir_used = f"{rir_set}/{rir_id}"

        noise_offset: int | None = None
        snr_used: float | None = None
        if noise is not None:
            y, noise_offset = transforms.mix_at_snr(y, _as_float(noise), snr_db, noise_rng)
            snr_used = snr_db

        y = transforms.apply_gain_db(y, gain_db)

        lowpass = cfg.device_path.lowpass
        audio = device.inmp441_path(
            y,
            gain_q8=cfg.device_path.gain_q8,
            lowpass_hz=None if lowpass is None else lowpass.cutoff_hz,
            lowpass_taps_count=device.DEFAULT_LOWPASS_TAPS if lowpass is None else lowpass.taps,
        )
        record: dict[str, Any] = {
            "clip_id": clip_id,
            "config_seed": cfg.seed,
            "clip_seed": derived,
            "samples": int(x.size),
            "time_shift_ms": round(shift_ms, 6),
            "time_shift_samples": shift_samples,
            "rir": rir_used,
            "noise_id": noise_id if noise is not None else "",
            "noise_offset": noise_offset,
            "snr_db": None if snr_used is None else round(snr_used, 6),
            "gain_db": round(gain_db, 6),
            "lowpass_hz": None if lowpass is None else lowpass.cutoff_hz,
            "lowpass_measured": None if lowpass is None else lowpass.measured,
            "device_gain_q8": cfg.device_path.gain_q8,
            "saturated_samples": int(
                np.count_nonzero((audio == device.INT16_MAX) | (audio == device.INT16_MIN))
            ),
        }
        LOGGER.info("augment %s", json.dumps(record, sort_keys=True))
        return AugmentResult(audio=audio, record=record)


def write_log(path: Path, records: list[dict[str, Any]]) -> None:
    """One JSON object per line (JSONL), sorted keys, UTF-8, LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
