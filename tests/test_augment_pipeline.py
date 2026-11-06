from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from mobyww.augment import (
    Augmenter,
    drop_warmup,
    load_config,
    parse_config,
    write_log,
)
from mobyww.augment import transforms as tr
from mobyww.augment.pipeline import clip_seed

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000


def base_config() -> dict[str, Any]:
    data = yaml.safe_load((ROOT / "configs" / "augmentation.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def speech(seconds: float = 1.0, amplitude: float = 6000.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * 300 * t) * np.hanning(t.size)


def noise(seconds: float = 3.0, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 2000, int(seconds * RATE))


def test_repo_config_loads_and_is_the_documented_plan() -> None:
    cfg = load_config(ROOT / "configs" / "augmentation.yaml")
    assert cfg.seed == 2024
    assert (cfg.snr_db.min, cfg.snr_db.max) == (0, 30)
    assert (cfg.gain_db.min, cfg.gain_db.max) == (-12, 6)
    assert (cfg.time_shift_ms.min, cfg.time_shift_ms.max) == (-100, 100)
    assert cfg.rir_sets == ()  # empty by default
    assert cfg.device_path.gain_q8 == 256
    assert cfg.device_path.lowpass is not None
    assert cfg.device_path.lowpass.cutoff_hz == 7200
    assert cfg.device_path.lowpass.measured is False  # the low-pass is an unmeasured assumption


def test_config_rejects_unapproved_rir_set_and_firmware_mismatch() -> None:
    data = base_config()
    data["rir"] = {"sets": ["mit_ir_survey"]}
    with pytest.raises(ValueError, match="not approved"):
        parse_config(data)

    data = base_config()
    data["rir"] = {"sets": ["but_reverbdb"]}
    assert parse_config(data).rir_sets == ("but_reverbdb",)

    data = base_config()
    data["device_path"]["bit_reduction"]["shift"] = 8
    with pytest.raises(ValueError, match="shift"):
        parse_config(data)

    data = base_config()
    data["device_path"]["bit_reduction"]["channel"] = "right"
    with pytest.raises(ValueError, match="channel"):
        parse_config(data)

    data = base_config()
    data["device_path"]["lowpass"]["cutoff_hz"] = "TBD"
    with pytest.raises(ValueError, match="cutoff_hz"):
        parse_config(data)

    data = base_config()
    data["device_path"]["lowpass"]["cutoff_hz"] = 9000  # above Nyquist
    with pytest.raises(ValueError, match="cutoff_hz"):
        parse_config(data)

    data = base_config()
    data["snr_db"] = {"min": 20, "max": 10}
    with pytest.raises(ValueError, match="above max"):
        parse_config(data)


def test_same_inputs_give_identical_output_and_record() -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))
    a = aug.augment("clip-1", speech(), noise=noise(), noise_id="n1")
    b = aug.augment("clip-1", speech(), noise=noise(), noise_id="n1")
    assert np.array_equal(a.audio, b.audio)
    assert a.record == b.record
    assert a.audio.dtype == np.int16
    assert a.audio.size == speech().size


def test_call_order_does_not_change_a_clip() -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))
    first = aug.augment("x", speech(), noise=noise())
    aug.augment("y", speech(), noise=noise())
    again = aug.augment("x", speech(), noise=noise())
    assert np.array_equal(first.audio, again.audio)


def test_different_clips_and_seeds_differ() -> None:
    cfg = load_config(ROOT / "configs" / "augmentation.yaml")
    a = Augmenter(cfg).augment("a", speech(), noise=noise())
    b = Augmenter(cfg).augment("b", speech(), noise=noise())
    assert a.record["clip_seed"] != b.record["clip_seed"]
    assert not np.array_equal(a.audio, b.audio)
    other = deepcopy(base_config())
    other["seed"] = 1
    c = Augmenter(parse_config(other)).augment("a", speech(), noise=noise())
    assert c.record["clip_seed"] != a.record["clip_seed"]
    assert clip_seed(2024, "a") == a.record["clip_seed"]  # stable, not Python's hash()


def test_skipping_noise_does_not_move_the_other_draws() -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))
    with_noise = aug.augment("k", speech(), noise=noise()).record
    without = aug.augment("k", speech()).record
    for key in ("gain_db", "time_shift_ms", "time_shift_samples", "clip_seed"):
        assert with_noise[key] == without[key]
    assert without["snr_db"] is None
    assert without["noise_offset"] is None


def test_record_has_seeds_and_parameters_inside_the_plan_ranges() -> None:
    cfg = load_config(ROOT / "configs" / "augmentation.yaml")
    aug = Augmenter(cfg)
    for i in range(50):
        rec = aug.augment(f"c{i}", speech(), noise=noise(), noise_id="n").record
        assert 0 <= rec["snr_db"] <= 30
        assert -12 <= rec["gain_db"] <= 6
        assert -100 <= rec["time_shift_ms"] <= 100
        assert rec["config_seed"] == 2024
        assert rec["device_gain_q8"] == 256
        assert rec["lowpass_hz"] == 7200
        assert rec["lowpass_measured"] is False
        json.dumps(rec)  # serialisable


def test_each_call_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))
    with caplog.at_level(logging.INFO, logger="mobyww.augment"):
        res = aug.augment("logged", speech(), noise=noise())
    lines = [r.getMessage() for r in caplog.records if r.name == "mobyww.augment"]
    assert len(lines) == 1
    payload = json.loads(lines[0].removeprefix("augment "))
    assert payload == res.record


def test_write_log_is_jsonl(tmp_path: Path) -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))
    records = [aug.augment(f"c{i}", speech()).record for i in range(3)]
    path = tmp_path / "logs" / "augment.jsonl"
    write_log(path, records)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["clip_id"] for line in lines] == ["c0", "c1", "c2"]


def test_rir_needs_an_enabled_approved_set() -> None:
    aug = Augmenter(load_config(ROOT / "configs" / "augmentation.yaml"))  # sets: []
    with pytest.raises(ValueError, match="not enabled"):
        aug.augment("r", speech(), rir=np.array([1.0, 0.5]), rir_set="but_reverbdb")

    data = base_config()
    data["rir"] = {"sets": ["but_reverbdb"]}
    enabled = Augmenter(parse_config(data))
    res = enabled.augment(
        "r", speech(), rir=np.array([1.0, 0.0, 0.5]), rir_set="but_reverbdb", rir_id="x"
    )
    assert res.record["rir"] == "but_reverbdb/x"


def test_mix_at_snr_hits_the_requested_snr() -> None:
    rng = np.random.default_rng(1)
    s, n = speech(), noise()
    for snr in (0.0, 10.0, 30.0):
        mixed, offset = tr.mix_at_snr(s, n, snr, rng)
        noise_part = mixed - s
        measured = 10 * np.log10(tr.power(s) / tr.power(noise_part))
        assert measured == pytest.approx(snr, abs=1e-9)
        assert 0 <= offset < n.size


def test_mix_at_snr_rejects_silence() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="silent"):
        tr.mix_at_snr(np.zeros(100), noise(), 10.0, rng)
    with pytest.raises(ValueError, match="silent"):
        tr.mix_at_snr(speech(), np.zeros(100), 10.0, rng)
    with pytest.raises(ValueError, match="empty"):
        tr.mix_at_snr(np.zeros(0), noise(), 10.0, rng)


def test_time_shift_keeps_length_and_fills_with_zeros() -> None:
    x = np.arange(1.0, 6.0)
    assert tr.time_shift(x, 2).tolist() == [0, 0, 1, 2, 3]
    assert tr.time_shift(x, -2).tolist() == [3, 4, 5, 0, 0]
    assert tr.time_shift(x, 0).tolist() == x.tolist()
    assert tr.time_shift(x, 9).tolist() == [0, 0, 0, 0, 0]
    assert tr.time_shift(x, -9).tolist() == [0, 0, 0, 0, 0]


def test_gain_db() -> None:
    assert tr.apply_gain_db(np.array([100.0]), 0.0)[0] == pytest.approx(100.0)
    assert tr.apply_gain_db(np.array([100.0]), 20.0)[0] == pytest.approx(1000.0)
    assert tr.apply_gain_db(np.array([100.0]), -6.0206)[0] == pytest.approx(50.0, rel=1e-4)


def test_rir_convolution_keeps_the_direct_path_and_length() -> None:
    x = np.zeros(100)
    x[10] = 1000.0
    rir = np.zeros(50)
    rir[5] = 0.5  # the strongest tap: the 5 samples before it are cut
    rir[8] = 0.25
    y = tr.convolve_rir(x, rir)
    assert y.size == x.size
    assert y[10] == pytest.approx(1000.0)  # direct path at gain 1, not delayed
    assert y[13] == pytest.approx(500.0)  # 0.25 / 0.5 * 1000, three samples later
    assert np.count_nonzero(np.abs(y) > 1e-6) == 2
    with pytest.raises(ValueError, match="empty"):
        tr.convolve_rir(x, np.zeros(0))
    with pytest.raises(ValueError, match="silent"):
        tr.convolve_rir(x, np.zeros(4))


def test_warmup_helper_is_exported() -> None:
    assert drop_warmup(np.zeros(5000)).size == 840
