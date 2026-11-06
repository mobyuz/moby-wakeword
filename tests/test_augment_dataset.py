from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from mobyww.audio import read_wav_int16, resample, write_wav_int16
from mobyww.augment import Augmenter, parse_config
from mobyww.augment.dataset import PrivatePathError, ambient_rows, augment_rows, refuse_private
from mobyww.data.manifest import Row
from mobyww.data.validate import sha256_file, validate

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000


def augmenter(rir_sets: list[str] | None = None) -> Augmenter:
    data = yaml.safe_load((ROOT / "configs" / "augmentation.yaml").read_text(encoding="utf-8"))
    data["rir"] = {"sets": rir_sets or []}
    return Augmenter(parse_config(data))


def tone(hz: float, seconds: float = 1.0, amplitude: float = 6000.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def make_source_rows(root: Path, count: int = 3) -> list[Row]:
    rows = []
    for i in range(count):
        rel = f"synthetic/clips/c{i}.wav"
        write_wav_int16(root / rel, tone(200 + 50 * i), RATE)
        rows.append(
            Row(
                id=f"c{i}",
                path=rel,
                sha256=sha256_file(root / rel),
                label="wake",
                phrase="hey moby",
                language="en",
                source="tts",
                source_name="piper-tts 1.3.0 / test",
                license="CC-BY-4.0",
                speaker=f"synth-t-s{i:03d}",
                split="train",
                duration_s=1.0,
                sample_rate=RATE,
            )
        )
    return rows


def noise_files(root: Path) -> list[Path]:
    paths = []
    for i in range(2):
        path = root / "open" / f"noise{i}.wav"
        write_wav_int16(
            path, np.random.default_rng(i).normal(0, 1500, 3 * RATE).astype(np.int16), RATE
        )
        paths.append(path)
    return paths


def test_resample_keeps_frequency_amplitude_and_length() -> None:
    rate_in, rate_out = 22050, 16000
    t = np.arange(rate_in) / rate_in
    x = 8000 * np.sin(2 * np.pi * 1000 * t)
    y = resample(x, rate_in, rate_out)
    assert y.size == rate_out
    spectrum = np.abs(np.fft.rfft(y))
    assert np.argmax(spectrum) == 1000  # 1 Hz bins at one second
    assert np.max(np.abs(y[2000:-2000])) == pytest.approx(8000, rel=0.02)
    assert resample(x, rate_in, rate_in).tolist() == x.tolist()
    assert resample(np.zeros(0), rate_in, rate_out).size == 0
    with pytest.raises(ValueError, match="positive"):
        resample(x, 0, 16000)


def test_refuse_private_names_any_case(tmp_path: Path) -> None:
    for name in ("recordings", "Consent", "CONSENT"):
        with pytest.raises(PrivatePathError):
            refuse_private(tmp_path / "data" / name / "a.wav")
    assert refuse_private(tmp_path / "synthetic" / "a.wav").name == "a.wav"


def test_augment_rows_is_deterministic_and_valid(tmp_path: Path) -> None:
    results = []
    for run in ("one", "two"):
        root = tmp_path / run
        rows = make_source_rows(root)
        new, records = augment_rows(
            rows,
            root,
            root / "augmented",
            augmenter(),
            copies=2,
            noise_files=noise_files(root),
            noise_license="CC-BY-4.0",
            noise_set="MUSAN",
        )
        assert len(new) == len(records) == 6
        assert validate(new, root).ok, validate(new, root).errors
        results.append(([r.sha256 for r in new], [r.id for r in new], records))
    assert results[0][0] == results[1][0]  # same inputs, same bytes
    assert results[0][1] == ["c0.a0", "c0.a1", "c1.a0", "c1.a1", "c2.a0", "c2.a1"]
    assert results[0][2][0]["source_id"] == "c0"
    assert results[0][2][0]["clip_seed"] == results[1][2][0]["clip_seed"]


def test_augmented_rows_keep_split_and_speaker_and_name_the_noise_licence(
    tmp_path: Path,
) -> None:
    rows = make_source_rows(tmp_path)
    new, _ = augment_rows(
        rows,
        tmp_path,
        tmp_path / "augmented",
        augmenter(),
        copies=1,
        noise_files=noise_files(tmp_path),
        noise_license="CC-BY-4.0",
        noise_set="MUSAN",
    )
    assert [r.speaker for r in new] == [r.speaker for r in rows]
    assert {r.split for r in new} == {"train"}
    assert {r.license for r in new} == {"CC-BY-4.0+CC-BY-4.0"}
    assert all("MUSAN" in r.source_name for r in new)
    audio, rate = read_wav_int16(tmp_path / new[0].path)
    assert rate == RATE
    assert audio.size == RATE
    assert audio.dtype == np.int16


def test_augment_rows_without_noise_uses_the_device_path_only(tmp_path: Path) -> None:
    rows = make_source_rows(tmp_path, 1)
    new, records = augment_rows(rows, tmp_path, tmp_path / "aug", augmenter(), copies=1)
    assert new[0].license == "CC-BY-4.0"
    assert records[0]["snr_db"] is None
    assert records[0]["lowpass_hz"] == 7200


def test_rir_is_refused_unless_the_set_is_enabled(tmp_path: Path) -> None:
    rows = make_source_rows(tmp_path, 1)
    rir = tmp_path / "open" / "rir.wav"
    write_wav_int16(rir, np.array([0, 3000, 1000, 500], dtype=np.int16), RATE)
    with pytest.raises(ValueError, match="not enabled"):
        augment_rows(
            rows, tmp_path, tmp_path / "aug", augmenter(), copies=1, rir_files=[rir], rir_set="x"
        )
    new, records = augment_rows(
        rows,
        tmp_path,
        tmp_path / "aug",
        augmenter(["but_reverbdb"]),
        copies=1,
        rir_files=[rir],
        rir_set="but_reverbdb",
    )
    assert records[0]["rir"] == "but_reverbdb/rir.wav"
    assert len(new) == 1


def test_augment_rows_rejects_other_sample_rates(tmp_path: Path) -> None:
    rows = make_source_rows(tmp_path, 1)
    write_wav_int16(tmp_path / rows[0].path, tone(300), 22050)
    with pytest.raises(ValueError, match="16000"):
        augment_rows(rows, tmp_path, tmp_path / "aug", augmenter(), copies=1)


def test_augment_rows_refuses_private_inputs(tmp_path: Path) -> None:
    rows = make_source_rows(tmp_path, 1)
    private = tmp_path / "data" / "recordings"
    private.mkdir(parents=True)
    with pytest.raises(PrivatePathError):
        augment_rows(rows, tmp_path, private / "out", augmenter(), copies=1)
    stray = private / "x.wav"
    write_wav_int16(stray, tone(300), RATE)
    with pytest.raises(PrivatePathError):
        augment_rows(rows, tmp_path, tmp_path / "aug", augmenter(), copies=1, noise_files=[stray])


def test_ambient_rows_chunks_validates_and_splits_per_source(tmp_path: Path) -> None:
    sources = []
    for i in range(6):
        path = tmp_path / "open" / f"long{i}.wav"
        noise = np.random.default_rng(i).normal(0, 2000, int(9.5 * RATE)).astype(np.int16)
        write_wav_int16(path, noise, RATE)
        sources.append(path)
    rows = ambient_rows(
        sources,
        tmp_path,
        tmp_path / "ambient",
        label="ambient",
        source_name="MUSAN noise",
        license="CC-BY-4.0",
        split_seed=1337,
        chunk_s=4.0,
        val_pct=40,
    )
    assert len(rows) == 6 * 3  # 4 s + 4 s + 1.5 s per file
    assert {r.label for r in rows} == {"ambient"}
    assert {r.speaker for r in rows} == {"n/a"}
    assert validate(rows, tmp_path).ok, validate(rows, tmp_path).errors
    by_source: dict[str, set[str]] = {}
    for r in rows:
        by_source.setdefault(r.id.split(".")[1], set()).add(r.split)
    assert all(len(s) == 1 for s in by_source.values())  # a source file never straddles splits
    assert "test" not in {r.split for r in rows}

    short = tmp_path / "open" / "short.wav"
    write_wav_int16(short, tone(200, seconds=0.5), RATE)
    assert (
        ambient_rows(
            [short],
            tmp_path,
            tmp_path / "ambient2",
            label="negative",
            source_name="x",
            license="CC-BY-4.0",
            split_seed=1,
        )
        == []
    )
    with pytest.raises(ValueError, match="label"):
        ambient_rows(
            sources[:1],
            tmp_path,
            tmp_path / "a3",
            label="wake",
            source_name="x",
            license="y",
            split_seed=1,
        )


def test_ambient_rows_applies_the_device_path(tmp_path: Path) -> None:
    path = tmp_path / "open" / "dc.wav"
    write_wav_int16(path, np.full(2 * RATE, 7, dtype=np.int16), RATE)
    rows = ambient_rows(
        [path],
        tmp_path,
        tmp_path / "amb",
        label="ambient",
        source_name="x",
        license="CC0-1.0",
        split_seed=1,
        gain_q8=512,
    )
    audio, _ = read_wav_int16(tmp_path / rows[0].path)
    assert set(audio.tolist()) == {14}  # DC kept, gain 2.0 from the firmware conversion
