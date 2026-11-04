from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import make_row
from mobyww.data.manifest import Row, read_manifest, write_manifest
from mobyww.data.validate import main, validate


def errors(rows: list[Row], root: Path, **kwargs: object) -> list[str]:
    return validate(rows, root, **kwargs).errors  # type: ignore[arg-type]


def test_good_manifest_passes(good_rows: list[Row], tmp_path: Path) -> None:
    report = validate(good_rows, tmp_path)
    assert report.ok, report.errors


def test_cli_passes_and_fails(good_rows: list[Row], tmp_path: Path) -> None:
    assert main([str(tmp_path / "manifest.jsonl")]) == 0
    bad = [replace(good_rows[0], license="")]
    write_manifest(tmp_path / "bad.jsonl", bad)
    assert main([str(tmp_path / "bad.jsonl"), "--root", str(tmp_path)]) == 1


def test_cli_rejects_unreadable_manifest(tmp_path: Path) -> None:
    (tmp_path / "m.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    assert main([str(tmp_path / "m.jsonl")]) == 2


def test_missing_license_fails(good_rows: list[Row], tmp_path: Path) -> None:
    for value in ("", "unknown", "TBD", "  "):
        rows = [replace(good_rows[0], license=value), *good_rows[1:]]
        assert any(e.startswith("missing license") for e in errors(rows, tmp_path)), value


def test_missing_consent_fails_for_recorded(good_rows: list[Row], tmp_path: Path) -> None:
    rows = [replace(good_rows[0], source="recorded", consent_id=""), *good_rows[1:]]
    assert any(e.startswith("missing consent") for e in errors(rows, tmp_path))


def test_consent_record_must_exist_and_match(good_rows: list[Row], tmp_path: Path) -> None:
    consent_dir = tmp_path / "consent"
    consent_dir.mkdir()
    row = replace(good_rows[0], source="recorded", consent_id="c-1")
    rows = [row, *good_rows[1:]]
    assert any(
        "no local consent record" in e for e in errors(rows, tmp_path, consent_dir=consent_dir)
    )

    record = consent_dir / f"{row.speaker}.json"
    record.write_text(json.dumps({"consent_id": "c-1", "granted": True}), encoding="utf-8")
    assert validate(rows, tmp_path, consent_dir=consent_dir).ok

    record.write_text(json.dumps({"consent_id": "other", "granted": True}), encoding="utf-8")
    assert any("does not match" in e for e in errors(rows, tmp_path, consent_dir=consent_dir))

    record.write_text(
        json.dumps({"consent_id": "c-1", "granted": True, "withdrawn": True}), encoding="utf-8"
    )
    assert any("withdrawn" in e for e in errors(rows, tmp_path, consent_dir=consent_dir))


def test_duplicate_recording_by_content_hash_fails(tmp_path: Path) -> None:
    first = make_row(tmp_path, "a", 220.0, speaker="spk-aaa", split="train")
    shutil.copy(tmp_path / first.path, tmp_path / "clips" / "copy.wav")
    copy = replace(first, id="copy", path="clips/copy.wav", speaker="spk-bbb", split="val")
    found = errors([first, copy], tmp_path)
    assert any(e.startswith("duplicate recording") for e in found)


def test_duplicate_id_and_path_fail(good_rows: list[Row], tmp_path: Path) -> None:
    rows = [*good_rows, replace(good_rows[0], speaker="spk-zzz")]
    found = errors(rows, tmp_path)
    assert any(e.startswith("duplicate id") for e in found)
    assert any(e.startswith("duplicate path") for e in found)


def test_speaker_leakage_fails(tmp_path: Path) -> None:
    rows = [
        make_row(tmp_path, "a", 220.0, speaker="spk-same", split="train"),
        make_row(tmp_path, "b", 330.0, speaker="spk-same", split="test"),
    ]
    found = errors(rows, tmp_path)
    assert any(e.startswith("speaker leakage") for e in found)


def test_non_speech_rows_do_not_count_as_leakage(tmp_path: Path) -> None:
    rows = [
        make_row(tmp_path, "a", 220.0, speaker="n/a", label="ambient", phrase="", split="train"),
        make_row(tmp_path, "b", 330.0, speaker="n/a", label="ambient", phrase="", split="test"),
    ]
    assert validate(rows, tmp_path).ok


def test_bad_paths_fail(good_rows: list[Row], tmp_path: Path) -> None:
    bad_paths = [
        "../outside.wav",
        "/abs/file.wav",
        "C:/abs/file.wav",
        "clips\\a.wav",
        "clips/missing.wav",
        "clips/a.mp3",
        "",
        "./clips/a.wav",
    ]
    for bad in bad_paths:
        rows = [replace(good_rows[0], path=bad), *good_rows[1:]]
        assert any(e.startswith("bad path") for e in errors(rows, tmp_path)), bad


def test_hash_mismatch_fails(good_rows: list[Row], tmp_path: Path) -> None:
    rows = [replace(good_rows[0], sha256="0" * 64), *good_rows[1:]]
    assert any(e.startswith("hash mismatch") for e in errors(rows, tmp_path))


def test_unassigned_split_fails_unless_allowed(good_rows: list[Row], tmp_path: Path) -> None:
    rows = [replace(good_rows[0], split="unassigned"), *good_rows[1:]]
    assert any(e.startswith("unassigned split") for e in errors(rows, tmp_path))
    assert validate(rows, tmp_path, allow_unassigned=True).ok


def test_enum_and_field_errors(good_rows: list[Row], tmp_path: Path) -> None:
    broken = replace(
        good_rows[0],
        label="x",
        language="xx",
        source="y",
        volume="z",
        mode="m",
        split="q",
        duration_s=0.0,
        source_name="",
        speaker="",
    )
    found = errors([broken, *good_rows[1:]], tmp_path)
    for needle in ("label", "language", "source ", "volume", "mode", "split", "duration_s"):
        assert any(needle in e for e in found), needle


def test_wake_clip_needs_phrase_and_empty_manifest_fails(
    good_rows: list[Row], tmp_path: Path
) -> None:
    rows = [replace(good_rows[0], phrase=""), *good_rows[1:]]
    assert any("needs its phrase" in e for e in errors(rows, tmp_path))
    assert errors([], tmp_path) == ["manifest has no rows"]


def test_unusual_distance_is_a_warning(good_rows: list[Row], tmp_path: Path) -> None:
    rows = [replace(good_rows[0], distance_cm=77), *good_rows[1:]]
    report = validate(rows, tmp_path)
    assert report.ok
    assert report.warnings


def test_manifest_round_trip_and_unknown_field(good_rows: list[Row], tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "manifest.jsonl") == good_rows
    (tmp_path / "x.jsonl").write_text(json.dumps({**good_rows[0].__dict__, "extra": 1}) + "\n")
    with pytest.raises(ValueError, match="unknown fields"):
        read_manifest(tmp_path / "x.jsonl")
