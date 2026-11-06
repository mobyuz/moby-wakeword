from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from conftest import make_row
from mobyww.data import sample
from mobyww.data.manifest import FIELD_NAMES, REQUIRED_FIELDS, Row, read_manifest
from mobyww.data.split import assign_splits
from mobyww.data.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def test_schema_matches_row_fields() -> None:
    schema = json.loads((ROOT / "data" / "manifest.schema.json").read_text(encoding="utf-8"))
    assert sorted(schema["properties"]) == sorted(FIELD_NAMES)
    assert sorted(schema["required"]) == sorted(REQUIRED_FIELDS)
    assert schema["additionalProperties"] is False


def test_schema_enums_match_code() -> None:
    from mobyww.data import manifest as m

    props = json.loads((ROOT / "data" / "manifest.schema.json").read_text(encoding="utf-8"))[
        "properties"
    ]
    assert tuple(props["label"]["enum"]) == m.LABELS
    assert tuple(props["source"]["enum"]) == m.SOURCES
    assert tuple(props["language"]["enum"]) == m.LANGUAGES
    assert tuple(props["split"]["enum"]) == (*m.SPLITS, m.UNASSIGNED)
    assert tuple(props["volume"]["enum"]) == ("", *m.VOLUMES)
    assert tuple(props["mode"]["enum"]) == m.MODES


def test_assign_splits_never_leaks(tmp_path: Path) -> None:
    rows: list[Row] = []
    for speaker in range(40):
        for clip in range(3):
            rows.append(
                make_row(
                    tmp_path,
                    f"s{speaker}c{clip}",
                    200.0 + speaker * 7 + clip,
                    speaker=f"spk-{speaker:03d}",
                    split="unassigned",
                )
            )
    out = assign_splits(rows, seed=1337)
    assert validate(out, tmp_path).ok
    assert {r.split for r in out} == {"train", "val", "test"}
    assert out == assign_splits(rows, seed=1337)  # deterministic


def test_assign_splits_keeps_existing_and_known_speaker(tmp_path: Path) -> None:
    fixed = make_row(tmp_path, "a", 220.0, speaker="spk-x", split="test")
    pending = replace(make_row(tmp_path, "b", 330.0, speaker="spk-x"), split="unassigned")
    out = assign_splits([fixed, pending], seed=1)
    assert [r.split for r in out] == ["test", "test"]


def test_sample_dataset_validates(tmp_path: Path) -> None:
    manifest = sample.write_sample(tmp_path)
    assert validate(read_manifest(manifest), tmp_path).ok
    assert sample.main() == 0


def test_configs_load_and_targets_are_confirmed() -> None:
    wake = yaml.safe_load((ROOT / "configs" / "wakeword.yaml").read_text(encoding="utf-8"))
    assert [c["phrase"] for c in wake["candidates"]] == ["hey moby"]
    assert "mobil" in wake["hard_negatives"]
    targets = yaml.safe_load((ROOT / "configs" / "targets.yaml").read_text(encoding="utf-8"))
    assert targets["status"] == "confirmed-by-owner"
    assert targets["false_wakes_per_hour_max"] == 1.0
    assert targets["recall_quiet_1m_min"] == 0.90
    assert targets["recall_3m_or_moderate_noise_min"] == 0.75
    aug = yaml.safe_load((ROOT / "configs" / "augmentation.yaml").read_text(encoding="utf-8"))
    assert aug["status"] == "implemented-w2"
