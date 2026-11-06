from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from mobyww.audio import write_wav_int16
from mobyww.augment.dataset import PrivatePathError
from mobyww.data.manifest import Row
from mobyww.data.validate import sha256_file
from mobyww.train import (
    FeatureSet,
    build_provenance,
    build_training_config,
    manifest_hash,
    organize_by_split,
    write_provenance,
    write_training_config,
)

ROOT = Path(__file__).resolve().parents[1]


def sets() -> list[FeatureSet]:
    return [
        FeatureSet(
            "wake", "f/wake", truth=True, sampling_weight=2.0, truncation_strategy="truncate_start"
        ),
        FeatureSet("hard", "f/hard", truth=False, sampling_weight=3.0),
    ]


def row(root: Path, clip_id: str, split: str) -> Row:
    rel = f"clips/{clip_id}.wav"
    write_wav_int16(root / rel, np.full(1600, 5, dtype=np.int16), 16000)
    return Row(
        id=clip_id,
        path=rel,
        sha256=sha256_file(root / rel),
        label="wake",
        phrase="hey moby",
        language="en",
        source="tts",
        source_name="t",
        license="CC0-1.0",
        speaker=f"s-{clip_id}",
        split=split,
        duration_s=0.1,
        sample_rate=16000,
    )


def test_training_config_has_the_upstream_keys_and_roundtrips(tmp_path: Path) -> None:
    config = build_training_config(train_dir="trained", feature_sets=sets())
    assert config["window_step_ms"] == 10
    assert config["training_steps"] == [10_000]
    assert config["clip_duration_ms"] == 1500
    assert [f["truth"] for f in config["features"]] == [True, False]
    assert {f["type"] for f in config["features"]} == {"mmap"}
    write_training_config(tmp_path / "t.yaml", config)
    assert yaml.safe_load((tmp_path / "t.yaml").read_text(encoding="utf-8")) == config


def test_training_config_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="positives"):
        build_training_config(train_dir="t", feature_sets=[replace(sets()[1])])
    with pytest.raises(ValueError, match="negatives"):
        build_training_config(train_dir="t", feature_sets=[sets()[0]])


def test_organize_by_split_copies_into_split_folders(tmp_path: Path) -> None:
    rows = [row(tmp_path, "a", "train"), row(tmp_path, "b", "val"), row(tmp_path, "c", "test")]
    counts = organize_by_split(rows, tmp_path, tmp_path / "sets" / "wake")
    assert counts == {"train": 1, "validation": 1, "test": 1}
    assert (tmp_path / "sets" / "wake" / "validation" / "b.wav").is_file()
    with pytest.raises(ValueError, match="split"):
        organize_by_split([row(tmp_path, "d", "unassigned")], tmp_path, tmp_path / "x")


def test_organize_by_split_refuses_private_folders(tmp_path: Path) -> None:
    rows = [row(tmp_path, "a", "train")]
    with pytest.raises(PrivatePathError):
        organize_by_split(rows, tmp_path, tmp_path / "data" / "recordings" / "wake")
    with pytest.raises(PrivatePathError):
        organize_by_split(rows, tmp_path / "consent", tmp_path / "wake")


def test_manifest_hash_ignores_row_and_file_order(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    b.write_text('{"id": 3}\r\n', encoding="utf-8")
    assert manifest_hash([a, b]) == manifest_hash([b, a])
    b.write_text('{"id": 4}\n', encoding="utf-8")
    assert manifest_hash([a]) != manifest_hash([a, b])


def test_provenance_records_hashes_seeds_and_no_accuracy(tmp_path: Path) -> None:
    model = tmp_path / "stream_state_internal_quant.tflite"
    model.write_bytes(b"not a real model")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"id": "x"}\n', encoding="utf-8")
    config = build_training_config(train_dir="trained", feature_sets=sets())
    prov = build_provenance(
        model_path=model,
        repo=ROOT,
        seeds={"split": 1337, "augmentation": 2024, "training": 4242, "synthesis": 3030},
        training_config=config,
        manifests=[manifest],
        config_files={"augmentation.yaml": ROOT / "configs" / "augmentation.yaml"},
        upstream="4665173cd35f1cff9a61e06fc427f124766c488e",
        extra={"colab": {"untested": True}},
    )
    assert prov["model"]["bytes"] == len(b"not a real model")
    assert len(prov["model"]["sha256"]) == 64
    assert prov["upstream"]["commit"] == "4665173cd35f1cff9a61e06fc427f124766c488e"
    assert prov["seeds"]["training"] == 4242
    assert prov["hyperparameters"]["training_steps"] == [10_000]
    assert prov["data"]["manifest_hash"] == manifest_hash([manifest])
    assert len(prov["config_hashes"]["augmentation.yaml"]) == 64
    assert prov["accuracy_measured"] is False
    assert "repository" in prov
    assert prov["colab"] == {"untested": True}
    path = write_provenance(tmp_path / "out", prov)
    assert json.loads(path.read_text(encoding="utf-8")) == prov
