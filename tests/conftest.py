from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from mobyww.data.manifest import Row, write_manifest
from mobyww.data.sample import write_tone
from mobyww.data.validate import sha256_file


def make_row(root: Path, clip_id: str, hz: float, **overrides: object) -> Row:
    """A valid synthetic row whose file really exists under `root`."""
    rel = f"clips/{clip_id}.wav"
    write_tone(root / rel, hz)
    row = Row(
        id=clip_id,
        path=rel,
        sha256=sha256_file(root / rel),
        label="wake",
        phrase="hey moby",
        language="en",
        source="tts",
        source_name="test",
        license="CC0-1.0",
        speaker=f"spk-{clip_id}",
        split="train",
        duration_s=0.5,
        sample_rate=16000,
    )
    return replace(row, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def good_rows(tmp_path: Path) -> list[Row]:
    rows = [
        make_row(tmp_path, "a", 220.0, split="train"),
        make_row(tmp_path, "b", 330.0, split="val"),
        make_row(tmp_path, "c", 440.0, split="test"),
    ]
    write_manifest(tmp_path / "manifest.jsonl", rows)
    return rows


def pcm_tone(amplitude: int = 8000, samples: int = 800) -> bytes:
    return b"".join(struct.pack("<h", amplitude if i % 2 else -amplitude) for i in range(samples))
