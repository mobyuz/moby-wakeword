"""A tiny synthetic sample dataset (sine tones, no real voice) and its validation.

Used by CI and `just validate-sample`. It downloads nothing and the tones are generated
here, so the sample carries no third-party license.
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

from mobyww.data.manifest import Row, read_manifest, write_manifest
from mobyww.data.validate import sha256_file, validate


def write_tone(path: Path, hz: float, seconds: float = 0.5, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / rate)))
        for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)


def write_sample(root: Path) -> Path:
    plan = [
        ("s001", "wake", "hey moby", "train", "synthetic-a", 220.0),
        ("s002", "wake", "hey moby", "val", "synthetic-b", 330.0),
        ("s003", "wake", "hey moby", "test", "synthetic-c", 440.0),
        ("s004", "ambient", "", "train", "n/a", 550.0),
    ]
    rows: list[Row] = []
    for clip_id, label, phrase, split, speaker, hz in plan:
        rel = f"clips/{clip_id}.wav"
        write_tone(root / rel, hz)
        rows.append(
            Row(
                id=clip_id,
                path=rel,
                sha256=sha256_file(root / rel),
                label=label,
                phrase=phrase,
                language="none",
                source="other",
                source_name="mobyww.data.sample sine tone",
                license="CC0-1.0",
                speaker=speaker,
                split=split,
                duration_s=0.5,
                sample_rate=16000,
            )
        )
    manifest = root / "manifest.jsonl"
    write_manifest(manifest, rows)
    return manifest


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mobyww-sample-") as tmp:
        root = Path(tmp)
        report = validate(read_manifest(write_sample(root)), root)
        for error in report.errors:
            print(f"error: {error}")
        print(f"sample manifest: {len(report.errors)} errors")
        return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
