"""Manifest rows: one JSON object per line (JSONL), one row per audio clip."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

LABELS = ("wake", "negative", "hard_negative", "ambient", "silence")
SOURCES = ("recorded", "tts", "public_dataset", "other")
SPLITS = ("train", "val", "test")
UNASSIGNED = "unassigned"
LANGUAGES = ("uz", "ru", "en", "other", "none")
VOLUMES = ("quiet", "normal", "loud")
MODES = ("prompted", "free", "none")
DISTANCES_CM = (30, 100, 300)
NO_SPEAKER = "n/a"
AUDIO_SUFFIXES = (".wav",)


@dataclass(frozen=True)
class Row:
    """One clip. Field names are the manifest keys and match data/manifest.schema.json."""

    id: str
    path: str  # relative to the data root, forward slashes, no ".."
    sha256: str  # hex SHA-256 of the file bytes
    label: str
    phrase: str  # what was said, lower case; "" for non-speech
    language: str
    source: str
    source_name: str  # dataset, TTS engine or "recording"
    license: str  # SPDX id or a named license; required
    speaker: str  # pseudonym, or NO_SPEAKER for non-speech
    split: str
    duration_s: float
    sample_rate: int
    consent_id: str = ""  # required when source == "recorded"
    distance_cm: int | None = None
    volume: str = ""
    room: str = ""
    device: str = ""
    recorded_on: str = ""  # ISO date
    mode: str = "none"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


FIELD_NAMES = tuple(f.name for f in fields(Row))
REQUIRED_FIELDS = tuple(f.name for f in fields(Row) if f.default is MISSING)


class ManifestError(ValueError):
    """A line of the manifest cannot be read as a row."""


def row_from_dict(data: dict[str, Any]) -> Row:
    unknown = sorted(set(data) - set(FIELD_NAMES))
    if unknown:
        raise ManifestError(f"unknown fields: {', '.join(unknown)}")
    try:
        return Row(**data)
    except TypeError as exc:
        raise ManifestError(str(exc)) from exc


def read_manifest(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ManifestError(f"{path}:{number}: a row must be a JSON object")
            try:
                rows.append(row_from_dict(obj))
            except ManifestError as exc:
                raise ManifestError(f"{path}:{number}: {exc}") from exc
    return rows


def write_manifest(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.to_json() + "\n")


def append_row(path: Path, row: Row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(row.to_json() + "\n")


def iter_dicts(rows: Iterable[Row]) -> Iterator[dict[str, Any]]:
    for row in rows:
        yield asdict(row)
