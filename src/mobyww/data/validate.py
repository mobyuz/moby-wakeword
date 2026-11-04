"""Manifest validator.

It fails on missing license, missing consent, duplicate recordings (content hash), speaker
leakage across splits and bad paths. Exit code 0 means no errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from mobyww.data.manifest import (
    AUDIO_SUFFIXES,
    DISTANCES_CM,
    LABELS,
    LANGUAGES,
    MODES,
    NO_SPEAKER,
    SOURCES,
    SPLITS,
    UNASSIGNED,
    VOLUMES,
    ManifestError,
    Row,
    read_manifest,
)

# Placeholder license values that mean "unknown": treated as missing.
UNKNOWN_LICENSES = frozenset({"", "unknown", "unclear", "tbd", "todo", "none", "n/a", "?"})


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_path(row: Row, root: Path, report: Report) -> Path | None:
    raw = row.path
    if not raw or "\\" in raw or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        report.errors.append(f"bad path: {row.id}: {raw!r} must be relative with forward slashes")
        return None
    if any(part in ("", ".", "..") for part in raw.split("/")):
        report.errors.append(f"bad path: {row.id}: {raw!r} must not contain '.', '..' or '//'")
        return None
    pure = PurePosixPath(raw)
    if pure.suffix.lower() not in AUDIO_SUFFIXES:
        report.errors.append(f"bad path: {row.id}: {raw!r} is not a {AUDIO_SUFFIXES} file")
        return None
    full = (root / raw).resolve()
    if not full.is_relative_to(root.resolve()):
        report.errors.append(f"bad path: {row.id}: {raw!r} escapes the data root")
        return None
    if not full.is_file():
        report.errors.append(f"bad path: {row.id}: file {raw!r} does not exist")
        return None
    return full


def _consent_problem(consent_dir: Path, row: Row) -> str | None:
    """Return an error text when the local consent record does not allow the row."""
    record = consent_dir / f"{row.speaker}.json"
    if not record.is_file():
        return f"no local consent record for speaker {row.speaker!r}"
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"unreadable consent record for speaker {row.speaker!r}"
    if data.get("consent_id") != row.consent_id:
        return f"consent_id does not match the record of speaker {row.speaker!r}"
    if not data.get("granted") or data.get("withdrawn"):
        return f"consent of speaker {row.speaker!r} is not granted or was withdrawn"
    return None


def _check_fields(row: Row, report: Report) -> None:
    enums = (
        ("label", row.label, LABELS),
        ("language", row.language, LANGUAGES),
        ("source", row.source, SOURCES),
        ("mode", row.mode, MODES),
    )
    for name, value, allowed in enums:
        if value not in allowed:
            report.errors.append(f"{row.id}: {name} {value!r} not in {allowed}")
    if row.volume and row.volume not in VOLUMES:
        report.errors.append(f"{row.id}: volume {row.volume!r} not in {VOLUMES}")
    if row.distance_cm is not None and row.distance_cm not in DISTANCES_CM:
        report.warnings.append(f"{row.id}: unusual distance_cm {row.distance_cm}")
    if row.duration_s <= 0 or row.sample_rate <= 0:
        report.errors.append(f"{row.id}: duration_s and sample_rate must be positive")
    if not row.source_name:
        report.errors.append(f"{row.id}: source_name is empty")
    if not row.speaker:
        report.errors.append(f"{row.id}: speaker is empty (use {NO_SPEAKER!r} for non-speech)")
    if row.label == "wake" and not row.phrase:
        report.errors.append(f"{row.id}: a wake clip needs its phrase")


def validate(
    rows: list[Row],
    root: Path,
    *,
    consent_dir: Path | None = None,
    allow_unassigned: bool = False,
    check_files: bool = True,
) -> Report:
    report = Report()
    if not rows:
        report.errors.append("manifest has no rows")
        return report

    seen_ids: dict[str, int] = defaultdict(int)
    seen_paths: dict[str, str] = {}
    by_hash: dict[str, list[str]] = defaultdict(list)
    speaker_splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        seen_ids[row.id] += 1
        if not row.id:
            report.errors.append("empty id")
        _check_fields(row, report)

        if row.license.strip().lower() in UNKNOWN_LICENSES:
            report.errors.append(f"missing license: {row.id}")

        if row.source == "recorded":
            if not row.consent_id:
                report.errors.append(f"missing consent: {row.id} has no consent_id")
            elif consent_dir is not None:
                problem = _consent_problem(consent_dir, row)
                if problem:
                    report.errors.append(f"missing consent: {row.id}: {problem}")

        if row.split == UNASSIGNED:
            if not allow_unassigned:
                report.errors.append(f"unassigned split: {row.id}")
        elif row.split not in SPLITS:
            report.errors.append(f"{row.id}: split {row.split!r} not in {SPLITS}")
        elif row.speaker != NO_SPEAKER:
            speaker_splits[row.speaker].add(row.split)

        if row.path in seen_paths:
            report.errors.append(f"duplicate path: {row.path} ({seen_paths[row.path]}, {row.id})")
        seen_paths[row.path] = row.id

        if check_files:
            full = _check_path(row, root, report)
            if full is not None:
                actual = sha256_file(full)
                if row.sha256 and actual != row.sha256:
                    report.errors.append(f"hash mismatch: {row.id}: file differs from manifest")
                by_hash[actual].append(row.id)
        elif row.sha256:
            by_hash[row.sha256].append(row.id)

    for clip_id, count in seen_ids.items():
        if count > 1:
            report.errors.append(f"duplicate id: {clip_id} appears {count} times")
    for digest, ids in by_hash.items():
        if len(ids) > 1:
            report.errors.append(
                f"duplicate recording: {', '.join(ids)} share content hash {digest[:12]}"
            )
    for speaker, splits in sorted(speaker_splits.items()):
        if len(splits) > 1:
            report.errors.append(
                f"speaker leakage: {speaker!r} appears in {', '.join(sorted(splits))}"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mobyww-validate", description="Validate a manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, help="data root (default: manifest directory)")
    parser.add_argument("--consent-dir", type=Path, help="local consent records directory")
    parser.add_argument("--allow-unassigned", action="store_true")
    args = parser.parse_args(argv)
    root: Path = args.root or args.manifest.parent
    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = validate(
        rows, root, consent_dir=args.consent_dir, allow_unassigned=args.allow_unassigned
    )
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}")
    print(f"{len(rows)} rows, {len(report.errors)} errors, {len(report.warnings)} warnings")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
