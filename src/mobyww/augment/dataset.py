"""Build the augmented training clips and the ambient/negative clips from WAV folders.

Used by the Colab notebook (scripts/colab/), unit-tested here on tiny synthetic audio. Every
output clip goes through the device path (`mobyww.augment.device`), so the model trains on what the
robot would send. Nothing here reads data/recordings or data/consent: callers pass explicit file
lists, and `mobyww.augment.dataset` refuses paths that contain those directory names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from mobyww.audio import read_wav_int16, wav_info, write_wav_int16
from mobyww.augment import device
from mobyww.augment.pipeline import Augmenter
from mobyww.data.manifest import NO_SPEAKER, UNASSIGNED, Row
from mobyww.data.split import assign_splits
from mobyww.data.validate import sha256_file

# Directory names that hold the owner's real recordings and consent records. They never enter
# training data (docs/privacy.md); the real recordings are the local test set.
PRIVATE_DIR_NAMES = frozenset({"recordings", "consent"})


class PrivatePathError(ValueError):
    """A path points into the private recordings or consent folders."""


def refuse_private(path: Path) -> Path:
    """Return `path` resolved, or raise PrivatePathError when any part names a private folder."""
    resolved = path.resolve()
    parts = {part.lower() for part in resolved.parts} | {part.lower() for part in path.parts}
    hit = parts & PRIVATE_DIR_NAMES
    if hit:
        raise PrivatePathError(f"{path}: refusing to read from a private folder ({sorted(hit)})")
    return resolved


def _pick(files: Sequence[Path], seed: int, salt: str) -> Path:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return files[int.from_bytes(digest[:8], "big") % len(files)]


def augment_rows(
    rows: Sequence[Row],
    data_root: Path,
    out_dir: Path,
    augmenter: Augmenter,
    *,
    copies: int,
    noise_files: Sequence[Path] = (),
    noise_license: str = "",
    noise_set: str = "",
    rir_files: Sequence[Path] = (),
    rir_set: str = "",
) -> tuple[list[Row], list[dict[str, Any]]]:
    """`copies` augmented copies of every row. Returns (new rows, per-call records).

    Noise and RIR files are chosen per clip from the clip's seed, so the choice is reproducible.
    A RIR is used only when `rir_set` is enabled in the augmentation config (otherwise the
    augmenter refuses). The new rows keep label, speaker and split; their licence names the noise
    licence too, because the audio now contains that material.
    """
    for folder in (data_root, out_dir):
        refuse_private(folder)
    data_root, out_dir = data_root.resolve(), out_dir.resolve()
    prefix = out_dir.relative_to(data_root).as_posix()
    new_rows: list[Row] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        source_path = refuse_private(data_root / row.path)
        speech, rate = read_wav_int16(source_path)
        if rate != device.SAMPLE_RATE_HZ:
            raise ValueError(f"{row.id}: expected {device.SAMPLE_RATE_HZ} Hz, got {rate}")
        for k in range(copies):
            clip_id = f"{row.id}.a{k}"
            seed = augmenter.config.seed
            kwargs: dict[str, Any] = {}
            if noise_files:
                noise_path = refuse_private(_pick(noise_files, seed, f"{clip_id}:noise"))
                kwargs.update(noise=read_wav_int16(noise_path)[0], noise_id=noise_path.name)
            if rir_files:
                rir_path = refuse_private(_pick(rir_files, seed, f"{clip_id}:rir"))
                kwargs.update(
                    rir=read_wav_int16(rir_path)[0], rir_set=rir_set, rir_id=rir_path.name
                )
            result = augmenter.augment(clip_id, speech, **kwargs)
            target = out_dir / f"{clip_id}.wav"
            write_wav_int16(target, result.audio, rate)
            licence = "+".join(x for x in (row.license, noise_license if noise_files else "") if x)
            new_rows.append(
                replace(
                    row,
                    id=clip_id,
                    path=f"{prefix}/{clip_id}.wav",
                    sha256=sha256_file(target),
                    license=licence,
                    source_name=f"{row.source_name} + mobyww.augment"
                    + (f" + {noise_set}" if noise_files and noise_set else ""),
                )
            )
            records.append({**result.record, "source_id": row.id, "output": new_rows[-1].path})
    return new_rows, records


def ambient_rows(
    files: Sequence[Path],
    data_root: Path,
    out_dir: Path,
    *,
    label: str,
    source_name: str,
    license: str,
    split_seed: int,
    chunk_s: float = 4.0,
    max_chunks_per_file: int = 20,
    gain_q8: int = device.GAIN_Q8_UNITY,
    lowpass_hz: float | None = None,
    val_pct: int = 10,
    test_pct: int = 0,
) -> list[Row]:
    """Cut long recordings into chunks, run the device path, write them and return manifest rows.

    Chunks of one source file share a pseudonym (`src-<hash of the file name>`), so the split is
    per source file and a recording never lands in two splits. A trailing piece shorter than one
    second is dropped.
    """
    if label not in {"negative", "ambient", "silence"}:
        raise ValueError("label must be negative, ambient or silence")
    for folder in (data_root, out_dir):
        refuse_private(folder)
    data_root, out_dir = data_root.resolve(), out_dir.resolve()
    prefix = out_dir.relative_to(data_root).as_posix()
    chunk = int(chunk_s * device.SAMPLE_RATE_HZ)
    rows: list[Row] = []
    for source in files:
        refuse_private(source)
        samples, rate = read_wav_int16(source)
        if rate != device.SAMPLE_RATE_HZ:
            raise ValueError(f"{source.name}: expected {device.SAMPLE_RATE_HZ} Hz, got {rate}")
        stem = hashlib.sha256(source.name.encode()).hexdigest()[:10]
        pieces = range(0, max(samples.size - rate + 1, 0), chunk)
        for index, start in enumerate(pieces):
            if index >= max_chunks_per_file:
                break
            piece = samples[start : start + chunk]
            audio = device.inmp441_path(piece, gain_q8=gain_q8, lowpass_hz=lowpass_hz)
            clip_id = f"{label}.{stem}.{index:03d}"
            target = out_dir / f"{clip_id}.wav"
            write_wav_int16(target, audio, rate)
            rows.append(
                Row(
                    id=clip_id,
                    path=f"{prefix}/{clip_id}.wav",
                    sha256=sha256_file(target),
                    label=label,
                    phrase="",
                    language="none",
                    source="public_dataset",
                    source_name=source_name,
                    license=license,
                    speaker=f"src-{stem}" if label == "negative" else NO_SPEAKER,
                    split=UNASSIGNED,
                    duration_s=round(wav_info(target).duration_s, 6),
                    sample_rate=rate,
                )
            )
    # Non-speech rows have no speaker, so assign_splits would hash each clip alone. Group the
    # chunks of one source file instead, by giving them the source pseudonym while splitting.
    keyed = [replace(r, speaker=f"src-{r.id.split('.')[1]}") for r in rows]
    assigned = assign_splits(keyed, split_seed, val_pct, test_pct)
    return [replace(orig, split=new.split) for orig, new in zip(rows, assigned, strict=True)]


__all__ = [
    "PRIVATE_DIR_NAMES",
    "PrivatePathError",
    "ambient_rows",
    "augment_rows",
    "refuse_private",
]
