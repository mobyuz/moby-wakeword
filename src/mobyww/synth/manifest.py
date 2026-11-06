"""Turn a folder of generated WAVs into manifest rows.

The rows follow data/manifest.schema.json and pass `mobyww.data.validate`. The licence of a row is
the one in the job file's voice entry (the licence of the voice's training data), the source is
`tts` (the schema's value for synthetic speech), and the speaker is a stable pseudonym for the
(voice, speaker id) pair, so `assign_splits` keeps one synthetic speaker in one split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from mobyww.audio import wav_info
from mobyww.data.manifest import Row, write_manifest
from mobyww.data.split import assign_splits
from mobyww.data.validate import sha256_file
from mobyww.synth.spec import (
    OUTPUT_SAMPLE_RATE_HZ,
    JobFile,
    JobFileError,
    expand_job,
    speaker_pseudonym,
)


@dataclass
class ManifestResult:
    rows: list[Row] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # expected clips with no WAV
    unexpected: list[str] = field(default_factory=list)  # WAVs the job file does not list
    bad_format: list[str] = field(default_factory=list)  # not 16 kHz 16-bit mono

    @property
    def complete(self) -> bool:
        return not (self.missing or self.unexpected or self.bad_format)


def build_manifest(
    job_file: JobFile,
    wav_dir: Path,
    data_root: Path,
    *,
    split_seed: int,
    val_pct: int = 10,
    test_pct: int = 0,
) -> ManifestResult:
    """Rows for every expected clip that exists. `test_pct` is 0 by default: the owner's real
    recordings are the test set, a synthetic test split would only give optimistic numbers."""
    wav_dir = wav_dir.resolve()
    data_root = data_root.resolve()
    if not wav_dir.is_relative_to(data_root):
        raise JobFileError(f"{wav_dir} is not inside the data root {data_root}")
    prefix = PurePosixPath(wav_dir.relative_to(data_root).as_posix())

    result = ManifestResult()
    expected: set[str] = set()
    engine = job_file.engine
    for job in job_file.jobs:
        voice = job_file.voice(job.voice_id)
        for clip in expand_job(job, job_file.seed):
            name = f"{clip.clip_id}.wav"
            expected.add(name)
            path = wav_dir / name
            if not path.is_file():
                result.missing.append(name)
                continue
            try:
                info = wav_info(path)
            except (ValueError, EOFError) as exc:
                result.bad_format.append(f"{name}: {exc}")
                continue
            if info.sample_rate != OUTPUT_SAMPLE_RATE_HZ or info.frames == 0:
                result.bad_format.append(f"{name}: {info.sample_rate} Hz, {info.frames} frames")
                continue
            result.rows.append(
                Row(
                    id=clip.clip_id,
                    path=str(prefix / name),
                    sha256=sha256_file(path),
                    label=job.label,
                    phrase=job.text.lower(),
                    language=job.language,
                    source="tts",
                    source_name=f"{engine.name} {engine.version} / {voice.id}",
                    license=voice.license,
                    speaker=speaker_pseudonym(voice, job.speaker_id),
                    split="unassigned",
                    duration_s=round(info.duration_s, 6),
                    sample_rate=info.sample_rate,
                )
            )
    result.unexpected = sorted(p.name for p in wav_dir.glob("*.wav") if p.name not in expected)
    result.rows = assign_splits(result.rows, split_seed, val_pct, test_pct)
    return result


def write_synthetic_manifest(path: Path, result: ManifestResult) -> None:
    write_manifest(path, result.rows)
