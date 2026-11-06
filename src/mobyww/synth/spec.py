"""The synthesis job file: what the Colab notebook must generate, and how to read it back.

Nothing here downloads a model or makes audio. The job file lists voices (with the licence,
revision and SHA-256 that docs/licenses.md recorded), jobs (voice x speaker x text) and the
per-clip parameter grid. The Colab notebook and the manifest writer both call `expand_job`, so
they agree on every clip name and every synthesis parameter without further coordination.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

SCHEMA = "mobyww-synth-job/1"
OUTPUT_SAMPLE_RATE_HZ = 16_000

Label = Literal["wake", "hard_negative"]
LABELS: tuple[Label, ...] = ("wake", "hard_negative")
LANGUAGES = ("en", "ru", "uz")

# Piper's own default grid in piper-sample-generator's CLI (length scale < 1 is faster speech).
DEFAULT_LENGTH_SCALES = (1.0, 0.75, 1.25, 1.4)
DEFAULT_NOISE_SCALES = (0.667, 0.75, 0.85, 0.9, 1.0, 1.4)
DEFAULT_NOISE_SCALE_WS = (0.8,)

UNKNOWN_LICENSES = frozenset({"", "unknown", "unclear", "tbd", "todo", "none", "n/a", "?"})


@dataclass(frozen=True)
class Voice:
    """One Piper voice. Values come from docs/licenses.md (verified 2026-11-06)."""

    id: str
    short: str  # used in clip ids and speaker pseudonyms
    hf_repo: str
    hf_revision: str
    onnx_path: str
    onnx_bytes: int
    onnx_sha256: str
    sample_rate_hz: int
    speaker_count: int
    text_language: str  # language of the texts this voice reads: en or ru
    license: str  # licence of the voice's training data (SPDX id), from its MODEL_CARD
    dataset: str
    attribution_required: bool


# The only voices docs/licenses.md marks "allowed". Adding one needs a licences.md row first.
APPROVED_VOICES: tuple[Voice, ...] = (
    Voice(
        id="en_US-libritts_r-medium",
        short="lr",
        hf_repo="rhasspy/piper-voices",
        hf_revision="e7229f7d33a10e145e109b886d661984201ac3c4",
        onnx_path="en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
        onnx_bytes=78_580_914,
        onnx_sha256="10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195",
        sample_rate_hz=22_050,
        speaker_count=904,
        text_language="en",
        license="CC-BY-4.0",
        dataset="LibriTTS-R (OpenSLR 141)",
        attribution_required=True,
    ),
    Voice(
        id="ru_RU-denis-medium",
        short="rd",
        hf_repo="rhasspy/piper-voices",
        hf_revision="e7229f7d33a10e145e109b886d661984201ac3c4",
        onnx_path="ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        onnx_bytes=63_201_294,
        onnx_sha256="15fab56e11a097858ee115545d0f697fc2a316c41a291a5362349fb870411b0a",
        sample_rate_hz=22_050,
        speaker_count=1,
        text_language="ru",
        license="CC0-1.0",
        dataset="OHF-Voice/voice-datasets ru_RU denis",
        attribution_required=False,
    ),
    Voice(
        id="ru_RU-dmitri-medium",
        short="rm",
        hf_repo="rhasspy/piper-voices",
        hf_revision="e7229f7d33a10e145e109b886d661984201ac3c4",
        onnx_path="ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        onnx_bytes=63_201_294,
        onnx_sha256="f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb",
        sample_rate_hz=22_050,
        speaker_count=1,
        text_language="ru",
        license="CC0-1.0",
        dataset="OHF-Voice/voice-datasets ru_RU dmitri",
        attribution_required=False,
    ),
)
VOICES_BY_ID = {v.id: v for v in APPROVED_VOICES}


@dataclass(frozen=True)
class Engine:
    name: str = "piper-tts"
    version: str = "1.3.0"
    generator_repo: str = "https://github.com/rhasspy/piper-sample-generator"
    generator_commit: str = "2971426a55072f7d22fec416ca7800df8bd23207"


@dataclass(frozen=True)
class Job:
    """`count` clips of one text by one speaker of one voice."""

    job_id: str
    voice_id: str
    speaker_id: int
    label: Label
    language: str  # language recorded in the manifest: en, ru or uz (accent variants)
    variant_id: str
    text: str
    count: int
    length_scales: tuple[float, ...] = DEFAULT_LENGTH_SCALES
    noise_scales: tuple[float, ...] = DEFAULT_NOISE_SCALES
    noise_scale_ws: tuple[float, ...] = DEFAULT_NOISE_SCALE_WS


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    job_id: str
    index: int
    text: str
    speaker_id: int
    length_scale: float
    noise_scale: float
    noise_scale_w: float
    torch_seed: int  # seed for the generator's noise; the same clip always gets the same one


@dataclass(frozen=True)
class JobFile:
    seed: int
    voices: tuple[Voice, ...]
    jobs: tuple[Job, ...]
    engine: Engine = field(default_factory=Engine)
    notes: tuple[str, ...] = ()
    schema: str = SCHEMA

    def voice(self, voice_id: str) -> Voice:
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        raise KeyError(voice_id)

    @property
    def total_clips(self) -> int:
        return sum(job.count for job in self.jobs)


class JobFileError(ValueError):
    """The job file is malformed or breaks a licence rule."""


def _digest_int(text: str, bits: int = 63) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> (64 - bits)


def speaker_pseudonym(voice: Voice, speaker_id: int) -> str:
    """The manifest `speaker` for a synthetic speaker: stable, never a real name."""
    return f"synth-{voice.short}-s{speaker_id:03d}"


def clip_id_for(job_id: str, index: int) -> str:
    return f"{job_id}.{index:04d}"


def expand_job(job: Job, seed: int) -> list[ClipSpec]:
    """The clips of one job with their synthesis parameters, in a seed-determined order."""
    grid = list(itertools.product(job.length_scales, job.noise_scales, job.noise_scale_ws))
    if not grid:
        raise JobFileError(f"{job.job_id}: empty parameter grid")
    grid.sort(key=lambda p: _digest_int(f"{seed}:{job.job_id}:{p}"))
    clips: list[ClipSpec] = []
    for index in range(job.count):
        length, noise, noise_w = grid[index % len(grid)]
        clip_id = clip_id_for(job.job_id, index)
        clips.append(
            ClipSpec(
                clip_id=clip_id,
                job_id=job.job_id,
                index=index,
                text=job.text,
                speaker_id=job.speaker_id,
                length_scale=length,
                noise_scale=noise,
                noise_scale_w=noise_w,
                torch_seed=_digest_int(f"{seed}:{clip_id}", 31),
            )
        )
    return clips


def validate(job_file: JobFile) -> None:
    """Licence and consistency rules. Raises JobFileError."""
    if job_file.schema != SCHEMA:
        raise JobFileError(f"unknown schema {job_file.schema!r}")
    approved = {v.id: v for v in APPROVED_VOICES}
    seen_jobs: set[str] = set()
    for voice in job_file.voices:
        known = approved.get(voice.id)
        if known is None:
            raise JobFileError(f"voice {voice.id!r} is not approved in docs/licenses.md")
        if voice != known:
            raise JobFileError(f"voice {voice.id!r} differs from the approved entry")
        if voice.license.strip().lower() in UNKNOWN_LICENSES:
            raise JobFileError(f"voice {voice.id!r} has no licence")
    for job in job_file.jobs:
        voice = job_file.voice(job.voice_id)
        if job.job_id in seen_jobs:
            raise JobFileError(f"duplicate job id {job.job_id!r}")
        seen_jobs.add(job.job_id)
        if job.label not in LABELS:
            raise JobFileError(f"{job.job_id}: label {job.label!r} not in {LABELS}")
        if job.language not in LANGUAGES:
            raise JobFileError(f"{job.job_id}: language {job.language!r} not in {LANGUAGES}")
        if not 0 <= job.speaker_id < voice.speaker_count:
            raise JobFileError(f"{job.job_id}: speaker {job.speaker_id} outside the voice")
        if job.count < 1 or not job.text.strip():
            raise JobFileError(f"{job.job_id}: needs a text and a positive count")


def to_dict(job_file: JobFile) -> dict[str, Any]:
    data = json.loads(json.dumps(asdict(job_file)))
    data["total_clips"] = job_file.total_clips
    return cast(dict[str, Any], data)


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def from_dict(data: dict[str, Any]) -> JobFile:
    try:
        job_file = JobFile(
            schema=str(data["schema"]),
            seed=int(data["seed"]),
            engine=Engine(**data["engine"]),
            voices=tuple(Voice(**v) for v in data["voices"]),
            jobs=tuple(
                Job(
                    **{
                        **j,
                        "length_scales": _floats(j["length_scales"]),
                        "noise_scales": _floats(j["noise_scales"]),
                        "noise_scale_ws": _floats(j["noise_scale_ws"]),
                    }
                )
                for j in data["jobs"]
            ),
            notes=tuple(str(n) for n in data.get("notes", ())),
        )
    except (KeyError, TypeError) as exc:
        raise JobFileError(f"malformed job file: {exc!r}") from exc
    validate(job_file)
    return job_file


def save_job_file(path: Path, job_file: JobFile) -> None:
    """Write YAML (`.yaml`/`.yml`) or JSON (anything else). Deterministic bytes, LF newlines."""
    validate(job_file)
    data = to_dict(job_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    else:
        text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_job_file(path: Path) -> JobFile:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)  # JSON is valid YAML
    if not isinstance(data, dict):
        raise JobFileError(f"{path}: not a mapping")
    return from_dict(data)


def job_file_sha256(job_file: JobFile) -> str:
    """Hash of the canonical content, independent of YAML/JSON formatting."""
    canonical = json.dumps(to_dict(job_file), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
