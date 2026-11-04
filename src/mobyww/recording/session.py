"""Guided recording sessions. Capture and user prompting are injected, so tests need no
microphone and no terminal."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Protocol

from mobyww.data.manifest import Row, append_row, read_manifest
from mobyww.data.validate import sha256_file
from mobyww.recording.capture import AudioSource, peak_amplitude, write_wav
from mobyww.recording.consent import require_active_consent

SOURCE_NAME = "mobyww-record"
RECORDING_LICENSE = "LicenseRef-Moby-Consented-Recording"
MIN_PEAK = 64  # below this the clip is treated as silence / a dead microphone

Answer = Literal["record", "skip", "quit"]


@dataclass(frozen=True)
class Prompt:
    phrase: str
    label: Literal["wake", "hard_negative", "free"]
    distance_cm: int
    volume: str
    repeat: int


class Prompter(Protocol):
    def ask(self, prompt: Prompt, index: int, total: int) -> Answer: ...
    def info(self, message: str) -> None: ...


def build_prompts(
    phrases: list[str],
    hard_negatives: list[str],
    distances_cm: list[int],
    volumes: list[str],
    repeats: int,
) -> list[Prompt]:
    """Wake phrases at every distance and volume; hard negatives at every distance, normal
    volume, once per repeat. Order is by distance so the speaker moves as rarely as possible."""
    prompts: list[Prompt] = []
    for distance in distances_cm:
        for phrase in phrases:
            for volume in volumes:
                prompts.extend(Prompt(phrase, "wake", distance, volume, r) for r in range(repeats))
        for phrase in hard_negatives:
            prompts.append(Prompt(phrase, "hard_negative", distance, "normal", 0))
    return prompts


@dataclass
class SessionResult:
    recorded: int = 0
    skipped: int = 0
    rejected_silent: int = 0
    stopped_early: bool = False


def _next_index(manifest: Path, speaker: str) -> int:
    if not manifest.exists():
        return 1
    return sum(1 for r in read_manifest(manifest) if r.speaker == speaker) + 1


def _save(
    *,
    data_root: Path,
    manifest: Path,
    pcm: bytes,
    sample_rate: int,
    speaker: str,
    consent_id: str,
    label: str,
    phrase: str,
    language: str,
    distance_cm: int | None,
    volume: str,
    room: str,
    device: str,
    mode: str,
    today: date,
) -> Row:
    clip_id = f"{speaker}-{_next_index(manifest, speaker):05d}"
    rel = f"recordings/{speaker}/{clip_id}.wav"
    write_wav(data_root / rel, pcm, sample_rate)
    row = Row(
        id=clip_id,
        path=rel,
        sha256=sha256_file(data_root / rel),
        label=label,
        phrase=phrase,
        language=language,
        source="recorded",
        source_name=SOURCE_NAME,
        license=RECORDING_LICENSE,
        speaker=speaker,
        split="unassigned",
        duration_s=round(len(pcm) / 2 / sample_rate, 3),
        sample_rate=sample_rate,
        consent_id=consent_id,
        distance_cm=distance_cm,
        volume=volume,
        room=room,
        device=device,
        recorded_on=today.isoformat(),
        mode=mode,
    )
    append_row(manifest, row)
    return row


def run_prompted_session(
    *,
    source: AudioSource,
    prompter: Prompter,
    data_root: Path,
    speaker: str,
    language: str,
    room: str,
    device: str,
    prompts: list[Prompt],
    clip_seconds: float,
    sample_rate: int,
    today: Callable[[], date] = lambda: datetime.now(UTC).date(),
) -> SessionResult:
    """Record each prompt after the speaker confirms. Needs an active consent record."""
    consent = require_active_consent(data_root / "consent", speaker)
    manifest = data_root / "manifests" / "recorded.jsonl"
    result = SessionResult()
    for index, prompt in enumerate(prompts, start=1):
        answer = prompter.ask(prompt, index, len(prompts))
        if answer == "quit":
            result.stopped_early = True
            break
        if answer == "skip":
            result.skipped += 1
            continue
        pcm = source.record(clip_seconds, sample_rate)
        if peak_amplitude(pcm) < MIN_PEAK:
            prompter.info("The clip is silent; check the microphone. It was not saved.")
            result.rejected_silent += 1
            continue
        _save(
            data_root=data_root,
            manifest=manifest,
            pcm=pcm,
            sample_rate=sample_rate,
            speaker=speaker,
            consent_id=consent.consent_id,
            label=prompt.label,
            phrase=prompt.phrase,
            language=language,
            distance_cm=prompt.distance_cm,
            volume=prompt.volume,
            room=room,
            device=device,
            mode="prompted",
            today=today(),
        )
        result.recorded += 1
    return result


def run_free_session(
    *,
    source: AudioSource,
    prompter: Prompter,
    data_root: Path,
    speaker: str,
    language: str,
    room: str,
    device: str,
    segments: int,
    segment_seconds: float,
    sample_rate: int,
    distance_cm: int | None = None,
    today: Callable[[], date] = lambda: datetime.now(UTC).date(),
) -> SessionResult:
    """Free speech mode: ordinary talk that does NOT contain the wake phrase, saved as
    negatives (label `negative`, no phrase)."""
    consent = require_active_consent(data_root / "consent", speaker)
    manifest = data_root / "manifests" / "recorded.jsonl"
    result = SessionResult()
    for index in range(1, segments + 1):
        prompt = Prompt("(free speech, do not say the wake word)", "free", 0, "", 0)
        answer = prompter.ask(prompt, index, segments)
        if answer == "quit":
            result.stopped_early = True
            break
        if answer == "skip":
            result.skipped += 1
            continue
        pcm = source.record(segment_seconds, sample_rate)
        if peak_amplitude(pcm) < MIN_PEAK:
            prompter.info("The clip is silent; check the microphone. It was not saved.")
            result.rejected_silent += 1
            continue
        _save(
            data_root=data_root,
            manifest=manifest,
            pcm=pcm,
            sample_rate=sample_rate,
            speaker=speaker,
            consent_id=consent.consent_id,
            label="negative",
            phrase="",
            language=language,
            distance_cm=distance_cm,
            volume="",
            room=room,
            device=device,
            mode="free",
            today=today(),
        )
        result.recorded += 1
    return result
