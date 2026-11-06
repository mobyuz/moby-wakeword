"""Plan the synthesis jobs: phrase variants, hard negatives, speakers, counts. Deterministic.

Pronunciation variants are SPELLINGS chosen to make the voice say "hey moby" a little differently
(an English-looking, an Uzbek-looking and a Russian spelling). They are guesses: no native
speaker has reviewed them and no audio has been generated, so nothing here claims that a variant
sounds like an Uzbek or Russian speaker. The Uzbek-accent variants are read by an English voice
(the Piper catalogue has no Uzbek voice, docs/licenses.md), which is a weak stand-in.

Why the Russian voices also read every hard negative: if only the Russian voices said "wake"
texts, the model could learn "Russian voice" instead of the phrase. Each voice therefore reads the
wake variants of its language AND the hard negatives, in the same proportions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from mobyww.synth.spec import (
    APPROVED_VOICES,
    DEFAULT_LENGTH_SCALES,
    DEFAULT_NOISE_SCALE_WS,
    DEFAULT_NOISE_SCALES,
    VOICES_BY_ID,
    Job,
    JobFile,
    Label,
    Voice,
    validate,
)


@dataclass(frozen=True)
class Variant:
    id: str
    text: str
    language: str  # recorded in the manifest
    voice_language: str  # which voices read it: en or ru
    note: str


WAKE_VARIANTS: tuple[Variant, ...] = (
    Variant("hey_moby", "hey moby", "en", "en", "plain English spelling"),
    Variant("hay_moby", "hay moby", "en", "en", "English respelling, vowel of 'hey' spelled out"),
    Variant("hey_mobee", "hey mobee", "en", "en", "English respelling, final vowel spelled out"),
    Variant("hei_mobi", "hei mobi", "uz", "en", "Uzbek-looking Latin spelling, English voice"),
    Variant("xey_mobi", "xey mobi", "uz", "en", "Uzbek x-spelling of 'h', English voice (guess)"),
    Variant("hey_mobi", "hey mobi", "uz", "en", "mixed spelling, English voice"),
    Variant("ru_hei_mobi", "хей моби", "ru", "ru", "Cyrillic transcription"),
    Variant("ru_hae_mobi", "хэй моби", "ru", "ru", "Cyrillic transcription with э"),
    Variant("ru_ei_mobi", "эй моби", "ru", "ru", "Cyrillic transcription without the h"),
)

# Russian renderings of the hard negatives in configs/wakeword.yaml, by the same Cyrillic
# transcription idea. Native-speaker review is missing. A hard negative with no entry here is
# not read by the Russian voices, and `plan` records that in the job file notes.
RU_HARD_NEGATIVES: dict[str, str] = {
    "mobil": "мобил",
    "mobile": "мобайл",
    "mobi": "моби",
    "moby dick": "моби дик",
    "hey mom": "хей мом",
    "hey bobby": "хей бобби",
    "hey toby": "хей тоби",
}


@dataclass(frozen=True)
class PlanParams:
    seed: int = 3030
    libritts_speakers: int = 100
    libritts_speaker_pool: int = 800  # the first 800 of 904; late speakers are rare in the corpus
    wake_clips_per_speaker_variant: int = 6
    hard_negative_clips_per_speaker_text: int = 4
    ru_wake_clips_per_variant: int = 120
    ru_hard_negative_clips_per_text: int = 60
    length_scales: tuple[float, ...] = DEFAULT_LENGTH_SCALES
    noise_scales: tuple[float, ...] = DEFAULT_NOISE_SCALES
    noise_scale_ws: tuple[float, ...] = DEFAULT_NOISE_SCALE_WS


# A tiny plan for checking the pipeline end to end (a couple of hundred clips).
SMOKE_PARAMS = PlanParams(
    libritts_speakers=4,
    wake_clips_per_speaker_variant=2,
    hard_negative_clips_per_speaker_text=1,
    ru_wake_clips_per_variant=4,
    ru_hard_negative_clips_per_text=2,
)


def _slug(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isascii() and ch.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part) or "x"


def choose_speakers(voice: Voice, count: int, pool: int, seed: int) -> list[int]:
    """`count` distinct speaker ids from the first `pool`, picked by hash order (seeded)."""
    pool = min(pool, voice.speaker_count)
    count = min(count, pool)
    ranked = sorted(
        range(pool),
        key=lambda s: hashlib.sha256(f"{seed}:{voice.id}:{s}".encode()).hexdigest(),
    )
    return sorted(ranked[:count])


def _job(
    voice: Voice,
    speaker: int,
    label: Label,
    variant_id: str,
    text: str,
    language: str,
    count: int,
    params: PlanParams,
) -> Job:
    prefix = "wk" if label == "wake" else "hn"
    return Job(
        job_id=f"{prefix}.{voice.short}.s{speaker:03d}.{variant_id}",
        voice_id=voice.id,
        speaker_id=speaker,
        label=label,
        language=language,
        variant_id=variant_id,
        text=text,
        count=count,
        length_scales=params.length_scales,
        noise_scales=params.noise_scales,
        noise_scale_ws=params.noise_scale_ws,
    )


def build_plan(hard_negatives: Sequence[str], params: PlanParams | None = None) -> JobFile:
    """The job file for the wake variants and the hard negatives. Same inputs, same bytes."""
    params = params or PlanParams()
    if not hard_negatives:
        raise ValueError("hard_negatives is empty: configs/wakeword.yaml must list them")
    jobs: list[Job] = []
    notes = [
        "Pronunciation variants are spellings; no native speaker reviewed them (docs/licenses.md).",
        "Uzbek-accent variants are read by an English voice: no Uzbek voice is approved.",
        "Synthetic clips must not be used as the test set; the owner's real recordings are.",
    ]

    for voice in APPROVED_VOICES:
        if voice.text_language == "en":
            speakers = choose_speakers(
                voice, params.libritts_speakers, params.libritts_speaker_pool, params.seed
            )
            wake_count = params.wake_clips_per_speaker_variant
            neg_count = params.hard_negative_clips_per_speaker_text
        else:
            speakers = list(range(voice.speaker_count))
            wake_count = params.ru_wake_clips_per_variant
            neg_count = params.ru_hard_negative_clips_per_text

        for speaker in speakers:
            for variant in WAKE_VARIANTS:
                if variant.voice_language == voice.text_language:
                    jobs.append(
                        _job(
                            voice,
                            speaker,
                            "wake",
                            variant.id,
                            variant.text,
                            variant.language,
                            wake_count,
                            params,
                        )
                    )
            for phrase in hard_negatives:
                text = phrase if voice.text_language == "en" else RU_HARD_NEGATIVES.get(phrase)
                if text is None:
                    continue
                jobs.append(
                    _job(
                        voice,
                        speaker,
                        "hard_negative",
                        _slug(phrase) if voice.text_language == "en" else f"ru_{_slug(phrase)}",
                        text,
                        voice.text_language,
                        neg_count,
                        params,
                    )
                )

    missing_ru = [p for p in hard_negatives if p not in RU_HARD_NEGATIVES]
    if missing_ru:
        notes.append(
            "Russian voices do not read these hard negatives (no Cyrillic rendering): "
            + ", ".join(missing_ru)
        )
    job_file = JobFile(
        seed=params.seed,
        voices=tuple(VOICES_BY_ID[v.id] for v in APPROVED_VOICES),
        jobs=tuple(jobs),
        notes=tuple(notes),
    )
    validate(job_file)
    return job_file
