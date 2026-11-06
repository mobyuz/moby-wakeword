from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from mobyww.audio import write_wav_int16
from mobyww.data.manifest import read_manifest
from mobyww.data.validate import validate
from mobyww.synth import (
    PlanParams,
    build_manifest,
    build_plan,
    expand_job,
    job_file_sha256,
    load_job_file,
    save_job_file,
)
from mobyww.synth import cli as synth_cli
from mobyww.synth.plan import RU_HARD_NEGATIVES, WAKE_VARIANTS, choose_speakers
from mobyww.synth.spec import (
    APPROVED_VOICES,
    JobFileError,
    from_dict,
    speaker_pseudonym,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "wakeword.yaml").read_text(encoding="utf-8"))
HARD_NEGATIVES: list[str] = CONFIG["hard_negatives"]

TINY = PlanParams(
    libritts_speakers=3,
    wake_clips_per_speaker_variant=2,
    hard_negative_clips_per_speaker_text=1,
    ru_wake_clips_per_variant=2,
    ru_hard_negative_clips_per_text=1,
)


def write_all_clips(job_file_path: Path, wav_dir: Path) -> int:
    job_file = load_job_file(job_file_path)
    count = 0
    for job in job_file.jobs:
        for clip in expand_job(job, job_file.seed):
            t = np.arange(1600) / 16000
            tone = 3000 * np.sin(2 * np.pi * (200 + 3 * count) * t)
            write_wav_int16(wav_dir / f"{clip.clip_id}.wav", tone.astype(np.int16), 16000)
            count += 1
    return count


def test_plan_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    a, b = build_plan(HARD_NEGATIVES), build_plan(HARD_NEGATIVES)
    assert a == b
    assert job_file_sha256(a) == job_file_sha256(b)
    save_job_file(tmp_path / "a.yaml", a)
    save_job_file(tmp_path / "b.yaml", b)
    save_job_file(tmp_path / "a.json", a)
    assert (tmp_path / "a.yaml").read_bytes() == (tmp_path / "b.yaml").read_bytes()
    assert load_job_file(tmp_path / "a.yaml") == a
    assert load_job_file(tmp_path / "a.json") == a
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["total_clips"] == (
        a.total_clips
    )


def test_seed_changes_the_plan() -> None:
    a = build_plan(HARD_NEGATIVES, PlanParams(seed=1))
    b = build_plan(HARD_NEGATIVES, PlanParams(seed=2))
    assert job_file_sha256(a) != job_file_sha256(b)
    voice = APPROVED_VOICES[0]
    assert choose_speakers(voice, 10, 800, 1) != choose_speakers(voice, 10, 800, 2)


def test_plan_covers_variants_hard_negatives_and_only_approved_voices() -> None:
    plan = build_plan(HARD_NEGATIVES)
    wake_texts = {j.text for j in plan.jobs if j.label == "wake"}
    assert {"hey moby", "hei mobi", "хей моби"} <= wake_texts
    assert {v.language for v in WAKE_VARIANTS} == {"en", "ru", "uz"}
    negatives = {j.text for j in plan.jobs if j.label == "hard_negative" and j.language == "en"}
    assert negatives == set(HARD_NEGATIVES)
    assert {v.id for v in plan.voices} == {
        "en_US-libritts_r-medium",
        "ru_RU-denis-medium",
        "ru_RU-dmitri-medium",
    }
    assert all(j.label != "wake" or j.text not in HARD_NEGATIVES for j in plan.jobs)
    assert plan.total_clips == sum(j.count for j in plan.jobs) > 0


def test_every_hard_negative_has_a_russian_rendering_today() -> None:
    assert set(HARD_NEGATIVES) <= set(RU_HARD_NEGATIVES)
    assert not any("Russian voices do not read" in n for n in build_plan(HARD_NEGATIVES).notes)


def test_every_voice_reads_wake_and_hard_negatives_so_voice_is_not_the_label() -> None:
    plan = build_plan(HARD_NEGATIVES)
    for voice in plan.voices:
        labels = {j.label for j in plan.jobs if j.voice_id == voice.id}
        assert labels == {"wake", "hard_negative"}, voice.id
        speakers = {j.speaker_id for j in plan.jobs if j.voice_id == voice.id}
        for speaker in speakers:
            mine = {
                j.label for j in plan.jobs if j.voice_id == voice.id and j.speaker_id == speaker
            }
            assert mine == {"wake", "hard_negative"}, (voice.id, speaker)
        wake = sum(j.count for j in plan.jobs if j.voice_id == voice.id and j.label == "wake")
        neg = sum(
            j.count for j in plan.jobs if j.voice_id == voice.id and j.label == "hard_negative"
        )
        assert 0.4 < wake / neg < 2.5, (voice.id, wake, neg)


def test_missing_russian_rendering_is_recorded() -> None:
    plan = build_plan([*HARD_NEGATIVES, "hey google"], TINY)
    assert any("hey google" in n for n in plan.notes)
    assert not any(j.text == "hey google" and j.language == "ru" for j in plan.jobs)


def test_speakers_are_distinct_and_in_the_pool() -> None:
    voice = APPROVED_VOICES[0]
    chosen = choose_speakers(voice, 100, 800, 3030)
    assert len(chosen) == len(set(chosen)) == 100
    assert chosen == sorted(chosen)
    assert max(chosen) < 800
    assert choose_speakers(APPROVED_VOICES[1], 5, 800, 1) == [0]  # single-speaker voice


def test_expand_job_is_deterministic_and_uses_the_grid() -> None:
    plan = build_plan(HARD_NEGATIVES, TINY)
    job = plan.jobs[0]
    a, b = expand_job(job, plan.seed), expand_job(job, plan.seed)
    assert a == b
    assert [c.clip_id for c in a] == [f"{job.job_id}.{i:04d}" for i in range(job.count)]
    for clip in a:
        assert clip.length_scale in job.length_scales
        assert clip.noise_scale in job.noise_scales
        assert clip.noise_scale_w in job.noise_scale_ws
        assert 0 <= clip.rng_seed < 2**31
    other = expand_job(job, plan.seed + 1)
    assert [c.rng_seed for c in other] != [c.rng_seed for c in a]
    big = expand_job(replace(job, count=50), plan.seed)
    assert len({c.clip_id for c in big}) == 50
    assert len({c.rng_seed for c in big}) == 50


def test_clip_ids_are_unique_across_the_whole_plan() -> None:
    plan = build_plan(HARD_NEGATIVES)
    ids = [c.clip_id for job in plan.jobs for c in expand_job(job, plan.seed)]
    assert len(ids) == len(set(ids)) == plan.total_clips


def test_licence_rules_reject_unapproved_or_edited_voices() -> None:
    data = to_dict(build_plan(HARD_NEGATIVES, TINY))
    data["voices"][0]["id"] = "ru_RU-ruslan-medium"  # CC BY-NC-SA: stopped in licenses.md
    with pytest.raises(JobFileError, match="not approved"):
        from_dict(data)

    data = to_dict(build_plan(HARD_NEGATIVES, TINY))
    data["voices"][0]["license"] = "unknown"
    with pytest.raises(JobFileError, match=r"differs|no licence"):
        from_dict(data)

    data = to_dict(build_plan(HARD_NEGATIVES, TINY))
    data["jobs"][1]["job_id"] = data["jobs"][0]["job_id"]
    with pytest.raises(JobFileError, match="duplicate job id"):
        from_dict(data)

    data = to_dict(build_plan(HARD_NEGATIVES, TINY))
    data["jobs"][0]["speaker_id"] = 10_000
    with pytest.raises(JobFileError, match="outside the voice"):
        from_dict(data)

    with pytest.raises(JobFileError, match="malformed"):
        from_dict({"schema": "mobyww-synth-job/1"})
    with pytest.raises(ValueError, match="empty"):
        build_plan([], TINY)


def test_manifest_from_generated_wavs_passes_the_validator(tmp_path: Path) -> None:
    job_path = tmp_path / "job.yaml"
    plan = build_plan(HARD_NEGATIVES, TINY)
    save_job_file(job_path, plan)
    data_root = tmp_path / "data"
    wav_dir = data_root / "synthetic" / "clips"
    total = write_all_clips(job_path, wav_dir)
    assert total == plan.total_clips

    result = build_manifest(plan, wav_dir, data_root, split_seed=1337, val_pct=30)
    assert result.complete
    assert len(result.rows) == total
    report = validate(result.rows, data_root)
    assert report.ok, report.errors

    by_label = {r.label for r in result.rows}
    assert by_label == {"wake", "hard_negative"}
    assert {r.source for r in result.rows} == {"tts"}
    assert {r.license for r in result.rows} == {"CC-BY-4.0", "CC0-1.0"}  # from the job file
    assert {r.language for r in result.rows} == {"en", "uz", "ru"}
    assert all(r.path.startswith("synthetic/clips/") for r in result.rows)
    assert all(r.speaker.startswith("synth-") for r in result.rows)
    assert all(r.split != "unassigned" for r in result.rows)
    assert "test" not in {r.split for r in result.rows}  # synthetic data is not a test set
    assert {r.split for r in result.rows} <= {"train", "val"}

    # one synthetic speaker, one split
    splits: dict[str, set[str]] = {}
    for row in result.rows:
        splits.setdefault(row.speaker, set()).add(row.split)
    assert all(len(s) == 1 for s in splits.values())
    voice = plan.voice(plan.jobs[0].voice_id)
    assert speaker_pseudonym(voice, plan.jobs[0].speaker_id) in splits


def test_manifest_reports_missing_unexpected_and_bad_clips(tmp_path: Path) -> None:
    job_path = tmp_path / "job.yaml"
    plan = build_plan(HARD_NEGATIVES, TINY)
    save_job_file(job_path, plan)
    data_root = tmp_path / "data"
    wav_dir = data_root / "clips"
    write_all_clips(job_path, wav_dir)

    first = next(iter(sorted(wav_dir.glob("*.wav"))))
    first.unlink()
    write_wav_int16(wav_dir / "stray.wav", np.zeros(160, dtype=np.int16), 16000)
    second = sorted(wav_dir.glob("wk.*"))[0]
    write_wav_int16(second, np.ones(160, dtype=np.int16), 22050)  # wrong rate

    result = build_manifest(plan, wav_dir, data_root, split_seed=1)
    assert [m for m in result.missing] == [first.name]
    assert result.unexpected == ["stray.wav"]
    assert len(result.bad_format) == 1
    assert not result.complete
    assert len(result.rows) == plan.total_clips - 2


def test_manifest_rejects_a_folder_outside_the_data_root(tmp_path: Path) -> None:
    plan = build_plan(HARD_NEGATIVES, TINY)
    with pytest.raises(JobFileError, match="not inside"):
        build_manifest(plan, tmp_path / "x", tmp_path / "y", split_seed=1)


def test_cli_plan_then_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_path = tmp_path / "job.json"
    config = ROOT / "configs" / "wakeword.yaml"
    assert synth_cli.main(["plan", "--out", str(job_path), "--config", str(config)]) == 0
    assert "sha256" in capsys.readouterr().out
    assert load_job_file(job_path).seed == 3030  # seeds.synthesis in configs/wakeword.yaml

    small = build_plan(HARD_NEGATIVES, TINY)
    save_job_file(job_path, small)
    data_root = tmp_path / "data"
    write_all_clips(job_path, data_root / "clips")
    out = data_root / "manifests" / "synthetic.jsonl"
    code = synth_cli.main(
        [
            "manifest",
            "--job",
            str(job_path),
            "--wav-dir",
            str(data_root / "clips"),
            "--data-root",
            str(data_root),
            "--out",
            str(out),
            "--config",
            str(config),
            "--require-all",
        ]
    )
    assert code == 0
    assert len(read_manifest(out)) == small.total_clips
    assert (
        synth_cli.main(
            [
                "manifest",
                "--job",
                str(tmp_path / "none.json"),
                "--wav-dir",
                "a",
                "--data-root",
                "b",
                "--out",
                "c",
            ]
        )
        == 2
    )
