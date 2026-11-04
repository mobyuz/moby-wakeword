from __future__ import annotations

import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import pcm_tone
from mobyww.data.manifest import read_manifest
from mobyww.data.validate import validate
from mobyww.recording import cli
from mobyww.recording.capture import ScriptedSource, peak_amplitude
from mobyww.recording.consent import (
    ConsentError,
    create_consent,
    new_pseudonym,
    require_active_consent,
    withdraw_consent,
)
from mobyww.recording.session import (
    Answer,
    Prompt,
    build_prompts,
    run_free_session,
    run_prompted_session,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wakeword.yaml"


class ScriptedPrompter:
    def __init__(self, answers: list[Answer]) -> None:
        self.answers = list(answers)
        self.seen: list[Prompt] = []
        self.messages: list[str] = []

    def ask(self, prompt: Prompt, index: int, total: int) -> Answer:
        self.seen.append(prompt)
        return self.answers.pop(0) if self.answers else "record"

    def info(self, message: str) -> None:
        self.messages.append(message)


def grant(data_root: Path, speaker: str = "spk-abc") -> str:
    return create_consent(data_root / "consent", speaker, granted=True).consent_id


def test_prompt_plan_covers_distances_volumes_and_hard_negatives() -> None:
    prompts = build_prompts(["hey moby"], ["mobil"], [30, 100, 300], ["quiet", "normal", "loud"], 2)
    wake = [p for p in prompts if p.label == "wake"]
    assert len(wake) == 3 * 3 * 2
    assert {p.distance_cm for p in wake} == {30, 100, 300}
    assert {p.volume for p in wake} == {"quiet", "normal", "loud"}
    assert [p.phrase for p in prompts if p.label == "hard_negative"] == ["mobil"] * 3


def test_prompted_session_writes_clips_manifest_and_validates(tmp_path: Path) -> None:
    consent_id = grant(tmp_path)
    prompts = build_prompts(["hey moby"], ["mobil"], [30, 100], ["normal"], 1)
    chunks = [pcm_tone(samples=1600 + i) for i in range(len(prompts))]
    source = ScriptedSource(chunks)
    result = run_prompted_session(
        source=source,
        prompter=ScriptedPrompter([]),
        data_root=tmp_path,
        speaker="spk-abc",
        language="en",
        room="kitchen",
        device="usb-mic",
        prompts=prompts,
        clip_seconds=2.5,
        sample_rate=16000,
        today=lambda: datetime(2026, 11, 5, tzinfo=UTC).date(),
    )
    assert result.recorded == len(prompts) == 4
    assert source.calls == [(2.5, 16000)] * 4
    rows = read_manifest(tmp_path / "manifests" / "recorded.jsonl")
    assert [r.id for r in rows][:2] == ["spk-abc-00001", "spk-abc-00002"]
    assert {r.consent_id for r in rows} == {consent_id}
    assert {r.distance_cm for r in rows} == {30, 100}
    assert all(r.split == "unassigned" and r.source == "recorded" for r in rows)
    assert rows[0].recorded_on == "2026-11-05"
    assert rows[0].room == "kitchen"
    with wave.open(str(tmp_path / rows[0].path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
    report = validate(rows, tmp_path, consent_dir=tmp_path / "consent", allow_unassigned=True)
    assert report.ok, report.errors


def test_session_refuses_without_active_consent(tmp_path: Path) -> None:
    kwargs = {
        "source": ScriptedSource([pcm_tone()]),
        "prompter": ScriptedPrompter([]),
        "data_root": tmp_path,
        "speaker": "spk-abc",
        "language": "en",
        "room": "r",
        "device": "d",
        "prompts": build_prompts(["hey moby"], [], [30], ["normal"], 1),
        "clip_seconds": 1.0,
        "sample_rate": 16000,
    }
    with pytest.raises(ConsentError):
        run_prompted_session(**kwargs)  # type: ignore[arg-type]
    grant(tmp_path)
    withdraw_consent(tmp_path / "consent", "spk-abc")
    with pytest.raises(ConsentError):
        run_prompted_session(**kwargs)  # type: ignore[arg-type]
    assert not (tmp_path / "recordings").exists()


def test_skip_quit_and_silent_clips(tmp_path: Path) -> None:
    grant(tmp_path)
    prompts = build_prompts(["hey moby"], [], [30], ["normal"], 4)
    source = ScriptedSource([bytes(3200), pcm_tone()])
    prompter = ScriptedPrompter(["skip", "record", "record", "quit"])
    result = run_prompted_session(
        source=source,
        prompter=prompter,
        data_root=tmp_path,
        speaker="spk-abc",
        language="uz",
        room="r",
        device="d",
        prompts=prompts,
        clip_seconds=1.0,
        sample_rate=16000,
    )
    assert (result.skipped, result.rejected_silent, result.recorded) == (1, 1, 1)
    assert result.stopped_early
    assert prompter.messages
    assert "silent" in prompter.messages[0]
    assert len(read_manifest(tmp_path / "manifests" / "recorded.jsonl")) == 1


def test_free_mode_saves_negatives(tmp_path: Path) -> None:
    grant(tmp_path)
    result = run_free_session(
        source=ScriptedSource([pcm_tone(), pcm_tone(samples=900)]),
        prompter=ScriptedPrompter([]),
        data_root=tmp_path,
        speaker="spk-abc",
        language="ru",
        room="r",
        device="d",
        segments=2,
        segment_seconds=8.0,
        sample_rate=16000,
    )
    assert result.recorded == 2
    rows = read_manifest(tmp_path / "manifests" / "recorded.jsonl")
    assert {(r.label, r.mode, r.phrase) for r in rows} == {("negative", "free", "")}


def test_consent_lifecycle(tmp_path: Path) -> None:
    record = create_consent(tmp_path, "spk-abc", granted=True, identity_note="local only")
    assert record.consent_id.startswith("c-")
    assert require_active_consent(tmp_path, "spk-abc").granted
    with pytest.raises(ConsentError):
        create_consent(tmp_path, "spk-abc", granted=True)
    with pytest.raises(ConsentError):
        create_consent(tmp_path, "spk-new", granted=False)
    with pytest.raises(ConsentError):
        create_consent(tmp_path, "John Smith", granted=True)  # a real name is not a pseudonym
    withdraw_consent(tmp_path, "spk-abc")
    with pytest.raises(ConsentError):
        require_active_consent(tmp_path, "spk-abc")
    assert new_pseudonym().startswith("spk-")


def test_peak_amplitude() -> None:
    assert peak_amplitude(bytes(100)) == 0
    assert peak_amplitude(pcm_tone(1234)) == 1234


def test_cli_plan_consent_and_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--data-root", str(tmp_path), "--config", str(CONFIG)]
    assert cli.main([*base, "plan", "--phrase", "hey moby", "--repeats", "1"]) == 0
    assert "prompts" in capsys.readouterr().out

    assert cli.main([*base, "consent", "--speaker", "spk-cli"], read=lambda _: "no") == 1
    assert not (tmp_path / "consent").exists()
    assert cli.main([*base, "consent", "--speaker", "spk-cli"], read=lambda _: "AGREE") == 0

    code = cli.main(
        [
            *base,
            "session",
            "--speaker",
            "spk-cli",
            "--language",
            "en",
            "--room",
            "r",
            "--device",
            "d",
            "--phrase",
            "hey moby",
            "--repeats",
            "1",
        ],
        source=ScriptedSource([pcm_tone()] * 40),
        read=lambda _: "",
    )
    assert code == 0
    assert len(read_manifest(tmp_path / "manifests" / "recorded.jsonl")) == 3 * 3 + 7 * 3

    code = cli.main(
        [
            *base,
            "free",
            "--speaker",
            "spk-cli",
            "--language",
            "en",
            "--room",
            "r",
            "--device",
            "d",
            "--segments",
            "1",
        ],
        source=ScriptedSource([pcm_tone()]),
        read=lambda _: "",
    )
    assert code == 0
    assert cli.main([*base, "withdraw", "--speaker", "spk-cli"]) == 0
    assert cli.main([*base, "withdraw", "--speaker", "spk-nobody"]) == 2
    code = cli.main(
        [
            *base,
            "session",
            "--speaker",
            "spk-cli",
            "--language",
            "en",
            "--room",
            "r",
            "--device",
            "d",
        ],
        source=ScriptedSource([pcm_tone()]),
    )
    assert code == 2  # consent withdrawn: refuses to record
