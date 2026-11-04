"""mobyww-record: guided recording command line.

Subcommands: consent, withdraw, plan, session, free. Nothing here records without a granted
local consent record, and the recordings, consent files and manifests stay under data/ (ignored
by git).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from mobyww.recording.capture import AudioSource, SoundDeviceSource
from mobyww.recording.consent import (
    CONSENT_TEXT,
    ConsentError,
    create_consent,
    new_pseudonym,
    withdraw_consent,
)
from mobyww.recording.session import (
    Answer,
    Prompt,
    build_prompts,
    run_free_session,
    run_prompted_session,
)

DEFAULT_CONFIG = Path("configs/wakeword.yaml")


class ConsolePrompter:
    def __init__(
        self,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self._read = read
        self._write = write

    def info(self, message: str) -> None:
        self._write(message)

    def ask(self, prompt: Prompt, index: int, total: int) -> Answer:
        if prompt.label == "free":
            self._write(f"[{index}/{total}] Speak freely for the whole clip. {prompt.phrase}")
        else:
            kind = "SAY" if prompt.label == "wake" else "SAY (this is a trap word)"
            self._write(
                f'[{index}/{total}] {kind}: "{prompt.phrase}"  distance {prompt.distance_cm} cm, '
                f"volume {prompt.volume}"
            )
        reply = self._read("Enter = record, s = skip, q = quit: ").strip().lower()
        if reply == "q":
            return "quit"
        if reply == "s":
            return "skip"
        return "record"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def _prompts_from_config(
    config: dict[str, Any], phrases: list[str] | None, repeats: int | None
) -> list[Prompt]:
    recording = config["recording"]
    chosen = phrases or [c["phrase"] for c in config["candidates"]]
    return build_prompts(
        chosen,
        list(config["hard_negatives"]),
        list(recording["distances_cm"]),
        list(recording["volumes"]),
        repeats if repeats is not None else int(recording["repeats_per_cell"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mobyww-record", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    consent = sub.add_parser("consent", help="show the consent text and store a local record")
    consent.add_argument("--speaker", help="pseudonym (default: a random one)")
    consent.add_argument("--identity-note", default="", help="local-only note, never committed")

    withdraw = sub.add_parser("withdraw", help="mark a speaker's consent as withdrawn")
    withdraw.add_argument("--speaker", required=True)

    plan = sub.add_parser("plan", help="print the prompt plan without recording anything")
    plan.add_argument("--phrase", action="append", help="limit to these phrases")
    plan.add_argument("--repeats", type=int)

    for name, help_text in (("session", "guided prompted session"), ("free", "free speech mode")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--speaker", required=True)
        cmd.add_argument("--language", choices=["uz", "ru", "en", "other"], required=True)
        cmd.add_argument("--room", required=True)
        cmd.add_argument("--device", required=True, help="microphone name")
        cmd.add_argument("--input-device", help="sounddevice device index or name")
        if name == "session":
            cmd.add_argument("--phrase", action="append", help="limit to these phrases")
            cmd.add_argument("--repeats", type=int)
        else:
            cmd.add_argument("--segments", type=int, default=20)
            cmd.add_argument("--segment-seconds", type=float, default=8.0)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    source: AudioSource | None = None,
    prompter: ConsolePrompter | None = None,
    read: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root
    try:
        config = load_config(args.config)
        if args.command == "consent":
            print(CONSENT_TEXT)
            if read("Type AGREE to give consent: ").strip() != "AGREE":
                print("No consent given. Nothing was stored.")
                return 1
            speaker = args.speaker or new_pseudonym()
            record = create_consent(
                data_root / "consent", speaker, granted=True, identity_note=args.identity_note
            )
            print(f"Consent stored for pseudonym {record.speaker} ({record.consent_id}).")
            return 0
        if args.command == "withdraw":
            withdraw_consent(data_root / "consent", args.speaker)
            print(
                f"Consent of {args.speaker} marked withdrawn. Delete their clips as described "
                "in docs/privacy.md."
            )
            return 0
        if args.command == "plan":
            prompts = _prompts_from_config(config, args.phrase, args.repeats)
            for prompt in prompts:
                print(f"{prompt.label}\t{prompt.phrase}\t{prompt.distance_cm} cm\t{prompt.volume}")
            print(f"{len(prompts)} prompts")
            return 0

        audio = config["audio"]
        mic = source or SoundDeviceSource(getattr(args, "input_device", None))
        talker = prompter or ConsolePrompter(read)
        if args.command == "session":
            result = run_prompted_session(
                source=mic,
                prompter=talker,
                data_root=data_root,
                speaker=args.speaker,
                language=args.language,
                room=args.room,
                device=args.device,
                prompts=_prompts_from_config(config, args.phrase, args.repeats),
                clip_seconds=float(audio["clip_seconds"]),
                sample_rate=int(audio["sample_rate_hz"]),
            )
        else:
            result = run_free_session(
                source=mic,
                prompter=talker,
                data_root=data_root,
                speaker=args.speaker,
                language=args.language,
                room=args.room,
                device=args.device,
                segments=args.segments,
                segment_seconds=args.segment_seconds,
                sample_rate=int(audio["sample_rate_hz"]),
            )
        print(
            f"recorded {result.recorded}, skipped {result.skipped}, silent {result.rejected_silent}"
        )
        return 0
    except (ConsentError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
