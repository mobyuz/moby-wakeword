"""Local consent records. They live under data/consent/, which git ignores.

A record holds a pseudonym and a consent id, never a real name by default. The optional
`identity_note` is local-only free text and is the only place a person can be identified.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CONSENT_TEXT_VERSION = "2026-11-v1"
SCOPES = ("train-wake-word-model", "evaluate-wake-word-model")
PSEUDONYM_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")

CONSENT_TEXT = f"""\
Moby wake word recordings - consent (version {CONSENT_TEXT_VERSION})

I agree that short audio clips of my voice are recorded to train and evaluate the Moby wake
word model. My name is not stored with the clips, only a pseudonym. The clips and this record
stay on the project's own machines and are never published. I may withdraw at any time and ask
for my clips to be deleted (see docs/privacy.md). A model already trained on the clips cannot
be un-trained, but it will not be retrained on them after I withdraw.
"""


class ConsentError(Exception):
    """Consent is missing, invalid or withdrawn."""


@dataclass
class ConsentRecord:
    speaker: str
    consent_id: str
    text_version: str
    scopes: list[str]
    granted: bool
    granted_at: str
    withdrawn: bool = False
    withdrawn_at: str = ""
    identity_note: str = field(default="", repr=False)


def new_pseudonym(rng: Callable[[int], str] = secrets.token_hex) -> str:
    return f"spk-{rng(3)}"


def record_path(consent_dir: Path, speaker: str) -> Path:
    if not PSEUDONYM_RE.fullmatch(speaker):
        raise ConsentError(f"invalid pseudonym {speaker!r}: use 3-32 chars of [a-z0-9-]")
    return consent_dir / f"{speaker}.json"


def create_consent(
    consent_dir: Path,
    speaker: str,
    *,
    granted: bool,
    scopes: tuple[str, ...] = SCOPES,
    identity_note: str = "",
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    new_id: Callable[[], str] = lambda: "c-" + secrets.token_hex(8),
) -> ConsentRecord:
    path = record_path(consent_dir, speaker)
    if path.exists():
        raise ConsentError(f"a consent record for {speaker!r} already exists")
    if not granted:
        raise ConsentError("consent was not granted; nothing is recorded")
    record = ConsentRecord(
        speaker=speaker,
        consent_id=new_id(),
        text_version=CONSENT_TEXT_VERSION,
        scopes=list(scopes),
        granted=True,
        granted_at=now().isoformat(timespec="seconds"),
        identity_note=identity_note,
    )
    consent_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def load_consent(consent_dir: Path, speaker: str) -> ConsentRecord:
    path = record_path(consent_dir, speaker)
    if not path.is_file():
        raise ConsentError(f"no consent record for {speaker!r}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ConsentRecord(**data)


def require_active_consent(consent_dir: Path, speaker: str) -> ConsentRecord:
    record = load_consent(consent_dir, speaker)
    if not record.granted or record.withdrawn:
        raise ConsentError(f"consent of {speaker!r} is not active")
    return record


def withdraw_consent(
    consent_dir: Path,
    speaker: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ConsentRecord:
    record = load_consent(consent_dir, speaker)
    record.withdrawn = True
    record.withdrawn_at = now().isoformat(timespec="seconds")
    path = record_path(consent_dir, speaker)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
