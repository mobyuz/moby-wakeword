"""Speaker-level split assignment: one speaker never lands in two splits."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from mobyww.data.manifest import NO_SPEAKER, UNASSIGNED, Row


def assign_split(key: str, seed: int, val_pct: int = 10, test_pct: int = 10) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def assign_splits(rows: list[Row], seed: int, val_pct: int = 10, test_pct: int = 10) -> list[Row]:
    """Fill splits that are still unassigned.

    Rows that already have a split keep it, and a speaker that already has a split elsewhere
    keeps that split, so assignment never creates leakage.
    """
    known = {r.speaker: r.split for r in rows if r.split != UNASSIGNED and r.speaker != NO_SPEAKER}
    out: list[Row] = []
    for row in rows:
        if row.split != UNASSIGNED:
            out.append(row)
        elif row.speaker == NO_SPEAKER:
            out.append(replace(row, split=assign_split(row.id, seed, val_pct, test_pct)))
        else:
            split = known.setdefault(
                row.speaker, assign_split(row.speaker, seed, val_pct, test_pct)
            )
            out.append(replace(row, split=split))
    return out
