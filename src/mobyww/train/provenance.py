"""Provenance written next to a trained model: how it was made, from what, with which seeds.

`provenance.json` holds the repository and upstream commits, the seeds, the hyperparameters, the
hash of the data manifests, the augmentation and job-file hashes, library versions and the model
hash. It never states an accuracy: `accuracy_measured` is false until W4 measures one on the
owner's real recordings.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "tensorflow",
    "keras",
    "numpy",
    "scipy",
    "ai-edge-litert",
    "pymicro-features",
    "microwakeword",
    "audiomentations",
    "mmap_ninja",
    "datasets",
    "piper-tts",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash(paths: Sequence[Path]) -> str:
    """Order-independent hash of manifest rows: sorted lines of all files, LF-normalised."""
    lines: list[str] = []
    for path in paths:
        lines.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))


def upstream_commit(distribution: str = "microwakeword") -> str | None:
    """The git commit microWakeWord was installed from (pip's direct_url.json), if recorded."""
    try:
        text = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    info = json.loads(text).get("vcs_info", {})
    commit = info.get("commit_id")
    return str(commit) if commit else None


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def build_provenance(
    *,
    model_path: Path,
    repo: Path,
    seeds: dict[str, int],
    training_config: dict[str, Any],
    manifests: Sequence[Path],
    config_files: dict[str, Path],
    upstream: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to say how `model_path` came to be. Hashes are of the files as given."""
    return {
        "schema": "mobyww-provenance/1",
        "model": {
            "file": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": sha256_path(model_path),
        },
        "repository": git_state(repo),
        "upstream": {
            "name": "OHF-Voice/micro-wake-word",
            "commit": upstream if upstream is not None else upstream_commit(),
        },
        "seeds": seeds,
        "hyperparameters": training_config,
        "data": {
            "manifest_files": [p.name for p in manifests],
            "manifest_hash": manifest_hash(manifests),
        },
        "config_hashes": {name: sha256_path(path) for name, path in sorted(config_files.items())},
        "package_versions": package_versions(),
        "accuracy_measured": False,
        "note": "Accuracy is unmeasured; the owner's real recordings are the test set (W4).",
        **(extra or {}),
    }


def write_provenance(directory: Path, provenance: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "provenance.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return path
