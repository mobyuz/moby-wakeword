"""Privacy guard: keeps recordings, speaker identities, consent files, features, and model
weights out of git.

.gitignore already ignores them, but `git add -f` bypasses it, so the pre-commit hook and CI run
this check on the staged or tracked paths. Reviewed, licensed test audio is allowed only under
tests/golden/ and only when tests/golden/PROVENANCE.yaml lists the file with a license.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".wma"}
)
WEIGHT_SUFFIXES = frozenset(
    {".tflite", ".h5", ".keras", ".ckpt", ".pt", ".pth", ".onnx", ".safetensors", ".npy", ".npz"}
)
DATA_ALLOWED = frozenset({"data/README.md", "data/manifest.schema.json"})
GOLDEN_DIR = "tests/golden/"
PROVENANCE = "tests/golden/PROVENANCE.yaml"
UNKNOWN_LICENSES = frozenset({"", "unknown", "unclear", "tbd", "todo", "none", "n/a", "?"})


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _golden_licenses(root: Path) -> dict[str, str]:
    provenance = root / PROVENANCE
    if not provenance.is_file():
        return {}
    data = yaml.safe_load(provenance.read_text(encoding="utf-8")) or {}
    files = data.get("files", {}) if isinstance(data, dict) else {}
    return {
        str(name): str(info.get("license", ""))
        for name, info in files.items()
        if isinstance(info, dict)
    }


def check_paths(paths: list[str], root: Path) -> list[Violation]:
    violations: list[Violation] = []
    golden = _golden_licenses(root)
    for raw in paths:
        path = raw.replace("\\", "/")
        pure = PurePosixPath(path)
        suffix = pure.suffix.lower()
        parts = {p.lower() for p in pure.parts}
        name = pure.name.lower()

        if path.startswith(GOLDEN_DIR) and path != PROVENANCE and suffix != ".md":
            relative = path[len(GOLDEN_DIR) :]
            license_id = golden.get(relative, "").strip().lower()
            if relative not in golden:
                violations.append(Violation(path, f"not listed in {PROVENANCE}"))
            elif license_id in UNKNOWN_LICENSES:
                violations.append(Violation(path, f"unknown license in {PROVENANCE}"))
            continue

        if suffix in AUDIO_SUFFIXES:
            violations.append(Violation(path, "audio files are never committed"))
        elif suffix in WEIGHT_SUFFIXES or path.endswith(("_mmap", ".mmap")) or "_mmap/" in path:
            violations.append(Violation(path, "model weights, checkpoints and features stay out"))
        elif "consent" in parts or ".consent." in name or name.endswith(".consent.json"):
            violations.append(Violation(path, "consent records stay local"))
        elif name.startswith(("speakers", "identities", "identity")) and suffix in {
            ".csv",
            ".json",
            ".jsonl",
            ".txt",
            ".yaml",
            ".yml",
        }:
            violations.append(Violation(path, "speaker identity files stay local"))
        elif path.startswith("data/") and path not in DATA_ALLOWED:
            violations.append(Violation(path, "data/ holds only its README and the schema"))
    return violations


def _git_paths(root: Path, args: list[str]) -> list[str]:
    out = subprocess.run(  # noqa: S603
        ["git", *args, "-z"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


def staged_paths(root: Path) -> list[str]:
    return _git_paths(root, ["diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def tracked_paths(root: Path) -> list[str]:
    return _git_paths(root, ["ls-files"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mobyww-guard", description="Privacy guard for git.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="check staged paths (pre-commit)")
    group.add_argument("--tracked", action="store_true", help="check every tracked path (CI)")
    group.add_argument("--paths", nargs="+", help="check explicit paths")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root: Path = args.root
    if args.staged:
        paths = staged_paths(root)
    elif args.tracked:
        paths = tracked_paths(root)
    else:
        paths = list(args.paths)
    violations = check_paths(paths, root)
    for violation in violations:
        print(f"blocked: {violation}", file=sys.stderr)
    if violations:
        print(
            "The privacy guard blocked this. Recordings, consent files, speaker identities, "
            "weights and unlicensed data must not be committed (docs/privacy.md).",
            file=sys.stderr,
        )
        return 1
    print(f"privacy guard: {len(paths)} paths ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
