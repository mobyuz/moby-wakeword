from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mobyww import guard

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")


def blocked(paths: list[str], root: Path = ROOT) -> list[str]:
    return [v.path for v in guard.check_paths(paths, root)]


def test_blocks_audio_weights_consent_and_identities() -> None:
    bad = [
        "data/recordings/spk-1/a.wav",
        "somewhere/clip.FLAC",
        "model.tflite",
        "runs/x/best.ckpt",
        "weights/net.safetensors",
        "features/train_mmap/data.npy",
        "data/consent/spk-1.json",
        "notes/spk-1.consent.json",
        "speakers.csv",
        "docs/identities.yaml",
        "data/manifests/recorded.jsonl",
        "data/hidden.txt",
    ]
    assert blocked(bad) == bad


def test_allows_normal_files() -> None:
    ok = [
        "README.md",
        "data/README.md",
        "data/manifest.schema.json",
        "src/mobyww/guard.py",
        "docs/privacy.md",
        "configs/wakeword.yaml",
        "tests/golden/PROVENANCE.yaml",
    ]
    assert blocked(ok) == []


def test_golden_audio_needs_provenance_with_license(tmp_path: Path) -> None:
    golden = tmp_path / "tests" / "golden"
    golden.mkdir(parents=True)
    path = "tests/golden/ours.wav"
    assert blocked([path], tmp_path) == [path]  # no provenance file at all

    (golden / "PROVENANCE.yaml").write_text(
        "files:\n  ours.wav: {license: unknown, origin: x}\n", encoding="utf-8"
    )
    assert blocked([path], tmp_path) == [path]  # unknown license

    (golden / "PROVENANCE.yaml").write_text(
        "files:\n  ours.wav: {license: CC0-1.0, origin: synthetic}\n", encoding="utf-8"
    )
    assert blocked([path], tmp_path) == []
    assert blocked(["tests/golden/other.wav"], tmp_path) == ["tests/golden/other.wav"]


def test_committed_tree_is_clean() -> None:
    if GIT is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert guard.check_paths(guard.tracked_paths(ROOT), ROOT) == []


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert GIT is not None
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    return subprocess.run(  # noqa: S603
        [GIT, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture
def hooked_repo(tmp_path: Path) -> Path:
    if GIT is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-b", "main").returncode == 0
    hooks = repo / ".githooks"
    hooks.mkdir()
    shutil.copy(ROOT / ".githooks" / "pre-commit", hooks / "pre-commit")
    (repo / ".gitignore").write_text("*.wav\n", encoding="utf-8")
    assert _git(repo, "config", "core.hooksPath", ".githooks").returncode == 0
    return repo


def _hook_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOBYWW_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))


def test_hook_blocks_force_added_recording_and_allows_clean_commit(
    hooked_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hook_env(monkeypatch)
    (hooked_repo / "clip.wav").write_bytes(b"RIFF")
    assert _git(hooked_repo, "add", "clip.wav").returncode != 0  # .gitignore blocks a plain add
    assert _git(hooked_repo, "add", "-f", "clip.wav").returncode == 0  # -f bypasses .gitignore
    blocked_commit = _git(hooked_repo, "commit", "-m", "add recording")
    assert blocked_commit.returncode != 0
    assert "blocked" in blocked_commit.stderr
    assert "clip.wav" in blocked_commit.stderr
    assert _git(hooked_repo, "rev-parse", "HEAD").returncode != 0  # no commit was created

    assert _git(hooked_repo, "rm", "--cached", "clip.wav").returncode == 0
    (hooked_repo / "notes.md").write_text("ok\n", encoding="utf-8")
    assert _git(hooked_repo, "add", "notes.md").returncode == 0
    assert _git(hooked_repo, "commit", "-m", "notes").returncode == 0


def test_hook_blocks_consent_file(hooked_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hook_env(monkeypatch)
    consent = hooked_repo / "data" / "consent"
    consent.mkdir(parents=True)
    (consent / "spk-1.json").write_text("{}", encoding="utf-8")
    assert _git(hooked_repo, "add", "-f", "data/consent/spk-1.json").returncode == 0
    result = _git(hooked_repo, "commit", "-m", "consent")
    assert result.returncode != 0
    assert "consent records stay local" in result.stderr


def test_real_gitignore_ignores_private_paths() -> None:
    if GIT is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    for path in (
        "data/recordings/a/b.wav",
        "data/consent/x.json",
        "model.tflite",
        ".venv/x.py",
        "speakers.csv",
    ):
        assert _git(ROOT, "check-ignore", "-q", path).returncode == 0, path
    assert _git(ROOT, "check-ignore", "-q", "data/README.md").returncode == 1
    assert _git(ROOT, "check-ignore", "-q", "data/manifest.schema.json").returncode == 1
