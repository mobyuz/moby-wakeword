"""The Colab notebook must never take or upload anything private, and must say it is untested."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "scripts" / "colab"
NOTEBOOK = COLAB / "moby_wakeword_colab.py"

# Anything that moves a file from the user's machine or Drive into the runtime.
UPLOAD_PATTERNS = (
    r"files\.upload",
    r"drive\.mount",
    r"google\.colab\.drive",
    r"\bupload\(",
    r"\bUploadButton\b",
    r"\bfiles\.upload_file",
)
PRIVATE_LITERALS = ("data/recordings", "data\\recordings", "data/consent", "data\\consent")


def load_make_ipynb() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_ipynb", COLAB / "make_ipynb.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells() -> list[tuple[str, str]]:
    return load_make_ipynb().split_cells(NOTEBOOK.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def colab_sources() -> list[Path]:
    return sorted(COLAB.glob("*.py"))


def test_notebook_says_untested_in_its_header_and_in_the_docs() -> None:
    header = "\n".join(NOTEBOOK.read_text(encoding="utf-8").splitlines()[:12])
    assert "UNTESTED" in header
    assert "No Colab run has happened" in header
    docs = (ROOT / "docs" / "upstream-toolchain.md").read_text(encoding="utf-8")
    assert "UNTESTED" in docs
    assert "moby_wakeword_colab" in docs


def test_no_script_uploads_or_mounts_anything() -> None:
    for path in colab_sources():
        text = path.read_text(encoding="utf-8")
        for pattern in UPLOAD_PATTERNS:
            assert not re.search(pattern, text), f"{path.name}: {pattern}"


def test_no_script_names_the_private_data_paths() -> None:
    for path in colab_sources():
        text = path.read_text(encoding="utf-8").lower()
        for literal in PRIVATE_LITERALS:
            assert literal not in text, f"{path.name}: {literal}"


def test_every_cell_is_valid_python() -> None:
    for kind, text in cells():
        if kind == "code":
            ast.parse(text)


def guard_namespace() -> dict[str, Any]:
    guard = [text for kind, text in cells() if kind == "code" and text.startswith("# GUARD CELL")]
    assert len(guard) == 1
    namespace: dict[str, Any] = {
        "WORK_DIR": "/content/moby-ww",
        "REPO_URL": "u",
        "REPO_REF": "main",
    }
    exec(compile(guard[0], "guard-cell", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    "path",
    [
        "data/recordings/spk-1/a.wav",
        "/content/data/consent/spk-1.json",
        "C:/Users/x/Recordings/a.wav",
        "somewhere/CONSENT/a.json",
    ],
)
def test_guard_cell_refuses_private_paths(path: str) -> None:
    refuse = guard_namespace()["refuse_private_path"]
    with pytest.raises(PermissionError, match="private"):
        refuse(path)


def test_guard_cell_accepts_ordinary_paths_and_runs_on_colab_only() -> None:
    namespace = guard_namespace()
    assert namespace["refuse_private_path"]("/content/moby-ww/data/synthetic/a.wav").name == "a.wav"
    assert namespace["FORBIDDEN_DIR_NAMES"] == ("recordings", "consent")
    with pytest.raises(RuntimeError, match="Colab"):
        namespace["assert_colab"]()  # this machine is not Colab


def test_guard_cell_runs_before_anything_that_touches_files() -> None:
    names = [text for kind, text in cells() if kind == "code"]
    guard_index = next(i for i, text in enumerate(names) if text.startswith("# GUARD CELL"))
    for text in names[:guard_index]:
        assert "subprocess" not in text
        assert "open(" not in text
        assert "shutil" not in text


def test_notebook_converts_to_a_valid_ipynb_without_outputs(tmp_path: Path) -> None:
    module = load_make_ipynb()
    notebook = module.to_notebook(NOTEBOOK.read_text(encoding="utf-8"))
    target = tmp_path / "n.ipynb"
    target.write_text(json.dumps(notebook), encoding="utf-8")
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["nbformat"] == 4
    assert loaded["cells"][0]["cell_type"] == "markdown"
    assert "UNTESTED" in "".join(loaded["cells"][0]["source"])
    assert all(c.get("outputs") == [] for c in loaded["cells"] if c["cell_type"] == "code")
    assert loaded["metadata"]["accelerator"] == "GPU"
    assert "UNTESTED" in loaded["metadata"]["colab"]["name"]


def test_notebook_pins_the_upstream_and_generator_commits() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
    commit = re.search(r'PINNED_UPSTREAM_COMMIT = "([0-9a-f]{40})"', text)
    assert commit
    assert f"micro-wake-word.git@{commit.group(1)}" in requirements
    assert re.search(
        r'PINNED_PIPER_GENERATOR_COMMIT = "2971426a55072f7d22fec416ca7800df8bd23207"', text
    )
    assert 'PIPER_TTS_VERSION = "1.3.0"' in text


def test_notebook_does_not_measure_or_claim_accuracy() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8").lower()
    assert "no model quality is claimed" in text
    assert "accuracy_measured" not in text or "false" in text
