"""Convert the percent-format notebook source (`# %%` cells) into an .ipynb for Colab.

    python scripts/colab/make_ipynb.py SOURCE.py TARGET.ipynb
    (SOURCE is scripts/colab/moby_wakeword_colab.py)

The notebook carries no outputs. Upload the .ipynb to Colab yourself (it contains no data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARK = "# %%"


def split_cells(source: str) -> list[tuple[str, str]]:
    """[(kind, text)] with kind 'markdown' or 'code'."""
    cells: list[tuple[str, list[str]]] = []
    for line in source.splitlines():
        if line.startswith(MARK):
            kind = "markdown" if "[markdown]" in line else "code"
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            raise ValueError("text before the first '# %%' marker")
    out: list[tuple[str, str]] = []
    for kind, lines in cells:
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.removeprefix("#") for ln in lines]
        text = "\n".join(lines).strip("\n")
        if text.strip():
            out.append((kind, text))
    return out


def to_notebook(source: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for kind, text in split_cells(source):
        lines = text.split("\n")
        body = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell: dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": body}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "moby_wakeword_colab (UNTESTED)", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    notebook = to_notebook(args.source.read_text(encoding="utf-8"))
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{args.target}: {len(notebook['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
