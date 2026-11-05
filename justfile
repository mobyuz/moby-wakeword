set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

py := if os() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

train_py := if os() == "windows" { ".venv-train/Scripts/python.exe" } else { ".venv-train/bin/python" }

setup:
    python -m venv .venv
    {{py}} -m pip install -r requirements-dev.txt
    {{py}} -m pip install -e . --no-deps
    git config core.hooksPath .githooks

lint:
    {{py}} -m ruff check .
    {{py}} -m ruff format --check .
    {{py}} -m mypy

test:
    {{py}} -m pytest

validate-sample:
    {{py}} -m mobyww.data.sample

guard:
    {{py}} -m mobyww.guard --tracked

check: lint test validate-sample guard

# Needs the .venv-train environment, see docs/upstream-toolchain.md. Slow on a cold start.
upstream-smoke:
    {{train_py}} -u scripts/upstream_smoke.py --workdir build/smoke --steps 40
