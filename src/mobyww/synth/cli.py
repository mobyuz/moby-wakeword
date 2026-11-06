"""Command line: write a job file (`plan`) and turn generated WAVs into a manifest (`manifest`).

    python -m mobyww.synth plan --out build/synth_job.yaml
    python -m mobyww.synth manifest --job build/synth_job.yaml --wav-dir data/synthetic/clips \
        --data-root data --out data/manifests/synthetic.jsonl

Both are cheap and need no model, no GPU and no network.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from mobyww.data.manifest import read_manifest
from mobyww.data.validate import validate
from mobyww.synth.manifest import build_manifest, write_synthetic_manifest
from mobyww.synth.plan import PlanParams, build_plan
from mobyww.synth.spec import JobFileError, job_file_sha256, load_job_file, save_job_file

DEFAULT_CONFIG = Path("configs/wakeword.yaml")


def _config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise JobFileError(f"{path}: not a mapping")
    return data


def _plan(args: argparse.Namespace) -> int:
    config = _config(args.config)
    params = PlanParams(seed=int(config.get("seeds", {}).get("synthesis", PlanParams.seed)))
    if args.seed is not None:
        params = replace(params, seed=args.seed)
    if args.en_speakers is not None:
        params = replace(params, libritts_speakers=args.en_speakers)
    job_file = build_plan([str(p) for p in config["hard_negatives"]], params)
    save_job_file(args.out, job_file)
    print(
        f"{args.out}: {len(job_file.jobs)} jobs, {job_file.total_clips} clips, "
        f"seed {job_file.seed}, sha256 {job_file_sha256(job_file)}"
    )
    return 0


def _manifest(args: argparse.Namespace) -> int:
    job_file = load_job_file(args.job)
    split_seed = int(_config(args.config).get("seeds", {}).get("split", 1337))
    result = build_manifest(
        job_file, args.wav_dir, args.data_root, split_seed=split_seed, test_pct=args.test_pct
    )
    for label, items in (
        ("missing", result.missing),
        ("unexpected", result.unexpected),
        ("bad format", result.bad_format),
    ):
        if items:
            print(f"{label}: {len(items)} (first: {items[0]})", file=sys.stderr)
    if not result.rows:
        print("error: no clips found", file=sys.stderr)
        return 1
    if args.require_all and not result.complete:
        print("error: --require-all and the folder does not match the job file", file=sys.stderr)
        return 1
    write_synthetic_manifest(args.out, result)
    report = validate(read_manifest(args.out), args.data_root)
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    print(f"{args.out}: {len(result.rows)} rows, {len(report.errors)} validator errors")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mobyww.synth", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write the synthesis job file for Colab")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--seed", type=int)
    plan.add_argument("--en-speakers", type=int, help="LibriTTS-R speakers to use (default 100)")
    plan.set_defaults(func=_plan)

    manifest = sub.add_parser("manifest", help="manifest rows from a folder of generated WAVs")
    manifest.add_argument("--job", type=Path, required=True)
    manifest.add_argument("--wav-dir", type=Path, required=True)
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    manifest.add_argument("--test-pct", type=int, default=0)
    manifest.add_argument("--require-all", action="store_true")
    manifest.set_defaults(func=_manifest)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (JobFileError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
