"""Download licence-approved open data into a cache OUTSIDE the repository.

Only sets that docs/licenses.md marks "allowed" are in the registry; a "candidate" entry is
refused. Downloads resume (HTTP Range), are verified (size always; MD5 or SHA-256 when known) and
become visible under their final name only after the check. When no checksum is published, the
SHA-256 of the first complete download is recorded in `SHA256SUMS` next to the files and later
runs compare against it (trust on first use): that detects a changed or damaged file, not a file
that was wrong the first time.

    python scripts/fetch_open_data.py --dry-run
    python scripts/fetch_open_data.py                  # BUT ReverbDB RIR set (9.3 GB) + its read_me
    python scripts/fetch_open_data.py --musan          # also MUSAN (11 GB)

MUSAN is never downloaded unless `--musan` is passed. Common Voice is never downloaded here: the
Mozilla Data Collective needs the owner's account and an accepted agreement (`--common-voice-help`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DEST = Path(os.environ.get("MOBYWW_DATA_CACHE", r"D:\moby-cache\wakeword-data"))
CHUNK = 1 << 20
USER_AGENT = "moby-wakeword-fetch/0.1 (+https://github.com/mobyuz/moby-wakeword)"
PRIVATE_NAMES = frozenset({"recordings", "consent"})
REPO_ROOT = Path(__file__).resolve().parents[3]

COMMON_VOICE_HELP = """\
Common Voice Uzbek (status: candidate in docs/licenses.md; NOT downloaded by this script).
The Mozilla Data Collective answers 403 "Access to this dataset has not been granted" until the
account holder accepts the dataset terms. The owner must do this by hand:
  1. Open https://mozilladatacollective.com/datasets/cmn29fa4y0150o1073fqww73p in a browser
     (Common Voice Scripted Speech 25.0 - Uzbek), sign in or sign up, and READ the terms.
  2. If you accept them (they reportedly forbid identifying speakers and re-hosting the data),
     download the archive yourself into D:\\moby-cache\\wakeword-data\\common-voice-uz\\.
  3. Tell the agent; the licence row in docs/licenses.md changes to "allowed" only after the
     terms text was actually read. Never upload the archive to Colab or any other host."""


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    url: str
    filename: str
    size_bytes: int
    license: str
    status: str  # as in docs/licenses.md: allowed / candidate / stopped
    md5: str | None = None
    sha256: str | None = None
    default: bool = False
    note: str = ""


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="reverbdb_readme",
        title="BUT Speech@FIT Reverb Database, read_me.txt",
        url="https://merlin.fit.vutbr.cz/ReverbDB/read_me.txt",
        filename="but_reverbdb/read_me.txt",
        size_bytes=14_529,
        sha256="4f62b0015e5c61a9f34672430c9f3d96a210bae1530c9e29fa325c6e6969b9fa",
        license="CC-BY-4.0",
        status="allowed",
        default=True,
        note="describes the layout; kept with the data for attribution",
    ),
    Dataset(
        key="reverbdb_rir",
        title="BUT Speech@FIT Reverb Database, RIR-Only (RIRs, room noise, metadata)",
        url="https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
        filename="but_reverbdb/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
        size_bytes=9_308_593_693,
        license="CC-BY-4.0",
        status="allowed",
        default=True,
        note="no checksum is published: SHA-256 recorded on first download",
    ),
    Dataset(
        key="musan",
        title="MUSAN (music, speech, noise), OpenSLR 17",
        url="https://www.openslr.org/resources/17/musan.tar.gz",
        filename="musan/musan.tar.gz",
        size_bytes=11_086_114_085,
        md5="0c472d4fc0c5141eca47ad1ffeb2a7df",
        license="CC-BY-4.0",
        status="allowed",
        default=False,
        note="only with --musan; per-file LICENSE files inside the archive are unread",
    ),
)
BY_KEY = {d.key: d for d in DATASETS}


class FetchError(RuntimeError):
    """A download or check failed."""


def check_destination(dest: Path) -> Path:
    resolved = dest.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise FetchError(f"{resolved} is inside the repository; downloads never go there")
    hit = {p.lower() for p in resolved.parts} & PRIVATE_NAMES
    if hit:
        raise FetchError(f"{resolved}: path names a private folder {sorted(hit)}")
    return resolved


def selected(args_musan: bool, args_reverbdb: bool) -> list[Dataset]:
    chosen = [d for d in DATASETS if d.default and (args_reverbdb or d.key == "reverbdb_readme")]
    if args_musan:
        chosen.append(BY_KEY["musan"])
    return chosen


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sums(dest: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    path = dest / "SHA256SUMS"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, name = line.partition("  ")
                sums[name.strip()] = digest.strip()
    return sums


def record_sum(dest: Path, name: str, digest: str) -> None:
    sums = read_sums(dest)
    sums[name] = digest
    with (dest / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(sums):
            handle.write(f"{sums[key]}  {key}\n")


def verify(dataset: Dataset, path: Path, dest: Path) -> str:
    """Check size and checksums; returns a one-line description of what was checked."""
    size = path.stat().st_size
    if size != dataset.size_bytes:
        raise FetchError(f"{dataset.key}: size {size} != expected {dataset.size_bytes}")
    notes = [f"size {size}"]
    if dataset.md5:
        actual_md5 = hash_file(path, "md5")
        if actual_md5 != dataset.md5:
            raise FetchError(f"{dataset.key}: md5 {actual_md5} != published {dataset.md5}")
        notes.append("md5 matches the published one")
    sha = hash_file(path, "sha256")
    if dataset.sha256:
        if sha != dataset.sha256:
            raise FetchError(f"{dataset.key}: sha256 {sha} != pinned {dataset.sha256}")
        notes.append("sha256 matches the pinned one")
    else:
        recorded = read_sums(dest).get(dataset.filename)
        if recorded is None:
            record_sum(dest, dataset.filename, sha)
            notes.append(f"sha256 {sha} recorded (first download, not independently known)")
        elif recorded != sha:
            raise FetchError(f"{dataset.key}: sha256 {sha} != recorded {recorded}")
        else:
            notes.append("sha256 matches the one recorded on first download")
    return ", ".join(notes)


def download(dataset: Dataset, dest: Path, *, opener: Callable[..., Any] | None = None) -> Path:
    """Download (resuming a `.part` file), verify, then rename. Returns the final path."""
    if dataset.status != "allowed":
        raise FetchError(f"{dataset.key} is '{dataset.status}' in docs/licenses.md: not downloaded")
    target = dest / dataset.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print(f"{dataset.key}: present, {verify(dataset, target, dest)}")
        return target
    part = target.with_name(target.name + ".part")
    need = dataset.size_bytes - (part.stat().st_size if part.exists() else 0)
    free = shutil.disk_usage(target.parent).free
    if free < need + (1 << 30):
        raise FetchError(f"{dataset.key}: needs {need} bytes, only {free} free in {target.parent}")

    open_url = opener or urllib.request.urlopen
    for attempt in range(1, 4):
        offset = part.stat().st_size if part.exists() else 0
        if offset > dataset.size_bytes:
            part.unlink()
            offset = 0
        if offset == dataset.size_bytes:
            break
        request = urllib.request.Request(  # noqa: S310 (https only, registry URLs)
            dataset.url, headers={"User-Agent": USER_AGENT}
        )
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with open_url(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if offset and status != 206:
                    print(f"{dataset.key}: server ignored Range, restarting")
                    offset = 0
                mode = "ab" if offset else "wb"
                with part.open(mode) as out:
                    while block := response.read(CHUNK):
                        out.write(block)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"{dataset.key}: attempt {attempt} failed: {exc}", file=sys.stderr)
            continue
        break
    if not part.exists() or part.stat().st_size != dataset.size_bytes:
        got = part.stat().st_size if part.exists() else 0
        raise FetchError(
            f"{dataset.key}: incomplete ({got}/{dataset.size_bytes}); run again to resume"
        )
    try:
        note = verify(dataset, part, dest)
    except FetchError:
        part.replace(part.with_name(part.name + ".bad"))
        raise
    part.replace(target)
    print(f"{dataset.key}: downloaded, {note}")
    return target


def extract(
    archive: Path, dest: Path, wanted: Callable[[str], bool], limit: int | None = None
) -> int:
    """Stream-extract the members `wanted(name)` accepts (Python's safe 'data' filter)."""
    count = 0
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not wanted(member.name):
                continue
            tar.extract(member, dest, filter="data")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def is_rir_wav(name: str) -> bool:
    parts = name.split("/")
    return name.lower().endswith(".wav") and "RIR" in parts[:-1]


MUSAN_SUBSETS = ("noise", "speech", "music")


def musan_member_filter(subsets: Sequence[str]) -> Callable[[str], bool]:
    """Accept MUSAN files of the given subsets, plus the LICENSE, README and ANNOTATIONS files.

    The per-directory LICENSE files carry the file-by-file attribution, so they are always taken.
    """
    unknown = sorted(set(subsets) - set(MUSAN_SUBSETS))
    if unknown:
        raise FetchError(f"unknown MUSAN subsets {unknown}; choose from {MUSAN_SUBSETS}")
    wanted = tuple(f"/{s}/" for s in subsets)

    def accept(name: str) -> bool:
        low = name.lower()
        in_subset = any(w in low for w in wanted)
        return in_subset and low.endswith((".wav", "license", "readme", "annotations"))

    return accept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"default {DEFAULT_DEST}")
    parser.add_argument("--musan", action="store_true", help="also download MUSAN (11 GB)")
    parser.add_argument(
        "--no-reverbdb", action="store_true", help="skip the BUT ReverbDB RIR archive (9.3 GB)"
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    parser.add_argument(
        "--extract-rir", action="store_true", help="after download, extract only RIR/*.wav"
    )
    parser.add_argument(
        "--extract-musan",
        metavar="SUBSETS",
        help="after download, extract these MUSAN subsets (comma list of noise,speech,music) "
        "with their LICENSE, README and ANNOTATIONS files",
    )
    parser.add_argument("--common-voice-help", action="store_true")
    args = parser.parse_args(argv)

    if args.common_voice_help:
        print(COMMON_VOICE_HELP)
        return 0
    try:
        dest = check_destination(args.dest)
        plan = selected(args.musan, not args.no_reverbdb)
        for dataset in plan:
            gib = dataset.size_bytes / (1 << 30)
            print(
                f"{dataset.key}: {dataset.title}, {gib:.2f} GiB, {dataset.license}, {dataset.url}"
            )
        if args.dry_run:
            print(f"dry run: nothing downloaded; destination would be {dest}")
            return 0
        dest.mkdir(parents=True, exist_ok=True)
        for dataset in plan:
            path = download(dataset, dest)
            if dataset.key == "reverbdb_rir" and args.extract_rir:
                n = extract(path, dest / "but_reverbdb" / "rir", is_rir_wav)
                print(f"reverbdb_rir: extracted {n} RIR files")
            if dataset.key == "musan" and args.extract_musan:
                accept = musan_member_filter(args.extract_musan.split(","))
                n = extract(path, dest / "musan" / "extracted", accept)
                print(f"musan: extracted {n} files")
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
