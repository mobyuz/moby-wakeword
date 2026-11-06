from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from mobyww.data import fetch

PAYLOAD = bytes(range(256)) * 400  # 102 400 bytes


def dataset(**overrides: Any) -> fetch.Dataset:
    values: dict[str, Any] = {
        "key": "t",
        "title": "test",
        "url": "https://example.invalid/t.bin",
        "filename": "t/t.bin",
        "size_bytes": len(PAYLOAD),
        "license": "CC0-1.0",
        "status": "allowed",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }
    values.update(overrides)
    return fetch.Dataset(**values)


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class Server:
    """A fake urlopen that honours (or ignores) Range and records the requests."""

    def __init__(self, payload: bytes = PAYLOAD, *, honour_range: bool = True) -> None:
        self.payload, self.honour_range = payload, honour_range
        self.ranges: list[str | None] = []

    def __call__(self, request: urllib.request.Request, timeout: float = 0) -> Response:
        header = request.get_header("Range")
        self.ranges.append(header)
        if header and self.honour_range:
            start = int(header.removeprefix("bytes=").removesuffix("-"))
            return Response(self.payload[start:], 206)
        return Response(self.payload, 200)


def test_download_verifies_and_renames(tmp_path: Path) -> None:
    server = Server()
    path = fetch.download(dataset(), tmp_path, opener=server)
    assert path.read_bytes() == PAYLOAD
    assert not path.with_name(path.name + ".part").exists()
    assert server.ranges == [None]
    again = Server()
    fetch.download(dataset(), tmp_path, opener=again)  # present: nothing requested
    assert again.ranges == []


def test_download_resumes_from_the_part_file(tmp_path: Path) -> None:
    target = tmp_path / "t" / "t.bin"
    target.parent.mkdir(parents=True)
    target.with_name("t.bin.part").write_bytes(PAYLOAD[:50_000])
    server = Server()
    fetch.download(dataset(), tmp_path, opener=server)
    assert server.ranges == ["bytes=50000-"]
    assert target.read_bytes() == PAYLOAD


def test_download_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    target = tmp_path / "t" / "t.bin"
    target.parent.mkdir(parents=True)
    target.with_name("t.bin.part").write_bytes(PAYLOAD[:10_000])
    fetch.download(dataset(), tmp_path, opener=Server(honour_range=False))
    assert target.read_bytes() == PAYLOAD


def test_wrong_checksum_is_quarantined_not_installed(tmp_path: Path) -> None:
    tampered = PAYLOAD[:-1] + b"\x00"
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.download(dataset(), tmp_path, opener=Server(tampered))
    assert not (tmp_path / "t" / "t.bin").exists()
    assert (tmp_path / "t" / "t.bin.part.bad").exists()


def test_md5_is_checked_when_published(tmp_path: Path) -> None:
    good = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()
    fetch.download(dataset(sha256=None, md5=good), tmp_path, opener=Server())
    with pytest.raises(fetch.FetchError, match="md5"):
        fetch.download(
            dataset(sha256=None, md5="0" * 32, filename="t/other.bin"), tmp_path, opener=Server()
        )


def test_incomplete_download_is_reported_and_resumable(tmp_path: Path) -> None:
    with pytest.raises(fetch.FetchError, match="incomplete"):
        fetch.download(dataset(), tmp_path, opener=Server(PAYLOAD[:1000]))
    assert (tmp_path / "t" / "t.bin.part").stat().st_size == 1000
    fetch.download(dataset(), tmp_path, opener=Server())
    assert (tmp_path / "t" / "t.bin").read_bytes() == PAYLOAD


def test_trust_on_first_use_records_then_detects_change(tmp_path: Path) -> None:
    ds = dataset(sha256=None)
    path = fetch.download(ds, tmp_path, opener=Server())
    sums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums == f"{hashlib.sha256(PAYLOAD).hexdigest()}  t/t.bin\n"
    path.write_bytes(PAYLOAD[:-1] + b"\x01")  # same size, different bytes
    with pytest.raises(fetch.FetchError, match="recorded"):
        fetch.download(ds, tmp_path, opener=Server())


def test_only_allowed_sets_download(tmp_path: Path) -> None:
    for status in ("candidate", "stopped"):
        with pytest.raises(fetch.FetchError, match="not downloaded"):
            fetch.download(dataset(status=status), tmp_path, opener=Server())
    assert all(d.status == "allowed" for d in fetch.DATASETS)


def test_musan_needs_its_flag() -> None:
    default = {d.key for d in fetch.selected(args_musan=False, args_reverbdb=True)}
    assert "musan" not in default
    assert default == {"reverbdb_readme", "reverbdb_rir"}
    assert "musan" in {d.key for d in fetch.selected(args_musan=True, args_reverbdb=True)}
    assert {d.key for d in fetch.selected(args_musan=False, args_reverbdb=False)} == {
        "reverbdb_readme"
    }
    assert not fetch.BY_KEY["musan"].default


def test_registry_matches_what_was_read_from_the_official_pages() -> None:
    musan = fetch.BY_KEY["musan"]
    assert musan.url == "https://www.openslr.org/resources/17/musan.tar.gz"
    assert musan.md5 == "0c472d4fc0c5141eca47ad1ffeb2a7df"  # OpenSLR checksum.md5
    assert musan.size_bytes == 11_086_114_085  # Content-Length of the file
    assert fetch.BY_KEY["reverbdb_rir"].size_bytes == 9_308_593_693
    assert fetch.BY_KEY["reverbdb_rir"].url.endswith("BUT_ReverbDB_rel_19_06_RIR-Only.tgz")
    assert all(d.url.startswith("https://") for d in fetch.DATASETS)


def test_dry_run_prints_and_downloads_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = fetch.main(["--dest", str(tmp_path / "cache"), "--dry-run", "--musan"])
    out = capsys.readouterr().out
    assert code == 0
    assert "musan" in out
    assert "reverbdb_rir" in out
    assert "dry run" in out
    assert not (tmp_path / "cache").exists()


def test_destination_is_never_inside_the_repo_or_a_private_folder(tmp_path: Path) -> None:
    with pytest.raises(fetch.FetchError, match="inside the repository"):
        fetch.check_destination(fetch.REPO_ROOT / "data" / "open")
    with pytest.raises(fetch.FetchError, match="private"):
        fetch.check_destination(tmp_path / "Recordings")
    assert fetch.check_destination(tmp_path / "ok") == (tmp_path / "ok").resolve()


def test_common_voice_is_instructions_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.main(["--common-voice-help"]) == 0
    out = capsys.readouterr().out
    assert "403" in out
    assert "accept" in out
    assert not any("common" in d.key for d in fetch.DATASETS)


def make_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_rir_only_takes_rir_wavs(tmp_path: Path) -> None:
    archive = tmp_path / "rir.tgz"
    make_archive(
        archive,
        {
            "VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/RIR/a.v00.wav": b"rir",
            "VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/silence/b.v00.wav": b"noise",
            "VUT_FIT_L207/photo.jpg": b"jpg",
        },
    )
    count = fetch.extract(archive, tmp_path / "out", fetch.is_rir_wav)
    assert count == 1
    found = [p.name for p in (tmp_path / "out").rglob("*") if p.is_file()]
    assert found == ["a.v00.wav"]


def test_extract_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tgz"
    make_archive(archive, {"../RIR/evil.wav": b"x"})
    with pytest.raises(tarfile.FilterError):
        fetch.extract(archive, tmp_path / "out", fetch.is_rir_wav)
    assert not (tmp_path / "RIR").exists()


def test_musan_member_filter() -> None:
    assert fetch.is_musan_noise_or_licence("musan/noise/free-sound/noise-free-sound-0000.wav")
    assert fetch.is_musan_noise_or_licence("musan/noise/free-sound/LICENSE")
    assert fetch.is_musan_noise_or_licence("musan/music/jamendo/ANNOTATIONS")
    assert not fetch.is_musan_noise_or_licence("musan/music/jamendo/music-jamendo-0000.wav")
