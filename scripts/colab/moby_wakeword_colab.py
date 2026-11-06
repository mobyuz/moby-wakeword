# %% [markdown]
# # Moby "hey moby" wake word, W2 data + training run (Google Colab)
#
# **STATUS: UNTESTED.** No Colab run has happened. Nothing below has been executed on Colab, on a
# GPU, or with real speech. Locally (Windows, CPU) only these pieces were run: the augmentation and
# device path, the job planner and manifest writer, the open-data fetcher (resume + checksum on a
# 14 KB file), and the feature/training wrapper (`mww_steps.py`) on the W0 synthetic tones. The TTS
# step (Piper), the MUSAN and BUT ReverbDB downloads of full size, the GPU setup and the whole
# end-to-end flow are unverified. Expect to fix things on the first run.
#
# **No model quality is claimed.** This notebook measures no accuracy. The owner's own real
# recordings (about 20 min) stay LOCAL, are the test set, and are never uploaded here: this
# notebook has no upload or Drive step, takes no file from `recordings` or `consent` folders, and
# refuses any such path (see the guard cell and tests/test_colab_guard.py).
#
# Flow: install pinned stack -> plan jobs -> Piper TTS -> manifest -> open data (MUSAN noise and
# speech, BUT ReverbDB RIR) -> augmentation with the INMP441 device path -> features -> training ->
# `provenance.json` next to the model. Licences: docs/licenses.md and docs/ATTRIBUTION.md.

# %% [markdown]
# ## 0. Parameters (edit here)

# %%
# Parameters. Every path below is checked by the guard cell that follows.
REPO_URL = "https://github.com/mobyuz/moby-wakeword.git"
REPO_REF = "main"  # a branch, tag or commit that exists on GitHub; the commit is recorded
WORK_DIR = "/content/moby-ww"  # everything is created under here and is lost with the runtime
SMOKE = True  # True: tiny counts and 40 training steps, to find breakage cheaply; False: real run
USE_MUSAN = True  # 11 GB download (noise as ambient/noise mix, speech as negatives); CC BY 4.0
USE_RIR = False  # BUT ReverbDB RIR set (9.3 GB, CC BY 4.0); enables reverberation when True
AUGMENT_COPIES = 3  # augmented copies per training clip
EN_SPEAKERS = 100  # LibriTTS-R speakers (of 904)
TRAINING_STEPS = 10_000
REQUIRE_GPU = True
PINNED_UPSTREAM_COMMIT = "4665173cd35f1cff9a61e06fc427f124766c488e"  # OHF-Voice/micro-wake-word
PINNED_PIPER_GENERATOR_COMMIT = "2971426a55072f7d22fec416ca7800df8bd23207"
PIPER_TTS_VERSION = "1.3.0"

# %% [markdown]
# ## 1. Guard (GUARD CELL): no recordings, no consent files, Colab only

# %%
# GUARD CELL: standard library only, so tests/test_colab_guard.py can execute it on its own.
import os
import sys
from pathlib import Path

FORBIDDEN_DIR_NAMES = ("recordings", "consent")


def refuse_private_path(path: str | os.PathLike[str]) -> Path:
    """Return the resolved path, or raise when any part of it names a private folder."""
    candidate = Path(path)
    parts = {p.lower() for p in candidate.parts} | {p.lower() for p in candidate.resolve().parts}
    hit = sorted(parts & set(FORBIDDEN_DIR_NAMES))
    if hit:
        raise PermissionError(f"{path}: refusing a private folder {hit}; it never goes to Colab")
    return candidate.resolve()


def assert_colab() -> None:
    if "google.colab" not in sys.modules and not os.environ.get("COLAB_RELEASE_TAG"):
        raise RuntimeError(
            "Run this notebook on Google Colab only. Training is never done locally."
        )


WORK = refuse_private_path(WORK_DIR)
for _name in ("REPO_URL", "REPO_REF"):
    assert not any(n in globals()[_name].lower() for n in FORBIDDEN_DIR_NAMES), _name

# %% [markdown]
# ## 2. Environment: Colab, GPU, Python version

# %%
import platform
import shutil
import subprocess


def sh(command: list[str], *, cwd: Path | None = None, check: bool = True) -> None:
    """Run a command and stream its output."""
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=check)


assert_colab()
print("python", platform.python_version(), platform.platform())
if sys.version_info[:2] != (3, 12):
    print(
        "WARNING: the pinned stack was only proven on Python 3.12 (docs/upstream-toolchain.md). "
        "Colab currently gives 3.13, which is unproven here; expect version-related breakage."
    )
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
print(gpu.stdout.strip() or "no NVIDIA GPU visible")
if REQUIRE_GPU and not gpu.stdout.strip():
    raise RuntimeError("No GPU. Runtime > Change runtime type > GPU, or set REQUIRE_GPU = False.")
WORK.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(WORK).free / 2**30
print(f"free disk: {free_gb:.0f} GiB (MUSAN + ReverbDB + features need roughly 60 GiB)")

# %% [markdown]
# ## 3. Install the pinned training stack and this repository

# %%
REPO = WORK / "repo"
if not REPO.exists():
    sh(["git", "clone", REPO_URL, str(REPO)])
sh(["git", "checkout", REPO_REF], cwd=REPO)
repo_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
).stdout.strip()
print("repository commit:", repo_commit)

# requirements-train.txt is frozen from a Windows run; unknown Linux/Colab conflicts may appear.
sh([sys.executable, "-m", "pip", "install", "-r", "requirements-train.txt"], cwd=REPO)
# Colab ships Python 3.13 while pyproject.toml pins 3.12, so the version gate is bypassed here only.
sh(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--ignore-requires-python"],
    cwd=REPO,
)

import importlib

importlib.invalidate_caches()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "colab"))

import json
from importlib import metadata
from urllib.parse import urlparse
from urllib.request import url2pathname

# An editable install (-e git+...) keeps no commit in pip's metadata, only the checkout folder as a
# file:// URL, so the commit is the git HEAD of that folder. (Same logic as
# mobyww.train.provenance.commit_from_direct_url, inlined so this cell needs no newer repo code.)
_direct = json.loads(metadata.distribution("microwakeword").read_text("direct_url.json") or "{}")
upstream_commit = _direct.get("vcs_info", {}).get("commit_id")
if not upstream_commit and urlparse(_direct.get("url", "")).scheme == "file":
    _folder = url2pathname(urlparse(_direct["url"]).path)
    _head = subprocess.run(
        ["git", "-C", _folder, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    upstream_commit = _head.stdout.strip() or None
print("microwakeword commit:", upstream_commit)
assert upstream_commit == PINNED_UPSTREAM_COMMIT, "microwakeword is not the pinned commit"

import tensorflow as tf

print("TensorFlow", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))
if REQUIRE_GPU and not tf.config.list_physical_devices("GPU"):
    # The frozen TensorFlow wheel may lack CUDA libraries on Colab. UNVERIFIED fallback:
    # sh([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]==2.20.0"]) then restart.
    raise RuntimeError("TensorFlow sees no GPU; see the comment above.")

# %% [markdown]
# ## 4. Piper in its own virtualenv, at the pinned generator commit

# %%
GEN = WORK / "piper-sample-generator"
if not GEN.exists():
    sh(["git", "clone", "https://github.com/rhasspy/piper-sample-generator", str(GEN)])
sh(["git", "checkout", PINNED_PIPER_GENERATOR_COMMIT], cwd=GEN)
head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=GEN, capture_output=True, text=True, check=True
).stdout.strip()
assert head == PINNED_PIPER_GENERATOR_COMMIT, head
# The generator pins piper-tts==1.3.0 in its pyproject.toml; we use Piper voices (.onnx) directly.
assert f"piper-tts=={PIPER_TTS_VERSION}" in (GEN / "pyproject.toml").read_text(encoding="utf-8")

PIPER_VENV = WORK / "venv-piper"
if not (PIPER_VENV / "bin" / "python").exists():
    shutil.rmtree(PIPER_VENV, ignore_errors=True)  # a half-made venv from a failed run
    # `python3 -m venv` fails on Colab's Ubuntu image (no ensurepip / python3-venv), so virtualenv
    # is used; it ships its own pip and needs no system package.
    sh([sys.executable, "-m", "pip", "install", "virtualenv"])
    sh([sys.executable, "-m", "virtualenv", str(PIPER_VENV)])
PIPER_PY = str(PIPER_VENV / "bin" / "python")
sh([PIPER_PY, "-m", "pip", "install", f"piper-tts=={PIPER_TTS_VERSION}", "PyYAML==6.0.3"])
sh([PIPER_PY, "-m", "pip", "install", "-e", str(REPO), "--no-deps", "--ignore-requires-python"])

# %% [markdown]
# ## 5. Plan the synthesis jobs (deterministic, from configs/wakeword.yaml)

# %%
from mobyww.synth import job_file_sha256, load_job_file

JOB = WORK / "job.json"
plan_args = ["--out", str(JOB), "--config", str(REPO / "configs" / "wakeword.yaml")]
plan_args += ["--smoke"] if SMOKE else ["--en-speakers", str(EN_SPEAKERS)]
sh([sys.executable, "-m", "mobyww.synth", "plan", *plan_args], cwd=REPO)
job_file = load_job_file(JOB)
print("job file sha256:", job_file_sha256(job_file), "clips:", job_file.total_clips)

# %% [markdown]
# ## 6. Generate speech with Piper (22.05 kHz), then resample to 16 kHz

# %%
RAW = WORK / "raw_tts"
sh(
    [
        PIPER_PY,
        str(REPO / "scripts" / "colab" / "synthesize_piper.py"),
        "--job",
        str(JOB),
        "--voices-dir",
        str(WORK / "voices"),
        "--out",
        str(RAW),
    ]
)

# %%
import numpy as np

from mobyww.audio import read_wav_int16, resample, write_wav_int16

DATA = WORK / "data"
CLIPS = DATA / "synthetic" / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)
for path in sorted(RAW.glob("*.wav")):
    target = CLIPS / path.name
    if target.exists():
        continue
    samples, rate = read_wav_int16(path)
    x = resample(samples.astype(np.float64), rate, 16000)
    peak = float(np.max(np.abs(x))) or 1.0
    # Peak at half scale (-6 dB): leaves room for the +6 dB gain of the augmentation plan.
    write_wav_int16(target, np.rint(x * (16384.0 / peak)).astype(np.int16), 16000)
print(len(list(CLIPS.glob("*.wav"))), "clips at 16 kHz")

# %% [markdown]
# ## 7. Manifest for the synthetic clips, validated

# %%
from mobyww.data.manifest import read_manifest
from mobyww.data.validate import validate
from mobyww.synth import build_manifest
from mobyww.synth.manifest import write_synthetic_manifest

MANIFESTS = DATA / "manifests"
split_seed = 1337  # configs/wakeword.yaml seeds.split
result = build_manifest(job_file, CLIPS, DATA, split_seed=split_seed)
print(
    "rows",
    len(result.rows),
    "missing",
    len(result.missing),
    "unexpected",
    len(result.unexpected),
    "bad",
    len(result.bad_format),
)
write_synthetic_manifest(MANIFESTS / "synthetic.jsonl", result)
report = validate(read_manifest(MANIFESTS / "synthetic.jsonl"), DATA)
assert report.ok, report.errors[:5]

# %% [markdown]
# ## 8. Open data: MUSAN and BUT ReverbDB from their official URLs (size + checksum verified)

# %%
from mobyww.data import fetch

OPEN = WORK / "open"
fetch_args = ["--dest", str(OPEN), "--no-reverbdb"]
if USE_RIR:
    fetch_args = ["--dest", str(OPEN), "--extract-rir"]
if USE_MUSAN:
    fetch_args += ["--musan", "--extract-musan", "noise,speech"]
# Sizes and the MUSAN MD5 are checked inside; the ReverbDB SHA-256 has no published value and is
# recorded in SHA256SUMS on first download (trust on first use).
assert fetch.main(fetch_args) == 0

# %% [markdown]
# ## 9. Augmentation with the INMP441 device path (seeded, logged)

# %%
from mobyww.augment import Augmenter, load_config, write_log
from mobyww.augment.dataset import ambient_rows, augment_rows
from mobyww.data.manifest import write_manifest

aug_config_path = REPO / "configs" / "augmentation.yaml"
aug_config = load_config(aug_config_path)
if USE_RIR:
    import dataclasses

    aug_config = dataclasses.replace(aug_config, rir_sets=("but_reverbdb",))
augmenter = Augmenter(aug_config)

noise_files = sorted((OPEN / "musan" / "extracted").rglob("noise/**/*.wav")) if USE_MUSAN else []
speech_files = sorted((OPEN / "musan" / "extracted").rglob("speech/**/*.wav")) if USE_MUSAN else []
rir_files = sorted((OPEN / "but_reverbdb" / "rir").rglob("*.wav")) if USE_RIR else []
print("noise", len(noise_files), "speech", len(speech_files), "rir", len(rir_files))

rows = read_manifest(MANIFESTS / "synthetic.jsonl")
augmented, records = [], []
for split, copies in (("train", AUGMENT_COPIES), ("val", 1)):
    part = [r for r in rows if r.split == split]
    new, rec = augment_rows(
        part,
        DATA,
        DATA / "augmented",
        augmenter,
        copies=copies,
        noise_files=noise_files,
        noise_license="CC-BY-4.0",
        noise_set="MUSAN noise",
        rir_files=rir_files,
        rir_set="but_reverbdb" if USE_RIR else "",
    )
    augmented += new
    records += rec
write_manifest(MANIFESTS / "augmented.jsonl", augmented)
write_log(DATA / "augment_log.jsonl", records)
assert validate(augmented, DATA).ok

limit = 3 if SMOKE else 20
lp = aug_config.device_path.lowpass
lowpass_hz = lp.cutoff_hz if lp else None
negative_rows = ambient_rows(
    speech_files,
    DATA,
    DATA / "negatives_speech",
    label="negative",
    source_name="MUSAN speech",
    license="CC-BY-4.0",
    split_seed=split_seed,
    max_chunks_per_file=limit,
    lowpass_hz=lowpass_hz,
)
ambient = ambient_rows(
    noise_files,
    DATA,
    DATA / "ambient",
    label="ambient",
    source_name="MUSAN noise",
    license="CC-BY-4.0",
    split_seed=split_seed,
    max_chunks_per_file=limit,
    lowpass_hz=lowpass_hz,
)
write_manifest(MANIFESTS / "negatives_speech.jsonl", negative_rows)
write_manifest(MANIFESTS / "ambient.jsonl", ambient)
print("augmented", len(augmented), "speech negatives", len(negative_rows), "ambient", len(ambient))

# %% [markdown]
# ## 10. Features (microWakeWord), one folder per manifest split

# %%
import mww_steps

from mobyww.train import FeatureSet, organize_by_split

SETS = WORK / "sets"
FEATURES = WORK / "features"
groups = {
    "wake": [r for r in augmented if r.label == "wake"],
    "hard_negative": [r for r in augmented if r.label == "hard_negative"],
    "speech": negative_rows,
    "ambient": ambient,
}
feature_sets = []
# Synthetic clips get no test split on purpose (the real test set is the owner's local recording),
# but upstream's final evaluation divides by the number of test positives and crashes on none. So
# validation clips are copied into test/ for upstream's own diagnostics only. Those numbers say
# nothing about accuracy; provenance records the copy.
TEST_IS_VALIDATION_COPY = False
for name, group in groups.items():
    if not group:
        print("skipping empty set", name)
        continue
    split_counts = organize_by_split(group, DATA, SETS / name)
    print(name, split_counts)
    if split_counts["validation"] and not split_counts["test"]:
        (SETS / name / "test").mkdir(parents=True, exist_ok=True)
        for clip in (SETS / name / "validation").glob("*.wav"):
            shutil.copy2(clip, SETS / name / "test" / clip.name)
        TEST_IS_VALIDATION_COPY = True
        print(name, "test/ is a copy of validation/ (upstream diagnostics only, not accuracy)")
    counts = mww_steps.build_features(
        SETS / name, FEATURES / name, ambient=name in {"speech", "ambient"}
    )
    print(name, "features from", counts)
    feature_sets.append(
        FeatureSet(
            name=name,
            features_dir=str(FEATURES / name),
            truth=name == "wake",
            sampling_weight={"wake": 2.0, "hard_negative": 3.0, "speech": 6.0, "ambient": 5.0}[
                name
            ],
            truncation_strategy="truncate_start" if name == "wake" else "random",
        )
    )

# %% [markdown]
# ## 11. Train (seeded) and write provenance next to the model

# %%
import yaml

from mobyww.train import (
    build_provenance,
    build_training_config,
    write_provenance,
    write_training_config,
)

seeds = yaml.safe_load((REPO / "configs" / "wakeword.yaml").read_text(encoding="utf-8"))["seeds"]
TRAINED = WORK / "trained"
training_config = build_training_config(
    train_dir=str(TRAINED),
    feature_sets=feature_sets,
    training_steps=40 if SMOKE else TRAINING_STEPS,
    batch_size=32 if SMOKE else 128,
    eval_step_interval=20 if SMOKE else 500,
    negative_class_weight=5 if SMOKE else 20,
)
config_path = WORK / "training_parameters.yaml"
write_training_config(config_path, training_config)
code = mww_steps.train(config_path, seed=seeds["training"], small=SMOKE)
MODEL = TRAINED / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
assert code == 0, f"training failed (exit {code})"
assert MODEL.is_file(), "no model file was written"

provenance = build_provenance(
    model_path=MODEL,
    repo=REPO,
    seeds=seeds,
    training_config=training_config,
    manifests=sorted(MANIFESTS.glob("*.jsonl")),
    config_files={
        "augmentation.yaml": aug_config_path,
        "wakeword.yaml": REPO / "configs" / "wakeword.yaml",
        "targets.yaml": REPO / "configs" / "targets.yaml",
        "job.json": JOB,
    },
    upstream=upstream_commit,
    extra={
        "colab": {
            "notebook": "scripts/colab/moby_wakeword_colab.py",
            "untested_before_this_run": True,
        },
        "run": {
            "smoke": SMOKE,
            "use_musan": USE_MUSAN,
            "use_rir": USE_RIR,
            "copies": AUGMENT_COPIES,
            "upstream_test_is_validation_copy": TEST_IS_VALIDATION_COPY,
        },
        "piper": {"tts": PIPER_TTS_VERSION, "generator_commit": PINNED_PIPER_GENERATOR_COMMIT},
        "job_file_sha256": job_file_sha256(job_file),
        "open_data_sha256sums": (OPEN / "SHA256SUMS").read_text(encoding="utf-8")
        if (OPEN / "SHA256SUMS").is_file()
        else "",
        "synthetic_audio_reproducible": False,
    },
)
print("provenance written to", write_provenance(MODEL.parent, provenance))

# %% [markdown]
# ## 12. Collect the result (model, provenance, attribution, licence files)
#
# Downloading from Colab to your machine is fine; nothing is uploaded from your machine.
# The attribution file and the MUSAN LICENSE files travel with any release candidate.

# %%
OUT = WORK / "release-candidate"
OUT.mkdir(exist_ok=True)
for source in (MODEL, MODEL.parent / "provenance.json", REPO / "docs" / "ATTRIBUTION.md", JOB):
    shutil.copy2(source, OUT / source.name)
for licence in (OPEN / "musan" / "extracted").rglob("LICENSE"):
    shutil.copy2(licence, OUT / f"MUSAN-{licence.parent.name}-LICENSE")
archive = shutil.make_archive(str(WORK / "moby-ww-release-candidate"), "zip", OUT)
print(archive)

from google.colab import files

files.download(archive)
