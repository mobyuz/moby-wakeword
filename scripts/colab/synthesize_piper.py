"""Generate the clips of a job file with Piper voices (piper-tts 1.3.0). Colab only, UNTESTED.

Runs in a separate virtualenv that has piper-tts, PyYAML and `pip install -e . --no-deps` of this
repository, so Piper's onnxruntime never meets the TensorFlow stack. The audio is written at the
voice's native rate (22.05 kHz); the notebook resamples it to 16 kHz afterwards.

Piper ONNX voices take no random seed, so the audio itself is not reproducible. What is
reproducible: the job file, the clip names, the per-clip parameters and the voice hashes (checked
here). The calls mirror piper-sample-generator's generate_samples_onnx at the pinned commit
(PiperVoice.load, SynthesisConfig, synthesize_wav).

    python synthesize_piper.py --job job.json --voices-dir voices --out raw
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import wave
from pathlib import Path

from mobyww.synth.spec import JobFile, Voice, expand_job, load_job_file

BASE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_voice(voice: Voice, voices_dir: Path) -> Path:
    """Download the voice at its pinned revision and verify size and SHA-256."""
    target = voices_dir / Path(voice.onnx_path).name
    voices_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".json"):
        url = (
            BASE.format(repo=voice.hf_repo, revision=voice.hf_revision, path=voice.onnx_path)
            + suffix
        )
        destination = Path(str(target) + suffix)
        if not destination.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, destination)  # noqa: S310 (https, pinned revision)
    if target.stat().st_size != voice.onnx_bytes or sha256_of(target) != voice.onnx_sha256:
        raise SystemExit(f"{target.name}: size or SHA-256 differs from the job file; refusing it")
    print(f"{target.name}: size and SHA-256 match the job file")
    return target


def synthesize(job_file: JobFile, voices_dir: Path, out_dir: Path) -> int:
    from piper import PiperVoice, SynthesisConfig  # installed in the Piper venv only

    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, PiperVoice] = {}
    made = 0
    for job in job_file.jobs:
        voice = job_file.voice(job.voice_id)
        if voice.id not in loaded:
            model = fetch_voice(voice, voices_dir)
            loaded[voice.id] = PiperVoice.load(str(model))
        engine = loaded[voice.id]
        for clip in expand_job(job, job_file.seed):
            target = out_dir / f"{clip.clip_id}.wav"
            if target.exists():
                continue  # resume
            config = SynthesisConfig(
                speaker_id=clip.speaker_id,
                length_scale=clip.length_scale,
                noise_scale=clip.noise_scale,
                noise_w_scale=clip.noise_scale_w,
            )
            with wave.open(str(target), "wb") as handle:
                engine.synthesize_wav(clip.text, wav_file=handle, syn_config=config)
            made += 1
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job_file = load_job_file(args.job)
    made = synthesize(job_file, args.voices_dir, args.out)
    print(f"generated {made} new clips ({job_file.total_clips} in the job file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
