"""W0 upstream toolchain smoke run.

Runs the REAL microWakeWord pipeline (feature generation, training, streaming conversion,
int8 quantization) on a tiny synthetic sample. The "wake word" here is a made-up three-tone
pattern and the negatives are noise: this proves that the toolchain runs on this machine. It
says nothing about wake word accuracy and the resulting model is not a speech model.

Run it inside the training environment (see docs/upstream-toolchain.md):

    .venv-train/Scripts/python.exe scripts/upstream_smoke.py --workdir D:/moby-cache/smoke

Nothing is downloaded: no TTS, no negative datasets, no RIR or noise sets.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import math
import platform
import random
import struct
import subprocess
import sys
import wave
from importlib import metadata
from pathlib import Path

import yaml

RATE = 16000


def write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(
            b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 12000)) for s in samples)
        )


def pattern_clip(rng: random.Random) -> list[float]:
    """Three rising tones with small random jitter: the stand-in 'wake word'."""
    base = rng.uniform(0.9, 1.1)
    out: list[float] = []
    for hz in (400 * base, 700 * base, 1100 * base):
        n = int(rng.uniform(0.18, 0.24) * RATE)
        out += [math.sin(2 * math.pi * hz * i / RATE) * math.sin(math.pi * i / n) for i in range(n)]
    return out


def negative_clip(rng: random.Random) -> list[float]:
    kind = rng.choice(["noise", "tone", "reverse"])
    n = int(rng.uniform(0.5, 1.2) * RATE)
    if kind == "noise":
        return [rng.uniform(-0.5, 0.5) for _ in range(n)]
    if kind == "tone":
        hz = rng.uniform(200, 3000)
        return [0.5 * math.sin(2 * math.pi * hz * i / RATE) for i in range(n)]
    return list(reversed(pattern_clip(rng)))[: n or 1]


def make_clips(directory: Path, count: int, maker: object, seed: int) -> None:
    rng = random.Random(seed)
    for i in range(count):
        write_wav(directory / f"{i:04d}.wav", maker(rng))  # type: ignore[operator]


def build_features(clips_dir: Path, out_dir: Path, *, positive: bool, seed: int) -> None:
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    clips = Clips(
        input_directory=str(clips_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=seed,
        split_count=0.15,
    )
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.0,
            "TanhDistortion": 0.0,
            "PitchShift": 0.0,
            "BandStopFilter": 0.0,
            "AddColorNoise": 0.5,
            "AddBackgroundNoise": 0.0,
            "Gain": 1.0,
            "GainTransition": 0.0,
            "RIR": 0.0,
        },
        impulse_paths=[],
        background_paths=[],
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    for folder, split, repeat in (
        ("training", "train", 2),
        ("validation", "validation", 1),
        ("testing", "test", 1),
    ):
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=10 if folder != "testing" else 1,
            step_ms=10,
        )
        target = out_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(target / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(split=split, repeat=repeat),
            batch_size=50,
            verbose=False,
        )
    if not positive:
        # Negatives double as the "ambient" sets upstream uses for false accepts per hour.
        for ambient, split in (("validation_ambient", "validation"), ("testing_ambient", "test")):
            generator = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
            )
            (out_dir / ambient).mkdir(parents=True, exist_ok=True)
            RaggedMmap.from_generator(
                out_dir=str(out_dir / ambient / "ambient_mmap"),
                sample_generator=generator.spectrogram_generator(split=split, repeat=1),
                batch_size=50,
                verbose=False,
            )


def versions() -> dict[str, str]:
    names = [
        "tensorflow",
        "keras",
        "numpy",
        "ai-edge-litert",
        "pymicro-features",
        "microwakeword",
        "audiomentations",
        "mmap_ninja",
        "webrtcvad-wheels",
        "audio-metadata",
        "datasets",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True, help="scratch directory (use D:)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--count", type=int, default=80, help="clips per class")
    parser.add_argument("--trace-after", type=int, default=0, help="dump stacks after N seconds")
    args = parser.parse_args()
    if args.trace_after:
        faulthandler.dump_traceback_later(args.trace_after, exit=False)
    work: Path = args.workdir
    work.mkdir(parents=True, exist_ok=True)

    print("python", platform.python_version(), platform.platform())
    for name, version in versions().items():
        print(f"  {name}=={version}")

    make_clips(work / "pos", args.count, pattern_clip, seed=1)
    make_clips(work / "neg", args.count, negative_clip, seed=2)
    build_features(work / "pos", work / "features_pos", positive=True, seed=11)
    build_features(work / "neg", work / "features_neg", positive=False, seed=12)

    config = {
        "window_step_ms": 10,
        "train_dir": str(work / "trained"),
        "features": [
            {
                "features_dir": str(work / "features_pos"),
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(work / "features_neg"),
                "sampling_weight": 4.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
        ],
        "training_steps": [args.steps],
        "positive_class_weight": [1],
        "negative_class_weight": [5],
        "learning_rates": [0.001],
        "batch_size": 32,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": max(10, args.steps // 2),
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    (work / "training_parameters.yaml").write_text(yaml.dump(config), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={work / 'training_parameters.yaml'}",
        "--train",
        "1",
        "--restore_checkpoint",
        "0",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        "16,16,16,16",
        "--repeat_in_block",
        "1,1,1,1",
        "--mixconv_kernel_sizes",
        "[5],[7,11],[9,15],[23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "16",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]
    print("running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    model = (
        work
        / "trained"
        / "tflite_stream_state_internal_quant"
        / "stream_state_internal_quant.tflite"
    )
    if completed.returncode != 0 or not model.is_file():
        print(f"FAILED: exit code {completed.returncode}, model present: {model.is_file()}")
        return 1
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    print(f"OK: {model} {model.stat().st_size} bytes sha256={digest}")
    print("This is a toolchain check on synthetic tones; it is NOT a wake word model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
