"""Steps that need TensorFlow and microWakeWord: feature generation and (seeded) training.

Run inside the training environment (Colab, or .venv-train locally). Only the calls proven in
docs/upstream-toolchain.md are used (Clips, Augmentation, SpectrogramGeneration, RaggedMmap,
model_train_eval). Upstream's own random split is bypassed: every split folder is its own Clips
source, because our splits are per speaker and come from the manifest.

The upstream augmentation is switched OFF here. All augmentation (noise, RIR, gain, the INMP441
device path) is done earlier by mobyww.augment, so it is seeded, logged and bit-exact.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# (split folder, upstream split name, repeat, slide_frames); testing uses the streaming model.
SPLITS = (
    ("train", "training", 2, 10),
    ("validation", "validation", 1, 10),
    ("test", "testing", 1, 1),
)
AMBIENT_SPLITS = (("validation", "validation_ambient"), ("test", "testing_ambient"))

_OFF = {
    "SevenBandParametricEQ": 0.0,
    "TanhDistortion": 0.0,
    "PitchShift": 0.0,
    "BandStopFilter": 0.0,
    "AddColorNoise": 0.0,
    "AddBackgroundNoise": 0.0,
    "Gain": 0.0,
    "GainTransition": 0.0,
    "RIR": 0.0,
}


def _count_wavs(directory: Path) -> int:
    return len(list(directory.glob("*.wav"))) if directory.is_dir() else 0


def build_features(set_dir: Path, out_dir: Path, *, ambient: bool) -> dict[str, int]:
    """Spectrogram features for set_dir/{train,validation,test}/*.wav into out_dir.

    With `ambient=True` the validation and test clips are also written as the *_ambient sets that
    upstream uses for false accepts per hour. Returns clip counts per folder.
    """
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    counts: dict[str, int] = {}
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities=dict(_OFF),
        impulse_paths=[],
        background_paths=[],
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    def run(folder: Path, target: Path, repeat: int, slide: int) -> None:
        clips = Clips(
            input_directory=str(folder),
            file_pattern="*.wav",
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=None,
        )
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10
        )
        target.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(target / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(split=None, repeat=repeat),
            batch_size=50,
            verbose=False,
        )

    for folder_name, upstream_name, repeat, slide in SPLITS:
        folder = set_dir / folder_name
        counts[folder_name] = _count_wavs(folder)
        if counts[folder_name]:
            run(folder, out_dir / upstream_name, repeat, slide)
    if ambient:
        for folder_name, upstream_name in AMBIENT_SPLITS:
            if counts.get(folder_name):
                run(set_dir / folder_name, out_dir / upstream_name, 1, 1)
    return counts


def training_command(config_path: Path, *, seed: int) -> list[str]:
    """The command that trains the mixednet model (upstream notebook architecture)."""
    here = Path(__file__).resolve().parent / "train_seeded.py"
    command = [
        sys.executable,
        str(here),
        "--seed",
        str(seed),
        "--",
        f"--training_config={config_path}",
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
        "64,64,64,64",
        "--repeat_in_block",
        "1,1,1,1",
        "--mixconv_kernel_sizes",
        "[5],[7,11],[9,15],[23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]
    return command


def small_architecture(command: list[str]) -> list[str]:
    """Swap in the 16-filter network the W0 smoke run used (for quick pipeline checks only)."""
    swaps = {"64,64,64,64": "16,16,16,16", "32": "16"}
    return [swaps.get(token, token) if i > 0 else token for i, token in enumerate(command)]


def train(config_path: Path, *, seed: int, small: bool = False) -> int:
    command = training_command(config_path, seed=seed)
    if small:
        command = small_architecture(command)
    print("running:", " ".join(command))
    return subprocess.run(command, check=False).returncode
