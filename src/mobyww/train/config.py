"""The microWakeWord training configuration and the data layout it expects.

Nothing here imports TensorFlow or microWakeWord. The values are the upstream notebook's defaults
(OHF-Voice/micro-wake-word, commit 4665173, notebooks/basic_training_notebook.ipynb), not tuned
for "hey moby": no training run with real data has happened, so none of them is a measured choice.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mobyww.augment.dataset import refuse_private
from mobyww.data.manifest import Row

SPLIT_FOLDERS = {"train": "train", "val": "validation", "test": "test"}


@dataclass(frozen=True)
class FeatureSet:
    """One directory of precomputed features (training/, validation/, testing/, *_ambient/)."""

    name: str
    features_dir: str
    truth: bool
    sampling_weight: float
    penalty_weight: float = 1.0
    truncation_strategy: str = "random"

    def to_upstream(self) -> dict[str, Any]:
        return {
            "features_dir": self.features_dir,
            "sampling_weight": self.sampling_weight,
            "penalty_weight": self.penalty_weight,
            "truth": self.truth,
            "truncation_strategy": self.truncation_strategy,
            "type": "mmap",
        }


def build_training_config(
    *,
    train_dir: str,
    feature_sets: Sequence[FeatureSet],
    training_steps: int = 10_000,
    batch_size: int = 128,
    eval_step_interval: int = 500,
    clip_duration_ms: int = 1500,
    positive_class_weight: float = 1,
    negative_class_weight: float = 20,
    learning_rate: float = 0.001,
) -> dict[str, Any]:
    """The `training_parameters.yaml` content (upstream keys), hyperparameters included."""
    if not any(s.truth for s in feature_sets):
        raise ValueError("at least one feature set must hold positives")
    if not any(not s.truth for s in feature_sets):
        raise ValueError("at least one feature set must hold negatives")
    return {
        "window_step_ms": 10,
        "train_dir": train_dir,
        "features": [s.to_upstream() for s in feature_sets],
        "training_steps": [training_steps],
        "positive_class_weight": [positive_class_weight],
        "negative_class_weight": [negative_class_weight],
        "learning_rates": [learning_rate],
        "batch_size": batch_size,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": eval_step_interval,
        "clip_duration_ms": clip_duration_ms,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def write_training_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def organize_by_split(rows: Sequence[Row], data_root: Path, set_dir: Path) -> dict[str, int]:
    """Copy the rows' WAVs into set_dir/<train|validation|test>/ by their manifest split.

    microWakeWord's own random split is bypassed on purpose: our splits are per speaker and are
    already in the manifest. Returns the clip count per folder. Refuses private folders.
    """
    refuse_private(data_root)
    refuse_private(set_dir)
    counts = {folder: 0 for folder in SPLIT_FOLDERS.values()}
    for row in rows:
        folder = SPLIT_FOLDERS.get(row.split)
        if folder is None:
            raise ValueError(f"{row.id}: split {row.split!r} is not train, val or test")
        source = refuse_private(data_root / row.path)
        target = set_dir / folder / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        counts[folder] += 1
    return counts
