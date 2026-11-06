"""Training support that needs no TensorFlow: the training config, the split layout and provenance.

The upstream-dependent steps (features, training) live in scripts/colab/ and run on Colab or in
the .venv-train environment. No model has been trained with this package.
"""

from mobyww.train.config import (
    FeatureSet,
    build_training_config,
    organize_by_split,
    write_training_config,
)
from mobyww.train.provenance import build_provenance, manifest_hash, write_provenance

__all__ = [
    "FeatureSet",
    "build_provenance",
    "build_training_config",
    "manifest_hash",
    "organize_by_split",
    "write_provenance",
    "write_training_config",
]
