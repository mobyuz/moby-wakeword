"""W2 synthetic speech planning: the Colab job file and the manifest writer.

This package makes no audio and downloads no model. `plan` writes the job file the Colab
notebook (scripts/colab/) executes; `manifest` turns the WAVs it produced into manifest rows.
Stdlib, NumPy and PyYAML only. Nothing has been generated yet.
"""

from mobyww.synth.manifest import ManifestResult, build_manifest
from mobyww.synth.plan import PlanParams, build_plan
from mobyww.synth.spec import (
    ClipSpec,
    JobFile,
    expand_job,
    job_file_sha256,
    load_job_file,
    save_job_file,
)

__all__ = [
    "ClipSpec",
    "JobFile",
    "ManifestResult",
    "PlanParams",
    "build_manifest",
    "build_plan",
    "expand_job",
    "job_file_sha256",
    "load_job_file",
    "save_job_file",
]
