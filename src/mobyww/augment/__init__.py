"""W2 augmentation: SNR mixing, RIR convolution, gain, time shift and the INMP441 device path.

Pure NumPy, deterministic (seeded) and logged. The device path reproduces the firmware
conversion bit-exactly (`device`); everything else follows configs/augmentation.yaml
(`pipeline`). Implemented and unit-tested on this machine; nothing here has been run on
real INMP441 data or on the robot.
"""

from mobyww.augment.device import (
    WARMUP_SAMPLES,
    convert_stereo,
    drop_warmup,
    inmp441_path,
    lowpass,
    slot_to_pcm16,
    to_slot,
)
from mobyww.augment.pipeline import (
    AugmentConfig,
    Augmenter,
    AugmentResult,
    load_config,
    parse_config,
    write_log,
)

__all__ = [
    "WARMUP_SAMPLES",
    "AugmentConfig",
    "AugmentResult",
    "Augmenter",
    "convert_stereo",
    "drop_warmup",
    "inmp441_path",
    "load_config",
    "lowpass",
    "parse_config",
    "slot_to_pcm16",
    "to_slot",
    "write_log",
]
