"""Run microwakeword.model_train_eval after seeding Python, NumPy and TensorFlow.

Upstream has no seed option, so this wrapper calls tf.keras.utils.set_random_seed first and then
runs the upstream module in the same process. GPU kernels can still be non-deterministic: the seed
makes a run repeatable in intent, not guaranteed bit-identical.

    python train_seeded.py --seed 4242 -- --training_config=... --train 1 ... mixednet ...
"""

from __future__ import annotations

import argparse
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    print(f"seeded Python, NumPy and TensorFlow with {args.seed}")
    sys.argv = ["microwakeword.model_train_eval", *rest]
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
