# Upstream library and environments

Date 2026-11-05

Status candidate. The toolchain facts are in [upstream-toolchain.md](../upstream-toolchain.md). Nothing here says a wake word model works.

## Decision

- Train with microWakeWord (Apache-2.0), pinned by commit, through a thin wrapper added in W3. The code is not copied into this repo.
- Two Python 3.12 environments. `.venv` holds only the lint, type and test tooling plus PyYAML, so CI stays small and fast. `.venv-train` holds TensorFlow and the upstream package. Both live inside the repo and are git-ignored.
- Python 3.12 is pinned. Upstream declares `>=3.10` and its notebook says 3.10, but TensorFlow 2.20 has Windows wheels for 3.12 and that is the combination that was run.
- `datasets` is pinned below 4. Upstream's clip loader decodes audio through `datasets.Audio`, and `datasets` 4 and later need `torchcodec` (and with it PyTorch) for that, which is a large and unneeded dependency here.

## Alternatives

- **openWakeWord.** Different training flow and model format; the plan chose microWakeWord because ESPHome's component runs its streaming TFLite Micro models. Not evaluated further.
- **Espressif's custom wake word service.** Paid, needs at least 20 000 quality recordings, and Uzbek TTS training is not announced (plan).
- **One environment.** Rejected: it would put TensorFlow into every CI run.

## Consequences

The CI smoke job installs the training stack and is allowed to fail until it has been seen to pass on Linux.
