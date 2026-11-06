# Changelog

## Unreleased

- Added the repository skeleton, CI, configs and the W0 upstream toolchain report: a real microWakeWord run on synthetic tones works on native Windows (not a speech model).
- Added the manifest schema, the validator, the guided recording tool, the privacy guard and the data, privacy and license documents (W1).
- Verified the W2 data and TTS licenses from primary sources (docs/licenses.md, docs/ATTRIBUTION.md): Piper libritts_r and ru denis/dmitri voices, MUSAN and BUT ReverbDB are allowed with attribution; Navoiy TTS and Chatterbox stay candidate; Chatterbox-MTL Central-Asian is stopped.
- Added the W2 augmentation (`mobyww.augment`): seeded SNR mixing, RIR (approved sets only, none enabled by default), gain, time shift, and the INMP441 device path that matches the firmware `moby_pcm.c` conversion bit for bit (unit-tested against hand-derived pairs; not run on hardware). The 7200 Hz low-pass is an unmeasured assumption.
- Added `mobyww.synth` (job-file planner and manifest writer; no TTS run), `mobyww.train` helpers, `scripts/fetch_open_data.py` and the Colab notebook `scripts/colab/` (UNTESTED: never run on Colab).
- The release gates in `configs/targets.yaml` are now marked confirmed by the owner.
- No model has been trained. Wake word accuracy is unmeasured.
