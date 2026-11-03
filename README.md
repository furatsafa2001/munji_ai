# munji_ai

ECG analysis pipeline for the MUNJI single-lead chest patch: data
preparation, model training, rule-based rate analysis, and the decision
layer that turns model and rule outputs into alerts.

All tunable constants, with the published source for every clinical
threshold, are in `munji_ai/config.py`.

## Setup

Python 3.13. From the repository root:

```
pip install -r requirements.txt
```

Every command below is run from the repository root (the folder that
contains `requirements.txt`), as `python -m munji_ai.<module>`.

## Layout

| Folder | Contents |
|---|---|
| `munji_ai/config.py` | Sampling rate, filters, windows, splits, rule thresholds, notification policy |
| `munji_ai/data/` | Dataset registry, download, loading, preprocessing, patient-level splits, window caches, augmentation |
| `munji_ai/features/` | Signal quality indices |
| `munji_ai/models/` | Quality gate, beat segmenter, rhythm classifier (RhythmNet), shockable rhythm detector (ShockableNet) |
| `munji_ai/rules/` | QRS detection and the rate engine (bradycardia, tachycardia, pause, asystole) |
| `munji_ai/decision/` | Decision layer: combines gate, models and rules into findings and alerts |
| `munji_ai/train/` | Training and patient-level cross-validation |
| `munji_ai/eval/` | Evaluation, false-alarm measurement, diagnostics, plots |

## Signal

The patch samples at 250 Hz, the same rate used for training, so no
resampling is needed at inference. The MAX30003 on the patch applies a
0.5 Hz high-pass and a 40 Hz low-pass filter on chip, so the device signal,
including saved recordings, is in the monitoring band (0.5 to 40 Hz). The
rate engine takes input in millivolts, never normalised.

## Pipeline

1. Download datasets from PhysioNet

   ```
   python -m munji_ai.data.download --list
   python -m munji_ai.data.download nsrdb nstdb cudb vfdb
   ```

2. Build patient-level splits and window caches

   ```
   python -m munji_ai.data.build_cache --splits-only
   python -m munji_ai.data.build_cache all
   ```

3. Train

   ```
   python -m munji_ai.train.train_quality --cache data/cache
   python -m munji_ai.train.train_beat --cache data/cache
   python -m munji_ai.train.train_rhythm --cache data/cache
   python -m munji_ai.train.train_shockable --cache data/cache
   python -m munji_ai.train.cross_validate --stage shockable
   ```

4. Evaluate rules and the decision layer

   ```
   python -m munji_ai.eval.eval_rules_holter --datasets nsrdb
   python -m munji_ai.eval.eval_beats_model --part vt
   python -m munji_ai.eval.eval_beats_model --part nsrdb
   ```

Each module's docstring lists its options.

## Test split

The test split is held out. Model and threshold decisions use
cross-validation and validation results only. The test split is used once,
for an end-to-end test of all models together after conversion to the
deployment format. Scripts that can read it require an explicit flag
(for example `--allow-test`).

## Experimental

These files belong to a VT detector that was evaluated and not adopted.
They are kept as the record of that evaluation and are not part of the
delivered pipeline:

- `munji_ai/train/vt_detector.py`
- `munji_ai/data/vtac.py`
- `munji_ai/data/holter_vt.py`
- `munji_ai/eval/eval_vt_detector.py`
- `munji_ai/eval/vt_detector_check.py`

VF/VT is one ShockableNet alert. Slow VT is caught by the beat-count rule
in the decision layer.

## Datasets

All training and evaluation data come from PhysioNet. Each dataset carries
its own licence, listed on its PhysioNet page. Icentia11k is distributed
under CC BY-NC-SA 4.0 (non-commercial).
