# MUNJI AI — data foundation

Phase 1 of the AI track: dataset registry, unified loader, preprocessing, patient-level splits, and noise augmentation.

Nothing here trains a model. This is the layer everything else sits on.

## Setup

```bash
pip install -r requirements.txt
python -m munji_ai.data.download --list
python -m munji_ai.data.download icentia11k          # start first — hours
python -m munji_ai.data.download mitdb afdb nstdb cudb vfdb nsrdb svdb ltafdb
python tests/test_pipeline.py                        # no downloads needed
```

`icentia11k` is the long pole. Kick it off before anything else and let it run unattended.

## Layout

```
munji_ai/
  config.py              every constant — rates, bands, windows, thresholds
  data/
    registry.py          dataset definitions; the only file that knows corpus names
    download.py          PhysioNet fetch, idempotent
    preprocess.py        resample, dual filter paths, robust normalise, windowing
    loader.py            unified WFDB reader -> Record
    augment.py           NSTDB noise injection, synthesised quality labels
    splits.py            patient-level splits with leak assertion
tests/
  test_pipeline.py       synthetic-ECG verification, no network required
```

## Decisions encoded here

**250 Hz everywhere.** Icentia, CUDB, VFDB and AFDB are natively 250. The MAX30003 runs at 256 and is resampled down at inference, so training and deployment see the same signal characteristics. One resample, on the device side only.

**Two filter paths, not one.** `model_path` is 0.5–40 Hz, the standard monitoring band, and feeds every model. `diagnostic_path` is 0.05–40 Hz and preserves the ST segment for storage. ST is out of scope for user-facing output, but a 0.5 Hz high-pass destroys that information permanently — keeping the acquisition path at 0.05 Hz costs nothing and keeps the option open.

**Zero-phase filtering offline.** `filtfilt` means R-peak timing stays aligned with annotations (verified: 0 samples of drift). The on-device path must be causal and will shift peaks slightly — do not assume these two agree.

**Robust normalisation.** Median and IQR rather than mean and standard deviation. One motion spike inflates the standard deviation enough to flatten a whole window, which is exactly the case the quality gate needs to see clearly.

**Patient-level splits, hash-assigned.** Splitting on records leaks a subject across train and test and inflates every metric. Assignment is a deterministic hash of patient id, so adding datasets never reshuffles existing assignments. `_assert_disjoint` raises on any leak. MITDB records 201 and 202 are the same subject and are pinned together in `SAME_SUBJECT`.

**Benchmarks are test-only.** Anything registered with `role="val"` is forced entirely into test. Training on MITDB or AFDB destroys comparability with published results.

**Noise augmentation doubles as quality labels.** Injecting NSTDB artifact at a known SNR gives exactly-labelled quality data — continuous severity rather than three discrete classes, and no dependency on an external quality corpus.

**Unmapped beat symbols are dropped, not folded into normal.** An unrecognised WFDB symbol never silently pollutes a class.

## Known gaps

- `icentia11k` is registered with `verified=False`. Its record layout and the `patient_re` pattern need confirming against the actual download before the first training run — the loader will raise a clear error if the pattern misses.
- `nsrdb`, `svdb`, `ltafdb` native rates are set to 128 Hz from documentation; confirm on download.
- Rule thresholds in `config.py` are engineering proposals, not clinical guidance.
- Icentia11k is CC BY-NC-SA 4.0 — research and prototype use. Revisit before any commercial release. The registry indirection means swapping corpora later is a config change, not a rewrite.
