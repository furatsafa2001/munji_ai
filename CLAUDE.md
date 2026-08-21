# MUNJI — project context

Wearable cardiac monitor. A chest patch streams single-lead ECG over Bluetooth to a mobile app, which alerts a caregiver. Target users are ambulatory and elderly patients.

## Hardware — fixed, not negotiable

| | |
|---|---|
| Electrodes | 3, gel (hydrogel), CM5 placement |
| Channels | **1** — three electrodes give one recording pair plus a reference |
| AFE / MCU | MAX30003 + ESP32-S3 |
| Rate | 256 Hz on device → resampled to 250 Hz for inference |
| High-pass | 0.05 Hz — **do not raise this.** It destroys ST information irreversibly |
| Patch outputs | ECG, BPM, HRV. No sound, no vibration |
| Alerting | Caregiver only — SMS, call, emergency services |

**One lead means one viewing angle.** Every scope decision below follows from that.

## Scope — 11 outputs

**Rules (4)** — deterministic, no training data:
Bradycardia · Tachycardia · Pause/asystole · Fast narrow-complex rhythm

**Models (7)** — across 4 models:

| Model | Outputs |
|---|---|
| 1 — Quality gate | Signal quality (silent gate) |
| 2 — Beat classifier | PVC burden, PAC count |
| 3 — Rhythm classifier | Atrial fibrillation, Normal sinus rhythm |
| 4 — Shockable detector | Ventricular fibrillation, Wide-complex tachycardia |

## Permanently out of scope

**ST elevation, ST depression, T-wave inversion.** Not a signal-quality problem — CM5 is actually the most ST-sensitive single lead. The blockers are territory blindness (inferior and posterior ischemia are invisible), no localisation, and no validation path. Raw signal is still stored at diagnostic bandwidth for future work.

**AV block.** Requires reliable P-wave detection. CM5 is a chest lead with inherently small P waves. First-degree block *is* a PR interval measurement — impossible without P.

**VT vs SVT-with-aberrancy.** Every published criterion needs precordial leads. Both collapse into "wide-complex tachycardia" and alert wording must be descriptive, never a named diagnosis.

## Architecture

The four models are **independent**: separate data, separate losses, no shared weights. Train in any order or in parallel.

They are **not** independent at evaluation. In production nothing reaches models 2–4 except windows the quality gate passed, so evaluating them on all windows produces a number that doesn't match reality. Gate first, then evaluate the rest on what it passed.

Models run in parallel at inference and will disagree. A fixed-priority arbitration layer emits exactly one alert:

```
VF / asystole → wide-complex → extreme rate → AF / pause → PVC burden → fast narrow → info
```

**A single window must never fire a phone call.** Every critical alert requires N-of-M consecutive confirmations (see `CONFIRMATION` in config).

## Data layer — built, 45 checks passing

```
munji_ai/
  config.py            every constant lives here — nothing downstream hardcodes
  data/
    registry.py        the only file that names a dataset
    download.py        PhysioNet fetch
    preprocess.py      resample, dual filter paths, robust normalise, windowing
    loader.py          unified WFDB reader → Record
    augment.py         NSTDB noise injection, synthesised quality labels
    splits.py          patient-level splits + leak assertion
    windows.py         record → training-ready X/y caches
    build_cache.py     CLI
tests/                 synthetic ECG, no network needed
```

9 datasets. Icentia11k is primary (hydrogel patch, closest hardware match). MITDB and AFDB are benchmarks, forced to the test split.

Caches: `data/cache/{stage}_{split}.npz` with `X`, `y`, `patient`, `dataset`, `record`, `offset`, `cfg_hash`.

## Conventions

- **250 Hz everywhere.** Train and deploy see identical signal characteristics.
- **Patient-level splits only.** Record-level splitting leaks a subject across train and test; the model learns the person, not the arrhythmia. `_assert_disjoint` enforces this.
- **Never train on MITDB or AFDB.** They are comparability benchmarks.
- **Config hash guards caches.** Change preprocessing and `load_cache` refuses stale files.
- **Report against ANSI/AAMI EC57**, not raw accuracy.
- Models must eventually run on a phone. Keep them small — the reference architecture in the literature is ~770k parameters.

## Working style

- Discuss and plan before implementing. Do not jump to code.
- Complete only the requested task, then stop. Never auto-continue to the next phase.
- Challenge assumptions when there's a better approach — don't just agree.
- Explain trade-offs when multiple valid solutions exist.
- State assumptions explicitly. Distinguish facts from recommendations.
- MVP discipline: scope decisions serve prototype validation, not production worst-cases.

## Medical safety

- Every clinical threshold in `config.py` is an **engineering proposal**, not clinical guidance.
- Never present model output as a diagnosis or a substitute for clinical judgement.
- Alert copy is descriptive, never diagnostic.
- Public VF database performance is a **ceiling**, not a field expectation — real out-of-hospital VF is materially harder. This caveat accompanies any reported number.

## Current state

✅ Data layer built and verified
🔜 **Model 1 — quality gate** ← current task
⬜ Rule engine (4 outputs, no training data)
⬜ Models 2, 3, 4
⬜ Arbitration layer, on-device integration
