# Model 1 — signal quality gate

The first model in the pipeline and the one everything else depends on. It decides whether a 5-second window is clean enough to analyse. Windows it rejects never reach models 2–4.

## Task

| | |
|---|---|
| Input | `(n, 1250)` — 5 s at 250 Hz, robust-normalised |
| Output | 3 classes: `good` / `qrs_only` / `unusable` |
| Data | `data/cache/quality_{train,val,test}.npz` |
| Build with | `python -m munji_ai.data.download nsrdb nstdb` then `python -m munji_ai.data.build_cache quality` |

Class meanings:

- **good** — all waveform features clear. Safe for morphology analysis.
- **qrs_only** — R peaks findable, P and T unreliable. Safe for rate and rhythm timing, not for beat classification.
- **unusable** — reject.

## Where the labels come from

Not human annotation. Clean NSRDB signal is mixed with real recorded artifact from NSTDB (baseline wander, electrode motion, muscle noise) at a **chosen** SNR, so the label follows from the SNR by construction:

```
SNR ≥ 12 dB   → good
SNR ≥  3 dB   → qrs_only
otherwise     → unusable
```

Verified accurate to within 0.01 dB. This is exact ground truth, which is unusual and worth exploiting — but it also means the model is learning *synthetic* degradation. Real hydrogel dry-out over multi-day wear produces gradual impedance rise, which this does not simulate. Note it as a known limitation; don't try to solve it now.

## Target

```
specificity ≥ 0.95      of genuinely usable windows, how many were passed
sensitivity ≥ 0.90      of genuinely unusable windows, how many were caught
```

Measured on the `reject` view — positive class is `unusable`.

**Specificity is the priority.** A gate that throws away good signal blinds every downstream model and can cause a missed cardiac event. A gate that passes a slightly noisy window costs a small amount of downstream accuracy. These are not symmetric errors and the loss function should reflect that — weight false rejection of usable signal more heavily than false acceptance.

Use `munji_ai/eval/metrics.py`. It is already written:

```python
from munji_ai.eval.metrics import report, passes_target
print(report(y_true, y_pred, patients=d["patient"], title="quality gate — test"))
ok, detail = passes_target(y_true, y_pred)
```

Report **per-patient spread**, not just pooled numbers. A model at 0.90 pooled that collapses on 10% of patients is not acceptable, and pooled metrics hide exactly that.

## Constraints

- **PyTorch**, must run on CPU and Apple MPS. No CUDA-only ops.
- **Under ~1M parameters.** This eventually runs on a phone alongside three other models. The reference architecture in the single-lead literature is ~770k parameters at ~46 MFLOPs per 10-second window.
- **Inference under ~10 ms per window** on CPU. It runs on every window before anything else.
- No dependency on R-peak detection. The gate runs *before* any peak detector, and a peak detector on unusable signal is meaningless.

## Suggested approach

A small 1D CNN is the sensible baseline — dilated convolutions or a compact ConvNeXt-style 1D block. The discriminating features are largely spectral (noise energy outside the ECG band) and morphological (is there a repeating QRS structure at all), both of which a shallow convolutional stack captures well.

Start simple. Get a baseline number before adding anything. If a 4-layer CNN hits target, the job is done.

## Deliverables

```
munji_ai/models/quality_gate.py     architecture
munji_ai/train/train_quality.py     training loop, CLI, checkpointing
munji_ai/eval/eval_quality.py       loads a checkpoint, prints the report
```

Follow existing conventions: constants in `config.py`, no hardcoded paths, docstrings that explain *why* rather than restating the code.

## Do not

- Train on the test split, or peek at it while tuning. Use `val` for every decision.
- Optimise accuracy. The classes are imbalanced and accuracy will mislead.
- Add augmentation on top of the cache. The noise injection already happened at build time; layering more changes the SNR and invalidates the labels.
- Grow the model to chase the last percent. Under target with a small model beats over target with a model that won't fit on the phone.

## Order of work

1. Read `README.md`, `config.py`, `data/windows.py`, `eval/metrics.py`
2. Load the cache, check the class distribution, plot a few windows from each class
3. Propose the architecture and training setup — **stop and wait for approval**
4. Baseline train, evaluate on `val`
5. Iterate on `val` only
6. Single final evaluation on `test`

## Known issue to check first

Class balance skews toward `good` at the current SNR range (roughly 60/30/10 in synthetic testing). If the real distribution is worse than about 70/20/10, either widen `NOISE_SNR_DB_RANGE` in `config.py` and rebuild the cache, or use class weighting in the loss. Widening the range is cleaner — it fixes the data rather than compensating for it.

Report the actual distribution before proposing an architecture.
