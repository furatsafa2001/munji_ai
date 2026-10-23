"""False alarms per 24 hours, on whole recordings no fold model has seen.

    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --thresholds 0.5 0.55 0.6 --datasets nsrdb stdb

Why this number
---------------
Window specificity says how often one 5-second window is wrong. A patient does
not experience windows. They experience their phone going off, after the 2 of 3
confirmation rule, a number of times per day. In wearable defibrillator
studies that alarm burden is what decides whether patients keep wearing the
device, so it is the figure the client and a clinician can actually judge.

The cross-validation cache cannot give it: each patient contributes at most a
capped, shuffled sample of windows, not their day. So this script goes back to
the raw recordings and runs the model over every window, end to end, at the
deployment stride.

No leakage
----------
Each recording is scored by the fold model that held its patient out, using
the fold assignment saved in the out-of-fold file. Only patients in the train
and val splits are used; the test split stays untouched.

What it does not do
-------------------
The quality gate is not applied, so every window reaches the model. In the
field the gate removes the worst artifact first, so these figures are an upper
bound on the alarm rate, not an estimate of it.

Reads only. Writes nothing and trains nothing.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR, TARGET_FS
from ..data.loader import load_record
from ..data.windows import (SHOCKABLE, dominant_label, load_cache,
                            rhythm_intervals, split_vt_by_rate)
from ..models.shockable_net import build

DEFAULT_DATASETS = ("nsrdb", "stdb", "edb", "fantasia")


def fold_of_records(oof_path: Path, cache_dir: Path) -> dict:
    """(dataset, record) -> fold, from the saved out-of-fold assignment."""
    with np.load(oof_path, allow_pickle=False) as d:
        records = d["record"].astype(str)
        folds = d["fold"].astype(int)
    datasets = np.concatenate([load_cache("shockable", s, cache_dir)["dataset"].astype(str)
                               for s in ("train", "val")])
    if len(datasets) != len(records):
        raise SystemExit("the cache no longer matches the out-of-fold file; "
                         "re-run cross_validate")
    out = {}
    for ds, rec, f in zip(datasets, records, folds):
        out.setdefault((ds, rec), int(f))
    return out


def load_fold_models(models_dir: Path, device) -> dict:
    models = {}
    for p in sorted(models_dir.glob("shockable_fold*.pt")):
        f = int(p.stem.replace("shockable_fold", ""))
        m = build()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models[f] = m.to(device).eval()
    if not models:
        raise SystemExit(f"no shockable_fold*.pt in {models_dir}; run cross_validate "
                         f"with --save-models")
    return models


def _normalize_rows(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """preprocess.normalize, applied to every row at once."""
    q25, med, q75 = np.percentile(W, [25, 50, 75], axis=1)
    scale = (q75 - q25) / 1.349
    return (W - med[:, None]) / (scale[:, None] + eps)


@torch.no_grad()
def window_scores(model, signal: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    if len(signal) < win:
        return np.zeros(0, dtype=np.float32)
    starts = np.arange(0, len(signal) - win + 1, stride)
    out = np.empty(len(starts), dtype=np.float32)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        W = signal[s[:, None] + np.arange(win)[None, :]]
        x = torch.from_numpy(_normalize_rows(W).astype(np.float32)).to(device)
        out[i:i + len(s)] = model.predict_proba(x).cpu().numpy()
    return out


def truth_flags(rec, n: int) -> np.ndarray:
    """True where the annotated rhythm makes the window genuinely shockable."""
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    flags = np.zeros(n, dtype=bool)
    if not iv:
        return flags
    for i in range(n):
        lo = i * stride
        lab = dominant_label(iv, lo, lo + win, C.SHOCKABLE_PURITY,
                             excluded=C.SHOCKABLE_EXCLUDED)
        flags[i] = lab in SHOCKABLE
    return flags


def count_alarms(pred: np.ndarray, truth: np.ndarray, rule) -> tuple[int, int]:
    """(false, true) alarms under N-of-M; a run of confirmed windows is one alarm."""
    n, m = rule
    false_alarms = true_alarms = 0
    prev = False
    csum = np.concatenate([[0], np.cumsum(pred)])
    for i in range(len(pred)):
        lo = max(0, i - m + 1)
        fired = (csum[i + 1] - csum[lo]) >= n
        if fired and not prev:
            if truth[lo:i + 1].any():
                true_alarms += 1
            else:
                false_alarms += 1
        prev = fired
    return false_alarms, true_alarms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True,
                    help="folder with shockable_fold*.pt and shockable_oof.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55, 0.6])
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-records", type=int, default=None,
                    help="per dataset, for a quick trial run")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    rule = C.CONFIRMATION["vf"]
    folds = fold_of_records(a.models / "shockable_oof.npz", a.cache)
    models = load_fold_models(a.models, device)
    print(f"models      {len(models)} folds from {a.models}")
    print(f"rule        {rule[0]} of {rule[1]} windows, stride "
          f"{C.WINDOW_STRIDE_SEC['shockable']} s")
    print(f"gate        not applied (upper bound on alarms)\n")

    hours = defaultdict(float)
    n_rec = defaultdict(int)
    fa = defaultdict(int)                  # (dataset, threshold) -> false alarms
    ta = defaultdict(int)
    flagged = defaultdict(int)
    windows = defaultdict(int)
    t0 = time.time()

    for ds in a.datasets:
        keys = sorted(k for k in folds if k[0] == ds)[: a.max_records]
        if not keys:
            print(f"  {ds:<9} no records in the out-of-fold file, skipped")
            continue
        for j, (_, name) in enumerate(keys, 1):
            f = folds[(ds, name)]
            if f not in models:
                continue
            rec = load_record(ds, name, a.raw)
            s = window_scores(models[f], rec.signal, device)
            truth = truth_flags(rec, len(s))
            hours[ds] += len(rec.signal) / TARGET_FS / 3600
            n_rec[ds] += 1
            windows[ds] += len(s)
            for t in a.thresholds:
                pred = s >= t
                f_, t_ = count_alarms(pred, truth, rule)
                fa[(ds, t)] += f_
                ta[(ds, t)] += t_
                flagged[(ds, t)] += int((pred & ~truth).sum())
            print(f"  {ds:<9} {j:>3}/{len(keys)}  {name:<10} "
                  f"{len(rec.signal) / TARGET_FS / 3600:5.1f} h   "
                  f"{time.time() - t0:6.0f} s elapsed", flush=True)

    used = [ds for ds in a.datasets if n_rec[ds]]
    if not used:
        print("nothing scored")
        return 1

    for t in a.thresholds:
        print(f"\n  threshold {t:.2f}")
        print(f"  {'dataset':<10}{'records':>8}{'hours':>8}{'false alarms':>14}"
              f"{'per 24 h':>10}{'window FP rate':>16}")
        tot_h = tot_fa = tot_w = tot_fl = 0
        for ds in used:
            per_day = fa[(ds, t)] / hours[ds] * 24
            print(f"  {ds:<10}{n_rec[ds]:>8}{hours[ds]:>8.1f}{fa[(ds, t)]:>14}"
                  f"{per_day:>10.2f}{flagged[(ds, t)] / max(windows[ds], 1):>16.4f}")
            tot_h += hours[ds]
            tot_fa += fa[(ds, t)]
            tot_w += windows[ds]
            tot_fl += flagged[(ds, t)]
        print(f"  {'all':<10}{sum(n_rec[d] for d in used):>8}{tot_h:>8.1f}{tot_fa:>14}"
              f"{tot_fa / tot_h * 24:>10.2f}{tot_fl / max(tot_w, 1):>16.4f}")

    print("\n" + "=" * 66)
    print("  Per 24 h is the rate a patient would feel with the gate off.")
    print("  Alarms overlapping annotated VF or fast VT count as true, not false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
