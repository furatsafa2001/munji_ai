"""RhythmNet on whole recordings: AF false alarms per 24 h, and AF episodes caught.

    python -m munji_ai.eval.af_alarm_rate --models checkpoints/rhythm_cv
    python -m munji_ai.eval.af_alarm_rate --models checkpoints/rhythm_cv \
        --rules 2/3 4/6 10/12 20/24 --datasets nsrdb ltafdb --max-records 5

This is the rhythm counterpart of eval/false_alarm_rate.py, kept separate so
the two models' numbers are never mixed up.

Why
---
Per-window AF specificity says little about what a patient feels. A monitor
scores a 30 s window every 15 s, about 5,760 a day, and alerts after the
confirmation rule. Implantable-monitor studies report that the positive
predictive value of device-detected AF climbs steeply with episode length
(0.42 under 6 minutes, 0.89 over 60 minutes), so the confirmation length is a
real design choice, not a detail. This script measures it.

What it reports, per confirmation rule and threshold
-----------------------------------------------------
    false alarms per 24 h   on recordings and stretches with no AF
    AF episodes (EC57-style) episode sensitivity and positive predictivity,
                            duration sensitivity and positive predictivity
                            (duration = AF burden, what a doctor reads)
    episodes of 6 min+      the device-detected AF threshold used in the
                            ASSERT analyses, reported separately
    delay                   minutes from AF onset to the alert

No leakage: each recording is scored by the fold model that held its patient
out (cross_validate --stage rhythm --save-models). The test split is not used.
No quality gate is applied, so false alarms are an upper bound.
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
from ..data.windows import dominant_label, load_cache, rhythm_intervals
from ..models.rhythm_net import build

STAGE = "rhythm"
DEFAULT_DATASETS = ("nsrdb", "ltafdb")
LONG_EPISODE_S = 6 * 60.0           # device-detected AF threshold (ASSERT)
AF_LABELS = ("AF", "AFL")           # flutter is part of the AF output


def fold_of_records(oof_path: Path, cache_dir: Path) -> dict:
    """(dataset, record) -> fold, from the saved out-of-fold assignment."""
    with np.load(oof_path, allow_pickle=False) as d:
        records, folds = d["record"].astype(str), d["fold"].astype(int)
    datasets = np.concatenate([load_cache(STAGE, s, cache_dir)["dataset"].astype(str)
                               for s in ("train", "val")])
    if len(datasets) != len(records):
        raise SystemExit("the rhythm cache no longer matches the out-of-fold file")
    out = {}
    for ds, rec, f in zip(datasets, records, folds):
        out.setdefault((ds, rec), int(f))
    return out


def load_fold_models(models_dir: Path, device) -> dict:
    models = {}
    for p in sorted(models_dir.glob(f"{STAGE}_fold*.pt")):
        f = int(p.stem.replace(f"{STAGE}_fold", ""))
        m = build()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models[f] = m.to(device).eval()
    if not models:
        raise SystemExit(f"no {STAGE}_fold*.pt in {models_dir}; run cross_validate "
                         f"--stage rhythm --save-models")
    return models


def _normalize_rows(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q25, med, q75 = np.percentile(W, [25, 50, 75], axis=1)
    return (W - med[:, None]) / (((q75 - q25) / 1.349)[:, None] + eps)


@torch.no_grad()
def window_scores(model, signal: np.ndarray, device, batch: int = 256) -> np.ndarray:
    """P(AF) for every 30 s window at the deployment stride."""
    win, stride = C.window_samples(STAGE), C.stride_samples(STAGE)
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
    """True where the annotated rhythm makes the window AF (flutter included)."""
    iv = rhythm_intervals(rec)
    win, stride = C.window_samples(STAGE), C.stride_samples(STAGE)
    flags = np.zeros(n, dtype=bool)
    if not iv:
        return flags
    for i in range(n):
        lo = i * stride
        lab = dominant_label(iv, lo, lo + win, 0.5, excluded=C.AMBIGUOUS_RHYTHMS)
        flags[i] = lab in AF_LABELS
    return flags


def fired_mask(pred: np.ndarray, rule) -> np.ndarray:
    n, m = rule
    csum = np.concatenate([[0], np.cumsum(pred)])
    i = np.arange(len(pred))
    return (csum[i + 1] - csum[np.maximum(0, i - m + 1)]) >= n


def count(pred: np.ndarray, truth: np.ndarray, rule) -> dict:
    """Alerts, AF episodes and durations for one recording."""
    n, m = rule
    stride_s = C.WINDOW_STRIDE_SEC[STAGE]
    win_s = C.WINDOW_SEC[STAGE]
    fired = fired_mask(pred, rule)
    out = {"false": 0, "true": 0, "episodes": 0, "caught": 0, "long_episodes": 0,
           "long_caught": 0, "delays": [], "truth_w": int(truth.sum()),
           "fired_w": int(fired.sum()), "overlap_w": int((fired & truth).sum())}
    for i in np.flatnonzero(fired):
        if i == 0 or not fired[i - 1]:
            lo = max(0, i - m + 1)
            out["true" if truth[lo:i + 1].any() else "false"] += 1
    t = np.concatenate([[False], truth, [False]])
    starts = np.flatnonzero(~t[:-1] & t[1:])
    ends = np.flatnonzero(t[:-1] & ~t[1:])
    long_w = int(np.ceil((LONG_EPISODE_S - win_s) / stride_s)) + 1
    for a, b in zip(starts, ends):
        out["episodes"] += 1
        is_long = (b - a) >= long_w
        out["long_episodes"] += int(is_long)
        hit = np.flatnonzero(fired[a:b])
        if hit.size:
            out["caught"] += 1
            out["long_caught"] += int(is_long)
            out["delays"].append(int(hit[0]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True,
                    help="folder with rhythm_fold*.pt and rhythm_oof.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5],
                    help="thresholds on P(AF)")
    ap.add_argument("--rules", nargs="+", default=None, metavar="N/M",
                    help="confirmation rules, e.g. 2/3 4/6 10/12 20/24 "
                         "(default: config.CONFIRMATION['af'])")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    rules = ([tuple(int(v) for v in r.split("/")) for r in a.rules] if a.rules
             else [tuple(C.CONFIRMATION["af"])])
    for n_, m_ in rules:
        if not 1 <= n_ <= m_:
            raise SystemExit(f"bad rule {n_}/{m_}: need 1 <= N <= M")
    stride_s, win_s = C.WINDOW_STRIDE_SEC[STAGE], C.WINDOW_SEC[STAGE]
    folds = fold_of_records(a.models / f"{STAGE}_oof.npz", a.cache)
    models = load_fold_models(a.models, device)
    print(f"models      {len(models)} rhythm folds from {a.models}")
    print("rules       " + "  ".join(f"{n_}/{m_} (~{((m_ - 1) * stride_s + win_s) / 60:.1f} min)"
                                    for n_, m_ in rules))
    print("gate        not applied (upper bound on alarms)\n")

    hours = defaultdict(float)
    n_rec = defaultdict(int)
    acc = defaultdict(lambda: defaultdict(float))
    delays = defaultdict(list)
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
            for t in a.thresholds:
                for r in rules:
                    c = count(s >= t, truth, r)
                    k = acc[(ds, t, r)]
                    for key in ("false", "true", "episodes", "caught", "long_episodes",
                                "long_caught", "truth_w", "fired_w", "overlap_w"):
                        k[key] += c[key]
                    delays[(t, r)] += c["delays"]
            print(f"  {ds:<9} {j:>3}/{len(keys)}  {name:<10} "
                  f"{len(rec.signal) / TARGET_FS / 3600:5.1f} h   "
                  f"{time.time() - t0:6.0f} s elapsed", flush=True)

    used = [d for d in a.datasets if n_rec[d]]
    if not used:
        print("nothing scored")
        return 1
    print(f"\n  ({', '.join(f'{d} {hours[d]:.0f} h' for d in used)})")
    head = f"  {'thr':>5}{'rule':>7}" + "".join(f"{d + ' f/24h':>14}" for d in used)
    head += f"{'ESe':>7}{'E+P':>7}{'DSe':>7}{'D+P':>7}{'6min+':>9}{'delay':>9}"
    print(head)
    for t in a.thresholds:
        for r in rules:
            tot = defaultdict(float)
            row = f"  {t:>5.2f}{f'{r[0]}/{r[1]}':>7}"
            for d in used:
                for k, v in acc[(d, t, r)].items():
                    tot[k] += v
                row += f"{acc[(d, t, r)]['false'] / hours[d] * 24:>14.2f}"
            dl = delays[(t, r)]
            med = f"{(np.median(dl) * stride_s + win_s) / 60:.1f} m" if dl else "-"
            ep = max(int(tot["episodes"]), 1)
            alerts = max(tot["true"] + tot["false"], 1)
            long_txt = f"{int(tot['long_caught'])}/{int(tot['long_episodes'])}"
            row += (f"{tot['caught'] / ep:>7.3f}{tot['true'] / alerts:>7.3f}"
                    f"{tot['overlap_w'] / max(tot['truth_w'], 1):>7.3f}"
                    f"{tot['overlap_w'] / max(tot['fired_w'], 1):>7.3f}"
                    f"{long_txt:>9}{med:>9}")
            print(row)
        print()
    print("  false/24h: alerts with no annotated AF in their confirming windows.")
    print("  ESe/E+P: episode sensitivity / positive predictivity; DSe/D+P: duration")
    print("  (AF burden) sensitivity / positive predictivity; 6min+: episodes lasting")
    print("  6 minutes or more that were caught; delay: median minutes onset to alert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
