"""Measure false alarms per 24 hours on whole recordings, and save fold models.

Why
---
Window specificity is not what a patient feels. They feel their phone going
off, after the 2 of 3 confirmation rule, some number of times a day, and in
wearable defibrillator studies that burden decides whether people keep the
device on. The cross-validation cache cannot measure it, because it holds a
capped, shuffled sample of each patient rather than their day.

So:

    munji_ai/train/cross_validate.py   --save-models: keep each fold's model
    munji_ai/eval/false_alarm_rate.py  NEW: run each fold model over the full
                                       raw recordings of the patients it held
                                       out, at the deployment stride, apply
                                       the confirmation rule, count alarms per
                                       24 h

Each recording is scored only by the fold model that never saw its patient.
The test split is not used. The quality gate is not applied, so the figures
are an upper bound on the field alarm rate.

Without --save-models, cross_validate behaves exactly as before.

How to run
----------
From the repo root:

    python fix_false_alarm_rate.py
    python tests/test_false_alarm_rate.py

Revert with:

    git checkout munji_ai/train/cross_validate.py
    and delete munji_ai/eval/false_alarm_rate.py
"""

from pathlib import Path

NEW = Path("munji_ai/eval/false_alarm_rate.py")
CV = Path("munji_ai/train/cross_validate.py")

NEW_SOURCE = r'''"""False alarms per 24 hours, on whole recordings no fold model has seen.

    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --thresholds 0.55 0.6 --rules 2/3 4/6 6/9 --datasets nsrdb stdb
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --rules 2/3 4/6 6/9 --datasets cudb vfdb        # the cost: delay, misses

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

Longer confirmation
-------------------
--rules scores several N-of-M confirmation rules from one pass over the data.
On arrest recordings (cudb, vfdb) the same table shows what a longer rule
costs: episodes caught and the median delay from onset to alarm. Artifact is
usually brief and fibrillation is not, so a longer rule should remove false
alarms faster than it delays true ones; this measures whether it does.

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


def count_alarms(pred: np.ndarray, truth: np.ndarray, rule) -> dict:
    """Alarms under N-of-M; a run of confirmed windows is one alarm.

    false   alarms whose confirming windows contain no annotated shockable one
    true    alarms that do
    episodes / caught / delays
            contiguous runs of shockable windows, whether an alarm fired while
            each was running, and how many windows after onset it fired
    """
    n, m = rule
    csum = np.concatenate([[0], np.cumsum(pred)])
    idx = np.arange(len(pred))
    fired = (csum[idx + 1] - csum[np.maximum(0, idx - m + 1)]) >= n
    out = {"false": 0, "true": 0, "episodes": 0, "caught": 0, "delays": []}

    for i in np.flatnonzero(fired):
        if i == 0 or not fired[i - 1]:
            lo = max(0, i - m + 1)
            out["true" if truth[lo:i + 1].any() else "false"] += 1

    t = np.concatenate([[False], truth, [False]])
    starts = np.flatnonzero(~t[:-1] & t[1:])
    ends = np.flatnonzero(t[:-1] & ~t[1:])          # exclusive
    for a, b in zip(starts, ends):
        out["episodes"] += 1
        hit = np.flatnonzero(fired[a:b])
        if hit.size:
            out["caught"] += 1
            out["delays"].append(int(hit[0]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True,
                    help="folder with shockable_fold*.pt and shockable_oof.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55, 0.6])
    ap.add_argument("--rules", nargs="+", default=None, metavar="N/M",
                    help="confirmation rules to compare, e.g. 2/3 4/6 6/9 "
                         "(default: config.CONFIRMATION['vf'])")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-records", type=int, default=None,
                    help="per dataset, for a quick trial run")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    if a.rules:
        rules = [tuple(int(v) for v in r.split("/")) for r in a.rules]
    else:
        rules = [tuple(C.CONFIRMATION["vf"])]
    for n_, m_ in rules:
        if not 1 <= n_ <= m_:
            raise SystemExit(f"bad rule {n_}/{m_}: need 1 <= N <= M")
    step = C.WINDOW_STRIDE_SEC["shockable"]
    folds = fold_of_records(a.models / "shockable_oof.npz", a.cache)
    models = load_fold_models(a.models, device)
    print(f"models      {len(models)} folds from {a.models}")
    print(f"rules       " + "  ".join(f"{n_}/{m_} (~{(m_ - 1) * step + 5:g} s)"
                                      for n_, m_ in rules))
    print(f"gate        not applied (upper bound on alarms)\n")

    hours = defaultdict(float)
    n_rec = defaultdict(int)
    acc = defaultdict(lambda: {"false": 0, "true": 0, "episodes": 0,
                               "caught": 0, "delays": []})
    per_record = defaultdict(lambda: defaultdict(int))
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
                pred = s >= t
                for r in rules:
                    c = count_alarms(pred, truth, r)
                    k = acc[(ds, t, r)]
                    for key in ("false", "true", "episodes", "caught"):
                        k[key] += c[key]
                    k["delays"] += c["delays"]
                    per_record[(t, r)][f"{ds}:{name}"] += c["false"]
            print(f"  {ds:<9} {j:>3}/{len(keys)}  {name:<10} "
                  f"{len(rec.signal) / TARGET_FS / 3600:5.1f} h   "
                  f"{time.time() - t0:6.0f} s elapsed", flush=True)

    used = [ds for ds in a.datasets if n_rec[ds]]
    if not used:
        print("nothing scored")
        return 1

    print(f"\n  false alarms per 24 h ({', '.join(f'{d} {hours[d]:.0f} h' for d in used)})")
    head = f"  {'thr':>5}{'rule':>6}" + "".join(f"{d:>10}" for d in used) + f"{'all':>10}"
    has_ep = any(acc[(d, t, r)]["episodes"] for d in used for t in a.thresholds
                 for r in rules)
    if has_ep:
        head += f"{'caught':>10}{'median delay':>14}"
    print(head)
    for t in a.thresholds:
        for r in rules:
            row = f"  {t:>5.2f}{f'{r[0]}/{r[1]}':>6}"
            tot = 0
            for d in used:
                fa = acc[(d, t, r)]["false"]
                tot += fa
                row += f"{fa / hours[d] * 24:>10.1f}"
            row += f"{tot / sum(hours[d] for d in used) * 24:>10.1f}"
            if has_ep:
                ep = sum(acc[(d, t, r)]["episodes"] for d in used)
                ca = sum(acc[(d, t, r)]["caught"] for d in used)
                dl = [x for d in used for x in acc[(d, t, r)]["delays"]]
                med = f"{np.median(dl) * step + 5:.0f} s" if dl else "-"
                row += f"{f'{ca}/{ep}':>10}{med:>14}"
            print(row)

    t, r = a.thresholds[0], rules[0]
    worst = sorted(per_record[(t, r)].items(), key=lambda kv: -kv[1])[:5]
    if worst and worst[0][1]:
        total = sum(per_record[(t, r)].values())
        print(f"\n  records with the most false alarms (threshold {t:.2f}, rule {r[0]}/{r[1]})")
        for name, fa in worst:
            print(f"    {name:<16}{fa:>6}   {fa / max(total, 1):>5.0%} of all")

    print("\n" + "=" * 66)
    print("  Per 24 h is the rate a patient would feel with the gate off.")
    print("  Alarms overlapping annotated VF or fast VT count as true, not false.")
    if has_ep:
        print("  Delay is from episode onset to alarm, including the 5 s window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

ARG_OLD = r'''    ap.add_argument("--noisy-eval", type=float, default=None, metavar="SNR_DB",
                    help="also score each held-out fold with NSTDB noise at this SNR")'''
ARG_NEW = r'''    ap.add_argument("--noisy-eval", type=float, default=None, metavar="SNR_DB",
                    help="also score each held-out fold with NSTDB noise at this SNR")
    ap.add_argument("--save-models", action="store_true",
                    help="save each fold's selected model next to the OOF file")'''

SAVE_OLD = r'''        model, P = train_one_fold(a.stage, d["X"][tr], y[tr], d["X"][te],
                                  device, a, classes)
'''
SAVE_NEW = r'''        model, P = train_one_fold(a.stage, d["X"][tr], y[tr], d["X"][te],
                                  device, a, classes)
        if a.save_models:
            a.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.out / f"{a.stage}_fold{f}.pt")
'''

PLAN = [
    (ARG_OLD, ARG_NEW, "--save-models flag"),
    (SAVE_OLD, SAVE_NEW, "fold model saved after training"),
]


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    if not CV.exists():
        print(f"MISSING  {CV}\nrun this from the repo root")
        return 1
    text, eol = read(CV)
    if "--noisy-eval" not in text:
        print("STOP  the augmentation change (PR #2) is not on this branch.")
        print("      Branch from feat/train-augmentation, then run this again.")
        return 1
    if NEW.exists() and "--save-models" in text:
        print("already applied - nothing to do")
        return 0

    print("applying\n")
    ok = True
    for old, new, label in PLAN:
        if new in text:
            print(f"  present  cross_validate  {label}")
        elif text.count(old) == 1:
            text = text.replace(old, new, 1)
            print(f"  fixed    cross_validate  {label}")
        else:
            print(f"  FAILED   cross_validate  {label} - expected code not found")
            ok = False
    if not ok:
        print("\nnothing written - all files are untouched")
        return 1

    write(NEW, NEW_SOURCE, eol)
    print(f"  created  {NEW.as_posix()}")
    write(CV, text, eol)
    print("\napplied. config, caches and every model are untouched.\n")
    print("now run:")
    print("  python tests/test_false_alarm_rate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
