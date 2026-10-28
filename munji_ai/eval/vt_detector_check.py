"""Two checks before the VT detector can be used: naming, and daily false alerts.

    python -m munji_ai.eval.vt_detector_check --model checkpoints/vt_detector/vt_detector.pt

1. Can it name VF and VT apart?
   The detector learnt "real VT or not" from VTaC alone, where both sides come
   from the same ICU monitors, and never saw CUDB or VFDB. Scored on their
   annotated VF and VT windows (same recordings, so no source to learn), a
   high AUC for VT over VF means the decision layer can call a ShockableNet
   alert VT when this detector agrees and VF when it does not. Only the
   train and validation recordings are used; MUNJI's test split stays
   untouched.

2. How often does it fire in daily life?
   It learnt against ICU false alarms, not against a day of walking and
   sleeping. Every NSRDB alert is false (healthy subjects, 24 h Holter).
   Alerts need n of the last m windows above a threshold, then the episode
   must go quiet for 30 s before another can fire. Each setting is reported
   with and without a rate gate: the detector only counts when the rate
   engine measures 100 bpm or more, the client's own threshold for VT, since
   a heart below 100 bpm cannot be in VT by definition.

Windows are 10 s every 5 s, filtered and normalised as in training.
Reads only. Writes nothing.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR
from ..data.loader import load_record
from ..data.preprocess import normalize
from ..data.windows import dominant_label, rhythm_intervals, split_vt_by_rate
from ..models.shockable_net import build
from ..rules.rate_engine import analyse
from .false_alarm_rate import fold_of_records

FS = C.TARGET_FS
STRIDE_S = 5.0
QUIET = 6                      # windows without a positive (30 s) close an episode


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build()
    m.load_state_dict(ck["state_dict"])
    return m.eval(), float(ck.get("window_s", 10.0))


@torch.no_grad()
def scores(model, signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    n = 1 + (len(signal) - win) // stride if len(signal) >= win else 0
    out = []
    for i in range(0, n, 512):
        batch = np.stack([normalize(signal[j * stride:j * stride + win])
                          for j in range(i, min(i + 512, n))]).astype(np.float32)
        out.append(torch.softmax(model(torch.from_numpy(batch)), -1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def window_truth(rec, n: int, win: int, stride: int) -> list:
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    out = []
    for i in range(n):
        lab = dominant_label(iv, i * stride, i * stride + win, 0.5, excluded=()) if iv else None
        out.append("VF" if lab == "VF" else "VT" if lab in ("VT_FAST", "VT_SLOW")
                   else None if lab is None else "other")
    return out


def count_alerts(pos: np.ndarray, n: int, m: int) -> int:
    alerts, active, quiet, recent = 0, False, 0, []
    for v in pos:
        recent = (recent + [bool(v)])[-m:]
        if active:
            quiet = 0 if v else quiet + 1
            if quiet >= QUIET:
                active, quiet, recent = False, 0, []
            continue
        if sum(recent) >= n:
            alerts += 1
            active, quiet = True, 0
    return alerts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("checkpoints/vt_detector/vt_detector.pt"))
    ap.add_argument("--folds-from", type=Path, default=Path("checkpoints/shock_cv_v3"),
                    help="its out-of-fold file lists MUNJI's train and validation recordings")
    ap.add_argument("--arrest", nargs="+", default=["cudb", "vfdb"])
    ap.add_argument("--daily", nargs="+", default=["nsrdb"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--rules", nargs="+", default=["2/3", "3/4"])
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    a = ap.parse_args(argv)

    model, win_s = load_model(a.model)
    win, stride = int(win_s * FS), int(STRIDE_S * FS)
    allowed = fold_of_records(a.folds_from / "shockable_oof.npz", a.cache)
    print(f"model       {a.model} ({win_s:.0f} s windows every {STRIDE_S:.0f} s)\n")

    # 1. naming --------------------------------------------------------------
    p_cls = defaultdict(list)
    t0 = time.time()
    for ds in a.arrest:
        for name in sorted(n for (d, n) in allowed if d == ds):
            rec = load_record(ds, name, a.raw)
            s = scores(model, rec.signal, win, stride)
            for p, t in zip(s, window_truth(rec, len(s), win, stride)):
                if t:
                    p_cls[t].append(float(p))
        print(f"  {ds:<6} scored   {time.time() - t0:5.0f} s", flush=True)
    vf, vt, other = (np.array(p_cls[k]) for k in ("VF", "VT", "other"))
    print(f"\n1. naming on CUDB and VFDB (train and validation recordings)")
    print(f"   windows    VF {len(vf):,}   VT {len(vt):,}   other rhythms {len(other):,}")
    print(f"   AUC, VT above VF      {auc(np.r_[np.ones(len(vt)), np.zeros(len(vf))], np.r_[vt, vf]):.3f}")
    print(f"   median P(VT)          VT {np.median(vt) if len(vt) else float('nan'):.2f}   "
          f"VF {np.median(vf) if len(vf) else float('nan'):.2f}   "
          f"other {np.median(other) if len(other) else float('nan'):.2f}")
    for thr in a.thresholds:
        f = lambda x: float((x >= thr).mean()) if len(x) else float("nan")
        print(f"   at {thr:.1f}: called VT   VT {f(vt):.0%}   VF {f(vf):.0%}   other {f(other):.0%}")

    # 2. daily life ----------------------------------------------------------
    rules = [tuple(int(v) for v in r.split("/")) for r in a.rules]
    alerts = defaultdict(int)
    hours = 0.0
    t0 = time.time()
    for ds in a.daily:
        recs = sorted(n for (d, n) in allowed if d == ds)
        for k, name in enumerate(recs):
            rec = load_record(ds, name, a.raw)
            s = scores(model, rec.signal, win, stride)
            hours += len(rec.signal) / FS / 3600
            fast = np.zeros(len(s), bool)
            cand = np.flatnonzero(s >= min(a.thresholds))
            for j in cand:
                r = analyse(rec.signal[j * stride:j * stride + win])
                fast[j] = np.isfinite(r.bpm) and r.bpm >= 100
            for thr in a.thresholds:
                for n, m in rules:
                    alerts[(thr, n, m, "none")] += count_alerts(s >= thr, n, m)
                    alerts[(thr, n, m, "rate")] += count_alerts((s >= thr) & fast, n, m)
            if (k + 1) % 5 == 0 or k + 1 == len(recs):
                print(f"  {ds:<6} {k + 1:>3}/{len(recs)}   {time.time() - t0:5.0f} s", flush=True)
    per_day = 24.0 / hours if hours else float("nan")
    print(f"\n2. false alerts per 24 h in daily life ({', '.join(a.daily)}, {hours:.0f} h)")
    print(f"   {'thr':>5}{'rule':>6}{'no gate':>10}{'rate >= 100':>13}")
    for thr in a.thresholds:
        for n, m in rules:
            print(f"   {thr:>5.1f}{f'{n}/{m}':>6}{alerts[(thr, n, m, 'none')] * per_day:>10.2f}"
                  f"{alerts[(thr, n, m, 'rate')] * per_day:>13.2f}")
    print("\n   The ShockableNet figure for comparison: about 0.5 per patient-day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
