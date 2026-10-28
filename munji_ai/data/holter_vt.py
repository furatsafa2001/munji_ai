"""Holter-domain windows for the VT detector: VT, VF and everything else.

    python -m munji_ai.data.holter_vt

Why
---
Trained on VTaC alone, the VT detector learnt ICU-monitor VT well (event AUC
0.89) but recognised only 14% of VT windows in CUDB and VFDB at 0.7: the
ambulatory recordings it will meet on the patch look different. Adding
Holter VT as positives alone would bring back the source trap seen earlier
("Holter means not VT"), so every source here brings both sides:

    VT      CUDB, VFDB      annotated VT, slow and fast
    VF      CUDB, VFDB      annotated VF and flutter, labelled not-VT: this is
                            what lets the detector name a ShockableNet alert
    other   CUDB, VFDB      everything else in the arrest recordings
            EDB, STDB,      cardiac patients, exercise, rest
            Fantasia

NSRDB is left out on purpose: it is the daily-life false-alarm test.
Only MUNJI's train and validation recordings are read (from the shockable
out-of-fold file), so the test split stays untouched.

Output: data/cache/holter_vt_windows.npz
    X        (n, 2500) float16, 10 s filtered like every source, not normalised
    label    VT / VF / other
    group    dataset:record, the patient key
    dataset
    bpm      rate measured from the signal, for the slow/fast split
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR
from .loader import load_record
from .windows import _episode_bpm, dominant_label, rhythm_intervals, split_vt_by_rate

FS = C.TARGET_FS
WIN_S, STRIDE_S = 10.0, 5.0


def label_windows(rec, n: int, win: int, stride: int) -> list:
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    if not iv:
        return ["other"] * n            # no rhythm annotation: a normal recording
    out = []
    for i in range(n):
        lab = dominant_label(iv, i * stride, i * stride + win, 0.5, excluded=())
        out.append(None if lab is None else "VF" if lab == "VF"
                   else "VT" if lab in ("VT_FAST", "VT_SLOW") else "other")
    return out


def main(argv=None) -> int:
    from ..eval.false_alarm_rate import fold_of_records

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cudb", "vfdb", "edb", "stdb", "fantasia"])
    ap.add_argument("--folds-from", type=Path, default=Path("checkpoints/shock_cv_v3"))
    ap.add_argument("--other-per-record", type=int, default=60)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    a = ap.parse_args(argv)

    allowed = fold_of_records(a.folds_from / "shockable_oof.npz", a.cache)
    win, stride = int(WIN_S * FS), int(STRIDE_S * FS)
    rng = np.random.default_rng(0)
    X, lab, grp, dss, bpm = [], [], [], [], []
    t0 = time.time()
    for ds in a.datasets:
        names = sorted(n for (d, n) in allowed if d == ds)
        for name in names:
            rec = load_record(ds, name, a.raw)
            n = 1 + (len(rec.signal) - win) // stride if len(rec.signal) >= win else 0
            labels = label_windows(rec, n, win, stride)
            idx = {k: [i for i, l in enumerate(labels) if l == k] for k in ("VT", "VF", "other")}
            if len(idx["other"]) > a.other_per_record:
                idx["other"] = sorted(rng.choice(idx["other"], a.other_per_record, replace=False))
            for k, ii in idx.items():
                for i in ii:
                    x = np.asarray(rec.signal[i * stride:i * stride + win], dtype=np.float64)
                    X.append(x.astype(np.float16))
                    lab.append(k)
                    grp.append(f"{ds}:{name}")
                    dss.append(ds)
                    b = _episode_bpm(x, FS)
                    bpm.append(float(b) if b else np.nan)
        print(f"  {ds:<9} {len(names):>3} recordings   {time.time() - t0:5.0f} s", flush=True)

    out = a.cache / "holter_vt_windows.npz"
    np.savez_compressed(out, X=np.stack(X), label=np.array(lab), group=np.array(grp),
                        dataset=np.array(dss), bpm=np.array(bpm))
    lab = np.array(lab)
    c = Counter(lab)
    vt = lab == "VT"
    print(f"\nsaved       {out}")
    print(f"windows     VT {c['VT']:,} ({int((np.array(bpm)[vt] < 150).sum()):,} under 150 bpm), "
          f"VF {c['VF']:,}, other {c['other']:,}")
    print(f"recordings  with VT {len(set(np.array(grp)[vt])):,}, "
          f"with VF {len(set(np.array(grp)[lab == 'VF'])):,}, total {len(set(grp)):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
