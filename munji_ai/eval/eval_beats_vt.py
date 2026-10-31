"""Do the beat rules catch real VT? NSVT and sustained VT on known VT runs.

    python -m munji_ai.eval.eval_beats_vt
    python -m munji_ai.eval.eval_beats_vt --miss 0.10     # model misses 1 PVC in 10
    python -m munji_ai.eval.eval_beats_vt --miss 0.20

Reads data/cache/vt_runs.npz (written by munji_ai.data.vt_runs: VT runs read
off expert beat labels; test-split patients were already left out there) and,
for each recording, its expert beat annotations. The beats around every run
go through the decision layer the way the app sends them: 20 s windows every
10 s, overlapping. Then:

    NSVT    a report entry overlapping the run; expected for runs under 30 s
    WCT     an alert during the run or within 30 s after it; expected for runs
            of 30 s or more, and not for shorter ones

--miss p relabels each true PVC as normal with probability p, standing in for
the beat model's misses: the strict consecutive rule breaks first, the 80%
rule for sustained VT should hold. Expert labels play a perfect beat model
here. When Furat's model is plugged in (beats_from_mask on its masks) the same
tables measure the model.

Reads only. Trains nothing; the test split is never opened.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..data import registry
from ..data.loader import BEAT_MAP, dataset_dir
from ..data.vt_runs import read_beats
from ..decision.decision_layer import DecisionLayer

WIN, STRIDE, AROUND = 20.0, 10.0, 60.0
BEAT_BINS = ((3, 4), (5, 9), (10, 29), (30, 10 ** 9))


def run_through(beats, t0, t1):
    """Beats in [t0, t1] -> decision layer in overlapping windows."""
    dl = DecisionLayer()
    alerts = []
    te = t0 + STRIDE
    while te <= t1 + STRIDE:
        inside = [(bt, c) for bt, c in beats if te - WIN < bt <= te]
        alerts += dl.on_beats(te, inside, usable=True)
        te += STRIDE
    dl.on_beats(te, [], usable=False)          # closes a run still open at the end
    return dl, alerts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=C.CACHE_DIR / "vt_runs.npz")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--miss", type=float, default=0.0,
                    help="probability each true PVC is relabelled normal")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    z = np.load(a.runs)
    n = len(z["record"])
    print(f"{n} VT runs from {len(set(z['patient']))} patients in {a.runs}")
    rng = np.random.default_rng(a.seed)
    by_rec = defaultdict(list)
    for i in range(n):
        by_rec[(str(z["dataset"][i]), str(z["record"][i]))].append(i)

    headers = {}
    rows = []
    t_start = time.time()
    for (ds_key, rec), idx in sorted(by_rec.items()):
        if ds_key not in headers:
            ds = registry.get(ds_key)
            headers[ds_key] = {h.stem: h for h in dataset_dir(ds, a.raw).rglob("*.hea")}
        hea = headers[ds_key].get(rec)
        got = read_beats(hea.with_suffix("")) if hea else None
        if got is None:
            continue
        sym, samp, fs = got
        beats = [(s / fs, BEAT_MAP[x]) for x, s in zip(sym, samp) if x in BEAT_MAP]
        if a.miss > 0:
            beats = [(t, "N" if c == "PVC" and rng.random() < a.miss else c) for t, c in beats]
        times = np.array([b[0] for b in beats])
        for i in idx:
            rs, re_ = z["start"][i] / fs, z["end"][i] / fs
            lo, hi = np.searchsorted(times, [rs - AROUND, re_ + AROUND])
            dl, alerts = run_through(beats[lo:hi], rs - AROUND, re_ + AROUND)
            nsvt = any(s <= re_ and e >= rs for s, e, _, _ in dl.nsvt)
            wct = any(al.kind == "WCT" and rs <= al.t <= re_ + C.VT_SUSTAINED_SEC for al in alerts)
            rows.append({"beats": int(z["beats"][i]), "dur": float(re_ - rs),
                         "bpm": float(z["bpm"][i]), "nsvt": nsvt, "wct": wct,
                         "patient": str(z["patient"][i])})
        if len(rows) and len(rows) % 100 == 0:
            print(f"  {len(rows)} runs   {time.time() - t_start:.0f} s", flush=True)

    if not rows:
        print("no runs could be read (raw annotations missing?)")
        return 1
    print(f"\n{len(rows)} runs evaluated"
          + (f", each true PVC relabelled normal with p = {a.miss}" if a.miss else
             ", expert beat labels (a perfect beat model)"))

    def table(title, groups):
        print(f"\n{title:<16}{'runs':>6}{'patients':>10}{'NSVT reported':>15}{'WCT alert':>11}")
        for name, sel in groups:
            g = [r for r in rows if sel(r)]
            if not g:
                continue
            pats = len({r['patient'] for r in g})
            print(f"{name:<16}{len(g):>6}{pats:>10}{np.mean([r['nsvt'] for r in g]):>15.0%}"
                  f"{np.mean([r['wct'] for r in g]):>11.0%}")

    table("by length", [(f"{lo}-{hi} beats" if hi < 10 ** 9 else f"{lo}+ beats",
                         lambda r, lo=lo, hi=hi: lo <= r["beats"] <= hi) for lo, hi in BEAT_BINS])
    table("by duration", [("under 30 s", lambda r: r["dur"] < C.VT_SUSTAINED_SEC),
                          ("30 s or more", lambda r: r["dur"] >= C.VT_SUSTAINED_SEC)])
    table("by rate", [("100-149 bpm", lambda r: r["bpm"] < 150), ("150+ bpm", lambda r: r["bpm"] >= 150)])
    print("\nExpected: runs under 30 s -> NSVT reported, no WCT alert;"
          "\n          runs of 30 s or more -> WCT alert. The NSVT column for them is"
          "\n          the strict rule (it reports pieces when a PVC is missed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
