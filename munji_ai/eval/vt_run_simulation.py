"""How good must the beat model be for the VT-run rule to work? A simulation.

    python -m munji_ai.eval.vt_run_simulation
    python -m munji_ai.eval.vt_run_simulation --datasets ltafdb edb nsrdb icentia11k --max-records 3000

The rule (decision layer): three or more consecutive ventricular beats faster
than 100 bpm is VT; under 30 s it is NSVT for the report, 30 s or more an
alert. The beat model labels each beat; this asks what its errors do to the
rule, before the model is plugged in.

Expert beat labels stand in for a perfect beat model, then get corrupted:
each true ventricular beat is kept with probability `sensitivity`, each other
beat turns ventricular with probability `false_rate`. The rule runs on the
corrupted labels and is scored against runs found on the true labels:

    detected   share of true VT runs that a predicted run overlaps,
               split into short (3-4 beats) and long (5+)
    false/24h  predicted runs overlapping no true run, per 24 h of recording

Errors here are independent from beat to beat. A real model's errors cluster
(noise, one odd patient), so treat these figures as the best case for a
given beat-level accuracy, a target to beat rather than a prediction.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..config import RAW_DIR
from ..data import registry
from ..data.loader import _patient_id, dataset_dir, list_records
from ..data.splits import assign
from ..data.vt_runs import VENTRICULAR, find_runs, read_beats

SENS = (0.80, 0.90, 0.95, 0.99)
FALSE_RATES = (0.001, 0.005, 0.01)


def corrupt(symbols: np.ndarray, sens: float, false_rate: float, rng) -> np.ndarray:
    is_v = np.isin(symbols, list(VENTRICULAR))
    u = rng.random(len(symbols))
    pred_v = np.where(is_v, u < sens, u < false_rate)
    return np.where(pred_v, "V", "N")


def overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score(true_runs, pred_runs):
    hit = [any(overlaps(t, p) for p in pred_runs) for t in true_runs]
    false = sum(1 for p in pred_runs if not any(overlaps(p, t) for t in true_runs))
    return hit, false


def load(datasets, raw: Path, max_records: int, seed: int):
    rng = np.random.default_rng(seed)
    recs = []
    for ds_key in datasets:
        try:
            ds = registry.get(ds_key)
            names = list_records(ds_key, raw)
        except Exception as e:
            print(f"  {ds_key:<11} skipped: {e}")
            continue
        headers = {h.stem: h for h in dataset_dir(ds, raw).rglob("*.hea")}
        names = [n for n in names if ds.role != "val" and assign(_patient_id(ds, n)) != "test"]
        if max_records and len(names) > max_records:
            names = list(rng.choice(names, max_records, replace=False))
        n0 = len(recs)
        for n in names:
            h = headers.get(n)
            got = read_beats(h.with_suffix("")) if h else None
            if got is None or len(got[0]) < 10:
                continue
            sym, samp, fs = got
            beat = np.array([s in VENTRICULAR or s.isalpha() for s in sym])   # beats only
            recs.append((ds_key, sym[beat], samp[beat], fs))
        print(f"  {ds_key:<11} {len(recs) - n0:,} recordings", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ltafdb", "edb", "nsrdb"])
    ap.add_argument("--max-records", type=int, default=0, help="cap per dataset (0 = all)")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    recs = load(a.datasets, a.raw, a.max_records, a.seed)
    if not recs:
        print("no recordings")
        return 1
    hours = sum((s[-1] - s[0]) / fs / 3600 for _, _, s, fs in recs)
    truth = [find_runs(sym, samp, fs) for _, sym, samp, fs in recs]
    n_true = sum(len(t) for t in truth)
    n_long = sum(1 for t in truth for r in t if r[2] >= 5)
    print(f"\n{len(recs):,} recordings, {hours:,.0f} h, {n_true:,} true VT runs "
          f"({n_long:,} of 5+ beats)   {time.time() - t0:.0f} s")

    rng = np.random.default_rng(a.seed)
    print(f"\n{'PVC sensitivity':>16}{'false PVC rate':>16}{'runs found':>12}"
          f"{'3-4 beats':>11}{'5+ beats':>10}{'false/24h':>11}")
    for sens in SENS:
        for fr in FALSE_RATES:
            hits, long_hits, short_hits, false = [], [], [], 0
            for (ds, sym, samp, fs), t in zip(recs, truth):
                pred = find_runs(corrupt(sym, sens, fr, rng), samp, fs)
                h, f = score(t, pred)
                hits += h
                long_hits += [x for x, r in zip(h, t) if r[2] >= 5]
                short_hits += [x for x, r in zip(h, t) if r[2] < 5]
                false += f
            pct = lambda v: f"{np.mean(v):.0%}" if v else "  -"
            print(f"{sens:>16.0%}{fr:>16.1%}{pct(hits):>12}{pct(short_hits):>11}"
                  f"{pct(long_hits):>10}{false / hours * 24:>11.1f}")
    print("\nIndependent errors: a best case. Real beat-model errors cluster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
