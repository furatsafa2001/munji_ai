"""Fast narrow rhythm in daily life: Icentia11k.

Icentia11k is a single-lead chest patch, like MUNJI, worn in daily life by
older patients (average age 62), and every beat is labelled. svt_runs found
no SVT of 30 s or more there (the longest run of premature supraventricular
beats lasts 20 s), so this cannot score how many SVT the label catches. It
scores the other side: how often the label fires in daily life, where the
fast rhythms are sinus (walking, stairs, stress) and an SVT is rare.

Every label counts as false. Icentia has no SVT label, and a label may fall
on an SVT the technologists marked as normal beats: each label of the
PRIMARY row is printed with its rate, the rate just before, the onset drop,
the beats around it and the rhythm mark, so you can look.

Fixed before the run
--------------------
PRIMARY   "label: onset 150, slower before": its labels per 24 h at most 0.29
          (NSRDB's bar), counting every label as false. PASS or FAIL. The
          exact Poisson 95% upper bound is reported beside it.
REPORTED  every label row that needs no RhythmNet (no RhythmNet runs here):
          labels, per 24 h with the bound, patients with any; and the fast
          'sinus' episodes svt_runs found (30 s of normal beats at 150 bpm or
          more), each with its onset judged on the annotated beats (the
          cycle-length drop at the start of its run over 100 and over 150,
          the time from 100 to 150, the annotated 30 s rate before the 150
          run) and the rows that labelled it.

Patients in MUNJI's test split are skipped.

Run (about 2 h with 4 workers; set --workers to your cores):
    python -m munji_ai.data.svt_runs               # first, for the fast sinus episodes
    python experiments/fast_narrow_icentia.py --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from munji_ai import config as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("fast_narrow_template", ROOT / "experiments" / "fast_narrow_template.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["fast_narrow_template"] = G
_spec.loader.exec_module(G)
T = G.T
NAN = float("nan")

DS = "icentia11k"
PRIMARY = G.SLOWER
ROWS = [G.NO_AF, "label: first run (neither)", G.LORENZ, G.LORENZ_150, G.LORENZ_REACH, G.SLOWER, G.DOCTOR,
        G.DOCTOR_PSVT]
MAX_FALSE_PER_DAY = G.MAX_FALSE_NSRDB          # 0.29
STEP = 5.0                                     # seconds between decisions, as on the 24 h sets
NEAR_S = 30.0


def poisson_upper(k: int, days: float) -> float:
    """Exact (Garwood) 95% upper bound on a rate from k events in `days`."""
    from scipy.stats import chi2
    return float(chi2.ppf(0.975, 2 * (k + 1)) / 2.0 / days) if days > 0 else NAN


def beats_near(rec, t: float) -> str:
    s = np.asarray(rec.beat_samples) / rec.fs
    sym = np.asarray(rec.beat_symbols)[(s >= t - NEAR_S) & (s <= t + NEAR_S)]
    k, v = np.unique(sym, return_counts=True)
    return ", ".join(f"{a} {b}" for a, b in sorted(zip(k, v), key=lambda kv: -kv[1])) or "none"


def run_one(name: str, raw: Path) -> dict:
    from munji_ai.data.loader import load_record
    rec = load_record(DS, name, raw)
    r = G.process(rec, False, STEP)
    labels = {n: [float(a) for a in r["alerts"][n]] for n in ROWS}
    det = {a: {**r["detail_rows"][PRIMARY][a], "kind": G.classify(a, r["ivs"]), "near": beats_near(rec, a)}
           for a in labels[PRIMARY]}
    return {"name": r["name"], "patient": r["patient"], "hours": r["hours"], "labels": labels, "detail": det}


def _work(job) -> dict:
    name, raw = job
    t1 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                # noqa: BLE001  torch is optional here
        pass
    try:
        r = run_one(name, Path(raw))
    except Exception as e:                           # noqa: BLE001  one bad record must not end the run
        return {"name": name, "error": f"{type(e).__name__}: {e}"}
    r["secs"] = time.time() - t1
    return r


# ------------------------------------------------------------------ fast sinus, judged on the annotations
def onset_on_annotations(path: Path, start: int, end: int) -> dict:
    """For one fast 'sinus' episode (samples at the record's rate), on the
    annotated beats: its first beat whose running rate is 150 or more, the
    onset drop of the run over 150 and of the run over 100 that hold that
    beat, the seconds from the 100 run's start to 150, and the annotated 30 s
    rate before the 150 run (median beat-to-beat interval)."""
    import wfdb
    ann = wfdb.rdann(str(path), "atr")
    fs = float(ann.fs or wfdb.rdheader(str(path)).fs)
    sym = np.asarray(ann.symbol)
    keep = np.isin(sym, list("NLRBAaJSVrFejnE/fQ?"))
    ts = np.asarray(ann.sample)[keep] / fs
    out = {"drop100": NAN, "drop150": NAN, "reach": NAN, "before": NAN}
    st150, drop150 = G.fast_runs(ts, G.FAST_BPM)
    st100, drop100 = G.fast_runs(ts, G.SVT_BPM)
    lo, hi = np.searchsorted(ts, [start / fs, end / fs])
    hit = [j for j in range(lo, min(hi + 1, len(ts))) if st150[j] >= 0]
    if not hit:
        return out
    j = hit[0]
    s150, s100 = int(st150[j]), int(st100[j])
    out["drop150"] = float(drop150.get(s150, NAN))
    out["drop100"] = float(drop100.get(s100, NAN)) if s100 >= 0 else NAN
    if s100 >= 0:
        out["reach"] = float(G.reach_times(ts, st100, G.FAST_BPM).get(s100, NAN))
    m = (ts[1:] >= ts[s150] - 30.0) & (ts[1:] < ts[s150])
    if m.sum() >= 5:
        out["before"] = 60.0 / float(np.median(np.diff(ts)[m]))
    return out


def fast_sinus_table(npz: Path, raw: Path, results: list) -> None:
    if not Path(npz).exists():
        print(f"\n(no {npz}: run python -m munji_ai.data.svt_runs first for the fast sinus episodes)")
        return
    z = np.load(npz)
    m = (z["kind"] == "sinus") & (z["dataset"] == DS)
    if not m.any():
        print("\nno fast sinus episodes in svt_runs")
        return
    from munji_ai.data import registry
    from munji_ai.data.loader import dataset_dir
    d = dataset_dir(registry.get(DS), raw)
    headers = {h.stem: h for h in d.rglob("*.hea")}
    by_name = {r["name"]: r for r in results if "error" not in r}
    nums = {n: str(k) for k, n in enumerate(ROWS, 1)}
    print(f"\nFast 'sinus' episodes from svt_runs ({int(m.sum())}): their onset on the annotated beats")
    print(f"{'patient':<10}{'record':<14}{'start':>9}{'dur':>6}{'peak':>6}{'drop100':>9}{'reach':>7}{'drop150':>9}"
          f"{'before':>8}  labelled by rows")
    abrupt = 0
    for k in np.flatnonzero(m):
        name = str(z["record"][k])
        fs = float(z["fs"][k])
        s, e = int(z["start"][k]), int(z["end"][k])
        o = onset_on_annotations(headers[name].with_suffix(""), s, e) if name in headers else {}
        r = by_name.get(name)
        lab = ""
        if r:
            lab = ", ".join(nums[n] for n in ROWS if any(s / fs - 30 <= a <= e / fs + 30 for a in r["labels"][n]))
        if np.isfinite(o.get("drop100", NAN)) and o["drop100"] >= T.ONSET_DROP:
            abrupt += 1
        print(f"{str(z['patient'][k]).split(':')[-1]:<10}{name:<14}{T._mmss(s / fs):>9}{(e - s) / fs:>6.0f}"
              f"{z['bpm'][k]:>6.0f}{T._f(o.get('drop100', NAN), '{:.0%}'):>9}{T._f(o.get('reach', NAN)):>6}s"
              f"{T._f(o.get('drop150', NAN), '{:.0%}'):>9}{T._f(o.get('before', NAN)):>8}  {lab or '-'}")
    print(f"(drop100 / drop150: the cycle-length drop at the start of the run over 100 / over 150 that holds the\n"
          f" episode, 9% or more is abrupt; reach: seconds from the 100 run's start to 150; before: the annotated 30 s\n"
          f" rate before the 150 run. {abrupt} of {int(m.sum())} started abruptly over 100: those are worth a look as"
          f" SVT\n marked as normal beats; a gradual start is the sinus rhythm of effort)")


# ------------------------------------------------------------------ report
def report(results: list) -> int:
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    hours = sum(r["hours"] for r in ok)
    days = hours / 24.0
    pats = {r["patient"] for r in ok}
    print(f"\nICENTIA11K daily life: {len(ok):,} recordings from {len(pats):,} patients, {hours:,.0f} h"
          + (f"; {len(bad)} unreadable" if bad else ""))
    for r in bad[:5]:
        print(f"  unreadable {r['name']}: {r['error']}")
    if not ok:
        return 1
    print("\nLabels per row (every label counted as false)")
    print(f"{'':<4}{'configuration':<34}{'labels':>8}{'per 24 h':>10}{'95% upper':>11}{'patients':>10}")
    for k, n in enumerate(ROWS, 1):
        c = sum(len(r["labels"][n]) for r in ok)
        pp = len({r["patient"] for r in ok if r["labels"][n]})
        print(f"{k:<4}{n:<34}{c:>8}{c / days:>10.3f}{poisson_upper(c, days):>11.3f}{pp:>10}")

    lab = [(r, a, v) for r in ok for a, v in r["detail"].items()]
    kinds = defaultdict(int)
    for _, _, v in lab:
        kinds[v["kind"]] += 1
    print(f"\nLabels of '{PRIMARY}': {len(lab)}" + (" (" + ", ".join(f"{v} {k}" for k, v in kinds.items()) + ")"
                                                   if lab else ""))
    for r, a, v in sorted(lab, key=lambda x: (x[0]["patient"], x[0]["name"], x[1]))[:40]:
        print(f"  {r['patient'].split(':')[-1]:<8}{r['name']:<14}{T._mmss(a):>8}  {v['bpm']:.0f} bpm, before "
              f"{T._f(v['before'])}, onset {T._f(v['drop'], '{:.0%}')}, regular {T._f(v['stab_run'])} ms, lz "
              f"{T._f(v['lz'], '{:.2f}')}, {v['kind']}; beats {v['near']}")

    c = sum(len(r["labels"][PRIMARY]) for r in ok)
    rate = c / days
    print(f"\nDAILY LIFE, '{PRIMARY}'")
    print(f"  {c} labels in {days:,.1f} days: {rate:.3f} per 24 h (95% upper {poisson_upper(c, days):.3f})  ->  "
          f"{'PASS' if rate <= MAX_FALSE_PER_DAY else 'FAIL'} (at most {MAX_FALSE_PER_DAY})")
    return 0 if rate <= MAX_FALSE_PER_DAY else 2


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--svt-runs", type=Path, default=C.CACHE_DIR / "svt_runs.npz")
    ap.add_argument("--max-patients", type=int, default=0, help="0 = all (sorted order)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    from munji_ai.data import registry
    from munji_ai.data.loader import _patient_id, list_records
    from munji_ai.data.splits import assign
    ds = registry.get(DS)
    try:
        names = list_records(DS, a.raw)
    except Exception as e:                           # noqa: BLE001
        print(f"no {DS}: {e}")
        return 1
    jobs, kept = [], set()
    for n in names:
        pid = _patient_id(ds, n)
        if assign(pid) == "test":
            continue
        if a.max_patients and pid not in kept and len(kept) >= a.max_patients:
            continue
        kept.add(pid)
        jobs.append((n, str(a.raw)))
    print(f"{len(jobs):,} recordings from {len(kept):,} patients (test split skipped), {a.workers} at a time", flush=True)
    t0 = time.time()
    results = []
    if a.workers <= 1:
        for k, j in enumerate(jobs, 1):
            results.append(_work(j))
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k:,}/{len(jobs):,} recordings, {time.time() - t0:.0f} s", flush=True)
    else:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
                results.append(r)
                if k % 200 == 0 or k == len(jobs):
                    el = time.time() - t0
                    print(f"  {k:,}/{len(jobs):,} recordings, {el:.0f} s, about {el / k * (len(jobs) - k) / 60:.0f}"
                          f" min left", flush=True)
    code = report(results)
    fast_sinus_table(a.svt_runs, a.raw, results)
    print(f"\n{time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
