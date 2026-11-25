"""Find SVT in beat-annotated recordings, by its definition, and count the fast
sinus rhythm of daily life.

    python -m munji_ai.data.svt_runs                       # Icentia11k
    python -m munji_ai.data.svt_runs --datasets icentia11k ltafdb

Why
---
The fast narrow rhythm label was scored on 17 SVT episodes of 30 s or more
at 150 bpm or more, from 5 patients, 10 of them from one. Its exercise test
had young people jogging (GUDB) and stress tests (STDB), not the daily life
of the elderly. Icentia11k is a single-lead chest patch, like MUNJI, worn
for days by older patients, and every beat is labelled. It has no SVT label,
but SVT has a definition: a run of consecutive supraventricular premature
beats faster than 100 bpm (as vt_runs reads VT off ventricular beats). And
its normal beats show how fast these patients' sinus rhythm gets.

This script only counts, to decide whether Icentia is worth building on:
  1. how much recording each patient has (segments and hours);
  2. SVT runs: 3 or more consecutive supraventricular premature beats
     (S, A, a, J) at 100 bpm or more; how many last 30 s or more, how many
     reach 150 bpm, how many do both (the episodes the label is scored on),
     and their lengths. Runs inside an AF or flutter mark are counted apart;
  3. fast sinus: 30 s windows (every 10 s) of normal beats (N, 90% or more of
     the window, outside AF and flutter marks) at 150 bpm or more, merged
     into episodes; and each patient's fastest 30 s of sinus rhythm.

Caution, before building on it: technologists may label the beats inside a
long SVT as normal (N), as the beat model calls the beats of sustained VT
normal. The run-length table shows whether long S runs exist at all. A fast
"sinus" episode may also be an SVT labelled N. Icentia gives no per-patient
age.

Only annotation files are read, which is fast. Patients in MUNJI's test
split are skipped, so the test split stays untouched.

Output: data/cache/svt_runs.npz, one row per SVT run or fast sinus episode
    kind ("svt" or "sinus"), dataset, record, patient, start, end (samples
    at the record's rate), fs, beats, bpm, in_af
and a summary.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, RAW_DIR
from . import registry
from .loader import _patient_id, dataset_dir, list_records
from .splits import assign

SUPRA = {"S", "A", "a", "J"}            # supraventricular / atrial / nodal premature
BEATS = set("NLRBAaJSVrFejnE/fQ?")     # WFDB beat symbols; the rest are marks
NORMAL = {"N"}
AF_MARKS = ("(AFIB", "(AFL")
LONG_S = 30.0                           # sustained: the label's 30 s
FAST_BPM = 150.0                        # the label's line
SVT_BPM = 100.0                         # the definition
WIN_S, STEP_S = 30.0, 10.0              # fast sinus windows
SINUS_SHARE = 0.90
LENGTH_BINS = ((3, 4), (5, 9), (10, 74), (75, 10 ** 9))


def find_runs(symbols, samples, fs: float, min_beats: int = 3, min_bpm: float = SVT_BPM) -> list:
    """(i, j, start, end, beats, bpm) for each run of consecutive
    supraventricular premature beats; i, j index the first and last beat."""
    out, i, n = [], 0, len(symbols)
    while i < n:
        if symbols[i] not in SUPRA:
            i += 1
            continue
        j = i
        while j + 1 < n and symbols[j + 1] in SUPRA:
            j += 1
        beats = j - i + 1
        if beats >= min_beats:
            rr = np.diff(samples[i:j + 1]) / fs
            bpm = 60.0 / float(np.mean(rr)) if len(rr) and np.mean(rr) > 0 else 0.0
            if bpm >= min_bpm:
                out.append((i, j, int(samples[i]), int(samples[j]), beats, bpm))
        i = j + 1
    return out


def rhythm_at(ann_samples, aux, beat_samples) -> np.ndarray:
    """The rhythm mark in force at each beat: the last '(' mark at or before
    it ('' before the first). A ')' does not end it, as in the loader."""
    aux = np.asarray([str(a).strip("\x00").strip() for a in aux], dtype=object) if aux is not None else None
    if aux is None or not len(aux):
        return np.full(len(beat_samples), "", dtype=object)
    k = np.array([a.startswith("(") for a in aux])
    ms, my = np.asarray(ann_samples)[k], aux[k]
    if not ms.size:
        return np.full(len(beat_samples), "", dtype=object)
    idx = np.searchsorted(ms, beat_samples, side="right") - 1
    out = np.full(len(beat_samples), "", dtype=object)
    out[idx >= 0] = my[idx[idx >= 0]]
    return out


def fast_sinus(symbols, samples, fs: float, in_af: np.ndarray) -> tuple:
    """(episodes, fastest): 30 s windows every 10 s whose beats are 90% or
    more normal and outside AF/flutter, at FAST_BPM or more by the median
    normal-to-normal interval, merged into (start, end, beats, bpm) episodes;
    and the fastest such window's rate (NaN when no window qualifies)."""
    t = np.asarray(samples) / fs
    if t.size < 3:
        return [], float("nan")
    sym = np.asarray(symbols)
    nn = (sym[1:] == "N") & (sym[:-1] == "N") & ~in_af[1:] & ~in_af[:-1]
    rr = np.diff(t)
    ends = np.arange(t[0] + WIN_S, t[-1] + 1e-9, STEP_S)
    lo = np.searchsorted(t, ends - WIN_S, side="left")
    hi = np.searchsorted(t, ends, side="right")
    rates = np.full(ends.size, np.nan)
    for w in range(ends.size):
        a, b = lo[w], hi[w]
        if b - a < 10:
            continue
        s = sym[a:b]
        if np.mean(s == "N") < SINUS_SHARE or in_af[a:b].any():
            continue
        m = nn[a:b - 1]
        if m.sum() < 5:
            continue
        rates[w] = 60.0 / float(np.median(rr[a:b - 1][m]))
    fastest = float(np.nanmax(rates)) if np.isfinite(rates).any() else float("nan")
    with np.errstate(invalid="ignore"):
        fast = rates >= FAST_BPM
    eps, w = [], 0
    while w < ends.size:
        if not fast[w]:
            w += 1
            continue
        v = w
        while v + 1 < ends.size and fast[v + 1]:
            v += 1
        s0, e0 = ends[w] - WIN_S, ends[v]
        a, b = np.searchsorted(t, s0), np.searchsorted(t, e0, side="right")
        eps.append((int(round(s0 * fs)), int(round(e0 * fs)), int(b - a), float(np.nanmax(rates[w:v + 1]))))
        w = v + 1
    return eps, fastest


def read_annotations(path: Path):
    """(beat symbols, beat samples, fs, rhythm at each beat) or None."""
    import wfdb
    for ext in ("atr", "ecg", "qrs"):
        try:
            ann = wfdb.rdann(str(path), ext)
        except Exception:
            continue
        fs = float(ann.fs or wfdb.rdheader(str(path)).fs)
        sym = np.asarray(ann.symbol)
        samp = np.asarray(ann.sample)
        aux = getattr(ann, "aux_note", None)
        beat = np.isin(sym, list(BEATS))
        rh = rhythm_at(samp, aux, samp[beat]) if aux is not None else np.full(int(beat.sum()), "", dtype=object)
        return sym[beat], samp[beat], fs, rh
    return None


def summarize(rows: list, hours: dict, segs: dict, fastest: dict, skipped: dict) -> None:
    by_ds = defaultdict(list)
    for (d, p), v in hours.items():
        by_ds[d].append((p, v, segs[(d, p)]))
    for d, items in sorted(by_ds.items()):
        hh = np.array([v for _, v, _ in items])
        ss = np.array([s for _, _, s in items])
        print(f"\n{d}: {ss.sum():,} recordings from {len(items):,} patients ({skipped.get(d, 0):,} test-split "
              f"recordings skipped), {hh.sum():,.1f} h")
        print(f"  per patient: recordings median {np.median(ss):.0f} (min {ss.min()}, max {ss.max()}), "
              f"hours median {np.median(hh):.1f} (min {hh.min():.1f}, max {hh.max():.1f})")

        r = [x for x in rows if x[0] == "svt" and x[1] == d]
        print(f"\n  SVT runs (3+ supraventricular premature beats at 100+ bpm)")
        if not r:
            print("    none")
        else:
            dur = np.array([(x[5] - x[4]) / x[6] for x in r])
            bpm = np.array([x[8] for x in r])
            beats = np.array([x[7] for x in r])
            in_af = np.array([x[9] for x in r])
            pat = np.array([x[3] for x in r], dtype=object)
            ok = ~in_af
            print(f"    {'':<34}{'runs':>8}{'patients':>10}")
            for name, m in (("all, outside AF/flutter marks", ok),
                            ("lasting 30 s or more", ok & (dur >= LONG_S)),
                            ("150 bpm or more", ok & (bpm >= FAST_BPM)),
                            ("both (what the label is scored on)", ok & (dur >= LONG_S) & (bpm >= FAST_BPM)),
                            ("inside AF/flutter marks (apart)", in_af)):
                print(f"    {name:<34}{int(m.sum()):>8,}{len(set(pat[m])):>10,}")
            print("    run length (outside AF/flutter): " + ", ".join(
                f"{a}-{b if b < 10 ** 8 else ''} beats {int((ok & (beats >= a) & (beats <= b)).sum()):,}"
                for a, b in LENGTH_BINS) + f"; longest {int(beats[ok].max()) if ok.any() else 0} beats"
                f" ({dur[ok].max() if ok.any() else 0:.0f} s)")
        f = [x for x in rows if x[0] == "sinus" and x[1] == d]
        fr = np.array([v for (dd, _), v in fastest.items() if dd == d])
        fr = fr[np.isfinite(fr)]
        print(f"\n  Fast sinus (30 s of normal beats at 150+ bpm, outside AF/flutter marks)")
        print(f"    episodes {len(f):,} in {len(set(x[3] for x in f)):,} patients"
              + (f"; fastest {max(x[8] for x in f):.0f} bpm" if f else ""))
        if fr.size:
            print(f"    each patient's fastest 30 s of sinus: median {np.median(fr):.0f}, 90th percentile "
                  f"{np.percentile(fr, 90):.0f}, max {fr.max():.0f} bpm; reached 130: {int((fr >= 130).sum()):,}, "
                  f"150: {int((fr >= 150).sum()):,} of {fr.size:,} patients")
    print("\n(a fast 'sinus' episode may be an SVT labelled N; long SVT runs labelled S may be rare if the beats of a"
          "\n sustained SVT were labelled N: see the run lengths)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["icentia11k"])
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--max-patients", type=int, default=0, help="per dataset, 0 = all (sorted order)")
    a = ap.parse_args(argv)

    rows, hours, segs, fastest, skipped = [], defaultdict(float), defaultdict(int), {}, defaultdict(int)
    t0 = time.time()
    for ds_key in a.datasets:
        try:
            ds = registry.get(ds_key)
            names = list_records(ds_key, a.raw)
        except Exception as e:
            print(f"  {ds_key:<11} skipped: {e}")
            continue
        root = dataset_dir(ds, a.raw)
        headers = {h.stem: h for h in root.rglob("*.hea")}
        kept_patients = set()
        for k, name in enumerate(names):
            if k and k % 500 == 0:
                print(f"  {ds_key:<11} {k:,}/{len(names):,} recordings   {time.time() - t0:5.0f} s", flush=True)
            patient = _patient_id(ds, name)
            if ds.role == "val" or assign(patient) == "test":
                skipped[ds_key] += 1
                continue
            if a.max_patients and patient not in kept_patients and len(kept_patients) >= a.max_patients:
                continue
            hea = headers.get(name)
            if hea is None:
                continue
            got = read_annotations(hea.with_suffix(""))
            if got is None:
                continue
            sym, samp, fs, rh = got
            if samp.size < 2:
                continue
            kept_patients.add(patient)
            hours[(ds_key, patient)] += float(samp[-1] - samp[0]) / fs / 3600.0
            segs[(ds_key, patient)] += 1
            in_af = np.array([str(x).startswith(AF_MARKS) for x in rh])
            for i, j, s, e, beats, bpm in find_runs(sym, samp, fs):
                rows.append(("svt", ds_key, name, patient, s, e, fs, beats, bpm, bool(in_af[i:j + 1].any())))
            eps, fmax = fast_sinus(sym, samp, fs, in_af)
            for s, e, beats, bpm in eps:
                rows.append(("sinus", ds_key, name, patient, s, e, fs, beats, bpm, False))
            if np.isfinite(fmax):
                fastest[(ds_key, patient)] = max(fastest.get((ds_key, patient), 0.0), fmax)
        print(f"  {ds_key:<11} {sum(1 for (d, _) in segs if d == ds_key):,} patients read   "
              f"{time.time() - t0:5.0f} s", flush=True)

    out = a.cache / "svt_runs.npz"
    a.cache.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*rows)) if rows else [[]] * 10
    np.savez(out, kind=np.array(cols[0], dtype=str), dataset=np.array(cols[1], dtype=str),
             record=np.array(cols[2], dtype=str), patient=np.array(cols[3], dtype=str),
             start=np.array(cols[4], dtype=np.int64), end=np.array(cols[5], dtype=np.int64),
             fs=np.array(cols[6], dtype=float), beats=np.array(cols[7], dtype=int),
             bpm=np.array(cols[8], dtype=float), in_af=np.array(cols[9], dtype=bool))
    print(f"\nsaved       {out}")
    if not hours:
        print("no recordings read")
        return 1
    summarize(rows, hours, segs, fastest, skipped)
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
