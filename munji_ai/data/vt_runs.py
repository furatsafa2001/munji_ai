"""Find ventricular tachycardia in beat-annotated recordings, by its definition.

    python -m munji_ai.data.vt_runs                       # Icentia11k and LTAFDB
    python -m munji_ai.data.vt_runs --datasets icentia11k ltafdb edb

Why
---
The VT detector learnt ICU VT well but found under 15% of VT in ambulatory
recordings. The patch needs VT from its own kind of recording. Icentia11k is
a single-lead chest patch, like MUNJI, and every beat is labelled; it has no
VT label, but VT has a definition: three or more consecutive ventricular
beats faster than 100 bpm (VTaC uses five). So VT runs can be read straight
off the beat labels.

Only annotation files are read here, which is fast. Patients in MUNJI's test
split are skipped, so the test split stays untouched.

Output: data/cache/vt_runs.npz, one row per run
    dataset, record, patient, start, end (samples at the record's rate),
    fs, beats, bpm
and a summary: how many runs, from how many patients, how many slow
(100 to 149 bpm) and fast, and how many reach five beats.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, RAW_DIR
from . import registry
from .loader import _patient_id, dataset_dir, list_records
from .splits import assign

VENTRICULAR = {"V", "E"}


def find_runs(symbols, samples, fs: float, min_beats: int = 3, min_bpm: float = 100.0) -> list:
    """(start, end, beats, bpm) for each run of consecutive ventricular beats."""
    out, i, n = [], 0, len(symbols)
    while i < n:
        if symbols[i] not in VENTRICULAR:
            i += 1
            continue
        j = i
        while j + 1 < n and symbols[j + 1] in VENTRICULAR:
            j += 1
        beats = j - i + 1
        if beats >= min_beats:
            rr = np.diff(samples[i:j + 1]) / fs
            bpm = 60.0 / float(np.mean(rr)) if len(rr) and np.mean(rr) > 0 else 0.0
            if bpm >= min_bpm:
                out.append((int(samples[i]), int(samples[j]), beats, bpm))
        i = j + 1
    return out


def read_beats(path: Path):
    import wfdb

    for ext in ("atr", "ecg", "qrs"):
        try:
            ann = wfdb.rdann(str(path), ext)
            fs = ann.fs or wfdb.rdheader(str(path)).fs
            return np.asarray(ann.symbol), np.asarray(ann.sample), float(fs)
        except Exception:
            continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["icentia11k", "ltafdb"])
    ap.add_argument("--min-beats", type=int, default=3)
    ap.add_argument("--min-bpm", type=float, default=100.0)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    a = ap.parse_args(argv)

    rows = []
    t0 = time.time()
    for ds_key in a.datasets:
        try:
            ds = registry.get(ds_key)
            names = list_records(ds_key, a.raw)
        except Exception as e:
            print(f"  {ds_key:<11} skipped: {e}")
            continue
        root = dataset_dir(ds, a.raw)
        headers = {h.stem: h for h in root.rglob("*.hea")}    # one directory scan, not one per record
        scanned, test_skipped = 0, 0
        patients = set()
        for k, name in enumerate(names):
            if k and k % 200 == 0:
                print(f"  {ds_key:<11} {k:,}/{len(names):,} recordings   "
                      f"{len(rows):,} runs so far   {time.time() - t0:5.0f} s", flush=True)
            patient = _patient_id(ds, name)
            if ds.role == "val" or assign(patient) == "test":
                test_skipped += 1
                continue
            hea = headers.get(name)
            if hea is None:
                continue
            got = read_beats(hea.with_suffix(""))
            if got is None:
                continue
            sym, samp, fs = got
            scanned += 1
            patients.add(patient)
            for s, e, beats, bpm in find_runs(sym, samp, fs, a.min_beats, a.min_bpm):
                rows.append((ds_key, name, patient, s, e, fs, beats, bpm))
        print(f"  {ds_key:<11} {scanned:,} recordings from {len(patients):,} patients "
              f"({test_skipped} test-split recordings skipped)   {time.time() - t0:5.0f} s", flush=True)

    out = a.cache / "vt_runs.npz"
    a.cache.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*rows)) if rows else [[]] * 8
    np.savez(out, dataset=np.array(cols[0], dtype=str), record=np.array(cols[1], dtype=str),
             patient=np.array(cols[2], dtype=str), start=np.array(cols[3], dtype=np.int64),
             end=np.array(cols[4], dtype=np.int64), fs=np.array(cols[5], dtype=float),
             beats=np.array(cols[6], dtype=int), bpm=np.array(cols[7], dtype=float))

    print(f"\nsaved       {out}")
    if not rows:
        print("no VT runs found")
        return 0
    ds_arr = np.array(cols[0], dtype=str)
    pat = np.array(cols[2], dtype=str)
    beats = np.array(cols[6], dtype=int)
    bpm = np.array(cols[7], dtype=float)
    print(f"\n{'':<12}{'runs':>7}{'patients':>10}{'100-149':>9}{'150+':>7}{'5+ beats':>10}{'longest':>9}")
    for d in sorted(set(ds_arr)) + ["all"]:
        m = np.ones(len(ds_arr), bool) if d == "all" else ds_arr == d
        print(f"{d:<12}{int(m.sum()):>7,}{len(set(pat[m])):>10,}{int((m & (bpm < 150)).sum()):>9,}"
              f"{int((m & (bpm >= 150)).sum()):>7,}{int((m & (beats >= 5)).sum()):>10,}"
              f"{int(beats[m].max()):>8} b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
