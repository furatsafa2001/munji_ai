"""VTaC: ventricular tachycardia alarms from ICU monitors, read in small pieces.

    python -m munji_ai.data.vtac                         # true alarms (confirmed VT)
    python -m munji_ai.data.vtac --limit 700 --false 700 # plus 700 false alarms
    python -m munji_ai.data.vtac --all                   # everything
    python -m munji_ai.data.vtac --limit 700 --false 700 --baseline
                                   # the patients' own rhythm, a minute in

Source: PhysioNet VTaC v1.1, 5,037 VT alarms from ICU bedside monitors, each
labelled true or false by at least two expert annotators. A VT episode there
means five or more consecutive ventricular beats above 100 bpm, the clinical
definition the client uses. Each recording is 6 minutes with the alarm at the
5-minute mark.

What it reads
-------------
The full database is 18 GB. MUNJI needs one ECG lead and 40 s around each
alarm, so each event is read remotely in that slice only: about 20 kB per
event. The lead is chosen to resemble the patch: a precordial "V" lead first
(CM5 approximates V5), then II, then the others.

Output: data/cache/vtac_windows.npz
    X         (n, 10000) float32: 40 s, filtered like every other source
              (resampled to 250 Hz, band-passed, notch), alarm at 20 s
    record    patient waveform record (the grouping key: one patient)
    event     alarm event
    decision  True for a confirmed VT alarm, False for a false alarm
    split     train / val / test, from the authors' benchmark split
    lead      the ECG lead used

With --baseline, one 40 s slice per patient is read from 60 s into the
recording, four minutes before the alarm, into vtac_baseline.npz: the
patient's ordinary rhythm on the same monitor. A VT detector trained only on
true against false alarms never sees a normal heart, and in daily life it
called normal rhythm VT. Some of these slices will hold the patient's own
non-sustained VT; that label noise is accepted and reported.

Resumable: every event is saved in data/cache/vtac_parts/ as soon as it is
read, so an interrupted run continues where it stopped. Events whose slice is
mostly missing are skipped and counted.

The VTaC test split is kept apart downstream, as MUNJI's own test split is.
"""

from __future__ import annotations

import argparse
import csv
import io
import socket
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, TARGET_FS
from .preprocess import model_path

BASE_URL = "https://physionet.org/files/vtac/1.1/"
PN_DIR = "vtac/1.1/annotated_waveforms"
LEAD_ORDER = ("V", "V5", "V2", "II", "I", "III", "aVF", "aVL", "aVR", "MCL")
ALARM_S = 300.0
BEFORE_S, AFTER_S = 20.0, 20.0
BASELINE_FROM_S = 60.0


def _requests_timeout(seconds: float) -> None:
    """Give every requests call a timeout. wfdb reads PhysioNet through requests,
    whose default is to wait forever, so one dropped connection froze a run."""
    try:
        import requests
    except ImportError:
        return
    if getattr(requests.Session.request, "_munji_timeout", False):
        return
    original = requests.Session.request

    def request(self, method, url, **kw):
        kw.setdefault("timeout", seconds)
        return original(self, method, url, **kw)

    request._munji_timeout = True
    requests.Session.request = request


def fetch_csv(name: str) -> list[dict]:
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))


def pick_lead(names: list[str]) -> int | None:
    """Index of the ECG lead most like the patch, or None if there is no ECG."""
    clean = [n.strip() for n in names]
    for want in LEAD_ORDER:
        for i, n in enumerate(clean):
            if n.upper() == want.upper():
                return i
    return None


def read_event(record: str, event: str, start_s: float | None = None
               ) -> tuple[np.ndarray, int, str]:
    """One ECG lead, 40 s around the alarm (or from start_s), at the source rate."""
    import wfdb

    pn = f"{PN_DIR}/{record}"
    hdr = wfdb.rdheader(event, pn_dir=pn)
    ch = pick_lead(list(hdr.sig_name))
    if ch is None:
        raise ValueError("no ECG lead")
    fs = int(round(hdr.fs))
    a0 = ALARM_S - BEFORE_S if start_s is None else start_s
    lo, hi = int(a0 * fs), int((a0 + BEFORE_S + AFTER_S) * fs)
    rec = wfdb.rdrecord(event, pn_dir=pn, sampfrom=lo, sampto=hi, channels=[ch])
    return rec.p_signal[:, 0].astype(np.float64), fs, hdr.sig_name[ch]


def clean(x: np.ndarray, max_missing: float = 0.1) -> np.ndarray | None:
    bad = ~np.isfinite(x)
    if bad.mean() > max_missing:
        return None
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def label(row: dict) -> bool | None:
    v = str(row.get("decision", "")).strip().lower()
    if v in ("true", "1", "t", "yes"):
        return True
    if v in ("false", "0", "f", "no"):
        return False
    return None


def split_map(rows: list[dict]) -> dict:
    """event or record -> train / val / test, whatever the column is called."""
    if not rows:
        return {}
    col = next((c for c in rows[0] if c.lower() in ("split", "set", "subset")), None)
    out = {}
    for r in rows:
        s = str(r.get(col, "")).strip().lower() if col else ""
        s = "val" if s.startswith("val") else s
        for key in ("event", "record"):
            if r.get(key):
                out.setdefault(r[key].strip(), s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="false alarms too")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N true alarms")
    ap.add_argument("--false", type=int, default=0, dest="n_false",
                    help="add N false alarms, from the same patients first")
    ap.add_argument("--baseline", action="store_true",
                    help="read each patient's rhythm from 60 s in, not the alarms")
    a = ap.parse_args(argv)

    socket.setdefaulttimeout(60)       # a stalled connection fails instead of hanging
    _requests_timeout(60)              # wfdb reads through requests, which ignores it
    parts = a.cache / ("vtac_parts_baseline" if a.baseline else "vtac_parts")
    parts.mkdir(parents=True, exist_ok=True)
    labels = fetch_csv("event_labels.csv")
    splits = split_map(fetch_csv("benchmark_data_split.csv"))
    true_rows = [r for r in labels if label(r) is True]
    false_rows = [r for r in labels if label(r) is False]
    if a.all:
        rows = true_rows + false_rows
    else:
        rows = true_rows[:a.limit] if a.limit else true_rows
        if a.n_false:
            # False alarms from the same patients first: true and false then
            # differ in rhythm, not in who was monitored.
            mine = {r["record"] for r in rows}
            ranked = sorted(false_rows, key=lambda r: r["record"] not in mine)
            rows = rows + ranked[:a.n_false]
    if a.baseline:
        seen, first = set(), []
        for r in rows:
            if r["record"] not in seen:
                seen.add(r["record"])
                first.append(r)
        rows = first
    todo = [r for r in rows if not (parts / f"{r['record']}_{r['event']}.npz").exists()]
    n_f = sum(1 for r in rows if label(r) is False)
    print(f"events      {len(rows):,} ({len(rows) - n_f:,} true, {n_f:,} false), "
          f"{len(rows) - len(todo):,} already read")

    def work(r):
        x, fs, lead = read_event(r["record"], r["event"],
                                 BASELINE_FROM_S if a.baseline else None)
        x = clean(x)
        if x is None:
            return r, "missing"
        y = model_path(x, fs).astype(np.float32)
        np.savez(parts / f"{r['record']}_{r['event']}.npz", x=y, lead=lead)
        return r, "ok"

    t0, done, failed, shown = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(work, r): r for r in todo}
        for f in as_completed(futs):
            try:
                _, status = f.result()
                failed += status != "ok"
            except Exception as e:
                failed += 1
                if shown < 3:
                    r = futs[f]
                    print(f"  failed  {r['record']}/{r['event']}: {type(e).__name__}: {e}", flush=True)
                    shown += 1
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done:>5}/{len(todo)}   {failed} skipped   "
                      f"about {(len(todo) - done) / max(rate, 1e-6) / 60:.0f} min left", flush=True)

    X, rec, ev, dec, spl, leads = [], [], [], [], [], []
    n = int((BEFORE_S + AFTER_S) * TARGET_FS)
    for r in rows:
        p = parts / f"{r['record']}_{r['event']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        x = d["x"]
        if len(x) < n:
            continue
        X.append(x[:n])
        rec.append(r["record"])
        ev.append(r["event"])
        dec.append(bool(label(r)))
        spl.append(splits.get(r["event"], splits.get(r["record"], "")))
        leads.append(str(d["lead"]))
    out = a.cache / ("vtac_baseline.npz" if a.baseline else "vtac_windows.npz")
    np.savez_compressed(out, X=np.stack(X) if X else np.zeros((0, n), np.float32),
                        record=np.array(rec), event=np.array(ev), decision=np.array(dec),
                        split=np.array(spl), lead=np.array(leads))
    dec = np.array(dec, dtype=bool)
    print(f"\nsaved       {out}")
    print(f"events      {len(X):,} ({int(dec.sum()):,} true VT, {int((~dec).sum()):,} false alarms)"
          f" from {len(set(rec)):,} patients")
    print(f"leads       " + ", ".join(f"{k} {v}" for k, v in
                                      sorted(Counter(leads).items(), key=lambda kv: -kv[1])))
    print(f"splits      " + ", ".join(f"{k or '?'} {v}" for k, v in sorted(Counter(spl).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
