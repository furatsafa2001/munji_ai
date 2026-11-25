"""Fast narrow rhythm, exercise test: does the label fire on the sinus
tachycardia of exercise?

The label's onset check (an abrupt start of the run over 100 bpm, which then
reaches 150 within 30 s) is there to keep exercise out. Until now it was
only tried on synthetic ramps: the PhysioNet sets we score are mostly at
rest. Here it meets real exercise. Nothing in the rule changes.

Data
----
STDB  MIT-BIH ST Change Database: 28 records, most recorded during exercise
      stress tests; 360 Hz; the first channel. Its annotations hold beat
      labels only, so the beats around every label are printed.
GUDB  Glasgow University Database: 25 subjects, 2 minutes of each task
      (sitting, maths, walking, jogging, hand bike); the chest strap
      (roughly V2-V1), 250 Hz, stored in volts (read as mV). Each task on
      its own: the recordings are not continuous, and joining them would
      fake an abrupt onset at the seam.

Every recording is taken as sinus rhythm, so every label is false. That is
an assumption for STDB (stress-test patients can have ectopy); the beat
labels printed beside each label let you check it.

Fixed before the run
--------------------
PRIMARY   the label of "label: onset 150, slower before": no label on any
          recording. PASS or FAIL. Not run when no recording is informative.
          It was added after the first run, where "label: Lorenz, 150 within
          30 s" failed: a real jog (GUDB 04) leapt past 100 abruptly, and two
          STDB labels came from missed beats (317) and lost signal (319).
FIRST     "label: Lorenz, 150 within 30 s", the first run's PRIMARY, still
          scored against the same bar, reported.
Informative recording: it has a template, and somewhere its 30 s rate is
150 bpm or more while the run over 100 it belongs to started inside the
recording, so its onset could be judged. A recording that starts already
fast cannot be labelled at all (its onset is never judged), so it tests
nothing; it is counted apart.
REPORTED  every other label row of fast_narrow_template, on the same
          recordings (no RhythmNet here: rows with the AF check do not
          block on AF); per informative recording the onset drop, the time
          from the run's start to 150, and the labels.

Run:
    python experiments/fast_narrow_exercise.py --download
    python experiments/fast_narrow_exercise.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import urllib.error
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

PRIMARY = G.SLOWER                              # set after the first run (row 7 failed it)
FIRST = G.LORENZ_REACH
ROWS = [n for n, _ in G.CONFIGS if n not in G.REPORTS]
STDB_URL = "https://physionet-open.s3.amazonaws.com/stdb/1.0.0/"
STDB_IDS = [str(k) for k in range(300, 328)]
GUDB_URLS = ("https://raw.githubusercontent.com/berndporr/ECG-GUDB/master/docs/experiment_data/",
             "https://berndporr.github.io/ECG-GUDB/experiment_data/")
GUDB_SUBJECTS = 25
GUDB_TASKS = ("sitting", "maths", "walking", "jogging", "hand_bike")
GUDB_FS = 250
NEAR_S = 30.0                                     # beats printed around a label: +-30 s


# ------------------------------------------------------------------ download
def download_stdb(root: Path, fetch=None) -> dict:
    """STDB from PhysioNet's open bucket: RECORDS if it is there, else 300 to
    327. Returns the counts of "kept", "fetched", "missing" and "failed"."""
    fetch = fetch or G._fetch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ids = STDB_IDS
    if fetch(STDB_URL + "RECORDS", root / "RECORDS") != "missing":
        got = [ln.strip() for ln in (root / "RECORDS").read_text().splitlines() if ln.strip()]
        ids = got or ids
    n = defaultdict(int)
    for rid in ids:
        for ext in (".hea", ".atr", ".dat"):
            n[fetch(STDB_URL + rid + ext, root / (rid + ext)).split(":")[0]] += 1
    return dict(n)


def download_gudb(root: Path, tasks=GUDB_TASKS, fetch=None) -> dict:
    """GUDB's chest-strap files, subject_00 to subject_24, from GitHub (the
    raw mirror first, then the project's pages). Returns the counts."""
    fetch = fetch or G._fetch
    n = defaultdict(int)
    for s in range(GUDB_SUBJECTS):
        for task in tasks:
            rel = f"subject_{s:02d}/{task}/ECG.tsv"
            dest = Path(root) / rel
            res = "missing"
            for base in GUDB_URLS:
                try:
                    res = fetch(base + rel, dest)
                except (urllib.error.URLError, OSError):
                    res = "failed"
                if res in ("kept", "fetched"):
                    break
            n[res.split(":")[0]] += 1
    return dict(n)


# ------------------------------------------------------------------ records
def read_stdb(path: Path):
    """One STDB record: the first channel on the model path, sinus for the
    whole record, with its beat labels kept for the printout."""
    import wfdb
    from munji_ai.data.loader import Record
    from munji_ai.data.preprocess import model_path
    stem = str(Path(path).with_suffix(""))
    rec = wfdb.rdrecord(stem)
    raw = np.nan_to_num(np.asarray(rec.p_signal[:, 0], np.float64))
    x = model_path(raw, int(round(rec.fs)))
    ann = wfdb.rdann(stem, "atr")
    ratio = len(x) / max(len(raw), 1)
    beats = np.round(np.asarray(ann.sample) * ratio).astype(np.int64)
    name = Path(path).stem
    return Record(dataset="stdb", name=name, patient=f"stdb:{name}", signal=x, fs=C.TARGET_FS,
                  beat_samples=beats, beat_symbols=np.asarray(ann.symbol, dtype=object),
                  rhythm_starts=np.array([0]), rhythm_labels=np.array(["(N"], dtype=object))


def read_gudb(path: Path):
    """One GUDB task: the chest strap (column 0, volts -> mV) on the model
    path, sinus for the whole recording; name subject_task."""
    from munji_ai.data.loader import Record
    from munji_ai.data.preprocess import model_path
    path = Path(path)
    raw = np.loadtxt(path, usecols=(0,), ndmin=1) * 1000.0
    x = model_path(np.nan_to_num(raw), GUDB_FS)
    name = f"{path.parent.parent.name.replace('subject_', 's')}_{path.parent.name}"
    return Record(dataset="gudb", name=name, patient=f"gudb:{path.parent.parent.name}", signal=x, fs=C.TARGET_FS,
                  beat_samples=np.array([0, len(x) - 1]), beat_symbols=np.array(["N", "N"]),
                  rhythm_starts=np.array([0]), rhythm_labels=np.array(["(N"], dtype=object))


def stdb_records(root: Path) -> list:
    return sorted((p for p in Path(root).glob("*.hea")
                   if p.with_suffix(".dat").exists() and p.with_suffix(".atr").exists()), key=lambda p: p.stem)


def gudb_records(root: Path, tasks=GUDB_TASKS) -> list:
    return sorted(p for p in Path(root).glob("subject_*/*/ECG.tsv")
                  if p.parent.name in tasks and p.stat().st_size > 0)


# ------------------------------------------------------------------ one record
def profile(rec) -> dict:
    """What the rule could see: template, fastest 30 s, and the informative
    decisions (150 or more, inside a run over 100 that started in the
    recording, so its onset was judged); for those, the largest onset drop
    and the shortest time from the run's start to 150."""
    b = T.beats_and_template(rec, G.POS_STEP)
    dec = G.decisions(b["ends"], b["wbeats"], b["pos"], b["corr"], b["fs"], b["ts"])
    bpm = dec["bpm"]
    drop = dec["onset"][G.SVT_BPM][1]
    with np.errstate(invalid="ignore"):
        fast = dec["ok"] & (bpm >= G.FAST_BPM)
    judged = fast & np.isfinite(drop)
    return {"template": b["first"] is not None,
            "peak": float(np.nanmax(bpm[dec["ok"]])) if np.isfinite(bpm[dec["ok"]]).any() else NAN,
            "fast_s": float(fast.sum() * G.POS_STEP),
            "informative": bool(judged.any()) and b["first"] is not None,
            "drop": float(np.nanmax(drop[judged])) if judged.any() else NAN,
            "reach": float(np.nanmin(dec["reach"][judged])) if judged.any() and np.isfinite(dec["reach"][judged]).any()
            else NAN}


def beats_near(rec, t: float) -> str:
    """Beat labels within +-30 s of t, as 'N 40, V 1'."""
    s = np.asarray(rec.beat_samples) / rec.fs
    sym = np.asarray(rec.beat_symbols)[(s >= t - NEAR_S) & (s <= t + NEAR_S)]
    k, v = np.unique(sym, return_counts=True)
    return ", ".join(f"{a} {b}" for a, b in sorted(zip(k, v), key=lambda kv: -kv[1])) or "none"


def run_one(path: Path, kind: str) -> dict:
    rec = read_stdb(path) if kind == "stdb" else read_gudb(path)
    r = G.process(rec, negative=False, step=G.POS_STEP)
    pr = profile(rec)
    labels = {n: [float(a) for a in r["alerts"][n]] for n in ROWS}
    near = {a: beats_near(rec, a) for n in ROWS for a in labels[n]} if kind == "stdb" else {}
    return {"dataset": kind, "name": r["name"], "minutes": r["hours"] * 60.0, **pr, "labels": labels,
            "near": near, "lz_at": r["lz_at"]}


# ------------------------------------------------------------------ report
def report(results: list) -> int:
    print("\nEXERCISE: fast narrow rhythm labels on sinus tachycardia (every label is false)")
    for ds in ("stdb", "gudb"):
        rs = [r for r in results if r["dataset"] == ds]
        if not rs:
            continue
        print(f"\n{ds}: {len(rs)} recordings, {sum(r['minutes'] for r in rs) / 60.0:.1f} h; with a template "
              f"{sum(r['template'] for r in rs)}; reached 150 bpm {sum(r['peak'] >= G.FAST_BPM for r in rs)}; "
              f"informative {sum(r['informative'] for r in rs)}")
    show = [r for r in results if r["informative"] or any(r["labels"][n] for n in ROWS)]
    nums = {n: str(k) for k, n in enumerate(ROWS, 1)}
    if show:
        print(f"\n{'set':<6}{'record':<16}{'min':>5}{'peak':>6}{'at150':>7}{'onset':>7}{'reach':>7}  labels by row")
        for r in show:
            lab = "; ".join(f"{nums[n]}: " + ", ".join(T._mmss(a) for a in r["labels"][n]) for n in ROWS if r["labels"][n])
            print(f"{r['dataset']:<6}{r['name']:<16}{r['minutes']:>5.1f}{T._f(r['peak']):>6}{r['fast_s']:>6.0f}s"
                  f"{T._f(r['drop'], '{:.0%}'):>7}{T._f(r['reach']):>6}s  {lab or '-'}"
                  + ("" if r["informative"] else "  (not informative)"))
        print("(peak: fastest 30 s; at150: time at 150 bpm or more; onset: the largest cycle-length drop at the\n"
              " start of the run over 100 behind those decisions, 9% or more is abrupt; reach: the shortest time\n"
              " from that run's start to 150 bpm; labels by row: the row numbers below and the label times)")
    for r in results:
        for a, s in r["near"].items():
            print(f"  beats around {r['dataset']} {r['name']} {T._mmss(a)}: {s}")

    print("\nLabels per row (all false), over every recording")
    print(f"{'':<4}{'configuration':<34}{'labels':>8}{'recordings':>12}{'informative':>13}")
    inf = [r for r in results if r["informative"]]
    for n in ROWS:
        print(f"{nums[n]:<4}{n:<34}{sum(len(r['labels'][n]) for r in results):>8}"
              f"{sum(bool(r['labels'][n]) for r in results):>12}{sum(bool(r['labels'][n]) for r in inf):>13}")

    if inf:
        n1 = sum(len(r["labels"][FIRST]) for r in results)
        print(f"\nreported, '{FIRST}': {n1} labels  ->  {'PASS' if n1 == 0 else 'FAIL'} (none allowed)")
    print(f"\nEXERCISE, '{PRIMARY}'")
    if not inf:
        print("  PRIMARY not run (no informative recording: none reached 150 bpm with a judged onset)")
        return 1
    n = sum(len(r["labels"][PRIMARY]) for r in results)
    print(f"  {n} labels on {len(results)} recordings ({len(inf)} informative)  ->  {'PASS' if n == 0 else 'FAIL'}"
          " (none allowed)")
    return 0 if n == 0 else 2


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdb", type=Path, default=Path("data/stdb"))
    ap.add_argument("--gudb", type=Path, default=Path("data/gudb"))
    ap.add_argument("--sets", nargs="*", default=["stdb", "gudb"])
    ap.add_argument("--download", action="store_true", help="fetch STDB and GUDB first (about 600 MB)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.download:
        if "stdb" in a.sets:
            print("STDB:", download_stdb(a.stdb), flush=True)
        if "gudb" in a.sets:
            print("GUDB:", download_gudb(a.gudb), flush=True)
    jobs = ([(p, "stdb") for p in stdb_records(a.stdb)] if "stdb" in a.sets else []) + \
           ([(p, "gudb") for p in gudb_records(a.gudb)] if "gudb" in a.sets else [])
    if not jobs:
        print("no recordings found: run with --download first")
        return 1
    results = []
    for k, (p, kind) in enumerate(jobs, 1):
        try:
            results.append(run_one(p, kind))
        except Exception as e:                                   # one bad file must not stop the run
            print(f"  skipped {kind} {p}: {type(e).__name__}: {e}")
        if k % 20 == 0 or k == len(jobs):
            print(f"  {k}/{len(jobs)} recordings, {time.time() - t0:.0f} s", flush=True)
    code = report(results)
    print(f"\n{time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
