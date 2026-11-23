"""QRS onset, end and duration against cardiologists: can one lead measure width?

    pip install neurokit2                                    # once
    python experiments/qrs_delineation.py --download         # fetches a set only if it is missing
    python experiments/qrs_delineation.py --workers 6
    python experiments/qrs_delineation.py --max-records 5    # smoke test

Why
---
The slow-VT template found about half of real VT on VTaC: on one lead a VT
beat often still looks like the patient's own beat. Monitoring services alert
on width, not shape: the ARIC Zio protocol on "wide QRS tachycardia > 120 bpm
and sustained for > 30 seconds", MCOT on wide-complex tachycardia at 150 or
more. The engine's own width failed against the expert on QTDB (mean -50 ms,
SD 35 ms, eval_qrs_width_qtdb), so before width can join the template a
delineator has to pass here.

What the literature reports
---------------------------
Martinez et al. 2004 (wavelet delineator), QTDB, the better lead of each
beat: QRS onset 4.6 +/- 7.7 ms, end 0.8 +/- 8.7 ms. On LUDB, every lead
marked on its own (Kalyakulina et al., the LUDB paper): their own wavelet
method onset -8.1 +/- 7.7, end 3.8 +/- 8.8; ecg-kit's wavedet, an
implementation of Martinez's algorithm, -3.3 +/- 14.3 and 3.7 +/- 15.9. So
one lead on its own is expected to do worse than the better-lead figures.

What runs
---------
NeuroKit2 (MIT license) has two wavelet delineators that mark QRS onset and
end. "dwt" uses Martinez's wavelet (the quadratic spline filters) with
simpler search windows and thresholds; "cwt" uses Martinez's QRS thresholds
on a Gaussian-derivative transform. Neither is the published algorithm, so
both are measured. The engine's width (rate_engine.qrs_widths_ms) is shown
beside them for reference: it gives a duration only.

Every signal goes through the model path (250 Hz, band-passed, notch), as in
the app. Delineation runs at the expert's R peaks, so detection errors are
left out, on the whole signal around them so the delineator sees the rhythm:
the detector's beats, with the expert's peak in place of the beat it marked.
Beats within 0.5 s of a signal's edge are not scored.

Data
----
QTDB  105 two-lead Holter excerpts, 250 Hz. A cardiologist marked QRS onset
      and end on 30 or more beats of each (q1c), looking at both leads
      together, so one lead is expected to read shorter. Both leads are
      scored, each on its own.
LUDB  200 twelve-lead resting ECGs, 10 s, 500 Hz. Cardiologists marked every
      lead on its own. V5 is the lead nearest the patch's CM5. Bundle branch
      blocks and conduction delays bring wide QRS.

Scored
------
Per method, pooled and per lead, method minus expert in ms:
  onset, end, duration   mean +/- SD
  p95, >20 ms            95th percentile of the absolute duration error, and
                         the share of beats off by more than 20 ms
  found                  the share of beats the method marked at both ends
  IEC                    IEC 60601-2-25 for QRS duration: mean within +/-10
                         ms and SD 10 ms (config). It is the standard for
                         measuring duration on a diagnostic ECG; an alarm
                         needs the separation below more than the caliper
  rel SD                 each beat's duration minus the median of its record
                         and lead, against the same for the expert. A fixed
                         bias cancels here, as it would against the
                         patient's template, so only the SD counts
  wider                  beats the expert measured 20 ms or more wider than
                         their record's usual beat, as a VT beat would be:
                         how much wider the method measured them (median)
  wide >= 120            beats the expert measured at 120 ms or more, and how
                         well the method's value separates them (balanced
                         accuracy) at 120 ms and at its best cut

The better-lead figure of the literature is printed for QTDB, to compare with
Martinez; it is not what one lead can do. Reads only.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.data.preprocess import model_path  # noqa: E402
from munji_ai.rules.rate_engine import detect_beats, qrs_widths_ms  # noqa: E402

FS = C.TARGET_FS
METHODS = ("dwt", "cwt")             # NeuroKit2's wavelet delineators
ENGINE = "engine"                    # rate_engine.qrs_widths_ms: a duration only
ALL = (ENGINE,) + METHODS
WAVE_SYMBOLS = {"(", ")", "p", "t", "u", "+", "~", "|", '"'}
WIDE_MS = float(C.NARROW_QRS_MS)     # 120
IEC_MEAN_MS = IEC_SD_MS = 10.0       # IEC 60601-2-25, QRS duration (config)
BIG_MS = 20.0                        # "off by more than": the p95 companion
WIDER_MS = 20.0                      # a beat the expert measured this much wider than usual
EDGE_SEC = 0.5                       # beats this close to an edge are not scored
MATCH_SEC = 0.10                     # a detected beat this close is the expert's
BOUND_SEC = 0.25                     # an onset or end this close to a peak is that beat's
QTDB_MARGIN_SEC = 5.0                # signal kept around QTDB's marked beats
PATCH_LEAD = "v5"                    # the LUDB lead nearest CM5
LITERATURE = "Martinez 2004 (better lead): onset +4.6 +/- 7.7, end +0.8 +/- 8.7"


# ------------------------------------------------------------------- reading
def expert_qrs(symbols, samples) -> np.ndarray:
    """(onset, peak, end) samples of every QRS marked '(' peak ')'. P and T
    waves are marked the same way and are skipped by their symbol."""
    sym, smp = list(symbols), np.asarray(samples, np.int64)
    out = [(smp[j - 1], smp[j], smp[j + 1]) for j in range(1, len(sym) - 1)
           if sym[j] not in WAVE_SYMBOLS and sym[j - 1] == "(" and sym[j + 1] == ")"]
    return np.array(out, np.int64).reshape(-1, 3)


def _to_fs(marks: np.ndarray, fs: int, lo: int = 0) -> np.ndarray:
    return np.round((marks - lo) * (FS / fs)).astype(np.int64)


def _scored(marks: np.ndarray, n: int) -> np.ndarray:
    e = int(EDGE_SEC * FS)
    return marks[(marks[:, 0] >= e) & (marks[:, 2] < n - e)]


def qtdb_leads(path: Path) -> list:
    """[(lead, signal at FS, expert marks at FS)] of one QTDB record: both
    leads share the expert's marks."""
    import wfdb
    name = str(path.with_suffix(""))
    marks = expert_qrs(*_symbols(wfdb.rdann(name, "q1c")))
    if not len(marks):
        return []
    rec = wfdb.rdrecord(name)
    fs = int(rec.fs)
    lo = max(0, int(marks[:, 0].min() - QTDB_MARGIN_SEC * fs))
    hi = min(rec.p_signal.shape[0], int(marks[:, 2].max() + QTDB_MARGIN_SEC * fs))
    out = []
    for ch, lead in enumerate(rec.sig_name):
        y = model_path(np.nan_to_num(rec.p_signal[lo:hi, ch]), fs)
        out.append((str(lead).strip(), y, _scored(_to_fs(marks, fs, lo), len(y))))
    return out


def ludb_leads(path: Path) -> list:
    """[(lead, signal at FS, that lead's expert marks at FS)] of one LUDB
    record. Annotation files are named by lead: "v5" in LUDB 1.0.1, "atr_v5"
    in 1.0.0."""
    import wfdb
    name = str(path.with_suffix(""))
    rec = wfdb.rdrecord(name)
    fs = int(rec.fs)
    out = []
    for ch, lead in enumerate(rec.sig_name):
        ld = str(lead).strip().lower()
        ann = None
        for ext in (ld, f"atr_{ld}"):
            try:
                ann = wfdb.rdann(name, ext)
                break
            except Exception:                        # noqa: BLE001  no such file
                continue
        if ann is None:
            continue
        marks = expert_qrs(*_symbols(ann))
        if len(marks):
            y = model_path(np.nan_to_num(rec.p_signal[:, ch]), fs)
            out.append((ld, y, _scored(_to_fs(marks, fs), len(y))))
    return out


def _symbols(ann):
    return ann.symbol, ann.sample


READERS = {"qtdb": qtdb_leads, "ludb": ludb_leads}


def records(root: Path, dataset: str) -> list:
    """The record files of a dataset under root, in record order."""
    pattern = "*.q1c" if dataset == "qtdb" else "*.hea"
    return sorted(root.rglob(pattern), key=lambda p: (str(p.parent), len(p.stem), p.stem))


# --------------------------------------------------------------- delineation
def context_peaks(y: np.ndarray, expert_peaks: np.ndarray) -> np.ndarray:
    """The signal's beats for the delineator to see the rhythm: the
    detector's, with the expert's peak in place of the beat it marked; none
    within EDGE_SEC of an edge."""
    det = np.asarray(detect_beats(y, FS)[0], np.int64)
    if det.size and expert_peaks.size:
        det = det[np.abs(det[:, None] - expert_peaks[None, :]).min(axis=1) > MATCH_SEC * FS]
    p = np.unique(np.concatenate([det, expert_peaks]))
    e = int(EDGE_SEC * FS)
    return p[(p >= e) & (p < len(y) - e)]


def bounds_at(peaks: np.ndarray, onsets, ends) -> tuple:
    """Per peak, the method's QRS onset in the BOUND_SEC before it and end in
    the BOUND_SEC after it (NaN if none). Matched by position, not by index:
    NeuroKit2 drops some entries from its lists."""
    def finite(v):
        a = np.array([float(x) for x in v if x is not None], float)
        return np.sort(a[np.isfinite(a)])
    on, off = finite(onsets), finite(ends)
    w = BOUND_SEC * FS
    o, e = np.full(len(peaks), np.nan), np.full(len(peaks), np.nan)
    for i, p in enumerate(peaks):
        k = np.searchsorted(on, p, side="right") - 1
        if k >= 0 and p - on[k] <= w:
            o[i] = on[k]
        k = np.searchsorted(off, p, side="left")
        if k < off.size and off[k] - p <= w:
            e[i] = off[k]
    return o, e


def delineate(y: np.ndarray, peaks: np.ndarray, method: str):
    """NeuroKit2's QRS onsets and ends at the given peaks."""
    import neurokit2 as nk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, waves = nk.ecg_delineate(y, rpeaks=peaks, sampling_rate=FS, method=method)
    return waves["ECG_R_Onsets"], waves["ECG_R_Offsets"]


def measure(dataset: str, path: Path) -> dict:
    """Every lead of one record: the expert's marks and each method's, in
    samples at FS (the engine: a duration in ms)."""
    rows, failed = [], defaultdict(int)
    for lead, y, marks in READERS[dataset](path):
        if len(marks) < 3:
            continue
        peaks = marks[:, 1]
        ctx = context_peaks(y, peaks)
        row = {"set": dataset, "record": path.stem, "lead": lead, "marks": marks,
               ENGINE: qrs_widths_ms(y, peaks, FS)}
        for m in METHODS:
            try:
                row[m] = bounds_at(peaks, *delineate(y, ctx, m))
            except Exception:                        # noqa: BLE001  counted, the beats left unmarked
                failed[m] += 1
                row[m] = (np.full(len(peaks), np.nan), np.full(len(peaks), np.nan))
        rows.append(row)
    return {"rows": rows, "failed": dict(failed)}


def _work(job):
    dataset, path = job
    try:
        return measure(dataset, Path(path))
    except Exception as e:                           # noqa: BLE001  one bad record must not end the run
        return {"rows": [], "failed": {}, "error": f"{dataset} {Path(path).stem}: {type(e).__name__}: {e}"}


# ------------------------------------------------------------------- scoring
def _ms(samples):
    return np.asarray(samples, float) * (1000.0 / FS)


def beat_table(rows: list, method: str) -> dict:
    """Over the beats of rows: errors of onset, end and duration (ms), the
    expert's duration, the method's, and the relative-duration error."""
    cols = defaultdict(list)
    for r in rows:
        on_e, off_e = r["marks"][:, 0].astype(float), r["marks"][:, 2].astype(float)
        dur_e = _ms(off_e - on_e)
        if method == ENGINE:
            dur_m = np.asarray(r[ENGINE], float)
            err_on = err_off = np.full(dur_e.size, np.nan)
        else:
            on_m, off_m = r[method]
            dur_m = _ms(off_m - on_m)
            err_on, err_off = _ms(on_m - on_e), _ms(off_m - off_e)
        d_m = dur_m - (np.nanmedian(dur_m) if np.isfinite(dur_m).sum() >= 3 else np.nan)
        d_e = dur_e - np.median(dur_e)
        for k, v in (("on", err_on), ("off", err_off), ("dur", dur_m - dur_e), ("truth", dur_e),
                     ("meas", dur_m), ("rel", d_m - d_e), ("d_e", d_e), ("d_m", d_m)):
            cols[k].append(v)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}


def _mean_sd(e: np.ndarray):
    e = e[np.isfinite(e)]
    return (float(e.mean()), float(e.std())) if e.size else (float("nan"), float("nan"))


def separation(truth: np.ndarray, meas: np.ndarray):
    """(beats the expert measured >= 120 ms, balanced accuracy at 120 ms,
    best cut, its balanced accuracy)."""
    ok = np.isfinite(meas)
    wide, m = truth[ok] >= WIDE_MS, meas[ok]
    nan = float("nan")
    if not wide.any() or wide.all():
        return int(wide.sum()), nan, nan, nan

    def bal(cut):
        p = m >= cut
        return ((p & wide).sum() / wide.sum() + (~p & ~wide).sum() / (~wide).sum()) / 2.0
    cuts = np.unique(np.round(m))
    scores = [bal(c) for c in cuts]
    k = int(np.argmax(scores))
    return int(wide.sum()), float(bal(WIDE_MS)), float(cuts[k]), float(scores[k])


def metrics(rows: list, method: str) -> dict:
    t = beat_table(rows, method)
    n = int(t["truth"].size)
    out = {"n": n, "found": float(np.mean(np.isfinite(t["meas"]))) if n else float("nan")}
    out["on"], out["off"], out["dur"] = (_mean_sd(t[k]) for k in ("on", "off", "dur"))
    a = np.abs(t["dur"][np.isfinite(t["dur"])])
    out["p95"] = float(np.percentile(a, 95)) if a.size else float("nan")
    out["big"] = float(np.mean(a > BIG_MS)) if a.size else float("nan")
    out["iec"] = bool(abs(out["dur"][0]) <= IEC_MEAN_MS and out["dur"][1] <= IEC_SD_MS)
    out["rel_sd"] = _mean_sd(t["rel"])[1]
    wider = (t["d_e"] >= WIDER_MS) & np.isfinite(t["d_m"])
    out["wider"] = (int(wider.sum()), float(np.median(t["d_e"][wider])) if wider.any() else float("nan"),
                    float(np.median(t["d_m"][wider])) if wider.any() else float("nan"))
    out["sep"] = separation(t["truth"], t["meas"])
    return out


def better_lead(rows: list, method: str):
    """QTDB, the literature's way: per beat, the lead whose onset (and end)
    is nearer the expert's. Leads of a record share the expert's marks."""
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["record"]].append(r)
    on_all, off_all = [], []
    for rs in by_rec.values():
        if len(rs) < 2:
            continue
        on = np.stack([_ms(r[method][0] - r["marks"][:, 0]) for r in rs])
        off = np.stack([_ms(r[method][1] - r["marks"][:, 2]) for r in rs])
        for e, acc in ((on, on_all), (off, off_all)):
            ok = np.isfinite(e).any(axis=0)
            k = np.nanargmin(np.where(np.isfinite(e[:, ok]), np.abs(e[:, ok]), np.inf), axis=0)
            acc.append(e[:, ok][k, np.arange(int(ok.sum()))])
    cat = (lambda v: np.concatenate(v) if v else np.zeros(0))
    return _mean_sd(cat(on_all)), _mean_sd(cat(off_all))


# ------------------------------------------------------------------ printing
def _pm(v, signed=True):
    m, s = v
    if not np.isfinite(m):
        return "-"
    return f"{m:+.1f} +/- {s:.1f}" if signed else f"{m:.1f} +/- {s:.1f}"


def _f(v, fmt="{:.0f}"):
    return fmt.format(v) if v is not None and np.isfinite(v) else "-"


def table(rows: list, title: str) -> dict:
    print(f"\n{title}")
    print(f"{'method':<8}{'found':>6}{'onset':>16}{'end':>16}{'duration':>16}{'p95':>6}{'>20ms':>7}"
          f"{'IEC':>6}{'rel SD':>8}{'wider (exp/meas)':>18}{'wide>=120':>11}{'at 120':>8}{'best cut':>14}")
    res = {}
    for m in ALL:
        s = metrics(rows, m)
        res[m] = s
        nw, at, cut, best = s["sep"]
        wn, we, wm = s["wider"]
        wider = f"{wn}: {_f(we)}/{_f(wm)}" if wn else "-"
        best_cut = f"{cut:.0f} ms {best:.2f}" if np.isfinite(cut) else "-"
        print(f"{m:<8}{_f(100 * s['found'], '{:.0f}%'):>6}{_pm(s['on']):>16}{_pm(s['off']):>16}"
              f"{_pm(s['dur']):>16}{_f(s['p95']):>6}{_f(100 * s['big'], '{:.0f}%'):>7}"
              f"{('PASS' if s['iec'] else 'FAIL'):>6}{_f(s['rel_sd'], '{:.1f}'):>8}"
              f"{wider:>18}{nw:>11}{_f(at, '{:.2f}'):>8}{best_cut:>14}")
    return res


def per_lead(rows: list, title: str) -> None:
    leads = sorted({r["lead"] for r in rows})
    print(f"\n{title}: duration error, mean +/- SD (beats)")
    print(f"{'lead':<8}" + "".join(f"{m:>24}" for m in ALL))
    for ld in leads:
        rs = [r for r in rows if r["lead"] == ld]
        cells = []
        for m in ALL:
            s = metrics(rs, m)
            cells.append(f"{_pm(s['dur'])} ({int(round(s['found'] * s['n']))})")
        print(f"{ld:<8}" + "".join(f"{c:>24}" for c in cells))


def report(results: dict) -> None:
    for ds, out in results.items():
        rows = out["rows"]
        if not rows:
            print(f"\n{ds}: nothing read")
            continue
        n_beats = sum(len(r["marks"]) for r in rows)
        n_rec = len({r["record"] for r in rows})
        if ds == "qtdb":
            head = (f"QTDB: {n_beats:,} beat-leads from {n_rec} records, each lead on its own "
                    f"(the expert marked both leads together)")
        else:
            head = f"LUDB: {n_beats:,} beat-leads from {n_rec} records, every lead marked on its own"
        res, where = table(rows, head), "all leads"
        if ds == "qtdb":
            print("better lead of each beat, as the literature reports it (not what one lead can do):")
            for m in METHODS:
                on, off = better_lead(rows, m)
                print(f"  {m:<6} onset {_pm(on)}, end {_pm(off)}")
            print(f"  {LITERATURE}")
        if ds == "ludb":
            v5 = [r for r in rows if r["lead"] == PATCH_LEAD]
            if v5:
                res, where = table(v5, f"LUDB {PATCH_LEAD.upper()}, the lead nearest the patch's CM5"), "V5"
        per_lead(rows, f"{ds.upper()} per lead")
        if out["failed"]:
            print(f"delineator errors (signals left unmarked): {out['failed']}")
        verdict = ", ".join(f"{m} {'PASS' if res[m]['iec'] else 'FAIL'} ({_pm(res[m]['dur'])} ms)" for m in ALL)
        print(f"IEC 60601-2-25, {ds.upper()} {where}: {verdict}")
    print("\n(method minus expert, ms. p95: 95th percentile of the absolute duration error. rel SD: SD of the"
          "\n duration change against the record's usual beat, method against expert. wider: beats the expert"
          f"\n measured {WIDER_MS:.0f} ms or more wider than usual, their median widening by expert / by method."
          f"\n at 120 and best cut: balanced accuracy separating beats the expert measured >= {WIDE_MS:.0f} ms)")


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qtdb", type=Path, default=Path("data/qtdb"))
    ap.add_argument("--ludb", type=Path, default=Path("data/ludb"))
    ap.add_argument("--sets", nargs="*", default=["qtdb", "ludb"], choices=["qtdb", "ludb"])
    ap.add_argument("--download", action="store_true",
                    help="download each set that is not there yet (wfdb, PhysioNet)")
    ap.add_argument("--max-records", type=int, default=0, help="per set (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if importlib.util.find_spec("neurokit2") is None:
        print("NeuroKit2 is not installed:  pip install neurokit2")
        return 1
    roots = {"qtdb": a.qtdb, "ludb": a.ludb}
    if a.download:
        import wfdb
        for ds in a.sets:
            if records(roots[ds], ds):
                print(f"{ds}: already in {roots[ds]}")
                continue
            roots[ds].mkdir(parents=True, exist_ok=True)
            print(f"downloading {ds} into {roots[ds]}", flush=True)
            wfdb.dl_database(ds, str(roots[ds]))
    jobs = []
    for ds in a.sets:
        recs = records(roots[ds], ds)
        if a.max_records:
            recs = recs[:a.max_records]
        if not recs:
            print(f"{ds}: no records under {roots[ds]} (--download fetches them)")
        jobs += [(ds, str(p)) for p in recs]
    if not jobs:
        return 1
    print(f"{len(jobs)} records, {a.workers} at a time", flush=True)
    results = {ds: {"rows": [], "failed": defaultdict(int)} for ds in a.sets}
    errors = []

    def done(job, out):
        if "error" in out:
            errors.append(out["error"])
        results[job[0]]["rows"] += out["rows"]
        for m, k in out["failed"].items():
            results[job[0]]["failed"][m] += k

    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(_work, j): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                done(futs[f], f.result())
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}", flush=True)
    else:
        for j in jobs:
            done(j, _work(j))
    order = {(ds, Path(p).stem): i for i, (ds, p) in enumerate(jobs)}
    for ds in results:
        results[ds]["rows"].sort(key=lambda r: (order.get((ds, r["record"]), 0), r["lead"]))
        results[ds]["failed"] = dict(results[ds]["failed"])
    report(results)
    if errors:
        print(f"\n{len(errors)} records not read: " + "; ".join(errors[:5]) + (" ..." if len(errors) > 5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
