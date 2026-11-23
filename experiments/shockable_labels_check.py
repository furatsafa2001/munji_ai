"""Check the per-window shockable labels before the cache is rebuilt.

    python experiments/shockable_labels_check.py

Four checks, each skipped with a note when its data is missing:

ec13       the rate on the ANSI/AAMI EC13 ventricular tachycardia test
           waveforms (PhysioNet aami-ec13): Figure 4a, VT at 206 bpm and 1 mV,
           and 4b, 195 bpm and 2 mV, each also at half and double size. EC13
           and IEC 60601-2-27 test monitors' heart-rate alarms with them. The
           autocorrelation rate must read the same at all three sizes; the rate
           engine's detector is shown beside it. On PhysioNet the _d records
           are the half-size ones and _h the double-size ones (its note).
               aws s3 sync --no-sign-request s3://physionet-open/aami-ec13/1.0.0/ data/raw/aami-ec13/
fragments  the rate band against cardiologist labels: the ECG Fragment
           Database (PhysioNet ecg-fragment-high-risk-label) re-labels VFDB in
           2 s fragments, reviewed by cardiologists of the Almazov Centre, and
           splits VT into high rate (VTHR), low rate (VTLR) and torsade de
           pointes (VTTdP). Each fragment's name gives its VFDB record and
           second, so the 5 s window around it is read from our own VFDB copy.
           The database gives no bpm line between high and low rate, so this
           shows agreement, not a threshold. It also says what the windows the
           rule calls irregular are: VF or torsade de pointes every time, which
           is why VT_IRREGULAR is shockable (aha-window-v2). The fragments'
           seconds and samples are on a 360 Hz time base although VFDB is
           250 Hz (true time = 1.44 x the name), so each fragment is found in
           our copy by its waveform. Only the frag folders are needed:
               aws s3 sync --no-sign-request s3://physionet-open/ecg-fragment-high-risk-label/1.0.0/ \\
                   data/raw/ecg_fragments/ --exclude "*" --include "*/frag/*"
diff       what the new rules change in the shockable cache: every 5 s window
           of CUDB and VFDB under the old labels (episode rate, purity 0.5)
           and the new (data/shockable_labels.py), in minutes, by reason.
           Counts are before the cache's per-record sampling.
negatives  the mined hard negatives (data/cache/shockable_extra_neg_v2.npz,
           --extra-negatives in training) were kept by the old labels. This
           counts those the new labels call shockable or intermediate (VT
           windows of LTAFDB; Icentia11k has no VT labels and is not read).
           --write-clean PATH saves a copy without them, the one file this
           script ever writes, and only when asked.

Reads only, unless --write-clean is given. Test-split patients are skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.data import preprocess as pp  # noqa: E402
from munji_ai.data.loader import _patient_id, load_record  # noqa: E402
from munji_ai.data.registry import get  # noqa: E402
from munji_ai.data.shockable_labels import (INTERMEDIATE, acf_rates, label_windows,  # noqa: E402
                                            window_class)
from munji_ai.data.splits import assign  # noqa: E402
from munji_ai.data.windows import (_frames, dominant_label, rhythm_intervals,  # noqa: E402
                                   split_vt_by_rate)
from munji_ai.eval.eval_beats_model import eligible  # noqa: E402
from munji_ai.rules.rate_engine import detect_beats  # noqa: E402

FS = C.TARGET_FS
WIN = C.window_samples("shockable")
STEP_MIN = C.WINDOW_STRIDE_SEC["shockable"] / 60.0
EC13 = {"aami4a": (206, "1x"), "aami4a_d": (206, "half"), "aami4a_h": (206, "double"),
        "aami4b": (195, "1x"), "aami4b_d": (195, "half"), "aami4b_h": (195, "double")}
EC13_LEAD_IN_SEC = 30.0          # each record opens with 30 s of triangle waves
FRAGMENT = re.compile(r"^(\d+)_C_([A-Za-z]+)_(\d+)s_frag$", re.IGNORECASE)
BANDS = ("150+", "100-150", "<100", "irregular")


def band(rate: float) -> str:
    if not np.isfinite(rate):
        return "irregular"
    return "150+" if rate >= C.VT_FAST_BPM else ("100-150" if rate >= 100 else "<100")


# ------------------------------------------------------------------------ ec13
def part_ec13(root: Path) -> None:
    import wfdb

    print("\n1. EC13 VT test waveforms: the rate at half, normal and double size")
    found = {p.stem: p for p in root.rglob("aami4*.hea")} if root.exists() else {}
    if not found:
        print(f"   no aami4*.hea under {root}; download first (see the top of this file)")
        return
    print(f"   {'record':<10}{'size':>7}{'should':>8}{'autocorr':>10}{'range':>13}{'detector':>10}")
    for name, (bpm, size) in EC13.items():
        if name not in found:
            print(f"   {name:<10} not downloaded")
            continue
        r = wfdb.rdrecord(str(found[name].with_suffix("")))
        x = pp.model_path(np.nan_to_num(np.asarray(r.p_signal[:, 0], np.float64)), int(round(r.fs)))
        x = x[int(EC13_LEAD_IN_SEC * FS):]
        n = len(x) // WIN
        if n == 0:
            print(f"   {name:<10} too short")
            continue
        rates, _ = acf_rates(x[:n * WIN].reshape(n, WIN), FS)
        w40 = int(C.RULES_WINDOW_SEC * FS)
        dets = []
        for e in range(w40, len(x) + 1, int(10 * FS)):        # the rate engine's 40 s windows
            peaks, _ = detect_beats(x[e - w40:e], FS)
            dets.append(60.0 * FS / np.median(np.diff(peaks)) if len(peaks) > 2 else np.nan)
        dets = np.asarray(dets)
        det = float(np.nanmedian(dets)) if np.isfinite(dets).any() else float("nan")
        fin = rates[np.isfinite(rates)]
        med = float(np.median(fin)) if fin.size else float("nan")
        rng = f"{fin.min():.0f}-{fin.max():.0f}" if fin.size else "-"
        note = [] if fin.size == n else [f"{n - fin.size} of {n} windows irregular"]
        if not np.isfinite(dets).all():
            note.append(f"detector: no rate in {int((~np.isfinite(dets)).sum())} of {len(dets)} 40 s windows")
        print(f"   {name:<10}{size:>7}{bpm:>8}{med:>10.1f}{rng:>13}{det:>10.1f}"
              + (f"   ({'; '.join(note)})" if note else ""))
    print("   Pass: the autocorrelation column within 5% of 'should' at every size. The detector")
    print("   column is the rate engine on its 40 s windows, shown for the rate engine's sake.")


# ------------------------------------------------------------------- fragments
ALIGN_SCALES = (1.0, 360.0 / 250.0, 250.0 / 360.0)   # the name's seconds as given, or read at the other rate
ALIGN_SEARCH_SEC = 10.0
ALIGN_MIN_R = 0.8


def read_fragment(path: Path):
    """(samples, sampling rate if the file states one) of one fragment file."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".hea":
            import wfdb
            r = wfdb.rdrecord(str(path.with_suffix("")))
            return np.asarray(r.p_signal[:, 0], np.float64), float(r.fs)
        if suffix == ".mat":
            from scipy.io import loadmat
            arrays = [np.asarray(v, np.float64).ravel() for k, v in loadmat(str(path)).items()
                      if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)]
            return (max(arrays, key=len), None) if arrays else (None, None)
        vals = []
        for line in path.read_text(errors="ignore").split():
            try:
                vals.append(float(line.replace(",", ".")))
            except ValueError:
                continue
        return (np.asarray(vals, np.float64), None) if vals else (None, None)
    except Exception:
        return None, None


def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, [1.0 / (fs / 2), 30.0 / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def best_match(seg: np.ndarray, tpl: np.ndarray):
    """Highest Pearson correlation of tpl against every position in seg, and where."""
    n = len(tpl)
    if len(seg) < n or n < 10:
        return -1.0, 0
    t = tpl - tpl.mean()
    t = t / (np.sqrt((t * t).sum()) + 1e-12)
    c = np.correlate(seg, t, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
    s1, s2 = cs[n:] - cs[:-n], cs2[n:] - cs2[:-n]
    r = c / np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-12))
    k = int(np.argmax(r))
    return float(r[k]), k


def part_fragments(root: Path, raw: Path) -> None:
    """Each fragment is first found in our VFDB copy by its waveform (the
    seconds in its name are only a starting point), then the 5 s window
    around it is labelled as the cache would label it."""
    import wfdb
    from scipy.signal import resample_poly

    from munji_ai.data.loader import dataset_dir

    print("\n2. Cardiologist labels (ECG Fragment Database) against the rate band")
    if not root.exists():
        print(f"   no {root}; download first (see the top of this file)")
        return
    files = {}
    for path in sorted(root.rglob("*")):
        m = FRAGMENT.match(path.stem)
        if m and path.is_file() and path.suffix.lower() != ".dat":
            key = (m.group(1), m.group(2).upper(), int(m.group(3)))
            if key not in files or path.parent.name.lower() == "frag":
                files[key] = path
    if not files:
        print(f"   no fragment files (record_C_LABEL_seconds_frag) under {root}")
        return
    ds = get("vfdb")
    recs, rows, skipped, wins = {}, [], Counter(), Counter()
    best_r = []
    for (rec_name, lab, sec), path in sorted(files.items()):
        try:
            if assign(_patient_id(ds, rec_name)) == "test":
                skipped["test split"] += 1
                continue
        except ValueError:
            skipped["not a VFDB record"] += 1
            continue
        if rec_name not in recs:
            try:
                hit = next(iter(dataset_dir(ds, raw).rglob(f"{rec_name}.hea")))
                r = wfdb.rdrecord(str(hit.with_suffix("")))
                chans = [_bandpass(np.nan_to_num(np.asarray(r.p_signal[:, c], np.float64)), r.fs)
                         for c in range(r.p_signal.shape[1])]
                rec = load_record("vfdb", rec_name, raw)
                recs[rec_name] = (rec, rhythm_intervals(rec), chans, float(r.fs))
            except Exception:
                recs[rec_name] = None
        if recs[rec_name] is None:
            skipped["VFDB record not found"] += 1
            continue
        rec, iv, chans, fs_raw = recs[rec_name]
        vals, fs_file = read_fragment(path)
        if vals is None or len(vals) < 100:
            skipped["fragment unreadable"] += 1
            continue
        best = (-1.0, None)
        for fs_frag in ([fs_file] if fs_file else [360.0, 250.0]):
            from math import gcd
            up, down = int(fs_raw), int(fs_frag)
            g = gcd(up, down)
            tpl = _bandpass(resample_poly(vals, up // g, down // g), fs_raw)
            for k in ALIGN_SCALES:
                a = int(max(0.0, (sec * k - ALIGN_SEARCH_SEC) * fs_raw))
                b = int((sec * k + ALIGN_SEARCH_SEC) * fs_raw) + len(tpl)
                for ch, xc in enumerate(chans):
                    rr, pos = best_match(xc[a:b], tpl)
                    if rr > best[0]:
                        best = (rr, (k, fs_frag, ch, (a + pos + len(tpl) / 2) / fs_raw))
        best_r.append(best[0])
        if best[0] < ALIGN_MIN_R:
            skipped[f"not found in VFDB (best correlation under {ALIGN_MIN_R})"] += 1
            continue
        k, fs_frag, ch, centre = best[1]
        wins[(k, fs_frag, ch)] += 1
        lo = int(centre * FS) - WIN // 2               # 5 s around the matched fragment
        if lo < 0 or lo + WIN > len(rec.signal):
            skipped["at the record's edge"] += 1
            continue
        rate, reg = acf_rates(rec.signal[lo:lo + WIN][None, :], FS)
        orig = dominant_label(iv, lo, lo + WIN, 0.5) or "mixed"
        new = label_windows(rec.signal, FS, iv, [lo], WIN)[0]
        rows.append({"frag": lab, "band": band(rate[0]), "rate": rate[0], "reg": reg[0],
                     "orig": orig, "new": new, "cls": window_class(new)})
    if best_r:
        print(f"   found in our VFDB copy: {sum(r >= ALIGN_MIN_R for r in best_r)} of {len(best_r)} fragments "
              f"(median best correlation {np.median(best_r):.2f})")
    if wins:
        print("   how the name's seconds map to VFDB time: " + ", ".join(
            f"x{k:.2f} at {fs:g} Hz, channel {ch}: {c}" for (k, fs, ch), c in wins.most_common()))
    if skipped:
        print("   skipped: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    if not rows:
        return
    labs = sorted({r["frag"] for r in rows}, key=lambda k: -sum(r["frag"] == k for r in rows))
    print(f"   {'cardiologist':<14}{'n':>5}" + "".join(f"{b:>11}" for b in BANDS)
          + f"{'median bpm':>12}   VFDB label (most common)")
    for k in labs:
        g = [r for r in rows if r["frag"] == k]
        rates = [r["rate"] for r in g if np.isfinite(r["rate"])]
        common = ", ".join(f"{lab} {c}" for lab, c in Counter(r["orig"] for r in g).most_common(2))
        print(f"   {k:<14}{len(g):>5}" + "".join(f"{np.mean([r['band'] == b for r in g]):>11.0%}" for b in BANDS)
              + f"{(f'{np.median(rates):.0f}' if rates else '-'):>12}   {common}")
    print("   VTHR high-rate VT, VTLR low-rate VT, VTTdP torsade de pointes, VFL flutter, VF fibrillation;"
          "\n   the rest are not ventricular arrhythmias. Bands come from our autocorrelation rate.")
    print("   What the cache would do with each cardiologist class (our label for the window):")
    for k in labs:
        g = [r for r in rows if r["frag"] == k]
        if k in ("VTHR", "VTLR", "VTTDP", "VF", "VFL"):
            print(f"     {k:<6}{len(g):>5}: " + ", ".join(
                f"{lab or 'transition'} {c}" for lab, c in Counter(r["new"] for r in g).most_common()))


# ------------------------------------------------------------------------ diff
def old_class(lab) -> str:
    if lab is None:
        return "left out"
    return "shockable" if lab in C.SHOCKABLE_RHYTHMS else "not shockable"


def part_diff(raw: Path, max_records: int) -> None:
    print("\n3. What the new rules change in the shockable cache (CUDB + VFDB, minutes, before sampling)")
    moves = defaultdict(float)
    n_rec = 0
    for ds in ("cudb", "vfdb"):
        for name in eligible(ds, raw, max_records):
            rec = load_record(ds, name, raw)
            iv = rhythm_intervals(rec)
            if not iv:
                continue
            n_rec += 1
            frames = list(_frames(rec, "shockable"))
            old_iv = split_vt_by_rate(rec, iv)
            new = label_windows(rec.signal, rec.fs, iv, [lo for lo, _ in frames], WIN)
            for (lo, hi), lab in zip(frames, new):
                old = dominant_label(old_iv, lo, hi, C.SHOCKABLE_PURITY, excluded=C.SHOCKABLE_EXCLUDED)
                cls = window_class(lab)
                reason = lab if lab in ("VF", "VT_FAST", "VT_SLOW", "VF_FINE", "VT_IRREGULAR") else (
                    "transition" if lab is None else "ambiguous" if lab in C.AMBIGUOUS_RHYTHMS else "other")
                now = {"shockable": "shockable", "not_shockable": "not shockable"}.get(cls, "left out")
                moves[(old_class(old), now, reason)] += STEP_MIN
    if not n_rec:
        print("   no CUDB or VFDB records")
        return
    print(f"   {n_rec} records")
    print(f"   {'old':<15}{'new':<15}{'why':<14}{'minutes':>9}")
    for (o, n_, why), v in sorted(moves.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])):
        mark = "" if o == n_ else "   <- changes"
        print(f"   {o:<15}{n_:<15}{why:<14}{v:>9.1f}{mark}")
    tot = defaultdict(float)
    for (o, n_, _), v in moves.items():
        tot[("old", o)] += v
        tot[("new", n_)] += v
    print("   totals: " + "; ".join(
        f"{side} {c} {tot[(side, c)]:.0f}" for side in ("old", "new")
        for c in ("shockable", "not shockable", "left out")))


# ------------------------------------------------------------------- negatives
def part_negatives(path: Path, raw: Path, write_clean: Path | None) -> None:
    print("\n4. Mined hard negatives under the new labels")
    if not path.exists():
        print(f"   no {path}; pass --negatives PATH (the file given to --extra-negatives)")
        return
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    n = len(data["X"])
    keep = np.ones(n, dtype=bool)
    dropped = Counter()
    pairs = sorted(set(zip(data["dataset"].astype(str), data["record"].astype(str))))
    checked = 0
    for ds, name in pairs:
        if ds == "icentia11k":
            continue
        idx = np.flatnonzero((data["dataset"].astype(str) == ds) & (data["record"].astype(str) == name))
        try:
            rec = load_record(ds, name, raw)
        except Exception as e:
            print(f"   {ds} {name}: not read ({e})")
            continue
        iv = rhythm_intervals(rec)
        if not any(lab in ("VT", "VF") for _, _, lab in iv):
            continue
        checked += 1
        labs = label_windows(rec.signal, rec.fs, iv, [int(o) for o in data["offset"][idx]], WIN,
                             purity=C.SHOCKABLE_PURITY)
        for i, lab in zip(idx, labs):
            if lab in C.SHOCKABLE_WINDOW_RHYTHMS or lab in INTERMEDIATE:
                keep[i] = False
                dropped[lab] += 1
    print(f"   {n:,} windows; {checked} records with VT or VF labels re-read")
    print(f"   now shockable or intermediate: {int((~keep).sum())}"
          + (" (" + ", ".join(f"{k} {v}" for k, v in dropped.most_common()) + ")" if dropped else ""))
    if write_clean is not None:
        out = {k: (v[keep] if v.shape[:1] == (n,) else v) for k, v in data.items()}
        write_clean.parent.mkdir(parents=True, exist_ok=True)
        np.savez(write_clean, **out)
        print(f"   wrote {write_clean} ({int(keep.sum()):,} windows)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="*", default=["ec13", "fragments", "diff", "negatives"],
                    choices=["ec13", "fragments", "diff", "negatives"])
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--ec13", type=Path, default=None, help="default: <raw>/aami-ec13")
    ap.add_argument("--fragments", type=Path, default=None, help="default: <raw>/ecg_fragments")
    ap.add_argument("--max-records", type=int, default=0, help="per set, for the diff (0 = all)")
    ap.add_argument("--negatives", type=Path, default=Path(C.CACHE_DIR) / "shockable_extra_neg_v2.npz")
    ap.add_argument("--write-clean", type=Path, default=None,
                    help="save the hard negatives without the windows now shockable or intermediate")
    a = ap.parse_args(argv)
    print(f"labels {C.SHOCKABLE_LABELS}: VT at {C.VT_FAST_BPM:g} bpm or more and polymorphic"
          f" (irregular) VT shockable,\nslower VT and fine VF (under {C.FINE_VF_MAX_PTP_MV * 1000:g} uV)"
          f" left out, one rhythm per window")
    if "ec13" in a.part:
        part_ec13(a.ec13 or a.raw / "aami-ec13")
    if "fragments" in a.part:
        part_fragments(a.fragments or a.raw / "ecg_fragments", a.raw)
    if "diff" in a.part:
        part_diff(a.raw, a.max_records)
    if "negatives" in a.part:
        part_negatives(a.negatives, a.raw, a.write_clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
