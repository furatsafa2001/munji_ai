"""Check the engine's QRS width against hand-measured QRS durations.

    python -m munji_ai.eval.eval_qrs_width_qtdb --data-dir data/qtdb
    python -m munji_ai.eval.eval_qrs_width_qtdb --download --data-dir data/qtdb

Data: PhysioNet QT Database (Laguna et al. 1997). 105 two-lead Holter
excerpts at 250 Hz; for 30-50 beats per record an expert marked QRS onset and
end by hand (.q1c files, 3,622 beats in all). MITDB cannot be used for this:
it marks where each beat is and what type it is, not how long the QRS lasts.

What it answers
---------------
The narrow/wide decision (120 ms) sits on an estimate, not a caliper. On
synthetic complexes the estimate reads about a quarter short. This script
measures the real relationship: bias, error, correlation, and, most usefully,
which measured value best separates beats the expert called >= 120 ms from
the rest. It also checks the IEC 60601-2-25 acceptance limits for QRS
duration (mean difference within +/-10 ms, SD 10 ms against the expert) and
suggests QRS_UNCERTAIN_MS as 95% of the measurement error. The script
reports; it changes nothing.

Caveats printed with the result: the expert marked onset and end looking at
both leads together, so a single-lead estimate is expected to read shorter;
and wide beats are a minority, so the separation figure rests on few of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import NARROW_QRS_MS, QRS_UNCERTAIN_MS
from ..data.preprocess import model_path
from ..rules.rate_engine import qrs_widths_ms

WAVE_SYMBOLS = {"(", ")", "p", "t", "u", "+", "~", "|", '"'}


def annotated_qrs(ann, fs):
    """(peak sample, duration ms) for every QRS with onset and end marked."""
    out = []
    sym, smp = ann.symbol, ann.sample
    for j in range(1, len(sym) - 1):
        if sym[j] in WAVE_SYMBOLS:
            continue
        if sym[j - 1] == "(" and sym[j + 1] == ")":
            out.append((int(smp[j]), (smp[j + 1] - smp[j - 1]) / fs * 1000.0))
    return out


def evaluate(data_dir: Path):
    import wfdb

    truth, meas, leads = [], [], []
    for q in sorted(data_dir.glob("*.q1c")):
        name = str(q.with_suffix(""))
        try:
            ann = wfdb.rdann(name, "q1c")
            rec = wfdb.rdrecord(name)
        except Exception as exc:                       # noqa: BLE001
            print(f"skip {q.stem}: {exc}")
            continue
        fs = int(rec.fs)
        beats = annotated_qrs(ann, fs)
        if not beats:
            continue
        peaks = np.array([b[0] for b in beats])
        dur = np.array([b[1] for b in beats])
        for ch, lead in enumerate(rec.sig_name):
            x = rec.p_signal[:, ch]
            if np.isnan(x).any():
                x = np.nan_to_num(x)
            y = model_path(x, fs)
            w = qrs_widths_ms(y, peaks, fs)
            ok = ~np.isnan(w)
            truth.extend(dur[ok])
            meas.extend(w[ok])
            leads.extend([lead] * int(ok.sum()))
    return np.array(truth), np.array(meas), np.array(leads)


def best_threshold(truth, meas, line=NARROW_QRS_MS):
    """Measured cut that best separates expert >= line from < line."""
    wide = truth >= line
    if wide.sum() == 0 or (~wide).sum() == 0:
        return float("nan"), float("nan")
    best = (float("nan"), -1.0)
    for t in np.unique(meas):
        pred = meas >= t
        sens = (pred & wide).sum() / wide.sum()
        spec = (~pred & ~wide).sum() / (~wide).sum()
        bal = (sens + spec) / 2
        if bal > best[1]:
            best = (float(t), float(bal))
    return best


def _corr(a, b):
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report(truth, meas, leads):
    if truth.size == 0:
        print("No annotated beats found. Is --data-dir the QT Database folder?")
        return
    err = meas - truth
    print(f"\nbeats compared        {truth.size:,} (both leads)")
    print(f"expert QRS            median {np.median(truth):.0f} ms, "
          f"{(truth >= NARROW_QRS_MS).sum():,} beats >= {NARROW_QRS_MS} ms")
    print(f"bias (engine - expert) median {np.median(err):+.0f} ms")
    print(f"absolute error        median {np.median(np.abs(err)):.0f} ms")
    print(f"correlation           {_corr(truth, meas):.2f}")

    mean_d, sd_d = float(np.mean(err)), float(np.std(err))
    iec = abs(mean_d) <= 10.0 and sd_d <= 10.0
    print("\nIEC 60601-2-25 acceptance (mean difference within +/-10 ms, SD <= 10 ms):")
    print(f"  mean {mean_d:+.1f} ms, SD {sd_d:.1f} ms  ->  {'PASS' if iec else 'FAIL'}")
    band = 1.96 * sd_d
    print(f"  suggested QRS_UNCERTAIN_MS: {band:.0f} ms (95% of the measurement error;"
          f" currently {QRS_UNCERTAIN_MS})")
    if not iec:
        print("  The estimate does not meet the standard: do not rely on the 120 ms line"
              " to name a rhythm until it does.")

    wide, pred = truth >= NARROW_QRS_MS, meas >= NARROW_QRS_MS
    if wide.any():
        print(f"\nat the current {NARROW_QRS_MS} ms line:")
        print(f"  wide beats called wide     {(pred & wide).sum() / wide.sum():.2f}")
        print(f"  narrow beats called narrow {(~pred & ~wide).sum() / (~wide).sum():.2f}")
        t, bal = best_threshold(truth, meas)
        print(f"\nmeasured value that best separates expert >= {NARROW_QRS_MS} ms:"
              f" {t:.0f} ms (balanced accuracy {bal:.2f})")
    print("\nCaveats: experts marked onset/end on both leads together, so one lead"
          "\nreads shorter; wide beats are few. Nothing in config is changed.")

    print("\nper lead:")
    for lead in sorted(set(leads)):
        m = leads == lead
        print(f"  {lead:<8} n={m.sum():>5}  bias {np.median(err[m]):+5.0f} ms  "
              f"r={_corr(truth[m], meas[m]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/qtdb"))
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        import wfdb
        args.data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("qtdb", str(args.data_dir))
    report(*evaluate(args.data_dir))


if __name__ == "__main__":
    main()
