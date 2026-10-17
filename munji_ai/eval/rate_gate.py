"""Evaluate a rate gate on the shockable detector, after the fashion of an S-ICD.

    python -m munji_ai.eval.rate_gate
    python -m munji_ai.eval.rate_gate --rate 150 --cv 0.15

Where this comes from
---------------------
The detector decides from five seconds of signal and nothing else. Commercial
implantable defibrillators do not work that way. The subcutaneous ICD applies
a detector and then a set of discriminators on top of it — a rate zone, a
regularity check, a sudden-onset check, and a comparison against the patient's
own resting morphology — and only shocks when they agree. In the S-ICD IDE
trial, adding that discrimination layer took inappropriate shocks from 26.1%
of patients to 10.2% over two years, with no loss of appropriate therapy and
no delay in detection.

This script tests the cheapest of those discriminators: a rate gate.

The rule
--------
    rate measurable, below the threshold, and regular   ->  not shockable
    anything else                                         ->  defer to the model

The clinical ground is simple. Fibrillation has no organised rate — a peak
detector either finds nothing usable or finds chaos. A heart beating at 70
with evenly spaced complexes is not fibrillating, whatever a classifier thinks
of the waveform. So when the rate is clean, slow and regular, the model's
positive can be overruled without risk to the class we are protecting.

The regularity condition matters. Fibrillation can fool a peak detector into
reporting some rate; it cannot make the intervals regular. Requiring a low
coefficient of variation is what keeps the gate from ever touching real VF.

What to expect
--------------
Sinus false alarms should fall, because most of them are on regular rhythm at
ordinary rates. The VT (slow/excl) row — windows the labelling deliberately
excluded as not fast enough to shock, which the model still calls positive —
should go to near-perfect specificity, since those are by definition below the
rate threshold.

Shockable sensitivity should not move. That is the check. If it does, the gate
is suppressing fibrillation and the regularity condition is too loose.

Reads only. Writes nothing, trains nothing, does not touch the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from ..config import CACHE_DIR, CONFIRMATION, RAW_DIR, TARGET_FS, window_samples
from ..data.loader import load_record
from ..data.windows import load_cache, rhythm_intervals
from ..train.train_shockable import confirmed_alarms

POSITIVE = "shockable"


def rate_and_regularity(sig: np.ndarray, fs: int = TARGET_FS):
    """Median rate and RR coefficient of variation for one window.

    Follows _episode_bpm in data/windows.py in shape — rectified trace,
    peak spacing — but takes its prominence from the upper percentile of the
    trace instead of its MAD (see the comment below), and adds the regularity
    measure the gate depends on. Returns
    (None, None) where no reliable rate exists, which for this purpose is the
    right answer for fibrillation.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < fs:
        return None, None
    x = np.abs(sig - np.median(sig))
    # Prominence is set from the upper tail of the trace rather than from
    # its MAD. On a clean window the MAD is just the noise floor, so a
    # MAD-based threshold admits noise peaks and the shoulders of each QRS
    # and the rate comes out two or three times too high. The 98th
    # percentile sits on the QRS complexes themselves whatever the noise
    # level, and it is scale-free, which matters because the cached windows
    # are already median/IQR normalised.
    prom = 0.4 * float(np.percentile(x, 98))
    if not np.isfinite(prom) or prom <= 0:
        return None, None
    peaks, _ = find_peaks(x, distance=max(1, int(0.2 * fs)), prominence=prom)
    if len(peaks) < 4:
        return None, None
    rr = np.diff(peaks) / float(fs)
    rr = rr[rr > 0]
    if len(rr) < 3:
        return None, None
    bpm = float(60.0 / np.median(rr))
    cv = float(np.std(rr) / max(np.mean(rr), 1e-9))
    return bpm, cv


def gate(pred, bpm, cv, rate_thr: float, cv_thr: float):
    """Apply the rule. Overrules a positive only when all three hold."""
    # The prediction array is dtype <U9 — the length of "shockable" — so
    # writing "not_shockable" into it would be silently truncated to
    # "not_shock". Widen first.
    out = pred.astype("<U16")
    measurable = ~np.isnan(bpm)
    slow = bpm < rate_thr
    regular = cv < cv_thr
    overrule = (pred == POSITIVE) & measurable & slow & regular
    out[overrule] = "not_shockable"
    return out, overrule


def scores(y_true, y_pred, positive=POSITIVE) -> dict:
    t, p = y_true == positive, y_pred == positive
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())
    return {"sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "fp": fp, "fn": fn}


def rhythm_at(record_key, dataset, offset, n, cache, root) -> str:
    key = (dataset, record_key)
    if key not in cache:
        try:
            cache[key] = rhythm_intervals(load_record(dataset, record_key, root))
        except Exception:
            cache[key] = []
    best, best_cover = "UNLABELLED", 0
    for s, e, lab in cache[key]:
        cover = max(0, min(e, offset + n) - max(s, offset))
        if cover > best_cover:
            best, best_cover = lab, cover
    return best if best_cover > 0 else "UNLABELLED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path,
                    default=Path("checkpoints") / "shockable_oof.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--rate", type=float, default=None,
                    help="bpm below which a regular rhythm is not shockable")
    ap.add_argument("--cv", type=float, default=None,
                    help="RR coefficient of variation below which it is regular")
    a = ap.parse_args(argv)

    if not a.oof.exists():
        print(f"no out-of-fold predictions at {a.oof}\n"
              f"run:  python -m munji_ai.train.cross_validate --stage shockable")
        return 1

    d = np.load(a.oof, allow_pickle=False)
    y = np.asarray(d["y"], dtype=str)
    pred = np.asarray(d["pred"], dtype=str)
    records = np.asarray(d["record"], dtype=str)
    offsets = np.asarray(d["offset"], dtype=np.int64)

    parts = [load_cache("shockable", s, a.cache) for s in ("train", "val")]
    X = np.concatenate([p["X"] for p in parts])
    datasets = np.concatenate([np.asarray(p["dataset"], dtype=str)
                               for p in parts])
    if len(X) != len(y):
        print("cache no longer matches the saved predictions — re-run "
              "cross_validate")
        return 1

    print(f"source      {a.oof}")
    print(f"windows     {len(y):,}")
    print("measuring rate and regularity per window — a minute or so\n")
    bpm = np.full(len(y), np.nan)
    cv = np.full(len(y), np.nan)
    for i, x in enumerate(X):
        b, c = rate_and_regularity(x)
        if b is not None:
            bpm[i], cv[i] = b, c

    n = window_samples("shockable")
    iv: dict = {}
    labels = np.array([rhythm_at(r, ds, o, n, iv, a.raw)
                       for r, ds, o in zip(records, datasets, offsets)])
    sinus = np.isin(labels, ("N", "NSR"))
    is_pos = y == POSITIVE

    # Where does the model's positive class sit on the rate axis? If real
    # fibrillation is mostly unmeasurable or fast, the gate has room.
    meas = ~np.isnan(bpm)
    print("rate measurability by true class")
    for name, m in (("shockable", is_pos), ("not shockable", ~is_pos)):
        mm = m & meas
        print(f"  {name:<14} measurable {mm.sum() / max(m.sum(), 1):>6.1%}   "
              f"median bpm {np.nanmedian(bpm[mm]) if mm.any() else float('nan'):>6.0f}   "
              f"median CV {np.nanmedian(cv[mm]) if mm.any() else float('nan'):>5.2f}")

    base = scores(y, pred)
    base_s = scores(y[sinus], pred[sinus]) if sinus.any() else None
    rule = CONFIRMATION["vf"]
    c0 = confirmed_alarms(y, pred, records, offsets, rule)
    print("\nmodel alone")
    print(f"  sensitivity {base['sensitivity']:.3f}   "
          f"specificity {base['specificity']:.3f}   "
          f"PPV {base['ppv']:.3f}")
    if base_s:
        print(f"  sinus specificity {base_s['specificity']:.3f}")
    print(f"  episodes {c0['episodes_caught']}/"
          f"{c0['episodes_caught'] + c0['episodes_missed']}   "
          f"false alarms {c0['false_alarms']}")

    if a.rate is None or a.cv is None:
        print(f"\n{'rate':>6}{'cv':>6}{'sens':>8}{'spec':>8}{'sinus':>8}"
              f"{'PPV':>7}{'VF lost':>9}{'overruled':>11}")
        best = None
        for rate_thr in (120.0, 150.0, 180.0):
            for cv_thr in (0.10, 0.15, 0.20, 0.30):
                g, over = gate(pred, bpm, cv, rate_thr, cv_thr)
                s = scores(y, g)
                ss = scores(y[sinus], g[sinus]) if sinus.any() \
                    else {"specificity": float("nan")}
                lost = int((over & is_pos).sum())
                flag = ""
                if s["sensitivity"] >= base["sensitivity"] - 0.003 and \
                        (best is None or ss["specificity"] > best[2]):
                    best = (rate_thr, cv_thr, ss["specificity"])
                    flag = "  <--"
                print(f"{rate_thr:>6.0f}{cv_thr:>6.2f}{s['sensitivity']:>8.3f}"
                      f"{s['specificity']:>8.3f}{ss['specificity']:>8.3f}"
                      f"{s['ppv']:>7.3f}{lost:>9,}{int(over.sum()):>11,}{flag}")
        print("\n  'VF lost' is the number of truly shockable windows the gate")
        print("  overruled. It should be zero or very close to it. '<--' marks")
        print("  the setting with the best sinus specificity at no cost to")
        print("  sensitivity.")
        if best is None:
            print("\n  No setting held sensitivity. The regularity condition is")
            print("  letting fibrillation through as 'regular'. Stop here.")
            return 0
        a.rate, a.cv = best[0], best[1]
        print(f"\nusing rate < {a.rate:.0f} and CV < {a.cv:.2f}\n")

    g, over = gate(pred, bpm, cv, a.rate, a.cv)
    s = scores(y, g)
    c1 = confirmed_alarms(y, g, records, offsets, rule)

    print("=" * 62)
    print(f"{'':<18}{'model alone':>13}{'+ rate gate':>13}{'change':>10}")
    for name, key in (("sensitivity", "sensitivity"),
                      ("specificity", "specificity"), ("PPV", "ppv")):
        print(f"  {name:<16}{base[key]:>13.3f}{s[key]:>13.3f}"
              f"{s[key] - base[key]:>+10.3f}")
    if sinus.any():
        ss0 = scores(y[sinus], pred[sinus])["specificity"]
        ss1 = scores(y[sinus], g[sinus])["specificity"]
        print(f"  {'sinus specificity':<16}{ss0:>13.3f}{ss1:>13.3f}"
              f"{ss1 - ss0:>+10.3f}")
    print(f"  {'false alarms':<16}{base['fp']:>13,}{s['fp']:>13,}"
          f"{s['fp'] - base['fp']:>+10,}")
    print(f"  {'missed':<16}{base['fn']:>13,}{s['fn']:>13,}"
          f"{s['fn'] - base['fn']:>+10,}")

    print(f"\nafter {rule[0]}-of-{rule[1]} confirmation")
    print(f"  {'episodes caught':<18}{c0['episodes_caught']:>13}"
          f"{c1['episodes_caught']:>13}")
    print(f"  {'episodes missed':<18}{c0['episodes_missed']:>13}"
          f"{c1['episodes_missed']:>13}")
    print(f"  {'false alarms':<18}{c0['false_alarms']:>13}"
          f"{c1['false_alarms']:>13}")

    print("\nper rhythm, specificity before -> after (negatives only)")
    for lab in sorted(set(labels)):
        m = (labels == lab) & ~is_pos
        if m.sum() < 5:
            continue
        before = scores(y[m], pred[m])["specificity"]
        after = scores(y[m], g[m])["specificity"]
        tag = "  <-- below rate threshold by construction" \
            if lab == "VT" else ""
        print(f"  {lab:<12}{m.sum():>7,}   {before:.3f} -> {after:.3f}{tag}")

    print("\n" + "=" * 62)
    lost = int((over & is_pos).sum())
    if lost > 0:
        print(f"  {lost} truly shockable window(s) were overruled. Even one is")
        print("  worth looking at: it means a fibrillating trace produced a")
        print("  rate that looked slow and regular. Tighten --cv before use.")
    else:
        print("  No shockable window was overruled. The gate removed")
        print(f"  {base['fp'] - s['fp']:,} false alarms at zero cost to the")
        print("  class it exists to protect.")
    print("\n  This is the first of the S-ICD discriminators. Onset detection")
    print("  and a per-patient morphology template are the next two, and both")
    print("  need signal history the current pipeline does not yet keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
