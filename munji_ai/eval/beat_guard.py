"""Can the beat model's own beat count reveal when the model is lost?

    python -m munji_ai.eval.beat_guard --part val
    python -m munji_ai.eval.beat_guard --part val --limit 2000

Why
---
The beat model has a measured blind spot. Inside a run of consecutive
ventricular beats it loses the beats themselves and calls most of what is left
normal: in sustained VT across CUDB and VFDB it reported 48 bpm against a
signal rate of 114, with 12% of the beats it did find labelled PVC. Every rule
built on PVC counts therefore reads low exactly in the patients with the most
ectopy, which is the worst direction for the error to run.

The failure is loud, not silent: the beat count collapses. The rate engine
already counts beats a completely different way (Pan-Tompkins++, no model, no
learning), so the two counts can be compared on the device at no extra cost.
Where they disagree, the beat labels for that stretch are not to be trusted
and the window can be left out of the burden instead of quietly lowering it.

That is the guard. Before it can be built, two things have to hold, and
neither has been measured:

    detector / expert    the reference has to be right on ordinary windows,
                         or the guard is built on sand
    model / detector     what the guard actually sees at run time, where no
                         expert labels exist

This measures both on cached windows and splits them by the longest run of
consecutive ventricular beats in the expert mask, which is where the model is
expected to fail. The threshold is not chosen here: the sweep at the end
prints what each candidate would cost and catch, and the number is picked from
that table.

Reads the cache only. Trains nothing, and never opens the test split.

Reference used for the beat count: rules.qrs_detect.detect_qrs, the
Pan-Tompkins++ detector the rate engine runs, not windows._episode_bpm, whose
find_peaks pass is documented to read smooth or small VT far too slow (it is
what mislabelled 47% of the slow-VT windows in ShockableNet's training set).
The ANSI/AAMI amplitude floor is left off here because cached windows are IQR
normalised, not millivolts; the adaptive decisions are unaffected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import config as C
from ..decision.decision_layer import beats_from_mask
from ..rules.qrs_detect import detect_qrs

FS = C.TARGET_FS
RUN_LEN = 3                       # a ventricular run, by the VT definition
SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def longest_v_run(beats) -> int:
    """Longest stretch of consecutive PVC beats in [(time, class), ...]."""
    best = run = 0
    for _, c in beats:
        run = run + 1 if c == "PVC" else 0
        best = max(best, run)
    return best


def predict(ckpt: Path, X, batch: int = 16):
    """Cleaned per-sample masks for every window, as deployment would read them."""
    import torch

    from ..models.beat_segmenter import build, clean_mask

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"beat model  {ckpt}  epoch {blob.get('epoch')}  "
          f"val beat F1 {blob.get('val_beat_f1', float('nan')):.3f}")

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch], np.float32))
            out.append(clean_mask(model(xb).argmax(dim=1).numpy()))
            if i and i % (batch * 50) == 0:
                print(f"  {i:,}/{len(X):,} windows", flush=True)
    return np.concatenate(out)


def describe(label: str, vals) -> None:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        print(f"{label:<24}{0:>8}{'  no windows':>45}")
        return
    p = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"{label:<24}{v.size:>8}{p[0]:>9.2f}{p[1]:>9.2f}{p[2]:>9.2f}"
          f"{p[3]:>9.2f}{p[4]:>9.2f}")


def table(title: str, rows: list, key: str) -> None:
    print(f"\n{title}")
    print(f"{'group':<24}{'windows':>8}{'p05':>9}{'p25':>9}{'median':>9}"
          f"{'p75':>9}{'p95':>9}")
    describe("all", [r[key] for r in rows])
    describe("no ventricular run", [r[key] for r in rows if r["run"] < RUN_LEN])
    describe(f"run of {RUN_LEN} or more", [r[key] for r in rows if r["run"] >= RUN_LEN])
    describe("run of 5 or more", [r[key] for r in rows if r["run"] >= 5])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="val", choices=("val", "train"),
                    help="which cached split to read (test is never opened)")
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--beat-ckpt", type=Path,
                    default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--limit", type=int, default=0,
                    help="use at most this many windows (0 = all)")
    a = ap.parse_args(argv)

    path = a.cache / f"beat_{a.part}.npz"
    z = np.load(path)
    X, Y = z["X"], z["y"]
    if a.limit:
        X, Y = X[:a.limit], Y[:a.limit]
    secs = X.shape[-1] / FS
    print(f"{len(X):,} windows of {secs:.0f} s from {path}\n")

    P = predict(a.beat_ckpt, X)

    rows = []
    for x, y_true, y_pred in zip(X, Y, P):
        be = beats_from_mask(y_true, 0.0, FS)
        bm = beats_from_mask(y_pred, 0.0, FS)
        n_e, n_m = len(be), len(bm)
        n_q = int(detect_qrs(np.asarray(x, dtype=float).ravel(), FS).size)
        rows.append({
            "expert": n_e, "model": n_m, "qrs": n_q,
            "run": longest_v_run(be),
            "q_over_e": n_q / n_e if n_e else np.nan,
            "m_over_e": n_m / n_e if n_e else np.nan,
            "m_over_q": n_m / n_q if n_q else np.nan,
        })

    plain = [r for r in rows if r["run"] < RUN_LEN]
    runs = [r for r in rows if r["run"] >= RUN_LEN]
    print(f"\n{len(rows):,} windows   {len(plain):,} without a ventricular run   "
          f"{len(runs):,} with a run of {RUN_LEN} or more")

    table("REFERENCE CHECK: detector beats / expert beats", rows, "q_over_e")
    print("Near 1.00 with a tight spread means the Pan-Tompkins count can stand in for"
          "\nthe expert at run time. A wide spread means the guard has no reliable"
          "\nreference and the idea does not survive.")

    table("MODEL AGAINST TRUTH: model beats / expert beats", rows, "m_over_e")
    print("How much of the real beat count the model recovers. Not available on the"
          "\ndevice; it is here to say whether a drop in the next table is the model's"
          "\nfault or the detector's.")

    table("WHAT THE GUARD SEES: model beats / detector beats", rows, "m_over_q")
    print("The only one of the three the device can compute. For the guard to work this"
          "\nhas to sit near 1.00 on ordinary windows and fall on windows with a run.")

    print(f"\n{'OPERATING POINTS':<12}")
    print(f"{'flag below':<12}{'ordinary flagged':>18}{'runs flagged':>14}"
          f"{'runs 5+ flagged':>17}")
    big = [r for r in rows if r["run"] >= 5]
    for t in SWEEP:
        def share(g):
            v = [r["m_over_q"] for r in g if np.isfinite(r["m_over_q"])]
            return f"{np.mean([x < t for x in v]):.1%}" if v else "-"
        print(f"{t:<12.2f}{share(plain):>18}{share(runs):>14}{share(big):>17}")
    print("'ordinary flagged' is the cost: good windows the guard would throw away."
          "\n'runs flagged' is the catch. A threshold is worth adopting only if the"
          "\ntwo columns separate; if they move together there is nothing to gate on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
