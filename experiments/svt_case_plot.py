"""One stretch of a record, with everything the SVT rule read there.

    python experiments/svt_case_plot.py cudb/cu07 120 200
    python experiments/svt_case_plot.py svdb/809 1040 1110

Same models and flags as munji_ai.eval.eval_svt, which flagged the case.
Writes reports/svt_cases/<dataset>_<record>_<start>_<end>.png, four panels on
one time axis:

    ECG (mV), the rhythm annotation, and a red mark at the end of every rate
    window named SVT (each one covers the 40 s before its mark)
    beats: the beat model's (filled) and the experts' when there are any (open)
    ShockableNet P(shockable) every 2.5 s with the operating line, and
    RhythmNet P(AF) every 15 s
    the rate engine's fastest rate held over a tachy run, per 40 s window

and prints, for every named window, what each rule input said.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.eval import eval_svt as E  # noqa: E402

FS = C.TARGET_FS
ROW = {"N": 0, "PAC": 1, "PVC": 2}
COLOUR = {"N": "tab:blue", "PAC": "tab:orange", "PVC": "tab:red"}
RHYTHM_COLOUR = {"VT": "red", "VF": "red", "SVT": "orange", "NSR": "green",
                 "AF": "purple", "AFL": "purple"}


def model_beats(events) -> list:
    """The beat model's beats from every usable window, each beat once."""
    out = sorted(b for t, _, kind, x in events if kind == "beats" and x[1] for b in x[0])
    kept = []
    for bt, c in out:
        if not kept or bt > kept[-1][0] + C.BEAT_DEDUP_SEC:
            kept.append((bt, c))
    return kept


def summary(named, events, beats) -> list:
    """One line per named window: what each input of the rule said."""
    rows = []
    shock = [(t, x) for t, _, k, x in events if k == "shock"]
    rhythm = [(t, x[0][1]) for t, _, k, x in events if k == "rhythm" and x[1]]
    rate = {t: x[0] for t, _, k, x in events if k == "rate"}
    for tn in named:
        lo = tn - C.RULES_WINDOW_SEC
        p = max((x for t, x in shock if lo < t <= tn), default=float("nan"))
        af = [x for t, x in rhythm if t <= tn]
        cls = [c for bt, c in beats if lo < bt <= tn]
        rows.append(f"  {tn:7.1f} s  engine {rate[tn].fast_run_bpm:5.0f} bpm   "
                    f"ShockableNet max {p:.2f}   P(AF) {af[-1] if af else float('nan'):.2f}   "
                    f"beats N {cls.count('N')}  PAC {cls.count('PAC')}  PVC {cls.count('PVC')}")
    return rows


def plot_case(rec, label: str, start: float, end: float, models, intervals,
              expert, out_png: Path) -> dict:
    """Run the rule over [start, end] and draw it. expert: (times, classes) or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds, name = label.split("/", 1)
    span = (max(0, int((start - C.RULES_WINDOW_SEC) * FS)), min(len(rec.signal), int(end * FS)))
    r = E.run_record(ds, name, rec, models, span=span, keep=True)
    ev = r.get("events", [])
    beats = model_beats(ev)
    named = [t for t in r["named"] if start <= t <= end]

    fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.2, 1.4, 1.2]})
    t = np.arange(len(rec.signal)) / FS
    m = (t >= start) & (t <= end)
    ax[0].plot(t[m], rec.signal[m], lw=0.6, color="black")
    top = float(np.max(rec.signal[m])) if m.any() else 1.0
    for s, e, lab in intervals:
        if e > start and s < end:
            ax[0].axvspan(max(s, start), min(e, end), color=RHYTHM_COLOUR.get(lab, "grey"),
                          alpha=0.12)
            ax[0].text(max(s, start), top, lab, fontsize=8, va="bottom")
    if named:
        ax[0].plot(named, [top * 1.08] * len(named), "v", color="red", ms=6,
                   label="rate window named SVT (ends here)")
        ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_ylabel("ECG (mV)")
    ax[0].set_title(f"{label}  {start:.0f} to {end:.0f} s   named SVT: {len(named)} windows")

    for bt, c in beats:
        if start <= bt <= end:
            ax[1].plot(bt, ROW[c], "o", color=COLOUR[c], ms=3)
    if expert is not None:
        for bt, c in zip(*expert):
            if start <= bt <= end and c in ROW:
                ax[1].plot(bt, ROW[c] - 0.3, "o", mfc="none", color=COLOUR[c], ms=3)
    ax[1].set_yticks([0, 1, 2], ["N", "PAC", "PVC"])
    ax[1].set_ylabel("beats\n(open: experts)")

    sh = [(t_, x) for t_, _, k, x in ev if k == "shock" and start <= t_ <= end]
    if sh:
        ax[2].step(*zip(*sh), where="post", color="tab:red", lw=1, label="P(shockable)")
    rh = [(t_, x[0][1]) for t_, _, k, x in ev if k == "rhythm" and start <= t_ <= end]
    if rh:
        ax[2].plot(*zip(*rh), "s", color="purple", ms=4, label="P(AF)")
    ax[2].axhline(C.SHOCKABLE_THRESHOLD, ls="--", color="tab:red", lw=0.8)
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_ylabel("probability")
    ax[2].legend(loc="upper left", fontsize=8)

    ra = [(t_, x[0].fast_run_bpm) for t_, _, k, x in ev if k == "rate" and start <= t_ <= end]
    if ra:
        ax[3].plot(*zip(*ra), ".-", color="black", lw=0.8)
    ax[3].axhline(C.FAST_NARROW_BPM, ls="--", color="grey", lw=0.8)
    ax[3].set_ylabel("held rate (bpm)")
    ax[3].set_xlabel("time (s)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"named": named, "rows": summary(named, ev, beats), "png": out_png}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("record", help="dataset/record, e.g. cudb/cu07")
    ap.add_argument("start", type=float, help="seconds")
    ap.add_argument("end", type=float, help="seconds")
    ap.add_argument("--shock", type=Path, default=Path("checkpoints/shock_cv_aha"))
    ap.add_argument("--rhythm", type=Path, default=Path("checkpoints/rhythm_final/rhythm_net.pt"))
    ap.add_argument("--beat-ckpt", type=Path, default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("reports") / "svt_cases")
    a = ap.parse_args(argv)

    from munji_ai.data.loader import beat_classes, load_record
    from munji_ai.data.windows import rhythm_intervals
    ds, name = a.record.split("/", 1)
    rec = load_record(ds, name, a.raw)
    iv = [(s / FS, e / FS, lab) for s, e, lab in rhythm_intervals(rec)]
    expert = None
    if len(rec.beat_samples):
        expert = (np.asarray(rec.beat_samples, dtype=np.float64) / FS, beat_classes(rec))
    models = E.Models(a)
    png = a.out / f"{ds}_{name}_{a.start:.0f}_{a.end:.0f}.png"
    res = plot_case(rec, a.record, a.start, a.end, models, iv, expert, png)
    print(f"\nwrote {res['png']}")
    print(f"{len(res['named'])} rate windows named SVT in {a.start:.0f} to {a.end:.0f} s")
    print("\n".join(res["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
