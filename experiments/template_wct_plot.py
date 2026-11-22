"""Pictures of the VT episodes template_wct.py scores: what the detector and the template see.

    python experiments/template_wct_plot.py                   # every VT of 30 s or more
    python experiments/template_wct_plot.py --records 422 423 430
    python experiments/template_wct_plot.py --min-dur 0       # the short ones too

Writes one PDF (template_wct_plots.pdf), one page per episode:

    two strips   8 s early in the episode and 8 s in its middle. Red marks:
                 the beats the detector found (the stream template_wct.py
                 uses), each with its correlation with the template. Blue
                 marks: the peaks behind "rate sig" (data.windows._episode_bpm,
                 same rule). A red mark on every T wave means the detector
                 counts it as a beat; blue marks missing means the reference
                 undercounts.
    calibration  the 10 s the fixed template came from (the patient's normal).
    shape        what the rule compares: the template and the episode's beats,
                 100 ms either side of R at their best shift, each scaled to
                 the same size, as Pearson correlation sees them.
    wider view   the median normal beat and the median VT beat in millivolts,
                 from 250 ms before R to 500 ms after; shaded: the part
                 compared.

The beats, templates and correlations come from template_wct.beats_and_template,
the same code the experiment scores, at the rate engine's stride. Reads only;
test-split patients are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import template_wct as tw  # noqa: E402

C = tw.C
STRIP_SEC = 8.0
WIDE = (-0.25, 0.50)        # seconds around R in the wider view
MAX_OVERLAY = 40            # beats drawn in the shape panel


def reference_peaks(seg: np.ndarray, fs: int) -> np.ndarray:
    """The peaks behind "rate sig": data.windows._episode_bpm's rule."""
    from scipy.signal import find_peaks
    x = np.abs(seg - np.median(seg))
    scale = float(np.median(x)) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(0, np.int64)
    p, _ = find_peaks(x, distance=max(1, int(0.2 * fs)), prominence=2.0 * scale)
    return p


def wide_median(x: np.ndarray, peaks: np.ndarray, fs: int):
    a, b = int(WIDE[0] * fs), int(WIDE[1] * fs)
    p = peaks[(peaks + a >= 0) & (peaks + b < len(x))]
    if p.size == 0:
        return None
    return np.median(x[p[:, None] + np.arange(a, b)[None, :]], axis=0)


def strip(ax, b, t0: float, t1: float, ref_t=None, title=""):
    x, fs, ts = b["x"], b["fs"], b["ts"]
    t0, t1 = max(0.0, t0), min(len(x) / fs, t1)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    tt = np.arange(i0, i1) / fs
    seg = x[i0:i1]
    ax.plot(tt, seg, color="black", lw=0.7)
    lo, hi = np.percentile(seg, [0.5, 99.5]) if seg.size else (-1, 1)
    pad = 0.25 * (hi - lo + 1e-6)
    top, bot = hi + pad, lo - pad
    k = np.flatnonzero((ts >= t0) & (ts < t1))
    cf = b["corr"]["fixed"][b["si"][k]]
    ax.plot(ts[k], np.full(k.size, top), "v", color="#d62728", ms=4, label="detector beat (correlation)")
    for t, c in zip(ts[k], cf):
        if np.isfinite(c):
            ax.text(t, top + 0.06 * (top - bot), f"{c:.2f}", ha="center", va="bottom", fontsize=5.5,
                    color="#d62728")
    if ref_t is not None:
        r = ref_t[(ref_t >= t0) & (ref_t < t1)]
        ax.plot(r, np.full(r.size, bot), "^", color="#1f77b4", ms=4, label="reference peak (rate sig)")
    ax.set_xlim(t0, t1)
    ax.set_ylim(bot - 0.08 * (top - bot), top + 0.22 * (top - bot))
    ax.set_xticks(np.arange(np.ceil(t0), t1, 1.0))
    ax.set_xticks(np.arange(np.ceil(t0 * 5) / 5, t1, 0.2), minor=True)    # 0.2 s, as ECG paper
    ax.grid(True, which="major", axis="x", color="#e8a0a0", lw=0.5)
    ax.grid(True, which="minor", axis="x", color="#f6dede", lw=0.4)
    ax.set_ylabel("mV", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, loc="left")


def episode_page(pdf, plt, ds, name, b, s: float, e: float) -> dict:
    x, fs, ts, si, ext, L = b["x"], b["fs"], b["ts"], b["si"], b["ext"], b["L"]
    first, cf = b["first"], b["corr"]["fixed"]
    i, j = np.searchsorted(ts, [s, e])
    k = si[i:j]
    seg = x[int(s * fs):int(e * fs)]
    ref = reference_peaks(seg, fs)
    ref_t = s + ref / fs
    rate_sig = 60.0 / np.median(np.diff(ref)) * fs if ref.size >= 3 else float("nan")
    rate_det = 60.0 / np.median(np.diff(ts[i:j])) if j - i >= 3 else float("nan")
    unlike = float(np.mean(cf[k] < tw.SHOW_C)) if first is not None and k.size else float("nan")
    corr_med = float(np.median(cf[k])) if first is not None and k.size else float("nan")
    where = "none" if first is None else ("before" if first[1] <= s else "after")

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.25], hspace=0.55, wspace=0.18,
                          left=0.06, right=0.98, top=0.90, bottom=0.06)
    fig.suptitle(f"{ds} {name}   VT from {tw._mmss(s)} for {e - s:.0f} s   rate: signal {tw._f(rate_sig)}, "
                 f"detector {tw._f(rate_det)} bpm   beats unlike the template (corr < {tw.SHOW_C}): "
                 f"{tw._f(unlike, '{:.0%}')}   template: {where}", fontsize=10, x=0.06, ha="left")
    a1 = fig.add_subplot(gs[0, :])
    strip(a1, b, s + 2.0, s + 2.0 + STRIP_SEC, ref_t, "early in the episode")
    a1.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    mid = 0.5 * (s + e)
    a2 = fig.add_subplot(gs[1, :])
    strip(a2, b, mid - STRIP_SEC / 2, mid + STRIP_SEC / 2, ref_t, "middle of the episode")
    a3 = fig.add_subplot(gs[2, :])
    if first is not None:
        strip(a3, b, first[0], first[1], None, f"calibration: where the fixed template came from "
                                               f"({tw._mmss(first[0])} to {tw._mmss(first[1])})")
    else:
        a3.axis("off")
        a3.text(0.0, 0.5, "No template: never 10 s of one shape at 40 to 100 bpm in a conducted rhythm.",
                fontsize=9, transform=a3.transAxes)

    a4 = fig.add_subplot(gs[3, 0])
    ms = (np.arange(L) - (L - 1) // 2) / fs * 1000.0
    if first is not None and k.size:
        r, sh = tw.best_corr(ext[k], first[2], with_shift=True)
        pick = np.linspace(0, k.size - 1, min(MAX_OVERLAY, k.size)).astype(int)
        win = np.lib.stride_tricks.sliding_window_view(ext[k[pick]], L, axis=1)
        al = tw._z(win[np.arange(pick.size), sh[pick]])
        for row in al:
            a4.plot(ms, row, color="#d62728", lw=0.6, alpha=0.35)
        a4.plot(ms, tw._z(first[2][None, :])[0], color="black", lw=2.0, label="template")
        a4.plot([], [], color="#d62728", lw=1, label=f"VT beats (median corr {corr_med:.2f})")
        a4.legend(fontsize=6, loc="lower right")
    else:
        a4.set_xticks([])
        a4.set_yticks([])
        a4.text(0.5, 0.5, "no template", ha="center", va="center", transform=a4.transAxes, fontsize=9)
    a4.set_title(f"what the rule compares: {tw.PRE_SEC * 1000:.0f} ms either side of R, shape only",
                 fontsize=8, loc="left")
    a4.set_xlabel("ms from R", fontsize=7)
    a4.tick_params(labelsize=6)

    a5 = fig.add_subplot(gs[3, 1])
    wt = np.arange(int(WIDE[0] * fs), int(WIDE[1] * fs)) / fs * 1000.0
    vt = wide_median(x, b["pos"][k], fs) if k.size else None
    if first is not None:
        c0, c1 = np.searchsorted(ts, [first[0], first[1]])
        nm = wide_median(x, b["pos"][si[c0:c1]], fs)
        if nm is not None:
            a5.plot(wt, nm, color="black", lw=1.6, label="normal (calibration)")
    if vt is not None:
        a5.plot(wt, vt, color="#d62728", lw=1.6, label="VT (episode)")
    a5.axvspan(-tw.PRE_SEC * 1000, tw.POST_SEC * 1000, color="#999999", alpha=0.15)
    a5.axvline(0, color="#999999", lw=0.5)
    a5.set_title("wider view: median beats in mV (shaded: the part compared)", fontsize=8, loc="left")
    a5.set_xlabel("ms from R", fontsize=7)
    a5.tick_params(labelsize=6)
    a5.legend(fontsize=6, loc="lower right")
    pdf.savefig(fig)
    plt.close(fig)
    return {"set": ds, "record": name, "start": s, "dur": e - s, "rate_sig": rate_sig,
            "rate_det": rate_det, "unlike": unlike, "template": where}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["cudb", "vfdb"])
    ap.add_argument("--records", nargs="*", default=None, help="only these record names")
    ap.add_argument("--min-dur", type=float, default=C.VT_SUSTAINED_SEC, help="shortest episode drawn (s)")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per set (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("template_wct_plots.pdf"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    t0, rows = time.time(), []
    with PdfPages(a.out) as pdf:
        for ds in a.sets:
            for name in tw.eligible(ds, a.raw, a.max_records):
                if a.records and name not in a.records:
                    continue
                rec = tw.load_record(ds, name, a.raw)
                eps = [(s, e) for s, e, lab in tw.rhythm_intervals(rec) if lab == "VT"
                       and (e - s) / rec.fs >= a.min_dur]
                if not eps:
                    continue
                b = tw.beats_and_template(rec, tw.POS_STEP)
                for s, e in eps:
                    rows.append(episode_page(pdf, plt, ds, name, b, s / rec.fs, e / rec.fs))
                print(f"  {ds} {name}: {len(eps)} pages", flush=True)
    if not rows:
        a.out.unlink(missing_ok=True)
        print("no VT episode to draw")
        return 1
    print(f"\n{len(rows)} pages -> {a.out}  ({time.time() - t0:.0f} s)")
    for r in rows:
        print(f"  {r['set']:<5}{r['record']:<6}{tw._mmss(r['start']):>8}{r['dur']:>6.0f} s  rate sig "
              f"{tw._f(r['rate_sig']):>4}  det {tw._f(r['rate_det']):>4}  unlike {tw._f(r['unlike'], '{:.0%}'):>5}"
              f"  template {r['template']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
