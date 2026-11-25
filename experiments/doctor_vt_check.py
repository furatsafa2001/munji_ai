"""What sat inside the doctor's report entries that touched VT.

    python experiments/doctor_vt_check.py                       # LTAFDB 204 and 206
    python experiments/doctor_vt_check.py --plot                # and one page per entry
    python experiments/doctor_vt_check.py --records ltafdb:204

Why
---
The doctor's report row of fast_narrow_template.py ("doctor report:
SVT/AFL") made 9 entries whose 30 s touched an annotated VT, all in LTAFDB:
8 in record 204 and 1 in 206, at 160 to 188 bpm, regular, Lorenz score
under 0. Its VT check (no run of 3 or more consecutive beats unlike the
patient's own beat) let them through. Before choosing a fix, this looks at
each entry's 30 s:

  annotated   the VT rhythm marks it touches, and the beats the annotators
              marked in it: how many ventricular (V, E), the longest run
  detected    the beats the VT check judged (the decision window's own
              beats, as decisions() reads them), and the check's value there
  matched     each annotated ventricular beat against the detected beats,
              within 150 ms (the ANSI/AAMI EC57 match window): missed by the
              detector, or found and judged alike or unlike the template
              (correlation under 0.6 or size outside half to twice)

so each entry falls in one of:
  missed      ventricular beats the detector did not find
  alike       ventricular beats found but alike the template: on this lead
              the VT beat looks like the patient's own
  short       found and unlike, but no 3 in a row among the detected beats
  no V beat   a VT rhythm mark with no ventricular beat marked in the 30 s

The entries are found the way the main run finds them (the same beats,
template, decisions and row at the 24 h sets' 5 s stride), so the count is
checked against the main run's. Reads only; --plot writes a PDF under data/,
which git ignores.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from munji_ai import config as C  # noqa: E402


def _experiment(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _experiment("fast_narrow_template")
T = G.T

VENTRICULAR = {"V", "E"}       # premature ventricular and ventricular escape
MATCH_S = 0.150                # ANSI/AAMI EC57 beat match window
MAIN_RUN = {"ltafdb:204": 8, "ltafdb:206": 1}
KINDS = ("missed", "alike", "short", "no V beat", "other")


def doctor_entries(rec, step: float) -> tuple:
    """(beats and template, decisions, entry times) for the doctor's report
    row on one record, as fast_narrow_template.process() makes them."""
    b = T.beats_and_template(rec, step, annotated_span=rec.dataset in G.SPAN_SETS)
    dec = G.decisions(b["ends"], b["wbeats"], b["pos"], b["corr"], b["fs"], b["ts"])
    cond = G.condition(dec, dict(G.CONFIGS)[G.DOCTOR], np.zeros(len(dec["t"]), bool))
    return b, dec, T.alert_times(dec["t"], cond, step)


def longest(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def look(rec, b: dict, dec: dict, a: float) -> dict:
    """One entry at time a (seconds in the analysed span): what was in its 30 s."""
    fs = b["fs"]
    lo = int(rec.beat_samples.min()) if rec.dataset in G.SPAN_SETS and len(rec.beat_samples) >= 2 else 0
    t0, t1 = a - G.SUSTAIN, a
    vt = [(s, e) for s, e, lab in b["ivs"] if lab == "VT" and s < t1 and e > t0]
    bt = (np.asarray(rec.beat_samples) - lo) / fs
    m = (bt >= t0) & (bt <= t1)
    sym, bt = np.asarray(rec.beat_symbols)[m], bt[m]
    is_v = np.isin(sym, list(VENTRICULAR))

    w = int(np.argmin(np.abs(dec["t"] - a)))
    wb = b["wbeats"][w]
    wb = wb[wb >= b["ends"][w] - int(G.SUSTAIN * fs)]
    idx = T.index_of(b["pos"], wb)
    det_t = b["pos"][idx] / fs
    un = T.unlike(b["corr"], ("shape", True, G.UNLIKE_C))[idx]
    shape, size = b["corr"]["shape"][idx], b["corr"]["size"][idx]

    found, unlike_v, shape_v, size_v = [], [], [], []
    for t in bt[is_v]:
        k = int(np.argmin(np.abs(det_t - t))) if det_t.size else -1
        ok = k >= 0 and abs(det_t[k] - t) <= MATCH_S
        found.append(ok)
        if ok:
            unlike_v.append(bool(un[k]))
            shape_v.append(float(shape[k]))
            size_v.append(float(size[k]))
    n_v, n_found, n_unlike = int(is_v.sum()), int(sum(found)), int(sum(unlike_v))
    if n_v == 0:
        kind = "no V beat" if vt else "other"
    elif n_found < n_v:
        kind = "missed"
    elif n_unlike < n_v:
        kind = "alike"
    else:
        kind = "short"

    n_t = bt[~is_v]
    n_shape = [float(shape[int(np.argmin(np.abs(det_t - t)))]) for t in n_t
               if det_t.size and np.min(np.abs(det_t - t)) <= MATCH_S]
    first = b["first"]
    return {"a": a, "bpm": float(dec["bpm"][w]), "vt_run": int(dec["vt_run"][w]),
            "share": float(dec["share"][w]), "lz": float(dec["lz"][w]), "vt": vt,
            "counts": dict(Counter(sym.tolist())), "n_v": n_v, "v_run": longest(is_v),
            "n_found": n_found, "n_unlike": n_unlike,
            "shape_v": float(np.median(shape_v)) if shape_v else np.nan,
            "size_v": float(np.median(size_v)) if size_v else np.nan,
            "shape_n": float(np.median(n_shape)) if n_shape else np.nan,
            "n_det": int(idx.size), "det_unlike": int(un.sum()), "kind": kind,
            "template": None if first is None else (first[0], first[1]),
            "t0": t0, "t1": t1, "beats_t": bt, "beats_v": is_v, "det_t": det_t, "det_un": un}


def label_at(ivs, t: float) -> str:
    return next((lab for s, e, lab in ivs if s <= t < e), "unlabelled")


def show(ds: str, name: str, ivs, r: dict) -> None:
    f = T._f
    vt = ", ".join(f"{T._mmss(s)} to {T._mmss(e)} ({e - s:.0f} s)" for s, e in r["vt"]) or "none"
    cnt = ", ".join(f"{k} {v}" for k, v in sorted(r["counts"].items(), key=lambda kv: -kv[1]))
    print(f"\n{ds} {name} {T._mmss(r['a'])}  rate {f(r['bpm'])}, alike {f(r['share'], '{:.0%}')}, lz "
          f"{f(r['lz'], '{:.2f}')}; VT check: longest unlike run {r['vt_run']} (it blocks at {G.VT_RUN})")
    print(f"  annotated VT marks in the 30 s: {vt}")
    print(f"  annotated beats: {cnt or 'none'}; ventricular {r['n_v']}, longest run {r['v_run']}")
    if r["n_v"]:
        print(f"  ventricular beats: {r['n_found']} found by the detector, {r['n_v'] - r['n_found']} missed;"
              f" of those found, {r['n_unlike']} unlike the template (shape median {f(r['shape_v'], '{:.2f}')},"
              f" size median {f(r['size_v'], '{:.2f}')})")
    print(f"  other beats: shape median {f(r['shape_n'], '{:.2f}')}; detected beats {r['n_det']},"
          f" {r['det_unlike']} unlike")
    tp = r["template"]
    if tp is not None:
        print(f"  template from {T._mmss(tp[0])} to {T._mmss(tp[1])}, rhythm there: {label_at(ivs, tp[0])}")
    print(f"  -> {r['kind']}")


def plot(pdf, ds: str, name: str, b: dict, r: dict) -> None:
    import matplotlib.pyplot as plt
    fs, x = b["fs"], b["x"]
    i0, i1 = max(0, int(r["t0"] * fs)), min(len(x), int(r["t1"] * fs))
    tt = np.arange(i0, i1) / fs
    fig, ax = plt.subplots(figsize=(11.7, 3.6))
    ax.plot(tt, x[i0:i1], color="black", lw=0.6)
    for s, e in r["vt"]:
        ax.axvspan(max(s, r["t0"]), min(e, r["t1"]), color="#d62728", alpha=0.12)
    lo_, hi_ = (float(np.min(x[i0:i1])), float(np.max(x[i0:i1]))) if i1 > i0 else (0.0, 1.0)
    span = max(hi_ - lo_, 1e-6)
    for t, v in zip(r["beats_t"], r["beats_v"]):
        ax.text(t, hi_ + 0.08 * span, "V" if v else "·", color="#d62728" if v else "#777777", ha="center",
                va="bottom", fontsize=7)
    ax.set_ylim(lo_ - 0.05 * span, hi_ + 0.22 * span)
    for t, u in zip(r["det_t"], r["det_un"]):
        k = min(max(int(t * fs), 0), len(x) - 1)
        ax.plot(t, x[k], "x" if u else "o", color="#1f77b4", ms=4, mfc="none")
    ax.set_title(f"{ds} {name} {T._mmss(r['a'])}: entry's 30 s. Shaded: annotated VT. Letters on top: annotated"
                 f" beats (V ventricular). Blue: detected beats (o alike the template, x unlike). Kind: {r['kind']}",
                 fontsize=7, loc="left")
    ax.set_xlabel("seconds in the analysed span", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def summary(rows: list) -> dict:
    n = Counter(r["kind"] for r in rows)
    print(f"\nSUMMARY  {len(rows)} entries touching VT")
    for k in KINDS:
        print(f"  {k:<10} {n.get(k, 0)}")
    print("(missed: the detector did not find VT beats; alike: found, but alike the patient's own beat on this"
          " lead;\n short: found and unlike, never 3 in a row among the detected beats; no V beat: a VT mark"
          " with no\n ventricular beat marked in the 30 s)")
    return dict(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=list(MAIN_RUN), help="dataset:record")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--stride", type=float, default=5.0, help="the main run's stride on the 24 h sets")
    ap.add_argument("--plot", action="store_true", help="one page per entry")
    ap.add_argument("--out", type=Path, default=Path("data/plots/doctor_vt.pdf"))
    a = ap.parse_args(argv)
    from munji_ai.data.loader import load_record

    pdf = None
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        a.out.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(a.out)
    rows = []
    try:
        for key in a.records:
            ds, name = key.split(":", 1)
            rec = load_record(ds, name, a.raw)
            step = a.stride if ds in G.LONG_SETS else G.POS_STEP
            b, dec, al = doctor_entries(rec, step)
            unsafe = [float(t) for t in al if G.classify(float(t), b["ivs"]).startswith("unsafe:")]
            expect = MAIN_RUN.get(key)
            print(f"\n{ds} {name}: {len(al)} entries, {len(unsafe)} touching VT or VF"
                  + (f" (the main run: {expect})" if expect is not None else ""))
            for t in unsafe:
                r = look(rec, b, dec, t)
                show(ds, name, b["ivs"], r)
                rows.append(r)
                if pdf is not None:
                    plot(pdf, ds, name, b, r)
    finally:
        if pdf is not None:
            pdf.close()
            print(f"\nplots: {a.out}")
    summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
