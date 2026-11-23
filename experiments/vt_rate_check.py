"""How fast is the VT in CUDB and VFDB really? A rate that does not depend on beat size.

    python experiments/vt_rate_check.py

Why
---
The VT pictures (template_wct_plot.py) showed the episode rate used so far
(data.windows._episode_bpm) reading far under the visible rate, or nothing at
all, when the VT is a smooth wave or small: its peak threshold is twice the
median deviation of the whole episode, so a sine-like wave has no peak
prominent enough, and the quieter stretches of an episode drop out. The same
rate splits VT for ShockableNet (windows.split_vt_by_rate): 150 bpm or more
-> VT_FAST, trained as shockable; under 150 -> VT_SLOW, trained as NOT
shockable; unread -> "VT", left out. A rate that reads low teaches fast VT as
not shockable.

The measure
-----------
Autocorrelation of each 5 s shockable window (the windows ShockableNet sees
and is trained on, 2.5 s apart): how far apart the waveform repeats itself
(data/shockable_labels.acf_rates; settings in config, VT_RATE_*). Under the
regularity line the window gets no rate: fibrillation and noise do not
repeat. No beat detection and no threshold on beat size, so a small or smooth
VT reads the same as a large sharp one. Checked against the rate engine's
detector where both should agree: windows wholly in normal rhythm.

This is the check that found the labelling fault (47% of VT_SLOW windows were
150 bpm or more); data/shockable_labels.py is the fix.

What it prints
--------------
check     normal-rhythm windows: autocorrelation rate against the detector
VT time   minutes of VT by the new rate (150+, 100-150, under 100, irregular)
          for each label training used (VT_FAST, VT_SLOW, unread)
training  the VT windows ShockableNet's training draws from (label over half
          the window, as windows.extract_shockable), counted, and how many
          sit on the wrong side of 150
episodes  every VT of 30 s or more: old rate, detector, new rate, and the
          share of its windows in each band

Counts are of every candidate window; training samples some of them, so the
shares, not the counts, carry over. Reads only. Test-split patients are
skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import template_wct as tw  # noqa: E402
from munji_ai.data.shockable_labels import acf_rates  # noqa: E402
from munji_ai.data.windows import (_episode_bpm, _frames, dominant_label,  # noqa: E402
                                   split_vt_by_rate)

C = tw.C
BANDS = ("150+", "100-150", "<100", "irregular")
VT_LABELS = ("VT_FAST", "VT_SLOW", "VT")
SHOWN = {"VT_FAST": "VT_FAST (shockable)", "VT_SLOW": "VT_SLOW (not shockable)",
         "VT": "VT, rate unread (left out)"}


def band(rate: float) -> str:
    if not np.isfinite(rate):
        return "irregular"
    return "150+" if rate >= C.VT_FAST_BPM else ("100-150" if rate >= 100 else "<100")


def detector_rate(beats: np.ndarray, lo: int, hi: int, fs: int) -> float:
    b = beats[(beats >= lo) & (beats < hi)]
    return float(60.0 * fs / np.median(np.diff(b))) if b.size >= 3 else float("nan")


def record_rows(rec) -> tuple[list, list]:
    """Per shockable window: label, rates; and the VT episodes of 30 s or more."""
    fs = rec.fs
    ivs = split_vt_by_rate(rec, tw.rhythm_intervals(rec))
    ends, wbeats = tw.windowed_beats(rec.signal, fs, tw.POS_STEP)
    beats = tw.stream(ends, wbeats, tw.POS_STEP, fs)
    frames = list(_frames(rec, "shockable"))
    rows = []
    for lo, hi in frames:
        lab = dominant_label(ivs, lo, hi, C.SHOCKABLE_PURITY)
        whole = dominant_label(ivs, lo, hi, 1.0)
        if lab in VT_LABELS or lab == "VF" or whole == "NSR":
            rows.append({"lo": lo, "hi": hi, "label": lab, "whole": whole})
    if rows:
        rate, reg = acf_rates(np.stack([rec.signal[r["lo"]:r["hi"]] for r in rows]), fs)
        for r, v, g in zip(rows, rate, reg):
            r.update(rate=float(v), reg=float(g), band=band(v),
                     det=detector_rate(beats, r["lo"], r["hi"], fs))
    eps = []
    for s, e, lab in ivs:
        if lab not in VT_LABELS:
            continue
        inside = [r for r in rows if r["lo"] >= s and r["hi"] <= e]
        old = _episode_bpm(rec.signal[s:e], fs)
        eps.append({"set": rec.dataset, "record": rec.name, "s": s / fs, "dur": (e - s) / fs,
                    "label": lab, "old": float(old) if old else float("nan"),
                    "det": detector_rate(beats, s, e, fs),
                    "new": float(np.nanmedian([r["rate"] for r in inside]))
                    if any(np.isfinite(r["rate"]) for r in inside) else float("nan"),
                    "bands": {b: (sum(r["band"] == b for r in inside) / len(inside)) if inside else float("nan")
                              for b in BANDS}})
    return rows, eps


def report(rows: list, eps: list) -> None:
    step_min = C.WINDOW_STRIDE_SEC["shockable"] / 60.0     # each window adds one stride of time

    nsr = [r for r in rows if r["whole"] == "NSR" and np.isfinite(r["det"])]
    print(f"\n1. Check, normal rhythm: {len(nsr)} windows wholly in NSR with a detector rate")
    if nsr:
        reg = [r for r in nsr if np.isfinite(r["rate"])]
        agree = [abs(r["rate"] - r["det"]) <= 0.10 * r["det"] for r in reg]
        print(f"   regular by autocorrelation: {len(reg) / len(nsr):.0%}; of those, within 10% of the "
              f"detector: {np.mean(agree):.0%}" if reg else "   none regular")
    vt_det = [r for r in rows if r["label"] in VT_LABELS and np.isfinite(r["rate"]) and np.isfinite(r["det"])]
    if vt_det:
        agree = [abs(r["rate"] - r["det"]) <= 0.10 * r["rate"] for r in vt_det]
        lower = [r["det"] < 0.9 * r["rate"] for r in vt_det]
        print(f"   VT windows, regular, with a detector rate: {len(vt_det)}; detector within 10%: "
              f"{np.mean(agree):.0%}, detector lower by more than 10%: {np.mean(lower):.0%}")
    vf = [r for r in rows if r["label"] == "VF"]
    if vf:
        print(f"   VF windows: {len(vf)}, irregular by autocorrelation: "
              f"{np.mean([r['band'] == 'irregular' for r in vf]):.0%} (flutter is labelled VF and repeats)")

    vt = [r for r in rows if r["label"] in VT_LABELS]
    print(f"\n2. VT time by the new rate, minutes (windows labelled VT over half their length: {len(vt)})")
    print(f"   {'label training used':<28}" + "".join(f"{b:>11}" for b in BANDS) + f"{'total':>9}")
    tot = defaultdict(float)
    for lab in VT_LABELS:
        g = [r for r in vt if r["label"] == lab]
        cells = [sum(r["band"] == b for r in g) * step_min for b in BANDS]
        for b, c in zip(BANDS, cells):
            tot[b] += c
        print(f"   {SHOWN[lab]:<28}" + "".join(f"{c:>11.1f}" for c in cells) + f"{len(g) * step_min:>9.1f}")
    print(f"   {'all VT':<28}" + "".join(f"{tot[b]:>11.1f}" for b in BANDS) + f"{sum(tot.values()):>9.1f}")

    print("\n3. Training: windows on the wrong side of 150 (share of each label's windows)")
    for lab, wrong, what in (("VT_SLOW", "150+", "taught NOT shockable, but 150 or more"),
                             ("VT", "150+", "left out, but 150 or more"),
                             ("VT_FAST", None, "taught shockable, but under 150")):
        g = [r for r in vt if r["label"] == lab]
        if not g:
            continue
        bad = [r for r in g if (r["band"] == wrong if wrong else r["band"] in ("100-150", "<100"))]
        print(f"   {lab:<8} {len(g):>5} windows: {len(bad):>5} ({len(bad) / len(g):.0%}) {what}")
    print("   Episodes, every length, by the band most of their windows fall in:")
    print(f"   {'label':<10}" + "".join(f"{b:>11}" for b in BANDS) + f"{'too short':>11}")
    for lab in VT_LABELS:
        g = [e for e in eps if e["label"] == lab]
        if not g:
            continue
        top = [max(BANDS, key=lambda b: e["bands"][b]) if np.isfinite(e["bands"]["150+"]) else "too short"
               for e in g]
        print(f"   {lab:<10}" + "".join(f"{top.count(b):>11}" for b in BANDS + ("too short",)))

    long_ = [e for e in eps if e["dur"] >= C.VT_SUSTAINED_SEC]
    print("\n4. VT episodes of 30 s or more (rates in bpm; bands: share of the episode's windows)")
    print(f"   {'set':<6}{'record':<8}{'start':>8}{'dur':>6}{'label':>9}{'old':>6}{'det':>6}{'new':>6}"
          + "".join(f"{b:>11}" for b in BANDS))
    for e in long_:
        print(f"   {e['set']:<6}{e['record']:<8}{tw._mmss(e['s']):>8}{e['dur']:>6.0f}{e['label']:>9}"
              f"{tw._f(e['old']):>6}{tw._f(e['det']):>6}{tw._f(e['new']):>6}"
              + "".join(f"{tw._f(e['bands'][b], '{:.0%}'):>11}" for b in BANDS))
    print("   old: the rate training used (_episode_bpm); det: the rate engine's detector; new: autocorrelation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["cudb", "vfdb"])
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per set (0 = all)")
    a = ap.parse_args(argv)

    t0, rows, eps = time.time(), [], []
    for ds in a.sets:
        for name in tw.eligible(ds, a.raw, a.max_records):
            rec = tw.load_record(ds, name, a.raw)
            r, e = record_rows(rec)
            rows += r
            eps += e
            n_vt = sum(x["label"] in VT_LABELS for x in r)
            n_long = sum(x["dur"] >= C.VT_SUSTAINED_SEC for x in e)
            print(f"  {ds} {name}: {n_vt} VT windows, {len(e)} VT episodes ({n_long} of 30 s or more)",
                  flush=True)
    if not rows:
        print("no windows")
        return 1
    report(rows, eps)
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
