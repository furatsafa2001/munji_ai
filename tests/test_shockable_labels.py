"""Shockable labels per window (data/shockable_labels.py). Synthetic signals, no PhysioNet.

    python tests/test_shockable_labels.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import shockable_labels as SL
from munji_ai.data import windows as W
from munji_ai.data.loader import Record
from test_pipeline import synth_ecg

FS = C.TARGET_FS
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def vt_wave(seconds, bpm, amp=1.0, seed=0):
    """Smooth monomorphic VT: one wide cosine cycle per beat, no baseline
    between beats (the shape the old episode rate could not read)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = np.zeros(n)
    i = 0
    while i < n:
        rr = 60.0 / bpm * rng.normal(1.0, 0.01)
        m = int(rr * FS)
        x[i:i + m] = amp * np.cos(np.linspace(-np.pi, np.pi, m, endpoint=False))[:n - i]
        i += m
    return x + rng.normal(0, 0.01 * amp, n)


def vf_wave(seconds, ptp, seed=0):
    """Fibrillation: several 3-7 Hz waves of drifting amplitude, no repeat."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = sum(rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * rng.uniform(3, 7) * t + rng.uniform(0, 6))
            * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)) for _ in range(5))
    return x / np.ptp(x) * ptp


def poly_vt(seconds, ptp, seed=0):
    """Polymorphic VT: 3-7 Hz activity that never repeats (band-limited noise)."""
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(4, [3.0, 7.0], btype="band", fs=FS, output="sos"),
                    np.random.default_rng(seed).normal(0, 1, int(seconds * FS)))
    return x / np.ptp(x) * ptp


def record(parts, name="s1", annotate=True):
    """parts: (seconds, rhythm token, signal) in order."""
    sig = np.concatenate([p[2] for p in parts])
    starts, t = [], 0
    for sec, _, _ in parts:
        starts.append(int(t * FS))
        t += sec
    return Record("synthetic", name, f"synthetic:{name}", sig, FS,
                  np.array([], np.int64), np.array([], "<U2"),
                  np.array(starts if annotate else [], np.int64),
                  np.array([p[1] for p in parts] if annotate else [], dtype=object))


def main() -> int:
    win = C.window_samples("shockable")

    print("1. Rate by autocorrelation")
    for bpm in (110, 160, 200):
        r, g = SL.acf_rates(vt_wave(5, bpm, seed=bpm)[None, :win], FS)
        check(f"smooth VT at {bpm} bpm reads within 3%", abs(r[0] - bpm) <= 0.03 * bpm,
              f"{r[0]:.1f} bpm, regularity {g[0]:.2f}")

    print("\n2. Beat size does not change the rate (as EC13 tests heart-rate meters)")
    base = vt_wave(5, 180, seed=7)[:win]
    rates = [SL.acf_rates((k * base)[None, :], FS)[0][0] for k in (0.5, 1.0, 2.0)]
    check("half, normal and double size read the same rate", max(rates) - min(rates) < 1e-6,
          ", ".join(f"{v:.1f}" for v in rates))

    print("\n3. Sinus rhythm")
    x, _, peaks = synth_ecg(seconds=60, hr=72, fs=FS, seed=1)
    X = np.stack([x[i:i + win] for i in range(0, len(x) - win, win)])
    r, g = SL.acf_rates(X, FS)
    true = 60.0 * FS / np.median(np.diff(peaks))
    regular = np.isfinite(r)
    check("regular windows read the heart rate within 10%",
          regular.any() and np.all(np.abs(r[regular] - true) <= 0.1 * true),
          f"{regular.sum()} of {len(r)} regular, true {true:.0f} bpm, read {np.nanmedian(r):.0f}")

    print("\n4. No repeat, no rate")
    noise = np.random.default_rng(3).normal(0, 1, (20, win))
    rn, gn = SL.acf_rates(noise, FS)
    check("white noise is irregular in every window", np.all(~np.isfinite(rn)),
          f"highest regularity {gn.max():.2f}")

    print("\n5. Labels per window (the cache's rules: one rhythm per window)")
    nsr, _, _ = synth_ecg(seconds=60, hr=70, fs=FS, seed=2)
    rec = record([(60, "(N", nsr[:60 * FS]),
                  (60, "(VT", vt_wave(60, 180, seed=1)),
                  (60, "(VT", vt_wave(60, 120, seed=2)),
                  (60, "(VF", vf_wave(60, 1.5, seed=3)),
                  (60, "(VF", vf_wave(60, 0.12, seed=4)),
                  (60, "(VT", poly_vt(60, 2.0, seed=6)),
                  (20, "(NOISE", np.random.default_rng(5).normal(0, 0.3, 20 * FS))])
    iv = W.rhythm_intervals(rec)
    stride = C.stride_samples("shockable")
    starts = list(range(0, len(rec.signal) - win + 1, stride))
    labs = SL.label_windows(rec.signal, FS, iv, starts, win)

    def inside(a, b):
        return [lab for s, lab in zip(starts, labs) if s >= a * FS and s + win <= b * FS]

    for (a, b), want in (((0, 60), "NSR"), ((60, 120), "VT_FAST"), ((120, 180), "VT_SLOW"),
                         ((180, 240), "VF"), ((240, 300), "VF_FINE"), ((300, 360), "VT_IRREGULAR"),
                         ((360, 380), "NOISE")):
        got = inside(a, b)
        share = np.mean([g == want for g in got]) if got else 0.0
        check(f"{a}-{b} s gives {want}", share >= 0.9, f"{share:.0%} of {len(got)} windows")
    across = [lab for s, lab in zip(starts, labs) if s < 60 * FS < s + win]
    check("a window across a rhythm change is left out (None)", all(lab is None for lab in across),
          str(across))

    print("\n6. Classes")
    cls = {lab: SL.window_class(lab) for lab in
           ("VF", "VT_FAST", "VT_SLOW", "VF_FINE", "VT_IRREGULAR", "NOISE", "UNKNOWN", None,
            "NSR", "AF")}
    check("VF, fast VT and polymorphic (irregular) VT are shockable",
          cls["VF"] == cls["VT_FAST"] == cls["VT_IRREGULAR"] == "shockable")
    check("slow VT, fine VF, ambiguity and transitions are left out",
          all(cls[k] is None for k in ("VT_SLOW", "VF_FINE", "NOISE", "UNKNOWN", None)))
    check("other rhythms are not shockable", cls["NSR"] == cls["AF"] == "not_shockable")

    print("\n7. The cache (windows.extract_shockable)")
    X, Y, offs = W.extract_shockable(rec, np.random.default_rng(0), cap=10_000)
    offs, Y = np.asarray(offs), np.asarray(Y)
    check("both classes present", set(Y) == {"shockable", "not_shockable"}, str(sorted(set(Y))))
    in_slow = (offs >= 120 * FS) & (offs + win <= 180 * FS)
    in_fine = (offs >= 240 * FS) & (offs + win <= 300 * FS)
    check("nothing from the slow VT or the fine VF", not (in_slow | in_fine).any(),
          f"{int(in_slow.sum())} slow VT, {int(in_fine.sum())} fine VF windows")
    pos = offs[Y == "shockable"]
    in_slow_or_fine = [(o + win / 2) / FS for o in pos if 120 * FS < o + win / 2 < 180 * FS
                       or 240 * FS < o + win / 2 < 300 * FS]      # over half in it
    check("no shockable window is mostly slow VT or fine VF", not in_slow_or_fine,
          f"{len(pos)} shockable; centred in slow VT or fine VF: {in_slow_or_fine}")
    spans = ((60, 120), (180, 240), (300, 360))
    whole = [o for o in pos if any(a * FS <= o and o + win <= b * FS for a, b in spans)]
    check("every window wholly in the fast VT, the coarse VF or the polymorphic VT is shockable",
          len(whole) == sum(len(inside(a, b)) for a, b in spans), f"{len(whole)}")
    check("windows are normalised like every cache", np.allclose(np.median(X[0]), 0, atol=1e-6))

    print("\n8. Scoring flags (dominant rhythm over half the window, same rate rules)")
    n = len(starts)
    shock, inter = SL.scoring_flags(rec, n)
    st = np.asarray(starts)
    bounds = np.array([0, 60, 120, 180, 240, 300, 360, 380]) * FS
    seg = np.full(n, -1)                       # -1: the window crosses a change
    for k in range(7):
        seg[(st >= bounds[k]) & (st + win <= bounds[k + 1])] = k
    check("fast VT, coarse VF and polymorphic VT are shockable",
          shock[seg == 1].all() and shock[seg == 3].all() and shock[seg == 5].all())
    check("slow VT and fine VF are intermediate, not shockable",
          inter[seg == 2].all() and inter[seg == 4].all() and not shock[(seg == 2) | (seg == 4)].any())
    check("sinus and noise are neither", not (shock | inter)[(seg == 0) | (seg == 6)].any())
    check("a window across a change takes the rhythm over half of it",
          shock[[i for i, s in enumerate(starts) if s < 60 * FS < s + win and 60 * FS - s < win / 2]].all())

    print("\n9. A record with no rhythm annotation")
    bare = record([(60, "(N", nsr[:60 * FS])], name="bare", annotate=False)
    s9, i9 = SL.scoring_flags(bare, 10)
    check("scored as not shockable and not intermediate", not s9.any() and not i9.any())
    check("the cache takes nothing from it",
          len(W.extract_shockable(bare, np.random.default_rng(0), cap=100)[0]) == 0)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
