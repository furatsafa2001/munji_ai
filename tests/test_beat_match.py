"""rules/beat_match against experiments/template_wct, the code it was moved from.

Every SVT result came from template_wct, so the move must change nothing: the
same constants, beat windows, patch-on template, shape and size of every beat,
and alike and unlike. Checked on synthetic records (no PhysioNet), on random
beats, and, when they are on disk, on real CUDB and VFDB records. Plus the
stored form the phone keeps.

    python tests/test_beat_match.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from munji_ai import config as C  # noqa: E402
from munji_ai.rules import beat_match as bm  # noqa: E402


def _experiment(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _experiment("template_wct")
G = _experiment("fast_narrow_template")
FS = C.TARGET_FS
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def same(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


# ------------------------------------------------------------------ records
def beat(tt, c, kind, amp):
    if kind == "N":                                   # narrow, the patient's own
        return amp * np.exp(-0.5 * ((tt - c) / 0.012) ** 2)
    return amp * 0.8 * (np.exp(-0.5 * ((tt - c + 0.03) / 0.03) ** 2)          # wide, two humps
                        - 0.9 * np.exp(-0.5 * ((tt - c - 0.05) / 0.035) ** 2))


def ecg(parts, seed=0):
    """parts: (bpm, seconds, share of wide beats, amplitude). Returns signal, beat times, kinds."""
    rng = np.random.default_rng(seed)
    t, times, kinds, amps = 1.0, [], [], []
    for bpm, sec, wide, amp in parts:
        t0 = t
        while t < t0 + sec:
            times.append(t)
            kinds.append("V" if rng.random() < wide else "N")
            amps.append(amp)
            t += 60.0 / bpm
    dur = t + 1.0
    n = int(dur * FS)
    tt = np.arange(n) / FS
    x = np.zeros(n)
    for c, k, a in zip(times, kinds, amps):
        lo, hi = max(0, int((c - 0.3) * FS)), min(n, int((c + 0.3) * FS))
        x[lo:hi] += beat(tt[lo:hi], c, k, a)
    return x + rng.normal(0, 0.01, n), np.array(times), kinds


def record(x, times, kinds, marks, name):
    from munji_ai.data.loader import Record
    starts, labels = zip(*marks)
    return Record(dataset="cudb", name=name, patient=f"cudb:{name}", signal=x, fs=FS,
                  beat_samples=np.round(times * FS).astype(int), beat_symbols=np.array(kinds, dtype=object),
                  rhythm_starts=np.array([int(s * FS) for s in starts]), rhythm_labels=np.array(labels, dtype=object))


def compare(rec, step, label):
    """Every output of the move against template_wct's on one record."""
    b = T.beats_and_template(rec, step)
    fs = b["fs"]
    pos, ext = bm.beat_windows(b["x"], b["pos"], fs)
    lo = 0
    blocked = [(s, e) for s, e, tok in T.raw_intervals(rec, lo, len(rec.signal)) if tok in T.CAL_EXCLUDED]
    tpl = bm.first_template(b["ts"], ext[b["si"]], fs, blocked)
    cal, counts = bm.calibrations(b["ts"], ext[b["si"]], fs)
    tcal, tcounts = T.calibrations(b["ts"], b["ext"][b["si"]], b["L"], b["lag"], "dominant")
    shape, size = bm.match(ext, tpl, fs)
    check(f"{label}: beat windows", same(pos, b["pos"]) and same(ext, b["ext"]))
    check(f"{label}: every qualifying stretch and the counts", counts == tcounts and len(cal) == len(tcal)
          and all(a[0] == c[0] and a[1] == c[1] and same(a[2], c[2]) for a, c in zip(cal, tcal)),
          f"{counts} {tcounts}")
    if b["first"] is None:
        check(f"{label}: no template, as template_wct", tpl is None)
    else:
        check(f"{label}: the same patch-on template, from the same stretch", tpl is not None
              and same(tpl.waveform, b["first"][2]) and (tpl.start, tpl.end) == (b["first"][0], b["first"][1]),
              f"{None if tpl is None else (tpl.start, tpl.end)} vs {b['first'][:2]}")
    check(f"{label}: shape and size of every beat", same(shape, b["corr"]["shape"]) and same(size, b["corr"]["size"]))
    check(f"{label}: alike and unlike of every beat",
          same(bm.alike(shape, size), G.alike(b["corr"], G.ALIKE_C))
          and same(bm.unlike(shape, size), T.unlike(b["corr"], ("shape", True, G.UNLIKE_C))))
    return b, tpl


def main() -> int:
    print("constants")
    check("the same as template_wct and fast_narrow_template",
          (bm.PRE_SEC, bm.POST_SEC, bm.LAG_SEC, bm.CAL_SEC, bm.CAL_BPM, bm.CAL_MIN_BEATS, bm.CAL_MATCH,
           bm.CAL_SHARE, bm.CAL_DOMINANT, bm.SIZE_FACTOR, bm.ALIKE_C, bm.UNLIKE_C)
          == (T.PRE_SEC, T.POST_SEC, T.LAG_SEC, T.CAL_SEC, T.CAL_BPM, T.CAL_MIN_BEATS, T.CAL_MATCH,
              T.CAL_SHARE, T.CAL_DOMINANT, T.SIZE_FACTOR, G.ALIKE_C, G.UNLIKE_C))
    check("window sizes", all(bm.sizes(f) == T._sizes(f) for f in (128, 250, 360)))

    print("random beats")
    rng = np.random.default_rng(7)
    _, L, lag = bm.sizes(FS)
    n_none = n_quick = n_dom = 0
    agree = True
    u = np.arange(L + 2 * lag) - (L + 2 * lag) / 2
    base = np.exp(-0.5 * (u / 3.0) ** 2)                        # narrow peak
    other = np.exp(-0.5 * ((u + 6) / 8.0) ** 2) - 0.9 * np.exp(-0.5 * ((u - 12) / 9.0) ** 2)   # wide
    for k in range(60):
        n = int(rng.integers(4, 16))
        share = rng.uniform(0, 0.55)
        noise = rng.uniform(0.005, 0.2)
        ext = np.stack([(other if rng.random() < share else base) * rng.uniform(0.8, 1.2)
                        + rng.normal(0, noise, base.size) for _ in range(n)])
        a, b = bm.dominant_template(ext, L, lag), T.dominant_template(ext, L, lag)
        agree &= (a is None and b is None) or (a is not None and b is not None and same(a, b))
        n_none += a is None
        n_quick += a is not None and bm.stretch_template(ext, L, lag) is not None
        n_dom += a is not None and bm.stretch_template(ext, L, lag) is None
        r1, k1 = bm.best_corr(ext, base[lag:lag + L], with_shift=True)
        r2, k2 = T.best_corr(ext, base[lag:lag + L], with_shift=True)
        agree &= same(r1, r2) and same(k1, k2)
    check("the template and the best correlation, on 60 random stretches", agree,
          f"no template {n_none}, median taken {n_quick}, main shape {n_dom}")
    check("the random stretches cover all three outcomes", n_none and n_quick and n_dom)

    print("synthetic records")
    x, times, kinds = ecg([(75, 150, 0.0, 1.0), (165, 90, 0.0, 1.0), (75, 60, 0.05, 1.0)], seed=1)
    compare(record(x, times, kinds, [(0, "(N")], "901"), bm_step(), "sinus, SVT, a few wide beats")
    # the first stretch has 36% wide beats (seed 5): the median-beat rule (80%) refuses it, the
    # main-shape rule (60%) takes the narrow beat
    x, times, kinds = ecg([(70, 40, 0.3, 1.0), (75, 120, 0.0, 1.0)], seed=5)
    b, tpl = compare(record(x, times, kinds, [(0, "(N")], "902"), bm_step(), "frequent wide beats at patch-on")
    e0 = b["ext"][b["si"]][b["ts"] < bm.CAL_SEC]
    narrow = bm.beat_windows(x, np.round(times[np.array(kinds) == "N"][:1] * FS).astype(int), FS)[1]
    check("frequent wide beats at patch-on: the main-shape rule decides the first stretch, and takes the narrow beat",
          bm.stretch_template(e0, L, lag) is None and tpl is not None and tpl.start == 0.0
          and bm.best_corr(narrow, tpl.waveform)[0] > 0.9, "" if tpl is None else str(tpl.start))
    # VT marked at the start: its stretches are blocked, the template comes later
    x, times, kinds = ecg([(75, 30, 0.0, 1.0), (75, 150, 0.0, 1.0)], seed=3)
    _, tpl = compare(record(x, times, kinds, [(0, "(VT"), (25, "(N")], "903"), bm_step(), "VT marked at the start")
    check("VT marked at the start: the template starts after the mark", tpl is not None and tpl.start >= 25.0,
          "" if tpl is None else str(tpl.start))
    # too fast at the start of the wear (SVT before any calm stretch)
    x, times, kinds = ecg([(170, 60, 0.0, 1.0), (80, 60, 0.0, 1.0)], seed=4)
    _, tpl = compare(record(x, times, kinds, [(0, "(N")], "904"), bm_step(), "fast at patch-on")
    check("fast at patch-on: the template waits for a stretch at 40 to 100", tpl is not None and tpl.start >= 60.0,
          "" if tpl is None else str(tpl.start))
    # never a calm stretch: no template, nothing alike or unlike
    x, times, kinds = ecg([(170, 120, 0.0, 1.0)], seed=5)
    b, tpl = compare(record(x, times, kinds, [(0, "(N")], "905"), bm_step(), "never calm")
    shape, size = bm.match(b["ext"], tpl, FS)
    check("never calm: no template, every beat NaN, none alike or unlike", tpl is None
          and np.isnan(shape).all() and not bm.alike(shape, size).any() and not bm.unlike(shape, size).any())

    print("stored on the phone")
    x, times, kinds = ecg([(75, 60, 0.0, 1.0)], seed=6)
    b, tpl = compare(record(x, times, kinds, [(0, "(N")], "906"), bm_step(), "stored")
    d = json.loads(json.dumps(tpl.to_dict()))
    back = bm.Template.from_dict(d)
    check("to_dict is JSON, and from_dict gives the same template back", same(back.waveform, tpl.waveform)
          and (back.fs, back.start, back.end) == (tpl.fs, tpl.start, tpl.end))
    check("a template read back matches every beat as before",
          all(same(u, v) for u, v in zip(bm.match(b["ext"], back, FS), bm.match(b["ext"], tpl, FS))))
    for bad, why in (({**d, "version": 99}, "another version"), ({**d, "waveform": d["waveform"][:-1]}, "a short one"),
                     ({**d, "fs": 128}, "another fs")):
        try:
            bm.Template.from_dict(bad)
            check(f"refused: {why}", False)
        except ValueError:
            check(f"refused: {why}", True)
    try:
        bm.match(b["ext"], bm.Template(tpl.waveform, 128, 0.0, 10.0), FS)
        check("a template at another fs is refused by match", False)
    except ValueError:
        check("a template at another fs is refused by match", True)

    print("real records (when on disk)")
    from munji_ai.data.loader import load_record
    done = 0
    for ds, name in (("cudb", "cu01"), ("cudb", "cu05"), ("vfdb", "426")):
        try:
            rec = load_record(ds, name, C.RAW_DIR)
        except Exception as e:                        # noqa: BLE001  not downloaded here
            print(f"  skipped {ds} {name}: {type(e).__name__}")
            continue
        compare(rec, T.POS_STEP, f"{ds} {name}")
        done += 1
    if not done:
        print("  (no real records found: the synthetic checks above still hold)")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


def bm_step() -> float:
    return T.POS_STEP


if __name__ == "__main__":
    raise SystemExit(main())
