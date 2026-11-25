"""The SVT / fast narrow rhythm rule on synthetic records, without PhysioNet.

Gaussian beats the rate engine's detector finds: an SVT that starts and ends
at once, sinus tachycardia that speeds up and slows down gradually, fast
regular AF that only RhythmNet can tell apart, a fast rhythm with a short run
of wide beats inside it, and VT. The checks are about plumbing: each check,
SHDB-AF's tokens and reader, its download, the AF episode the decision layer
keeps, EC57 scoring, and the script end to end. They say nothing about real
ECG or about RhythmNet itself.

    python tests/test_fast_narrow_template.py
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("fast_narrow_template", ROOT / "experiments" / "fast_narrow_template.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

FS = 250
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beats(parts, seed=0):
    """[(bpm from, bpm to, seconds, sigma ms, jitter s)] -> [(time, sigma)].
    The rate moves linearly from one to the other; jitter makes RR irregular."""
    rng = np.random.default_rng(seed)
    t, out = 1.0, []
    for b0, b1, sec, sd, jit in parts:
        t0 = t
        while t < t0 + sec:
            out.append((t, sd))
            f = (t - t0) / sec
            t += 60.0 / (b0 + (b1 - b0) * f) + (rng.uniform(-jit, jit) if jit else 0.0)
    return out, t + 1.0


def ecg(parts, seed=0, fs=FS):
    bt, dur = beats(parts, seed)
    n = int(dur * fs)
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for c, sd in bt:
        lo, hi = max(0, int((c - 0.3) * fs)), min(n, int((c + 0.3) * fs))
        amp = -1.5 if sd < 0 else 1.0            # a negative sigma: an inverted, larger beat
        x[lo:hi] += amp * np.exp(-0.5 * ((tt[lo:hi] - c) / (abs(sd) / 1000.0)) ** 2)
    return x + np.random.default_rng(seed + 7).normal(0, 0.01, n), dur, np.array([c for c, _ in bt])


def record(x, starts, labels, name, ds="cudb"):
    from munji_ai.data.loader import Record
    return Record(dataset=ds, name=name, patient=f"{ds}:{name}", signal=x, fs=FS,
                  beat_samples=np.array([0, x.size - 1]), beat_symbols=np.array(["N", "N"]),
                  rhythm_starts=np.array([int(s * FS) for s in starts]),
                  rhythm_labels=np.array(labels, dtype=object))


REST = (75, 75, 150, 12, 0)
SVT = [REST, (180, 180, 90, 12, 0), (75, 75, 60, 12, 0)]
RAMP = [REST, (75, 170, 150, 12, 0), (170, 170, 60, 12, 0), (170, 75, 120, 12, 0)]
AF_IRREG = [REST, (158, 158, 90, 12, 0.10), (75, 75, 60, 12, 0)]
VT_NEW = [REST, (180, 180, 90, 35, 0), (75, 75, 60, 12, 0)]
SLOW = [REST, (130, 130, 90, 12, 0), (75, 75, 60, 12, 0)]
# an SVT that starts at once at 140, then wanders smoothly across 150
HOVER = [REST, (140, 140, 10, 12, 0), (140, 162, 15, 12, 0), (162, 148, 15, 12, 0), (148, 162, 15, 12, 0),
         (162, 149, 15, 12, 0), (149, 160, 15, 12, 0), (75, 75, 60, 12, 0)]
# sinus at the start of exercise: a leap to 110 in 10 s (vagal release), then on to 160 in a minute
EXER = [REST, (75, 110, 10, 12, 0), (110, 160, 60, 12, 0), (160, 160, 60, 12, 0), (160, 75, 90, 12, 0)]
IRREG = [REST, (170, 170, 90, 12, 0.10), (75, 75, 60, 12, 0)]
# fast rhythm from above with 4 beats of another shape inside it (a short VT run)
WITH_RUN = [REST, (180, 180, 5, 12, 0), (180, 180, 1.4, -20, 0), (180, 180, 85, 12, 0), (75, 75, 60, 12, 0)]


def ecg_beats(bt, dur, seed=0, sd=12, fs=FS):
    """An ECG from given beat times (to leave beats out: missed beats, lost signal)."""
    n = int(dur * fs)
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for c in bt:
        lo, hi = max(0, int((c - 0.3) * fs)), min(n, int((c + 0.3) * fs))
        x[lo:hi] += np.exp(-0.5 * ((tt[lo:hi] - c) / (sd / 1000.0)) ** 2)
    return x + np.random.default_rng(seed + 7).normal(0, 0.01, n)


def main() -> int:
    print("pieces")
    corr = {"shape": np.array([0.95, 0.95, 0.85, np.nan, 0.95]), "size": np.array([1.0, 0.3, 1.0, 1.0, np.nan])}
    check("alike: shape 0.90 or more and size half to twice; no template never alike",
          list(G.alike(corr, 0.90)) == [True, False, False, False, False])
    check("longest run of consecutive flags", G.longest_run(np.array([1, 1, 0, 1, 1, 1, 0], bool)) == 3)
    ivs = [(0, 100, "NSR"), (100, 200, "VT"), (200, 400, "SVT"), (400, 500, "NSR")]
    check("a label whose 30 s touches VT is unsafe, even after the VT ended",
          G.classify(150, ivs) == "unsafe:VT" and G.classify(220, ivs) == "unsafe:VT"
          and G.classify(260, ivs) == "SVT" and G.classify(460, ivs) == "false:NSR")
    ts = np.concatenate([np.arange(0, 60, 0.8), 60 + np.arange(1, 100) * 0.333, 93 + np.arange(1, 30) * 0.8])
    start, drop = G.fast_runs(ts, 150)
    s, d, s0 = G.sudden_onset(np.array([50.0, 80.0, 90.0]), ts, start, drop)
    check("sudden onset: none before the run, sudden once it is in the 30 s", list(s) == [False, True, True]
          and d[1] > 0.5, f"{s}, {d}")
    last = G.runs_of(start)
    rise = G.termination(ts, last[int(s0[1])])
    check("abrupt termination: the cycle length after the run is far longer", rise > 0.3, str(rise))
    ramp = np.cumsum(np.r_[np.full(60, 0.8), np.linspace(0.8, 0.35, 120), np.full(80, 0.35),
                           np.linspace(0.35, 0.8, 120), np.full(30, 0.8)])
    st, _ = G.fast_runs(ramp, 100)
    lr = G.runs_of(st)
    check("gradual slowing is not an abrupt end", all(G.termination(ramp, j) < 0.09 for j in lr.values()),
          str([round(G.termination(ramp, j), 3) for j in lr.values()]))
    idx, tok = G.shdb_tokens(["(N", "(N", "", "(AT", "(AT", "(PAT", "(NOD", "(AFIB", "(AFIB", "(N"])
    check("SHDB-AF marks: one per change, AT/PAT/NOD as SVTA", idx == [0, 3, 7, 9]
          and tok == ["(N", "(SVTA", "(AFIB", "(N"], f"{idx} {tok}")
    ends, blk = G.af_block(np.r_[np.zeros(4), np.ones(6), np.zeros(50)])
    check("AF: a window over 0.60 blocks; the episode (4 of 6) closes after 40 quiet windows",
          not blk[3] and blk[4] and blk[12] and blk[48] and not blk[49], f"{np.flatnonzero(blk)[[0, -1]]}")
    check("blocked_at reads the latest finished rhythm window", list(G.blocked_at(np.array([10.0, 100.0, 1000.0]),
          (ends, blk))) == [False, True, False])
    okw = np.ones(20, bool)
    okw[2] = False
    check("episodes from consecutive decisions; a window without 30 s of beats does not split one",
          G.episodes_of(np.arange(0, 50, 2.5), np.array([0, 1, 0, 1, 0, 0, 1, 1] + [0] * 12, bool), okw)
          == [(2.5, 7.5), (15.0, 17.5)])
    fake = [{"dataset": "x", "name": "a", "ivs": [(0, 100, "NSR"), (100, 160, "SVT"), (160, 300, "NSR"),
                                                  (300, 310, "SVT"), (310, 400, "NSR"), (400, 420, "VT")],
             "report": [{"s": 110.0, "e": 170.0, "psvt": True}, {"s": 200.0, "e": 230.0, "psvt": False},
                        {"s": 395.0, "e": 405.0, "psvt": True}]}]
    five = G.ROW_CHECKS[G.PRIMARY]
    pw = {k: np.array(v, bool) for k, v in zip(five, ([1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]))}
    check("why: 'slower' with no decision at 150, '-' when one passed all, else the check failing most",
          G.why(pw, np.array([], int), five) == "slower" and G.why(pw, np.array([0, 1, 2]), five) == "not AF"
          and G.why(pw, np.array([1, 2]), G.ROW_CHECKS[G.NO_AF]) == "regular"
          and G.why(pw, np.array([0, 2]), G.ROW_CHECKS[G.NO_AF]) == "regular+onset"
          and G.why({**pw, "onset": np.ones(3, bool)}, np.array([0, 1, 2]), G.ROW_CHECKS[G.NO_AF]) == "-")
    rng = np.random.default_rng(0)
    reg, afr = G.af_score(0.4 + rng.uniform(-0.002, 0.002, 300)), G.af_score(rng.uniform(0.3, 0.9, 200))
    check("Lorenz: regular RR far under 0 (all at the origin), irregular RR over 0, too few points unscored",
          reg < -0.9 and afr > 0.3 and np.isnan(G.af_score(np.full(20, 0.4))), f"{reg} {afr}")
    sp = np.r_[np.full(100, 0.4), [0.15, 0.15], np.full(100, 0.4)]
    check("Lorenz: intervals of 220 ms or less are dropped with their points",
          G.af_score(sp) == -1.0, str(G.af_score(sp)))
    tb = np.cumsum(np.r_[np.full(300, 0.8), rng.uniform(0.3, 0.9, 400)])
    lzs = G.lorenz_scores(np.array([200.0, tb[-1]]), tb, np.array([True, True]))
    check("Lorenz over the last 2 minutes: sinus then AF", lzs[0] < 0 < lzs[1], str(lzs))
    tr = np.cumsum(np.r_[np.full(20, 0.8), np.full(10, 0.5), np.linspace(0.5, 0.36, 40), np.full(20, 0.36),
                         np.full(10, 0.8), np.full(10, 0.36)])
    st100, _ = G.fast_runs(tr, 100)
    rt = G.reach_times(tr, st100, 150)
    vals = [rt[k] for k in sorted(rt)]
    check("reach: seconds from a run's start over 100 to its first running rate of 150, per run",
          len(vals) == 2 and 10 < vals[0] < 20 and vals[1] < 2, str(vals))
    rng2 = np.random.default_rng(5)
    at_rr = rng2.choice([1, 2], 60) * 0.25 + rng2.normal(0, 0.006, 60)
    ok_at, cov_at, cyc = G.organized_rr(at_rr)
    ok_af = any(G.organized_rr(rng2.normal(0.38, 0.06, 60).clip(0.25, 0.8))[0] for _ in range(50))
    ok_reg = G.organized_rr(0.375 + rng2.normal(0, 0.004, 60))[0]
    check("RR multiples: 1:1 and 2:1 of a 250 ms cycle pass; fast AF never; a regular rhythm is not 'organized';"
          " too few intervals not judged", ok_at and abs(cyc - 0.25) < 0.02 and not ok_af and not ok_reg
          and not G.organized_rr(at_rr[:8])[0], f"{ok_at} {cov_at:.2f} {cyc:.3f} {ok_af} {ok_reg}")
    tb = np.arange(0, 120, 0.5)
    okw = np.array([True, True, False, True])
    rb = G.rate_before(np.array([30.0, 60.0, 90.0, 115.0]), okw, np.array([120.0, 130.0, 140.0, 150.0]), tb,
                       np.array([-1, 125, 190, 2]))
    check("rate before: the last decision before the run's first beat, NaN without a run or a usable rate",
          np.isnan(rb[0]) and rb[1] == 130.0 and np.isnan(rb[2]) and np.isnan(rb[3]), str(rb))
    o = G.ec57(fake)
    check("EC57: episode Se 1/2, 30 s or more 1/1, +P 1/3, one unsafe, one false",
          o["ESe"] == 0.5 and o["ESe30"] == 1.0 and abs(o["E+P"] - 1 / 3) < 1e-9 and o["unsafe"] == 1
          and o["false_by"]["x"] == 1 and abs(o["DSe"] - 50 / 70) < 1e-9, str({k: o[k] for k in ("ESe", "E+P", "DSe")}))

    print("records")

    def run(parts, starts, labels, name, seed=0, af=None, af_fire=None):
        x, dur, _ = ecg(parts, seed)
        return G.process(record(x, starts, labels, name), negative=False, step=G.POS_STEP, af=af, af_fire=af_fire)
    r_svt = run(SVT, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu90")
    sc = G.score([r_svt], G.PRIMARY)
    check("SVT that starts at once, same shape, regular: labelled once, in the SVT",
          sc["long"]["SVT"] == [1, 1] and not sc["unsafe"] and not sc["false"], f"{dict(sc['long'])}")
    e0 = r_svt["episodes"][0]
    check("the episode reached 150, every check passed there, nothing kept it silent",
          e0["peak"] > 170 and e0["fast_s"] >= 60 and set(e0["why"].values()) == {"-"}
          and all(v == 1.0 for k_, v in e0["pass"].items() if k_ != "organized") and sc["fast"] == [1, 1] and e0["lz"] < -0.5,
          f"{e0['peak']:.0f} {e0['fast_s']} {e0['why']} {e0['pass']} lz {e0['lz']}")
    ep = r_svt["report"][0] if r_svt["report"] else {}
    check("and the report holds one PSVT episode, about 151 to 241 s (ended abruptly)", len(r_svt["report"]) == 1
          and ep["psvt"] and abs(ep["s"] - 151) < 3 and abs(ep["e"] - 241) < 3, str(r_svt["report"]))
    r_ramp = run(RAMP, [0], ["(N"], "cu91", seed=1)
    check("sinus tachycardia, gradual: no label and no report episode",
          r_ramp["alerts"][G.PRIMARY].size == 0 and not r_ramp["report"], f"{r_ramp['report']}")
    r_af = run(AF_IRREG, [0, 151, 241], ["(N", "(AFIB", "(N"], "cu92", seed=2)
    check("fast irregular AF: no label (regularity)", r_af["alerts"][G.PRIMARY].size == 0)
    p = np.zeros(40)
    p[9:16] = 1.0                                            # RhythmNet says AF from about 150 to 255 s
    r_afr = run(SVT, [0, 151, 241], ["(N", "(AFIB", "(N"], "cu93", seed=3, af=G.af_block(p))
    check("fast regular AF that RhythmNet calls AF: no label; without the AF check, a false label",
          r_afr["alerts"][G.PRIMARY].size == 0 and r_afr["alerts"]["label: no AF check"].size == 1,
          f"{r_afr['alerts']['label: no AF check']}")
    r_blk = run(SVT, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu96", seed=3, af=G.af_block(p))
    e1 = r_blk["episodes"][0]
    check("an SVT RhythmNet calls AF: row 1 silent for 'not AF', row 2 labels it",
          e1["why"][G.PRIMARY] == "not AF" and e1["why"][G.NO_AF] == "-" and e1["why"][G.LORENZ] == "-"
          and G.score([r_blk], G.PRIMARY)["fast"] == [0, 1] and G.score([r_blk], G.NO_AF)["fast"] == [1, 1]
          and G.score([r_blk], G.LORENZ)["fast"] == [1, 1], str(e1["why"]))
    r_slow = run(SLOW, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu97", seed=11)
    e2 = r_slow["episodes"][0]
    check("an SVT at 130: 'slower', not counted in SVT150", e2["why"][G.NO_AF] == "slower" and 120 < e2["peak"] < 140
          and G.score([r_slow], G.PRIMARY)["fast"] == [0, 0], f"{e2['peak']:.0f} {e2['why']}")
    real_cut = G.LZ_AF
    G.LZ_AF = -1e9                                          # the Lorenz score calls everything AF
    try:
        r_lz = run(SVT, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu99", seed=3)
    finally:
        G.LZ_AF = real_cut
    check("the Lorenz row is blocked by its own check alone, the others not",
          r_lz["alerts"][G.LORENZ].size == 0 and r_lz["alerts"][G.NO_AF].size == 1 and not r_lz["report_lz"]
          and len(r_lz["report"]) == 1 and r_lz["episodes"][0]["why"][G.LORENZ] == "Lorenz",
          str(r_lz["episodes"][0]["why"]))
    with contextlib.redirect_stdout(io.StringIO()):
        g = G.lorenz_cuts([r_svt, r_blk])
    check("the cut-off table sorts the row-2 labels by what they touched", len(g["SVT/AFL"]) == 2, str(g))
    r_hov = run(HOVER, [0, 151, 251], ["(N", "(SVTA", "(N"], "cu88", seed=13)
    check("an SVT that starts at 140 and wanders across 150: labelled with the onset over 100, not at 150",
          r_hov["alerts"][G.LORENZ].size == 1 and r_hov["alerts"][G.LORENZ_150].size == 0
          and r_hov["alerts"][G.LORENZ_REACH].size == 1 and r_hov["episodes"][0]["why"][G.LORENZ_150] == "onset",
          str({k: r_hov["alerts"][k].tolist() for k in (G.LORENZ, G.LORENZ_150, G.LORENZ_REACH)}))
    r_ex = run(EXER, [0], ["(N"], "cu89", seed=14)
    check("sinus leaping at the start of exercise, then on to 160: the onset over 100 labels it (the cost),"
          " 150 within 30 s and the onset at 150 do not", r_ex["alerts"][G.LORENZ].size == 1
          and r_ex["alerts"][G.LORENZ_REACH].size == 0 and r_ex["alerts"][G.LORENZ_150].size == 0,
          str({k: r_ex["alerts"][k].tolist() for k in (G.LORENZ, G.LORENZ_150, G.LORENZ_REACH)}))
    # sinus that climbs slowly to 160; then the detector misses every other beat for 6 s
    parts = [(75, 75, 150, 12, 0), (75, 160, 240, 12, 0), (160, 160, 150, 12, 0), (160, 80, 120, 12, 0)]
    bt, dur = beats(parts, 0)
    bt = np.array([c for c, _ in bt])
    keep = bt[~((bt > 460) & (bt < 466)) | (np.arange(bt.size) % 2 == 0)]
    r_mb = G.process(record(ecg_beats(keep, dur), [0], ["(N"], "cu70"), negative=False, step=G.POS_STEP)
    check("missed beats at 160 fake an abrupt onset: the rows before label it, 'slower before' does not",
          r_mb["alerts"][G.LORENZ_150].size == 1 and r_mb["alerts"][G.SLOWER].size == 0,
          str({k: r_mb["alerts"][k].tolist() for k in (G.LORENZ_150, G.SLOWER)}))
    # lost signal for two minutes, back at 166
    parts = [(75, 75, 150, 12, 0), (75, 75, 120, 12, 0), (166, 166, 90, 12, 0), (166, 100, 120, 12, 0)]
    bt, dur = beats(parts, 1)
    bt = np.array([c for c, _ in bt])
    r_gap = G.process(record(ecg_beats(bt[~((bt > 150) & (bt < 270))], dur, seed=1), [0], ["(N"], "cu71"),
                      negative=False, step=G.POS_STEP)
    check("lost signal, back at 166: the rows before label it, 'slower before' does not",
          r_gap["alerts"][G.LORENZ_150].size == 1 and r_gap["alerts"][G.SLOWER].size == 0,
          str({k: r_gap["alerts"][k].tolist() for k in (G.LORENZ_150, G.SLOWER)}))
    check("a real SVT and the hovering one: 'slower before' labels the first; the second hovers round its line",
          r_svt["alerts"][G.SLOWER].size == 1 and r_hov["alerts"][G.SLOWER].size == 0
          and r_svt["episodes"][0]["why"][G.SLOWER] == "-",
          str({k: v[G.SLOWER].tolist() for k, v in (("svt", r_svt["alerts"]), ("hover", r_hov["alerts"]))}))
    check("the exercise leap past 100 (EXER): 'slower before' silent", r_ex["alerts"][G.SLOWER].size == 0)
    # atrial tachycardia with 1:1 and 2:1 conduction of a 250 ms cycle (about 160 bpm), irregular
    rng3 = np.random.default_rng(21)
    rest = list(np.arange(1.0, 150.0, 0.8))
    k = np.tile([1, 2], 130)                                 # 1:1 and 2:1 by turns: irregular, on multiples
    flip = rng3.random(k.size) < 0.1
    k[flip] = 3 - k[flip]
    at = list(rest[-1] + np.cumsum(k * 0.25 + rng3.normal(0, 0.005, k.size)))
    tail = list(at[-1] + np.arange(0.8, 60.0, 0.8))
    bt_at = np.array(rest + at + tail)
    r_at = G.process(record(ecg_beats(bt_at, bt_at[-1] + 1.0, seed=21), [0, rest[-1], at[-1]],
                            ["(N", "(SVTA", "(N"], "cu72"), negative=False, step=G.POS_STEP)
    check("atrial tachycardia with variable conduction: the doctor's report enters it; the alert (row 8) and the"
          " Lorenz rows do not (irregular)", r_at["alerts"][G.DOCTOR].size >= 1 and r_at["alerts"][G.SLOWER].size == 0
          and r_at["alerts"][G.LORENZ].size == 0, str({k_: r_at["alerts"][k_].tolist() for k_ in (G.DOCTOR, G.SLOWER)}))
    r_afi = run(AF_IRREG, [0, 151, 241], ["(N", "(AFIB", "(N"], "cu73", seed=5)
    check("irregular AF: no report entry", r_afi["alerts"][G.DOCTOR].size == 0
          and r_afi["alerts"][G.DOCTOR_PSVT].size == 0)
    check("abrupt start or end: an SVT that starts and ends at once is entered; so is atrial tachycardia with"
          " variable conduction (abrupt start)", r_svt["alerts"][G.DOCTOR_PSVT].size == 1
          and r_at["alerts"][G.DOCTOR_PSVT].size >= 1,
          str({k_: v[G.DOCTOR_PSVT].tolist() for k_, v in (("svt", r_svt["alerts"]), ("at", r_at["alerts"]))}))
    # climbs to 170 over 90 s, holds, then drops at once: the start is gradual, the end abrupt
    r_end = run([REST, (75, 170, 90, 12, 0), (170, 170, 90, 12, 0), (75, 75, 60, 12, 0)], [0], ["(N"], "cu74", seed=6)
    w_end = int(np.argmin(np.abs(r_end["alerts"][G.DOCTOR_PSVT][0] - np.array(list(r_end["detail_rows"][G.DOCTOR_PSVT]))))) \
        if r_end["alerts"][G.DOCTOR_PSVT].size else -1
    d_end = list(r_end["detail_rows"][G.DOCTOR_PSVT].values())[w_end] if w_end >= 0 else {}
    check("abrupt end alone is enough: a gradual start that ends at once is entered",
          r_end["alerts"][G.DOCTOR_PSVT].size >= 1 and d_end.get("start_drop", 1.0) < 0.09
          and d_end.get("end_rise", 0.0) >= 0.09, str(d_end))
    check("sinus climbing slowly to 170 (RAMP): the report enters it, the alert does not (the report's cost: no"
          " onset check)", r_ramp["alerts"][G.DOCTOR].size >= 1 and r_ramp["alerts"][G.SLOWER].size == 0,
          str(r_ramp["alerts"][G.DOCTOR].tolist()))
    check("the same sinus, gradual up and down: the row 'abrupt start or end' does not enter it",
          r_ramp["alerts"][G.DOCTOR_PSVT].size == 0, str(r_ramp["alerts"][G.DOCTOR_PSVT].tolist()))
    r_irr = run(IRREG, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu98", seed=12)
    e3 = r_irr["episodes"][0]
    check("an irregular one at 170: silent, and 'regular' is why (its onset window, half calm sinus, does not"
          " count as regular: regularity is read on the fast run's own beats)",
          r_irr["alerts"][G.NO_AF].size == 0 and r_irr["alerts"][G.LORENZ].size == 0
          and "regular" in e3["why"][G.NO_AF], f"{e3['why']} {e3['pass']}")
    print("what the patient and the doctor get")
    stride, win = G.C.WINDOW_STRIDE_SEC["rhythm"], G.C.WINDOW_SEC["rhythm"]
    pa = np.zeros(6000)
    pa[4:12] = 1.0                                          # AF, then quiet: the episode closes
    pa[100:110] = 1.0                                       # AF again within 24 h: no second alert
    pa[5900:5910] = 1.0                                     # more than 24 h after the first: alerted
    fa = G.af_alert_times(pa)
    check("AF/AFL alert: at the 4th of 6 windows, at most one per 24 h",
          fa.size == 2 and fa[0] == 7 * stride + win and fa[1] == 5903 * stride + win, str(fa))
    check("AF/AFL alert times shift with the analysed span", G.af_alert_times(pa, offset_s=100.0)[0] == fa[0] - 100.0)
    p2 = np.zeros(40)
    p2[9:16] = 1.0
    r_afe = run(SVT, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu87", seed=3, af=G.af_block(p2),
                af_fire=G.af_alert_times(p2))
    t_af = 12 * stride + win
    check("an SVT RhythmNet calls AF: its AF/AFL alert counts for the episode, in seconds from its start",
          r_afe["episodes"][0]["af_alert"] == t_af - 151, str(r_afe["episodes"][0]["af_alert"]))
    check("RhythmNet not run: the AF/AFL alert is unknown (NaN), not 'none'", np.isnan(r_svt["episodes"][0]["af_alert"]))
    FAST190 = [REST, (190, 190, 90, 12, 0), (75, 75, 60, 12, 0)]
    r_190 = run(FAST190, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu86", seed=15)
    e190 = r_190["episodes"][0]
    # the rate engine's monitor-style averaging drops the longest 3/16 of a run's
    # intervals, so the calm beats before the onset fall out and "held 30 s" is
    # met about 20 s in: the app as it is, not changed here
    check("an SVT at 190 held 90 s: the rate notification starts inside it, with the report's 150 line",
          e190["rate_alert"] is not None and 10 <= e190["rate_alert"] <= 45 and e190["fast_report"], str(e190["rate_alert"]))
    check("an SVT at 130: no rate notification, no 150 line",
          r_slow["episodes"][0]["rate_alert"] is None and not r_slow["episodes"][0]["fast_report"])
    # the fast rhythm starts at 151, the annotation only at 200: the notification opened before it
    r_late = run(FAST190, [0, 200, 241], ["(N", "(SVTA", "(N"], "cu85", seed=15)
    check("a notification already open before the episode is not counted as new",
          r_late["episodes"][0]["rate_alert"] is None, str(r_late["episodes"][0]["rate_alert"]))
    # irregular at 160: no alert, no report entry (not regular, not on multiples), but the 150 line
    r_i160 = run([REST, (160, 160, 90, 12, 0.10), (75, 75, 60, 12, 0)], [0, 151, 241], ["(N", "(SVTA", "(N"], "cu84",
                 seed=12)
    sel = [r_190, r_at, r_i160, r_slow]
    scs = {n: G.score(sel, n) for n in (G.SLOWER, G.DOCTOR, G.DOCTOR_PSVT)}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        gets = G.patient_gets(sel, scs)
    a_, f_ = gets["all"], gets["fast"]
    check("gets: each episode once, at the first that applies (alert, report entry, 150 line, the rest)",
          (a_["n"], a_["alert"], a_["entry"], a_["fast line"], a_["rest"]) == (4, 1, 1, 1, 1)
          and (f_["n"], f_["rest"]) == (3, 0) and a_["rate"] == 1 and a_["AF unknown"] == 4,
          f"{a_} {f_}")
    a2 = gets[G.DOCTOR_PSVT]["all"]
    check("gets with the row 'abrupt start or end': the same here (each entry here started abruptly)",
          (a2["alert"], a2["entry"], a2["fast line"], a2["rest"]) == (1, 1, 1, 1), str(a2))
    check("gets: the table and the warning are printed", "WHAT THE PATIENT" in buf.getvalue()
          and "WARNING  RhythmNet did not run on 4" in buf.getvalue())
    r_run = run(WITH_RUN, [0, 151, 156, 157.5, 243], ["(N", "(SVTA", "(VT", "(SVTA", "(N"], "cu94", seed=4)
    sr = G.score([r_run], "label: no VT check")
    check("4 wide beats inside a fast rhythm: no label; without the VT check, an unsafe label",
          G.score([r_run], G.PRIMARY)["unsafe"] == [] and len(sr["unsafe"]) >= 1, f"{sr['unsafe']}")
    r_vt = run(VT_NEW, [0, 151, 241], ["(N", "(VT", "(N"], "cu95", seed=5)
    check("VT with a new shape: no label, no report episode", r_vt["alerts"][G.PRIMARY].size == 0 and not r_vt["report"])
    with contextlib.redirect_stdout(io.StringIO()):
        rep = G.report([r_svt, r_run])
    rep["hours"]["nsrdb"] = 24.0
    code, lines = G.verdict(rep, G.PRIMARY)
    check("verdict: a row with the AF check is not run without RhythmNet on every set",
          code == 1 and "RhythmNet" in lines[0], lines[0])
    check("the final rules: the alert is 'onset 150, slower before', the report 'abrupt start or end'; the alert"
          " needs no RhythmNet", G.FINAL_ALERT == G.SLOWER and G.FINAL_REPORT == G.DOCTOR_PSVT
          and not dict(G.CONFIGS)[G.FINAL_ALERT]["af"] and not dict(G.CONFIGS)[G.FINAL_REPORT]["af"])

    print("SHDB-AF")
    import wfdb
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "shdbaf"
        d.mkdir()
        x, dur, bt = ecg(SVT, seed=6, fs=200)
        wfdb.wrsamp("001", fs=200, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                    p_signal=np.stack([x, 0.5 * x], 1), fmt=["16", "16"], write_dir=str(d))
        aux = ["(N" if t < 151 else "(AT" if t < 241 else "(N" for t in bt]
        wfdb.wrann("001", "atr", np.round(bt * 200).astype(int), ["N"] * bt.size, aux_note=aux, write_dir=str(d))
        wfdb.wrsamp("002", fs=200, units=["mV"], sig_name=["ECG1"], p_signal=x[:, None], fmt=["16"], write_dir=str(d))
        recs = G.shdb_records(d)
        check("only annotated records with a signal are read", [p.stem for p in recs] == ["001"])
        rec = G.read_shdb(recs[0])
        from munji_ai.data.windows import rhythm_intervals
        iv = [(s / FS, e / FS, lab) for s, e, lab in rhythm_intervals(rec)]
        check("SHDB-AF record on the model path at 250 Hz; AT read as SVT, three intervals",
              rec.fs == 250 and [lab for _, _, lab in iv] == ["NSR", "SVT", "NSR"] and abs(iv[1][0] - 151) < 1.5,
              str(iv))

        x3, _, bt3 = ecg([REST], seed=7, fs=200)
        wfdb.wrsamp("003", fs=200, units=["mV"], sig_name=["ECG1"], p_signal=x3[:, None], fmt=["16"], write_dir=str(d))
        wfdb.wrann("003", "atr", np.round(bt3 * 200).astype(int), ["N"] * bt3.size, aux_note=["(N"] * bt3.size,
                   write_dir=str(d))
        files = {"RECORDS": b"001\n002\n003\n", "002.hea": b"h2", "002.dat": b"d2"}
        for n in ("001", "003"):
            for e in ("hea", "atr", "dat"):
                files[f"{n}.{e}"] = (d / f"{n}.{e}").read_bytes()

        class Resp(io.BytesIO):
            def __init__(self, body, size):
                super().__init__(body)
                self.headers = {"Content-Length": str(size)}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        calls = []

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            method = "GET" if isinstance(req, str) else req.get_method()
            name = url.rsplit("/", 1)[1]
            calls.append((method, name))
            if name not in files:
                import urllib.error
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return Resp(b"" if method == "HEAD" else files[name], len(files[name]))
        import urllib.request
        real = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            dl = Path(td) / "dl"
            with contextlib.redirect_stdout(io.StringIO()):
                failed = G.download_shdb(dl, workers=2, base="https://x/")
            got = sorted(p.name for p in dl.iterdir())
            calls.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                G.download_shdb(dl, workers=2, base="https://x/")
            again = sum(1 for m, n in calls if m == "GET" and n != "RECORDS")
            dl2 = Path(td) / "dl2"
            with contextlib.redirect_stdout(io.StringIO()):
                G.download_shdb(dl2, workers=2, base="https://x/", only_svt=False)
            got2 = sorted(p.name for p in dl2.glob("*.dat"))
        finally:
            urllib.request.urlopen = real
        check("download: .hea and .atr for all; .dat only for the record with SVT marks; a 404 is not a failure",
              got == ["001.atr", "001.dat", "001.hea", "002.hea", "003.atr", "003.hea"] and failed == [],
              f"{got} {failed}")
        check("a second run fetches nothing already whole", again == 0, str(again))
        check("--all-shdbaf: every annotated record's signal", got2 == ["001.dat", "003.dat"], str(got2))
        del files["RECORDS"]
        urllib.request.urlopen = fake_urlopen
        try:
            dl3 = Path(td) / "dl3"
            with contextlib.redirect_stdout(io.StringIO()):
                failed3 = G.download_shdb(dl3, workers=4, base="https://x/")
            got3 = sorted(p.name for p in dl3.iterdir())
        finally:
            urllib.request.urlopen = real
        check("no RECORDS file: every number is tried, the missing ones skipped",
              got3 == got and failed3 == [], f"{got3} {failed3}")

    print("end to end")
    from munji_ai.data.loader import _patient_id
    from munji_ai.data.registry import get
    from munji_ai.data.splits import assign

    def pick(ds, names, split=None):
        def ok_(n):
            s = assign(_patient_id(get(ds), n))
            return s == split if split else s != "test"
        return next(n for n in names if ok_(n))

    def write(raw, ds, name, parts, starts, labels, fs, seed, beats_too=False):
        x, dur, bt = ecg(parts, seed, fs)
        d = raw / ds
        d.mkdir(exist_ok=True)
        wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None], fmt=["16"], write_dir=str(d))
        if beats_too:
            wfdb.wrann(name, "atr", np.round(bt * fs).astype(int), ["N"] * bt.size, write_dir=str(d))
        else:
            wfdb.wrann(name, "atr", np.array([int(s * fs) for s in starts]), ["+"] * len(starts),
                       aux_note=labels, write_dir=str(d))
    import torch
    from munji_ai.eval import af_alarm_rate
    from munji_ai.models.rhythm_net import build as build_rhythm
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        write(raw, "cudb", pick("cudb", [f"cu{k:02d}" for k in range(50, 99)]), VT_NEW, [0, 151, 241],
              ["(N", "(VT", "(N"], 250, 7)
        write(raw, "svdb", pick("svdb", [str(k) for k in range(800, 900)]), SVT, [0, 151, 241],
              ["(N", "(SVTA", "(N"], 128, 8)
        write(raw, "nsrdb", pick("nsrdb", [str(k) for k in range(90000, 90400)], "val"), RAMP, [0], ["(N"], 128, 9,
              beats_too=True)
        write(raw, "nsrdb", pick("nsrdb", [str(k) for k in range(90000, 90400)], "train"), RAMP, [0], ["(N"], 128, 10,
              beats_too=True)
        ck = raw / "rhythm_net.pt"
        torch.save({"state_dict": build_rhythm().state_dict(), "classes": ["NSR", "AF", "OTHER"]}, ck)
        real = af_alarm_rate.window_scores
        af_alarm_rate.window_scores = lambda model, signal, device, batch=256: np.zeros(
            max(0, (len(signal) - 7500) // 3750 + 1), np.float32)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = G.main(["--raw", str(raw), "--sets", "cudb", "svdb", "nsrdb", "--workers", "1",
                             "--rhythm-ckpt", str(ck), "--rhythm-cv", str(raw / "none")])
        finally:
            af_alarm_rate.window_scores = real
        text = out.getvalue()
        line = next((ln for ln in text.splitlines() if ln.split()[:1] == ["1"] and G.PRIMARY in ln), "")
        check("NSRDB's training patient is skipped (RhythmNet saw it), its validation patient read",
              "nsrdb: 1 records skipped" in text, text[:300])
        check("through main: the SVT labelled, nothing unsafe, no false label", "1/1" in line
              and line.split()[-1] == "0.00" and " 0 " in line, line)
        check("the EC57 report is printed", "ANSI/AAMI EC57" in text and "PSVT (ended)" in text)
        check("the 150 bpm table, the false-label lists, the cut-off table and the Lorenz verdict are printed",
              "Reached 150 bpm in some 30 s: 1" in text and "False labels, 'label: no AF check'" in text
              and "SVT150" in text and f"False labels, '{G.LORENZ}'" in text and "AF over" in text
              and text.count("CANDIDATE  ->  PASS") == 4 and "Lorenz: PSVT" in text and "DOCTOR REPORT row" in text
              and f"Unsafe labels, '{G.FINAL_ALERT}'" in text,
              text[-900:])
        check("verdict PASS on these records: the alert and the report", rc == 0 and "ALERT  ->  PASS" in text
              and "REPORT  ->  PASS" in text and f"the alert: '{G.FINAL_ALERT}'" in text, f"rc {rc}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
