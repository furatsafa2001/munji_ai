"""Window-extraction checks. Synthetic records, no PhysioNet needed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import windows as W
from munji_ai.data.loader import Record
from test_pipeline import synth_ecg


def make_record(seconds=300, hr=72, ectopic_every=8, af_from=None, seed=0):
    """Synthetic record with beat and rhythm annotations."""
    x, fs, peaks = synth_ecg(seconds=seconds, hr=hr, fs=250, seed=seed)
    syms, keep = [], []
    for i, p in enumerate(peaks):
        if ectopic_every and i % ectopic_every == 0 and i > 0:
            syms.append("V" if i % (ectopic_every * 2) == 0 else "A")
        else:
            syms.append("N")
        keep.append(p)
    rs, rl = [0], ["(N"]
    if af_from is not None:
        rs.append(int(af_from * fs))
        rl.append("(AFIB")
    return Record(
        dataset="synthetic", name="s0", patient="synthetic:s0",
        signal=x, fs=fs,
        beat_samples=np.array(keep, dtype=np.int64),
        beat_symbols=np.array(syms, dtype="<U2"),
        rhythm_starts=np.array(rs, dtype=np.int64),
        rhythm_labels=np.array(rl, dtype=object),
    )


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("\n1. mask painting")
    m = W.paint_mask(2000, np.array([500, 1000]), np.array(["N", "PVC"]))
    n_w = int((m == C.BEAT_CLASS_ID["N"]).sum())
    v_w = int((m == C.BEAT_CLASS_ID["PVC"]).sum())
    check("both classes painted", n_w > 0 and v_w > 0, f"N={n_w} PVC={v_w} samples")
    check("PVC region wider than normal (physiological)", v_w > n_w,
          f"{v_w} > {n_w}")
    check("mask is mostly background", (m == 0).mean() > 0.8,
          f"{(m == 0).mean():.1%} background")
    check("no class id outside range", m.max() < len(C.BEAT_CLASSES))

    print("\n2. overlapping beats resolve to ectopic")
    m2 = W.paint_mask(2000, np.array([1000, 1010]), np.array(["N", "PVC"]))
    check("ectopic survives adjacent normal",
          (m2 == C.BEAT_CLASS_ID["PVC"]).sum() > 0)

    print("\n3. rhythm intervals")
    rec = make_record(seconds=300, af_from=150)
    iv = W.rhythm_intervals(rec)
    check("two episodes found", len(iv) == 2, str([(s, e, l) for s, e, l in iv]))
    check("episodes are contiguous and cover the record",
          iv[0][1] == iv[1][0] and iv[-1][1] == len(rec.signal))

    print("\n4. dominant label purity")
    check("pure NSR window labelled NSR",
          W.dominant_label(iv, 0, 7500, C.RHYTHM_PURITY) == "NSR")
    check("pure AF window labelled AF",
          W.dominant_label(iv, 60000, 67500, C.RHYTHM_PURITY) == "AF")
    boundary = W.dominant_label(iv, 37500 - 3750, 37500 + 3750, C.RHYTHM_PURITY)
    check("50/50 boundary window rejected as ambiguous", boundary is None,
          f"got {boundary!r}")

    print("\n5. beat extraction")
    rng = np.random.default_rng(0)
    X, Y, off = W.extract_beat(rec, rng, cap=150)
    check("windows produced", len(X) > 0, f"{len(X)} windows")
    check("signal shape correct", X[0].shape == (C.window_samples("beat"),),
          str(X[0].shape))
    check("mask shape matches signal", Y[0].shape == X[0].shape)
    stacked = np.concatenate(Y)
    present = {c: int((stacked == C.BEAT_CLASS_ID[c]).sum())
               for c in ("N", "PAC", "PVC")}
    check("all three beat classes present", all(v > 0 for v in present.values()),
          str(present))
    ect = sum(1 for m in Y if np.isin(m, [C.BEAT_CLASS_ID["PAC"],
                                          C.BEAT_CLASS_ID["PVC"]]).any())
    check("ectopic-bearing windows prioritised", ect / len(Y) > 0.7,
          f"{ect}/{len(Y)} = {ect / len(Y):.0%}")

    print("\n6. rhythm extraction")
    Xr, Yr, _ = W.extract_rhythm(rec, rng, cap=80)
    check("windows produced", len(Xr) > 0, f"{len(Xr)} windows")
    check("shape correct", Xr[0].shape == (C.window_samples("rhythm"),))
    check("both AF and non-AF present", {"AF"} <= set(Yr) and len(set(Yr)) > 1,
          str(dict(zip(*np.unique(Yr, return_counts=True)))))

    print("\n7. shockable extraction")
    vf = make_record(seconds=200, af_from=None, seed=3)
    vf.rhythm_starts = np.array([0, 100 * 250], dtype=np.int64)
    vf.rhythm_labels = np.array(["(N", "(VFIB"], dtype=object)
    Xs, Ys, _ = W.extract_shockable(vf, rng, cap=400)
    counts = dict(zip(*np.unique(Ys, return_counts=True)))
    check("both classes present", len(counts) == 2, str(counts))
    check("shape correct", Xs[0].shape == (C.window_samples("shockable"),))

    print("\n8. quality extraction")
    fake = {"em": rng.normal(0, 1, 300_000),
            "bw": np.sin(np.linspace(0, 1200, 300_000))}
    Xq, Yq, _ = W.extract_quality(rec, rng, 60, fake)
    dist = dict(zip(*np.unique(Yq, return_counts=True)))
    check("windows produced", len(Xq) == 60, f"{len(Xq)}")
    check("all three quality classes present",
          set(dist) == set(C.QUALITY_CLASSES), str(dist))

    print("\n9. config hash guards stale caches")
    h1 = W.config_hash()
    old = C.MODEL_BAND
    C.MODEL_BAND = (0.7, 40.0)
    h2 = W.config_hash()
    C.MODEL_BAND = old
    check("hash changes when preprocessing changes", h1 != h2, f"{h1} -> {h2}")
    check("hash restored when config restored", W.config_hash() == h1)

    print("\n10. memory estimate at configured caps")
    for stage in ("quality", "beat", "rhythm", "shockable"):
        n = C.MAX_WINDOWS_PER_STAGE[stage]
        w = C.window_samples(stage)
        gb = n * w * 2 / 1e9
        if stage == "beat":
            gb += n * w * 1 / 1e9
        check(f"{stage:<9} cache under 3 GB", gb < 3.0, f"~{gb:.2f} GB ({n:,} x {w})")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
