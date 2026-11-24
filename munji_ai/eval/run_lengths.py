"""PVC sensitivity inside ventricular runs, split by run length.

    python -m munji_ai.eval.run_lengths
    python -m munji_ai.eval.run_lengths --ckpt v3=checkpoints/v3/beat_segmenter.pt

compare_beat pools every PVC inside a run of 3 or more into one number. A long
run holds many beats, and beat windows overlap by half, so a few long runs can
outweigh every short one in that number. This splits it:

    beats   every PVC inside a run, counted once: a beat seen in two windows is
            read from the window whose centre is nearest, and a beat cut by a
            window edge is not read. Its run length is the length seen in that
            window, so a run longer than the window is counted as what fits.
    runs    every run lying whole inside a window, counted once, and whether the
            model's labels over it hold 3 consecutive PVCs: what the NSVT rule
            needs to fire.

Validation patients only (the beat val cache), the same windows for every model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.windows import load_cache, longest_pvc_run, mask_beat_classes
from . import compare_beat as CB

BINS = ((3, 4), (5, 9), (10, 29), (30, None))


def bin_name(k: int) -> str | None:
    for lo, hi in BINS:
        if k >= lo and (hi is None or k <= hi):
            return f"{lo}+" if hi is None else f"{lo}-{hi}"
    return None


def collect(Y, P: dict, ds, pat, rec, off):
    """(beats, runs): every in-run PVC once, every whole run once."""
    beats, runs = {}, {}
    for w in range(len(Y)):
        regs = CB.regions(Y[w])
        k_of = CB.stretch([c for _, _, c in regs])
        for (s, e, _), k in zip(regs, k_of):
            if k < CB.RUN or s == 0 or e >= CB.WIN:
                continue
            key = (str(ds[w]), str(rec[w]), int(off[w]) + s)
            dist = abs((s + e) / 2.0 - CB.WIN / 2.0)
            if key not in beats or dist < beats[key]["dist"]:
                beats[key] = {"dist": dist, "k": k, "ds": str(ds[w]), "pat": str(pat[w]),
                              "pred": {n: CB.label_at(P[n][w], s, e) for n in P}}
        i = 0
        while i < len(regs):
            k = k_of[i]
            if k < CB.RUN:
                i += 1
                continue
            j = i + k
            if i > 0 and j < len(regs):              # a beat before and after it: whole
                key = (str(ds[w]), str(rec[w]), int(off[w]) + regs[i][0])
                if key not in runs:
                    runs[key] = {"k": k, "ds": str(ds[w]), "pat": str(pat[w]),
                                 "pred": {n: [CB.label_at(P[n][w], s, e) for s, e, _ in regs[i:j]]
                                          for n in P}}
            i = j
    return list(beats.values()), list(runs.values())


def _sens(rows, n) -> float:
    return float(np.mean([r["pred"][n] == CB.ID["PVC"] for r in rows])) if rows else float("nan")


def _as(rows, n, cls) -> float:
    return float(np.mean([r["pred"][n] == cls for r in rows])) if rows else float("nan")


def _caught(rows, n) -> float:
    return (float(np.mean([longest_pvc_run(r["pred"][n]) >= CB.RUN for r in rows]))
            if rows else float("nan"))


def _pct(v: float) -> str:
    return "-" if np.isnan(v) else f"{v:.0%}"


def report(beats, runs, names) -> None:
    last = names[-1]
    groups = [bin_name(lo) for lo, _ in BINS] + ["all"]

    print("\nPVC beats inside runs, each beat once (sens = share the model calls PVC)")
    print(f"{'run length':<12}{'beats':>8}{'patients':>10}"
          + "".join(f"{n + ' sens':>11}" for n in names)
          + f"{last + ' ->N':>10}{last + ' ->PAC':>11}{last + ' missed':>12}")
    for g in groups:
        sel = [r for r in beats if g == "all" or bin_name(r["k"]) == g]
        print(f"{g:<12}{len(sel):>8,}{len({r['pat'] for r in sel}):>10}"
              + "".join(f"{_pct(_sens(sel, n)):>11}" for n in names)
              + f"{_pct(_as(sel, last, C.BEAT_CLASS_ID['N'])):>10}"
              + f"{_pct(_as(sel, last, C.BEAT_CLASS_ID['PAC'])):>11}"
              + f"{_pct(_as(sel, last, 0)):>12}")

    print("\nRuns lying whole inside a window (caught = 3 consecutive PVCs in the model's labels)")
    print(f"{'run length':<12}{'runs':>8}{'patients':>10}" + "".join(f"{n + ' caught':>13}" for n in names))
    for g in groups:
        sel = [r for r in runs if g == "all" or bin_name(r["k"]) == g]
        print(f"{g:<12}{len(sel):>8,}{len({r['pat'] for r in sel}):>10}"
              + "".join(f"{_pct(_caught(sel, n)):>13}" for n in names))

    print("\nBy source (all run lengths)")
    print(f"{'source':<12}{'beats':>8}{'runs':>8}{'patients':>10}"
          + "".join(f"{n + ' sens':>11}{n + ' caught':>13}" for n in names))
    for src in sorted({r["ds"] for r in beats} | {r["ds"] for r in runs}):
        b = [r for r in beats if r["ds"] == src]
        u = [r for r in runs if r["ds"] == src]
        print(f"{src:<12}{len(b):>8,}{len(u):>8,}{len({r['pat'] for r in b + u}):>10}"
              + "".join(f"{_pct(_sens(b, n)):>11}{_pct(_caught(u, n)):>13}" for n in names))

    by_pat = defaultdict(list)
    for r in beats:
        by_pat[r["pat"]].append(r)
    top = sorted(by_pat.items(), key=lambda kv: -len(kv[1]))[:8]
    print("\nPatients holding the most in-run PVC beats")
    print(f"{'patient':<22}{'beats':>8}{'longest':>9}" + "".join(f"{n + ' sens':>11}" for n in names))
    for p, rows in top:
        print(f"{p:<22}{len(rows):>8,}{max(r['k'] for r in rows):>9}"
              + "".join(f"{_pct(_sens(rows, n)):>11}" for n in names))
    share = sum(len(r) for _, r in top[:3]) / max(len(beats), 1)
    print(f"the top 3 patients hold {share:.0%} of the in-run PVC beats")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=None,
                    help="name=path, repeatable (default: v2 and v3)")
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    a = ap.parse_args(argv)

    models = []
    for item in a.ckpt or CB.DEFAULT_CKPTS:
        name, _, path = item.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"{name}: {p} not found, left out")
            continue
        models.append(CB.Model(name, p))
        print(f"{name:<4} {p}  epoch {models[-1].epoch}")
    if not models:
        print("no checkpoints")
        return 1

    d = load_cache("beat", "val", a.cache)
    Y = np.asarray(d["y"])
    keep = np.flatnonzero([longest_pvc_run(mask_beat_classes(y)) >= CB.RUN for y in Y])
    print(f"validation windows {len(Y):,}, with a run of {CB.RUN}+ PVCs {len(keep):,}")
    if not len(keep):
        return 0
    X = d["X"][keep]
    Y = Y[keep]
    ds, pat = d["dataset"][keep].astype(str), d["patient"][keep].astype(str)
    rec, off = d["record"][keep].astype(str), d["offset"][keep]
    P = {}
    for m in models:
        print(f"  predicting with {m.name} ...", flush=True)
        P[m.name] = m.masks(X)
    beats, runs = collect(Y, P, ds, pat, rec, off)
    report(beats, runs, [m.name for m in models])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
