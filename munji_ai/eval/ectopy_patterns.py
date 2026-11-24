"""Couplets, bigeminy and trigeminy: can the beat model's labels show them?

    python -m munji_ai.eval.ectopy_patterns
    python -m munji_ai.eval.ectopy_patterns --ckpt v3=checkpoints/v3/beat_segmenter.pt

This measures the beat model, not the decision layer's rules: the patterns are
read off the expert's labels and off the model's labels at the same beats, with
the definitions the decision layer uses (config):

    couplet     exactly 2 consecutive PVCs
    bigeminy    a PVC after every normal beat, for ECTOPY_PATTERN_MIN_CYCLES cycles or more
    trigeminy   a PVC after every 2 normal beats, for as many cycles

Two views, on validation patients only (the beat val cache):

    beats      every expert PVC once (read from the window whose centre is
               nearest, never at a window edge), by the pattern it sits in:
               alone, couplet, bigeminy, trigeminy, or a run of 3 or more.
    episodes   every expert pattern lying whole inside a window, once: caught
               when the model's labels show the same pattern there (a couplet
               at the same two beats; bigeminy or trigeminy sharing at least
               ECTOPY_PATTERN_MIN_CYCLES PVCs). And the model's own patterns
               that match no expert one: those are false entries in the
               doctor's report.

Validation windows are chosen around ectopic beats, so the counts are not a day
of monitoring; false entries per day on healthy people come from
eval_beats_model --part nsrdb.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.windows import load_cache, mask_beat_classes
from . import compare_beat as CB

ID = C.BEAT_CLASS_ID
V, N = ID["PVC"], ID["N"]
KINDS = ("couplet", "bigeminy", "trigeminy")
CONTEXTS = ("alone", "couplet", "bigeminy", "trigeminy", "run 3+")


def episodes(seq, min_cycles: int | None = None) -> list:
    """[(kind, [PVC indices], first index, last index)] in a sequence of class ids."""
    cyc = C.ECTOPY_PATTERN_MIN_CYCLES if min_cycles is None else min_cycles
    seq = [int(c) for c in seq]
    n = len(seq)
    k = CB.stretch(seq)
    out = []
    for i in range(n):
        if seq[i] == V and k[i] == 2 and (i == 0 or seq[i - 1] != V):
            out.append(("couplet", [i, i + 1], i, i + 1))
    for kind, step in (("bigeminy", 2), ("trigeminy", 3)):
        used = set()
        for s in range(n):
            if seq[s] != V or k[s] != 1 or s in used:
                continue
            lead = range(s - step + 1, s)
            if s - step + 1 < 0 or any(seq[j] != N for j in lead):
                continue
            chain = [s]
            while True:
                nx = chain[-1] + step
                if (nx < n and seq[nx] == V and k[nx] == 1
                        and all(seq[j] == N for j in range(chain[-1] + 1, nx))):
                    chain.append(nx)
                else:
                    break
            used.update(chain)
            if len(chain) >= cyc:
                out.append((kind, chain, s - step + 1, chain[-1]))
    return out


def collect(Y, P: dict, ds, pat, rec, off) -> tuple:
    """(beats, expert episodes, model episodes), each counted once."""
    beats, expert, model = {}, {}, {n: {} for n in P}
    owner = {"expert": {}, **{n: {} for n in P}}          # (kind, PVC position) -> episode key

    def add(store, who, kind, vpos, row):
        if any((kind, v) in owner[who] for v in vpos):
            return
        key = (kind,) + vpos[0]
        store[key] = row
        for v in vpos:
            owner[who][(kind, v)] = key

    for w in range(len(Y)):
        regs = CB.regions(Y[w])
        if not regs:
            continue
        seq = [c for _, _, c in regs]
        k_of = CB.stretch(seq)
        labs = {n: [CB.label_at(P[n][w], s, e) for s, e, _ in regs] for n in P}
        where = lambda i: (str(ds[w]), str(rec[w]), int(off[w]) + regs[i][0])
        last = len(regs) - 1

        ex = episodes(seq)
        ctx = {}
        for kind, vs, _, _ in ex:
            for v in vs:
                ctx.setdefault(v, kind)
        for i, ((s, e, c), k) in enumerate(zip(regs, k_of)):
            if c != V or s == 0 or e >= CB.WIN:
                continue
            key = where(i)
            dist = abs((s + e) / 2.0 - CB.WIN / 2.0)
            if key not in beats or dist < beats[key]["dist"]:
                cx = "run 3+" if k >= 3 else ("couplet" if k == 2 else ctx.get(i, "alone"))
                beats[key] = {"dist": dist, "ctx": cx, "ds": str(ds[w]), "pat": str(pat[w]),
                              "pred": {n: labs[n][i] for n in P}}

        mod = {n: episodes(labs[n]) for n in P}
        for kind, vs, a, b in ex:
            if a == 0 or b == last:                       # could be cut by the window edge
                continue
            got = {}
            for n in P:
                need = 2 if kind == "couplet" else C.ECTOPY_PATTERN_MIN_CYCLES
                got[n] = any(mk == kind and (mv == vs if kind == "couplet" else len(set(mv) & set(vs)) >= need)
                             for mk, mv, _, _ in mod[n])
            add(expert, "expert", kind, [where(v) for v in vs],
                {"kind": kind, "ds": str(ds[w]), "pat": str(pat[w]), "caught": got})
        for n in P:
            for kind, vs, a, b in mod[n]:
                if a == 0 or b == last:
                    continue
                real = any(ek == kind and set(ev) & set(vs) for ek, ev, _, _ in ex)
                add(model[n], n, kind, [where(v) for v in vs],
                    {"kind": kind, "ds": str(ds[w]), "pat": str(pat[w]), "real": real})
    return list(beats.values()), list(expert.values()), {n: list(v.values()) for n, v in model.items()}


def _pct(v) -> str:
    return "-" if v is None or np.isnan(v) else f"{v:.0%}"


def _share(rows, f) -> float:
    return float(np.mean([f(r) for r in rows])) if rows else float("nan")


def report(beats, expert, model, names) -> None:
    last = names[-1]
    print("\nPVC beats by the pattern they sit in, each beat once (sens = share called PVC)")
    print(f"{'pattern':<12}{'beats':>8}{'patients':>10}" + "".join(f"{n + ' sens':>11}" for n in names)
          + f"{last + ' ->N':>10}{last + ' ->PAC':>11}{last + ' missed':>12}")
    for cx in CONTEXTS + ("all",):
        sel = [r for r in beats if cx == "all" or r["ctx"] == cx]
        print(f"{cx:<12}{len(sel):>8,}{len({r['pat'] for r in sel}):>10}"
              + "".join(f"{_pct(_share(sel, lambda r, n=n: r['pred'][n] == V)):>11}" for n in names)
              + f"{_pct(_share(sel, lambda r: r['pred'][last] == N)):>10}"
              + f"{_pct(_share(sel, lambda r: r['pred'][last] == ID['PAC'])):>11}"
              + f"{_pct(_share(sel, lambda r: r['pred'][last] == 0)):>12}")

    print("\nPatterns lying whole inside a window, each once")
    print(f"{'pattern':<12}{'expert':>8}{'patients':>10}" + "".join(f"{n + ' caught':>12}" for n in names)
          + "".join(f"{n + ' false':>11}{n + ' correct':>12}" for n in names))
    for kind in KINDS:
        ex = [r for r in expert if r["kind"] == kind]
        cells = "".join(f"{_pct(_share(ex, lambda r, n=n: r['caught'][n])):>12}" for n in names)
        for n in names:
            mk = [r for r in model[n] if r["kind"] == kind]
            false = sum(not r["real"] for r in mk)
            cells += f"{false:>11,}{_pct(_share(mk, lambda r: r['real'])):>12}"
        print(f"{kind:<12}{len(ex):>8,}{len({r['pat'] for r in ex}):>10}" + cells)
    print("caught: the expert's pattern shows in the model's labels. false: a pattern in the model's\n"
          "labels the expert does not have there. correct: the share of the model's patterns that are real.")

    print("\nBy source")
    print(f"{'source':<12}{'PVCs':>7}" + "".join(f"{k[:5] + ' n':>9}{last + ' caught':>12}" for k in KINDS))
    for src in sorted({r["ds"] for r in beats} | {r["ds"] for r in expert}):
        row = f"{src:<12}{sum(r['ds'] == src for r in beats):>7,}"
        for kind in KINDS:
            ex = [r for r in expert if r["ds"] == src and r["kind"] == kind]
            row += f"{len(ex):>9,}{_pct(_share(ex, lambda r: r['caught'][last])):>12}"
        print(row)

    by_pat = defaultdict(list)
    for r in expert:
        by_pat[r["pat"]].append(r)
    top = sorted(by_pat.items(), key=lambda kv: -len(kv[1]))[:3]
    if expert:
        share = sum(len(v) for _, v in top) / len(expert)
        print(f"\nthe 3 patients with the most patterns hold {share:.0%} of them: "
              + ", ".join(f"{p} ({len(v)})" for p, v in top))


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
    keep = np.flatnonzero([(mask_beat_classes(y) == V).any() for y in Y])
    print(f"validation windows {len(Y):,}, with a PVC {len(keep):,}", flush=True)
    if not len(keep):
        return 0
    X, Y = d["X"][keep], Y[keep]
    ds, pat = d["dataset"][keep].astype(str), d["patient"][keep].astype(str)
    rec, off = d["record"][keep].astype(str), d["offset"][keep]
    P = {}
    for m in models:
        print(f"  predicting with {m.name} ...", flush=True)
        P[m.name] = m.masks(X)
    beats, expert, model = collect(Y, P, ds, pat, rec, off)
    report(beats, expert, model, [m.name for m in models])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
