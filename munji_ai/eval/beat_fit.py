"""Does the beat model generalise, or did it learn its training patients?

    python -m munji_ai.eval.beat_fit
    python -m munji_ai.eval.beat_fit --ckpt checkpoints/v3/beat_segmenter.pt --train-sample 10000

Three checks on one checkpoint:

1. Training against validation. The same scores on windows the model trained
   on and on validation windows, whose patients it never saw. A gap that is
   large, or large for one source, means it learnt those patients more than
   the beats.
2. A confusion matrix for each. Rows are the expert's beats (N, PAC, PVC) and
   "no beat" for model beats where the expert marked none; columns are the
   model's label and "missed". It is built with the training script's
   matching (one to one, 150 ms), so the sensitivities and precisions read off
   it are the ones training reported.
3. Spread over validation patients. A 95% interval for each score from
   resampling patients (validation is a sample of people, and a score is only
   as precise as the people behind it), the spread of per-patient scores, and
   the patients the model does worst on.

The training windows are a fixed random sample of the training cache (seed 0),
the same in every run. Beat windows overlap by half, so a beat can be counted
twice; that holds for both columns alike, and the patient resampling keeps a
patient's windows together.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.windows import load_cache
from ..train.train_beat import (BEAT_IGNORE_ID, EVENT_CLASSES, MATCH_TOLERANCE_MS,
                                mask_to_events)
from . import compare_beat as CB

ROWS = list(EVENT_CLASSES) + ["no beat"]
COLS = list(EVENT_CLASSES) + ["missed"]
K = len(EVENT_CLASSES)
_RI = {c: i for i, c in enumerate(EVENT_CLASSES)}


def window_matrix(true_mask, pred_mask, fs: int = C.TARGET_FS) -> np.ndarray:
    """Confusion counts for one window, with beat_scores' greedy matching."""
    tol = MATCH_TOLERANCE_MS * fs / 1000.0
    te, pe = mask_to_events(true_mask), mask_to_events(pred_mask)
    pe = [(p, c) for p, c in pe if int(true_mask[p]) != BEAT_IGNORE_ID]
    used = [False] * len(pe)
    m = np.zeros((K + 1, K + 1), np.int64)
    for pos, cls in te:
        best, best_d = -1, tol + 1
        for k, (ppos, _) in enumerate(pe):
            if used[k]:
                continue
            d = abs(ppos - pos)
            if d <= tol and d < best_d:
                best, best_d = k, d
        if best < 0:
            m[_RI[cls], K] += 1
            continue
        used[best] = True
        m[_RI[cls], _RI[pe[best][1]]] += 1
    for k, (_, cls) in enumerate(pe):
        if not used[k]:
            m[K, _RI[cls]] += 1
    return m


def matrices(Y, P) -> np.ndarray:
    return np.stack([window_matrix(t, p) for t, p in zip(Y, P)]) if len(Y) else \
        np.zeros((0, K + 1, K + 1), np.int64)


def scores(m: np.ndarray) -> dict:
    """Per-class sensitivity, precision, F1 and macro F1, as beat_scores defines them."""
    out = {}
    for i, c in enumerate(EVENT_CLASSES):
        tp = m[i, i]
        fn = m[i, :].sum() - tp
        fp = m[:, i].sum() - tp
        sens = tp / (tp + fn) if tp + fn else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        f1 = (2 * sens * prec / (sens + prec)
              if sens and prec and not np.isnan(sens + prec) else 0.0)
        out[c] = {"sensitivity": sens, "precision": prec, "f1": f1, "support": int(tp + fn)}
    tp, fn, fp = m[:K, :K].sum(), m[:K, K].sum(), m[K, :K].sum()
    s = tp / (tp + fn) if tp + fn else float("nan")
    p = tp / (tp + fp) if tp + fp else float("nan")
    out["QRS (any beat)"] = {"sensitivity": s, "precision": p,
                             "f1": 2 * s * p / (s + p) if s and p else 0.0, "support": int(tp + fn)}
    f1s = [out[c]["f1"] for c in EVENT_CLASSES if out[c]["support"] > 0]
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out


def by_patient(M: np.ndarray, patients) -> tuple[list, np.ndarray]:
    names = sorted(set(map(str, patients)))
    idx = {p: i for i, p in enumerate(names)}
    out = np.zeros((len(names), K + 1, K + 1), np.int64)
    for m, p in zip(M, patients):
        out[idx[str(p)]] += m
    return names, out


def bootstrap(per_pat: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """95% intervals from resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    keys = [("macro F1", None)] + [(f"{c} {s}", (c, s)) for c in EVENT_CLASSES
                                   for s in ("sensitivity", "precision")]
    draws = {k: [] for k, _ in keys}
    npat = len(per_pat)
    for _ in range(n):
        w = np.bincount(rng.integers(0, npat, npat), minlength=npat)
        sc = scores(np.tensordot(w, per_pat, axes=1))
        for k, sel in keys:
            draws[k].append(sc["macro_f1"] if sel is None else sc[sel[0]][sel[1]])
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                  if len(v) else (float("nan"), float("nan")))
    return out


def _f(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def print_matrix(title: str, m: np.ndarray) -> None:
    print(f"\n{title}: rows are the expert's beats, columns the model's label (share of each row)")
    corner = "expert / model"
    print(f"{corner:<16}" + "".join(f"{c:>10}" for c in COLS) + f"{'beats':>12}")
    for i, r in enumerate(ROWS[:K]):
        tot = m[i].sum()
        print(f"{r:<16}" + "".join(f"{m[i, j] / max(tot, 1):>10.1%}" for j in range(K + 1))
              + f"{tot:>12,}")
    beats = m[:K].sum()
    print(f"{'no beat':<16}" + "".join(f"{m[K, j]:>10,}" for j in range(K)) + f"{'':>10}"
          + f"   model beats where the expert marked none ({m[K].sum() / max(beats, 1) * 1000:.1f} per 1,000 beats)")


def report(tr: dict | None, va: dict) -> None:
    sv = scores(va["M"].sum(0))
    st = scores(tr["M"].sum(0)) if tr else None

    print("\n1. TRAINING AGAINST VALIDATION (per-beat, the training script's matching)")
    head = f"{'':<22}{'train':>9}{'val':>9}{'gap':>9}" if st else f"{'':<22}{'val':>9}"
    print(head)
    for c in list(EVENT_CLASSES) + ["QRS (any beat)"]:
        for s in ("sensitivity", "precision", "f1"):
            v = sv[c][s]
            row = f"{c + ' ' + s[:4]:<22}"
            row += (f"{_f(st[c][s]):>9}{_f(v):>9}{_f(st[c][s] - v):>9}" if st else f"{_f(v):>9}")
            print(row)
    print(f"{'macro F1':<22}" + (f"{st['macro_f1']:>9.3f}{sv['macro_f1']:>9.3f}"
                                 f"{st['macro_f1'] - sv['macro_f1']:>9.3f}" if st else f"{sv['macro_f1']:>9.3f}"))

    if st:
        print("\nmacro F1 by source")
        print(f"{'source':<14}{'train':>9}{'val':>9}{'gap':>9}{'train pts':>11}{'val pts':>9}")
        for src in sorted(set(va["ds"].tolist()) | set(tr["ds"].tolist())):
            a, b = tr["ds"] == src, va["ds"] == src
            ft = scores(tr["M"][a].sum(0))["macro_f1"] if a.any() else float("nan")
            fv = scores(va["M"][b].sum(0))["macro_f1"] if b.any() else float("nan")
            print(f"{src:<14}{_f(ft):>9}{_f(fv):>9}{_f(ft - fv):>9}"
                  f"{len(set(tr['pat'][a].tolist())):>11}{len(set(va['pat'][b].tolist())):>9}")

    print("\n2. CONFUSION MATRICES")
    if st:
        print_matrix("train", tr["M"].sum(0))
    print_matrix("val", va["M"].sum(0))

    names, per = by_patient(va["M"], va["pat"])
    print(f"\n3. VALIDATION PATIENTS ({len(names)})")
    ci = bootstrap(per)
    print(f"{'95% interval, resampling patients':<36}{'value':>9}{'low':>9}{'high':>9}")
    print(f"{'macro F1':<36}{sv['macro_f1']:>9.3f}{ci['macro F1'][0]:>9.3f}{ci['macro F1'][1]:>9.3f}")
    for c in EVENT_CLASSES:
        for s in ("sensitivity", "precision"):
            lo, hi = ci[f"{c} {s}"]
            print(f"{c + ' ' + s:<36}{_f(sv[c][s]):>9}{_f(lo):>9}{_f(hi):>9}")

    src_of = {}
    for p, s in zip(va["pat"], va["ds"]):
        src_of[str(p)] = str(s)
    rows = []
    for p, m in zip(names, per):
        sc = scores(m)
        beats = int(m[:K].sum())
        pvc, pac = sc["PVC"], sc["PAC"]
        false_pvc = (m[:, _RI["PVC"]].sum() - m[_RI["PVC"], _RI["PVC"]]) / max(beats, 1)
        rows.append((p, src_of[p], beats,
                     pvc["sensitivity"] if pvc["support"] >= 20 else float("nan"),
                     pac["sensitivity"] if pac["support"] >= 20 else float("nan"),
                     float(false_pvc), int(pvc["support"]), int(pac["support"])))

    def spread(vals, label):
        v = np.array([x for x in vals if not np.isnan(x)])
        if len(v):
            print(f"{label:<36}{len(v):>9}{np.median(v):>9.3f}{np.percentile(v, 10):>9.3f}{v.min():>9.3f}")

    print(f"\n{'per-patient spread':<36}{'patients':>9}{'median':>9}{'p10':>9}{'worst':>9}")
    spread([r[3] for r in rows], "PVC sens (patients with 20+ PVCs)")
    spread([r[4] for r in rows], "PAC sens (patients with 20+ PACs)")
    v = np.array([r[5] for r in rows])
    print(f"{'false PVC per beat':<36}{len(v):>9}{np.median(v):>9.3%}{np.percentile(v, 90):>9.3%}"
          f"{v.max():>9.3%}   (p90 and worst are the high end)")

    print("\nhardest validation patients (by PVC or PAC sensitivity, then false PVC)")
    print(f"{'patient':<22}{'source':<12}{'beats':>8}{'PVCs':>7}{'PVC sens':>10}{'PACs':>7}"
          f"{'PAC sens':>10}{'false PVC':>11}")
    key = lambda r: min(x for x in (r[3], r[4], 1.0) if not np.isnan(x))
    for r in sorted(rows, key=lambda r: (key(r), -r[5]))[:8]:
        print(f"{r[0]:<22}{r[1]:<12}{r[2]:>8,}{r[6]:>7,}{_f(r[3]):>10}{r[7]:>7,}{_f(r[4]):>10}{r[5]:>11.2%}")


def _predict(model, d: dict, sel=None) -> dict:
    sel = np.arange(len(d["y"])) if sel is None else sel
    X, Y = d["X"][sel], np.asarray(d["y"])[sel]
    P = model.masks(X)
    return {"M": matrices(Y, P), "ds": np.asarray(d["dataset"])[sel].astype(str),
            "pat": np.asarray(d["patient"])[sel].astype(str)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/v3/beat_segmenter.pt"))
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--train-sample", type=int, default=10000,
                    help="training windows to score (0 = skip the training side)")
    a = ap.parse_args(argv)

    if not a.ckpt.exists():
        print(f"{a.ckpt} not found")
        return 1
    model = CB.Model(a.ckpt.parent.name or "model", a.ckpt)
    print(f"model  {a.ckpt}  epoch {model.epoch}  val beat F1 at training "
          f"{None if model.val_f1 is None else round(model.val_f1, 3)}")

    tr = None
    if a.train_sample:
        dt = load_cache("beat", "train", a.cache)
        n = len(dt["y"])
        sel = np.sort(np.random.default_rng(0).choice(n, min(a.train_sample, n), replace=False))
        print(f"train  {len(sel):,} of {n:,} windows (fixed sample)", flush=True)
        tr = _predict(model, dt, sel)
        del dt
    dv = load_cache("beat", "val", a.cache)
    print(f"val    {len(dv['y']):,} windows", flush=True)
    va = _predict(model, dv)
    report(tr, va)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
