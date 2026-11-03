"""Mine hard negatives for the shockable detector from real ambulatory days.

    python -m munji_ai.data.hard_negatives --models checkpoints/shock_cv_v2
    python -m munji_ai.data.hard_negatives --models checkpoints/shock_cv_v2 \
        --datasets icentia11k --segments 2 --max-patients 20     # quick trial

Why
---
On full 24-hour recordings the detector raised dozens of false VF alarms a
day, because cross-validation only ever showed it a capped, shuffled sample of
each healthy subject, never a real day of walking, sleeping and loose
electrodes. Hard-negative mining is the standard way to spend long recordings
well: run the current model over them, keep the windows it gets most wrong,
plus a random sample so ordinary windows stay represented. Repeating it with
the improved model (a second round) targets the mistakes that model makes.

Sources
-------
    icentia11k   chest patch, ambulatory, elderly-skewing: the closest corpus
                 to the MUNJI device. Every downloaded segment can be scanned.
    ltafdb       24-hour Holter from patients with atrial fibrillation, so
                 irregular and ambulatory at once: the negatives most easily
                 mistaken for fibrillation.

Neither is used in any shockable evaluation, so the false-alarm measurement on
nsrdb stays an external test. Patients the project split assigns to test are
skipped. Any window annotated shockable (VF, or VT at 150 bpm or more) is
dropped, so every kept window is a true negative.

Long runs
---------
Scanning everything takes many hours on a laptop. Each patient's result is
saved as soon as it is done (--work-dir), and a rerun skips finished patients,
so the job can be stopped and resumed. The final .npz is assembled from those
parts at the end, or with --merge-only.

Output
------
An .npz shaped like a cache split (X, y, dataset, patient, record, offset)
plus each window's mining score, for cross_validate / train_shockable
--extra-negatives.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR, TARGET_FS
from .loader import _patient_id, list_records, load_record
from .registry import get
from .splits import assign

SOURCES = ("icentia11k", "ltafdb")
OUT_NAME = "shockable_extra_neg_v2.npz"
WORK_NAME = "hard_negative_parts"


def records_by_patient(dataset: str, raw: Path, segments: int, max_patients: int | None):
    """{patient: [record, ...]}, the first `segments` records each (0 = all), test out."""
    ds = get(dataset)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in list_records(dataset, raw):
        groups[_patient_id(ds, name)].append(name)
    out, skipped = {}, 0
    for pid in sorted(groups):
        if assign(pid) == "test":
            skipped += 1
            continue
        recs = sorted(groups[pid])
        out[pid] = recs[:segments] if segments else recs
        if max_patients and len(out) >= max_patients:
            break
    return out, skipped


def _normalize(w: np.ndarray) -> np.ndarray:
    q25, med, q75 = np.percentile(w, [25, 50, 75])
    return (w - med) / ((q75 - q25) / 1.349 + 1e-8)


def mine_patient(model, dataset, pid, recs, raw, device, hard_thr, k_hard, k_rand, rng):
    """Top-k hard and a random sample for one patient, streaming over records."""
    from ..eval.false_alarm_rate import truth_flags, window_scores

    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    hard: list = []            # (score, window, record, offset)
    rand: list = []            # (key, score, window, record, offset)
    stats = {"hours": 0.0, "windows": 0, "hard": 0}
    for name in recs:
        try:
            rec = load_record(dataset, name, raw)
        except Exception as e:
            print(f"    skip {name}: {e}")
            continue
        s = window_scores(model, rec.signal, device)
        keep = ~truth_flags(rec, len(s))
        stats["hours"] += len(rec.signal) / TARGET_FS / 3600
        stats["windows"] += int(keep.sum())
        h = np.flatnonzero(keep & (s >= hard_thr))
        stats["hard"] += int(h.size)
        for i in h[np.argsort(-s[h])][:k_hard]:
            hard.append((float(s[i]), _normalize(rec.signal[i * stride:i * stride + win]),
                         name, int(i * stride)))
        hard = sorted(hard, key=lambda r: -r[0])[:k_hard]
        e = np.flatnonzero(keep & (s < hard_thr))
        if e.size:
            keys = rng.random(e.size)
            for j in np.argsort(keys)[:k_rand]:
                i = e[j]
                rand.append((float(keys[j]), float(s[i]),
                             _normalize(rec.signal[i * stride:i * stride + win]),
                             name, int(i * stride)))
            rand = sorted(rand, key=lambda r: r[0])[:k_rand]
    rows = [(sc, w, r, o) for sc, w, r, o in hard] + [(sc, w, r, o) for _, sc, w, r, o in rand]
    return rows, stats


def save_part(path: Path, dataset: str, pid: str, rows, stats) -> None:
    if rows:
        X = np.stack([w for _, w, _, _ in rows]).astype(C.CACHE_DTYPE)
    else:
        X = np.zeros((0, C.window_samples("shockable")), dtype=C.CACHE_DTYPE)
    np.savez(path, X=X, score=np.array([s for s, _, _, _ in rows], dtype=np.float32),
             record=np.array([r for _, _, r, _ in rows]),
             offset=np.array([o for _, _, _, o in rows], dtype=np.int64),
             dataset=dataset, patient=pid, hours=stats["hours"],
             windows=stats["windows"], hard=stats["hard"])


def merge(work: Path, out: Path, hard_thr: float) -> Path:
    parts = sorted(work.glob("*.npz"))
    if not parts:
        raise SystemExit(f"no parts in {work}")
    X, S, R, O, D, P = [], [], [], [], [], []
    hours = windows = hard = 0.0
    for p in parts:
        with np.load(p, allow_pickle=False) as d:
            n = len(d["X"])
            hours += float(d["hours"]); windows += int(d["windows"]); hard += int(d["hard"])
            if n == 0:
                continue
            X.append(d["X"]); S.append(d["score"]); R.append(d["record"]); O.append(d["offset"])
            D.append(np.full(n, str(d["dataset"]))); P.append(np.full(n, str(d["patient"])))
    X, S = np.concatenate(X), np.concatenate(S)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=np.array(["not_shockable"] * len(X)), dataset=np.concatenate(D),
             patient=np.concatenate(P), record=np.concatenate(R),
             offset=np.concatenate(O), score=S, hard_thr=np.float32(hard_thr))
    D = np.concatenate(D)
    print(f"\nscored      {hours:,.0f} h, {int(windows):,} windows, {len(parts)} patients")
    print(f"above {hard_thr:<5} {int(hard):,} windows ({hard / max(windows, 1):.2%})"
          f" - the raw false-positive rate on these recordings")
    print(f"kept        {len(X):,} windows: {(S >= hard_thr).sum():,} hard, "
          f"{(S < hard_thr).sum():,} random  "
          + "  ".join(f"{k} {int((D == k).sum()):,}" for k in sorted(set(D))))
    print(f"saved       {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=False,
                    help="folder with shockable_fold*.pt (cross_validate --save-models)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--segments", type=int, default=0,
                    help="records per patient; 0 = every downloaded record")
    ap.add_argument("--hard", type=float, default=0.3)
    ap.add_argument("--per-patient-hard", type=int, default=400)
    ap.add_argument("--per-patient-random", type=int, default=100)
    ap.add_argument("--max-patients", type=int, default=None, help="per source")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--work-dir", type=Path, default=Path(CACHE_DIR) / WORK_NAME)
    ap.add_argument("--out", type=Path, default=Path(CACHE_DIR) / OUT_NAME)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    if a.merge_only:
        merge(a.work_dir, a.out, a.hard)
        return 0
    if a.models is None:
        ap.error("--models is required unless --merge-only")

    from ..eval.false_alarm_rate import load_fold_models
    device = torch.device(a.device)
    models = load_fold_models(a.models, device)
    if a.fold not in models:
        raise SystemExit(f"fold {a.fold} not in {a.models}")
    model = models[a.fold]
    a.work_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for ds in a.datasets:
        groups, skipped = records_by_patient(ds, a.raw, a.segments, a.max_patients)
        n_rec = sum(len(v) for v in groups.values())
        print(f"source      {ds}: {len(groups)} patients, {n_rec:,} records "
              f"({skipped} test-split patients skipped)")
        plan += [(ds, pid, recs) for pid, recs in groups.items()]
    done = {p.stem for p in a.work_dir.glob("*.npz")}
    todo = [(ds, pid, r) for ds, pid, r in plan if pid.replace(":", "_") not in done]
    print(f"model       fold {a.fold} from {a.models}")
    print(f"keep        up to {a.per_patient_hard} windows scoring >= {a.hard} and "
          f"{a.per_patient_random} random others per patient")
    print(f"resume      {len(plan) - len(todo)} of {len(plan)} patients already done "
          f"(parts in {a.work_dir})\n")

    rng = np.random.default_rng(0)
    t0, hours = time.time(), 0.0
    for j, (ds, pid, recs) in enumerate(todo, 1):
        rows, st = mine_patient(model, ds, pid, recs, a.raw, device, a.hard,
                                a.per_patient_hard, a.per_patient_random, rng)
        save_part(a.work_dir / f"{pid.replace(':', '_')}.npz", ds, pid, rows, st)
        hours += st["hours"]
        el = time.time() - t0
        eta = el / j * (len(todo) - j)
        print(f"  {j:>4}/{len(todo)}  {pid:<20}{st['hours']:6.1f} h  "
              f"{st['hard'] / max(st['windows'], 1):6.2%} hard   "
              f"{el / 3600:5.1f} h elapsed, about {eta / 3600:4.1f} h left", flush=True)

    merge(a.work_dir, a.out, a.hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
