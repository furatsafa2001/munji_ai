"""Mine hard negatives for the shockable detector from real ambulatory days.

    python -m munji_ai.data.hard_negatives --models checkpoints/shock_cv
    python -m munji_ai.data.hard_negatives --models checkpoints/shock_cv \
        --segments 2 --max-patients 20                       # quick trial

Why
---
Over full 24-hour recordings the detector raised about 96 false VF alarms a
day. In cross-validation it looked fine because each healthy subject gave it
at most 100 shuffled windows, so it rarely met what a real day contains:
walking, sleeping, eating, turning over, a loose electrode. A patient-template
veto did not help, which says the patient's own beats are not visible in those
windows; the model has to learn that such stretches are not fibrillation.

Ambulatory monitoring work trains on real-life recordings that keep their
noise, and hard-negative mining is the standard way to spend that data well:
run the current model over normal days and keep the windows it gets most
wrong, plus a random sample so ordinary windows stay represented.

Source
------
Icentia11k: the chest-patch, ambulatory, elderly-skewing corpus closest to the
MUNJI device. It contains no VF, so every window is a true negative (any window
annotated shockable is dropped all the same). It is not used in any shockable
evaluation, so the false-alarm measurement on nsrdb stays a fair, external test.
Patients the project split assigns to test are skipped.

Output
------
An .npz shaped like a cache split (X, y, dataset, patient, record, offset)
plus each window's mining score. cross_validate --extra-negatives adds it to
every training fold and never to a held-out fold.
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

DATASET = "icentia11k"
OUT_NAME = "shockable_extra_neg.npz"


def records_by_patient(raw: Path, segments: int, max_patients: int | None):
    """{patient: [record, ...]} with the first `segments` segments each, test split out."""
    ds = get(DATASET)
    groups: dict[str, list[str]] = defaultdict(list)
    skipped_test = 0
    for name in list_records(DATASET, raw):
        groups[_patient_id(ds, name)].append(name)
    out = {}
    for pid in sorted(groups):
        if assign(pid) == "test":
            skipped_test += 1
            continue
        out[pid] = sorted(groups[pid])[:segments]
        if max_patients and len(out) >= max_patients:
            break
    return out, skipped_test


def mine(models_dir: Path, raw: Path, out_path: Path, segments: int, hard_thr: float,
         per_patient_hard: int, per_patient_random: int, fold: int,
         max_patients: int | None, device, seed: int = 0) -> Path:
    from ..eval.false_alarm_rate import load_fold_models, truth_flags, window_scores

    models = load_fold_models(models_dir, device)
    if fold not in models:
        raise SystemExit(f"fold {fold} not in {models_dir}")
    model = models[fold]
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    rng = np.random.default_rng(seed)

    patients, skipped = records_by_patient(raw, segments, max_patients)
    print(f"source      {DATASET}: {len(patients)} patients, {segments} segments each "
          f"({skipped} test-split patients skipped)")
    print(f"model       fold {fold} from {models_dir}")
    print(f"keep        up to {per_patient_hard} windows scoring >= {hard_thr} and "
          f"{per_patient_random} random others per patient\n")

    X, P, R, O, S = [], [], [], [], []
    n_windows = n_hard = 0
    hours = 0.0
    t0 = time.time()
    for j, (pid, recs) in enumerate(patients.items(), 1):
        cand = []                                   # (score, record, offset, signal)
        for name in recs:
            try:
                rec = load_record(DATASET, name, raw)
            except Exception as e:                  # a damaged segment: skip it
                print(f"  skip {name}: {e}")
                continue
            s = window_scores(model, rec.signal, device)
            keep = ~truth_flags(rec, len(s))
            hours += len(rec.signal) / TARGET_FS / 3600
            n_windows += int(keep.sum())
            n_hard += int((keep & (s >= hard_thr)).sum())
            for i in np.flatnonzero(keep):
                cand.append((float(s[i]), name, i * stride, rec.signal))
        if not cand:
            continue
        scores = np.array([c[0] for c in cand])
        hard = np.flatnonzero(scores >= hard_thr)
        hard = hard[np.argsort(-scores[hard])][:per_patient_hard]
        rest = np.setdiff1d(np.arange(len(cand)), hard)
        easy = rng.choice(rest, size=min(per_patient_random, len(rest)), replace=False)
        for k in np.concatenate([hard, easy]).astype(int):
            sc, name, off, sig = cand[k]
            w = np.asarray(sig[off:off + win], dtype=np.float64)
            q25, med, q75 = np.percentile(w, [25, 50, 75])
            X.append(((w - med) / ((q75 - q25) / 1.349 + 1e-8)).astype(C.CACHE_DTYPE))
            P.append(pid)
            R.append(name)
            O.append(off)
            S.append(sc)
        if j % 10 == 0 or j == len(patients):
            print(f"  {j:>4}/{len(patients)} patients   {hours:7.1f} h   "
                  f"{len(X):>7,} windows kept   {time.time() - t0:6.0f} s", flush=True)

    if not X:
        raise SystemExit("nothing mined")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=np.stack(X), y=np.array(["not_shockable"] * len(X)),
             dataset=np.array([DATASET] * len(X)), patient=np.array(P),
             record=np.array(R), offset=np.array(O, dtype=np.int64),
             score=np.array(S, dtype=np.float32), hard_thr=np.float32(hard_thr))

    S = np.array(S)
    print(f"\nscored      {hours:.0f} h, {n_windows:,} windows")
    print(f"above {hard_thr:<5} {n_hard:,} windows ({n_hard / max(n_windows, 1):.2%}) "
          f"- the raw false-positive rate on a chest patch")
    print(f"kept        {len(X):,} windows: {(S >= hard_thr).sum():,} hard, "
          f"{(S < hard_thr).sum():,} random")
    print(f"saved       {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True,
                    help="folder with shockable_fold*.pt (from cross_validate --save-models)")
    ap.add_argument("--fold", type=int, default=0, help="which fold model mines")
    ap.add_argument("--segments", type=int, default=2,
                    help="segments per patient (about 70 minutes each)")
    ap.add_argument("--hard", type=float, default=0.3,
                    help="a window scoring at or above this is a hard negative")
    ap.add_argument("--per-patient-hard", type=int, default=150)
    ap.add_argument("--per-patient-random", type=int, default=50)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=Path(CACHE_DIR) / OUT_NAME)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    mine(a.models, a.raw, a.out, a.segments, a.hard, a.per_patient_hard,
         a.per_patient_random, a.fold, a.max_patients, torch.device(a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
