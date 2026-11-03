"""Add hard-negative mining from Icentia11k and --extra-negatives to cross_validate.

Why
---
Full-day recordings showed about 96 false VF alarms per 24 h. Longer
confirmation cut that to about 19 at a cost in missed episodes, and a
patient-template veto barely helped while missing true VF. The detector has to
learn what a real day looks like. Hard-negative mining on Icentia11k (chest
patch, ambulatory, elderly) gives it the windows it currently gets most wrong,
and Icentia is not used in any shockable evaluation, so the false-alarm test
on nsrdb remains external.

What changes
------------
    munji_ai/data/hard_negatives.py    NEW: mine windows from Icentia11k
    munji_ai/train/cross_validate.py   --extra-negatives: add them to every
                                       training fold, never to a held-out fold

Without --extra-negatives cross_validate behaves exactly as before. config,
registry, caches and every model are untouched.

How to run
----------
From the repo root:

    python fix_hard_negatives.py
    python tests/test_hard_negatives.py

Revert with:

    git checkout munji_ai/train/cross_validate.py
    and delete munji_ai/data/hard_negatives.py
"""

from pathlib import Path

NEW = Path("munji_ai/data/hard_negatives.py")
CV = Path("munji_ai/train/cross_validate.py")

NEW_SOURCE = r'''"""Mine hard negatives for the shockable detector from real ambulatory days.

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
'''

ARG_OLD = r'''    ap.add_argument("--save-models", action="store_true",
                    help="save each fold's selected model next to the OOF file")'''
ARG_NEW = r'''    ap.add_argument("--save-models", action="store_true",
                    help="save each fold's selected model next to the OOF file")
    ap.add_argument("--extra-negatives", type=Path, default=None,
                    help="shockable only: .npz of extra negative windows added to "
                         "every training fold (data/hard_negatives.py)")'''

PARSE_OLD = r'''    a = ap.parse_args(argv)
'''
PARSE_NEW = r'''    a = ap.parse_args(argv)
    if a.extra_negatives and a.stage != "shockable":
        ap.error("--extra-negatives applies to the shockable stage only")
'''

LOAD_OLD = r'''    if a.augment or a.noisy_eval is not None:
        from ..data.train_augment import BANK_NAME, load_noise_bank, split_bank
        a._bank_train, a._bank_eval = split_bank(
            load_noise_bank(a.cache / BANK_NAME))
'''
LOAD_NEW = r'''    if a.augment or a.noisy_eval is not None:
        from ..data.train_augment import BANK_NAME, load_noise_bank, split_bank
        a._bank_train, a._bank_eval = split_bank(
            load_noise_bank(a.cache / BANK_NAME))
    Xe = ye = None
    if a.extra_negatives:
        with np.load(a.extra_negatives, allow_pickle=False) as e:
            Xe, ye = e["X"], e["y"].astype(str)
            extra_patients = set(e["patient"].astype(str))
        if set(ye) - {"not_shockable"}:
            raise SystemExit("extra negatives must all be not_shockable")
        shared = extra_patients & set(np.asarray(d["patient"], dtype=str))
        if shared:
            raise SystemExit(f"extra negatives share {len(shared)} patients with "
                             f"the evaluation data - that would leak")
'''

HDR_OLD = r'''    print(f"augment     {'on (noise, jitter, cutout)' if a.augment else 'off'}")'''
HDR_NEW = r'''    print(f"augment     {'on (noise, jitter, cutout)' if a.augment else 'off'}")
    if Xe is not None:
        print(f"extra neg   {len(Xe):,} windows from {len(extra_patients)} patients, "
              f"training folds only")'''

CALL_OLD = r'''        model, P = train_one_fold(a.stage, d["X"][tr], y[tr], d["X"][te],
                                  device, a, classes)
'''
CALL_NEW = r'''        Xtr, ytr = d["X"][tr], y[tr]
        if Xe is not None:
            Xtr = np.concatenate([Xtr, Xe.astype(Xtr.dtype)])
            ytr = np.concatenate([ytr, ye])
        model, P = train_one_fold(a.stage, Xtr, ytr, d["X"][te],
                                  device, a, classes)
'''

PLAN = [
    (ARG_OLD, ARG_NEW, "--extra-negatives flag"),
    (PARSE_OLD, PARSE_NEW, "--extra-negatives is shockable only"),
    (LOAD_OLD, LOAD_NEW, "extra negatives loaded, leakage checked"),
    (HDR_OLD, HDR_NEW, "header line"),
    (CALL_OLD, CALL_NEW, "added to training folds only"),
]


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    if not CV.exists():
        print(f"MISSING  {CV}\nrun this from the repo root")
        return 1
    text, eol = read(CV)
    if "--save-models" not in text:
        print("STOP  the false-alarm change is not on this branch.")
        print("      Branch from feat/false-alarm-rate, then run this again.")
        return 1
    if NEW.exists() and "--extra-negatives" in text:
        print("already applied - nothing to do")
        return 0
    print("applying\n")
    ok = True
    for old, new, label in PLAN:
        if new in text:
            print(f"  present  cross_validate  {label}")
        elif text.count(old) == 1:
            text = text.replace(old, new, 1)
            print(f"  fixed    cross_validate  {label}")
        else:
            print(f"  FAILED   cross_validate  {label} - expected code not found")
            ok = False
    if not ok:
        print("\nnothing written - all files are untouched")
        return 1
    write(NEW, NEW_SOURCE, eol)
    print(f"  created  {NEW.as_posix()}")
    write(CV, text, eol)
    print("\napplied. config, registry, caches and models are untouched.\n")
    print("now run:")
    print("  python tests/test_hard_negatives.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
