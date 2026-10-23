"""Fuse RR-interval features into RhythmNet, to attack the OTHER class.

Why
---
After the focal-loss fix, the one weakness that did not move is OTHER: about
730 OTHER windows are still called NSR, and OTHER F1 sits at 0.71. Focal loss
changes which windows the model attends to; it cannot give the model
information it does not have. OTHER is mostly rhythms defined by beat timing
(ectopic beats, bigeminy, pauses, block, abnormal rates), and RR-interval
statistics measure timing directly. Combining such handcrafted features with
a learned representation is what the strongest CinC 2017 entries did, on the
same data that supplies OTHER here.

What changes
------------
    munji_ai/features/rr.py            NEW: 12 RR features per window
    munji_ai/models/rhythm_net.py      RhythmNet(n_rr=...) fuses them after pooling
    munji_ai/train/cross_validate.py   --rr flag; per-class F1 line in the report

n_rr = 0 (the default) builds exactly the original network, with the same
parameter names, so existing checkpoints still load. Without --rr,
cross_validate behaves exactly as before for both stages; the only visible
difference is one extra line printing per-class F1.

The features are computed once from the cached windows and saved next to the
cache as rhythm_trainval_rr.npz, keyed to the exact windows they came from.
No feature looks outside its own window, so there is no cross-fold leakage.
The config hash does not move and no window cache is invalidated.

train_rhythm.py is deliberately not touched. If the comparison favours RR
features, the final single-split training gets the same flag in a follow-up.

How to run
----------
From the repo root:

    python fix_rhythm_rr_features.py
    python tests/test_rr_features.py

Then on Colab (GPU), with the same flags as the gamma 2 run plus --rr:

    python -m munji_ai.train.cross_validate --stage rhythm --gamma 2 --rr
        --epochs 30 --batch 64 --lr 2e-3 --out <dir>/rhythm_g2_rr

Compare against the gamma 2 run without --rr: OTHER F1 (0.711), the
OTHER -> NSR cell (734), AF sensitivity at equal specificity, false alarms.

Revert with:

    git checkout munji_ai/models/rhythm_net.py munji_ai/train/cross_validate.py
    and delete munji_ai/features/rr.py
"""

from pathlib import Path

RR = Path("munji_ai/features/rr.py")
MODEL = Path("munji_ai/models/rhythm_net.py")
CV = Path("munji_ai/train/cross_validate.py")

RR_SOURCE = r'''"""RR-interval features for the rhythm classifier.

    from munji_ai.features.rr import RR_FEATURES, rr_features
    f = rr_features(window)          # one 30-second window -> (12,)

Why
---
RhythmNet reads the raw trace and has to infer beat spacing on its own. That
works well for AF (irregular, no pattern) but leaves the OTHER class weak: in
the pooled confusion matrix about 740 OTHER windows are called NSR. OTHER is
mostly rhythms whose signature is in the *timing* of beats rather than their
shape: ectopic beats and bigeminy (alternating short-long), pauses and
dropped beats (block), and rates outside the normal band.

These are exactly what classical RR statistics measure, and the best entries
of the PhysioNet/CinC 2017 challenge (the source of the OTHER class here)
combined such features with a learned representation. This module supplies
them; RhythmNet concatenates them with its pooled CNN embedding.

The CNN still sees the raw trace, so if the beat detector fails under motion
the network is not blind: `qsqi` and `valid` tell the head how far to trust
the RR numbers in each window.

Features, all computed on one window
------------------------------------
    hr         median heart rate / 100
    cv         SD of RR / mean RR                 overall irregularity
    rmssd_n    RMSSD / mean RR                    beat-to-beat irregularity
    pnn50      share of successive changes > 50 ms
    premature  share of RR < 0.85 x median        early (ectopic) beats
    long       share of RR > 1.5 x median         pauses, dropped beats
    tpr        turning-point ratio                ~0.67 random, ~1 bigeminy
    entropy    Shannon entropy of RR histogram    AF high, NSR low
               (bins fixed at 0.5x to 1.5x the median RR)
    sd_ratio   Poincare SD1 / SD2                 AF near 1, NSR small
    range      log(max RR / min RR)
    qsqi       agreement of two beat detectors    trust in the numbers above
    valid      1 if enough beats were found, else 0 and everything else is 0

RMSSD/mean, TPR and Shannon entropy are the three statistics of the Dash et al.
AF detector; the rest cover the non-AF rhythms that fill OTHER.

No feature uses anything outside its own window, so computing them before the
patient-level split cannot leak information across folds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import TARGET_FS
from .sqi import MIN_DYNAMIC_RANGE, detect_r_pantompkins, q_sqi

RR_FEATURES = ("hr", "cv", "rmssd_n", "pnn50", "premature", "long", "tpr",
               "entropy", "sd_ratio", "range", "qsqi", "valid")
RR_VERSION = 1            # bump when any feature definition changes

MIN_BEATS = 4             # fewer and the statistics are noise
RR_PLAUSIBLE = (0.2, 3.0) # seconds; 20 to 300 bpm. Outside is a detector error
HIST_BINS = 16
CLIP = 5.0                # no feature is allowed to blow up the head


def rr_features(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """All RR features for one window, as float32 in RR_FEATURES order."""
    out = np.zeros(len(RR_FEATURES), dtype=np.float32)
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) < MIN_DYNAMIC_RANGE or np.std(x) < MIN_DYNAMIC_RANGE:
        return out

    peaks = detect_r_pantompkins(x, fs)
    if len(peaks) < MIN_BEATS:
        return out

    rr = np.diff(peaks) / fs
    rr = rr[(rr >= RR_PLAUSIBLE[0]) & (rr <= RR_PLAUSIBLE[1])]
    if len(rr) < MIN_BEATS - 1:
        return out

    med, mean = float(np.median(rr)), float(np.mean(rr))
    d = np.diff(rr)

    f = dict.fromkeys(RR_FEATURES, 0.0)
    f["hr"] = 60.0 / med / 100.0
    f["cv"] = float(np.std(rr)) / mean
    f["rmssd_n"] = float(np.sqrt(np.mean(d ** 2))) / mean if d.size else 0.0
    f["pnn50"] = float(np.mean(np.abs(d) > 0.05)) if d.size else 0.0
    f["premature"] = float(np.mean(rr < 0.85 * med))
    f["long"] = float(np.mean(rr > 1.5 * med))

    if d.size >= 2:
        turns = (d[:-1] * d[1:]) < 0
        f["tpr"] = float(np.mean(turns))

    # Bins span a fixed band around the median (0.5x to 1.5x), not min..max.
    # Scaling the bins to the data would make the tiny jitter of a sinus
    # rhythm look as spread out as fibrillation.
    edges = (0.5 * med, 1.5 * med)
    counts, _ = np.histogram(np.clip(rr, *edges), bins=HIST_BINS, range=edges)
    p = counts[counts > 0] / counts.sum()
    f["entropy"] = float(-(p * np.log(p)).sum() / np.log(HIST_BINS))

    lo, hi = float(rr.min()), float(rr.max())
    f["range"] = float(np.log(hi / lo))

    if d.size >= 2:
        sd1 = float(np.std(d)) / np.sqrt(2.0)
        sd2_sq = 2.0 * float(np.var(rr)) - sd1 ** 2
        if sd2_sq > 1e-12:
            f["sd_ratio"] = sd1 / np.sqrt(sd2_sq)

    f["qsqi"] = float(q_sqi(x, fs))
    f["valid"] = 1.0

    out[:] = [f[k] for k in RR_FEATURES]
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=CLIP, neginf=0.0),
                   0.0, CLIP).astype(np.float32)


def rr_features_batch(X: np.ndarray, fs: int = TARGET_FS,
                      progress: bool = True) -> np.ndarray:
    """rr_features over every row of X."""
    F = np.zeros((len(X), len(RR_FEATURES)), dtype=np.float32)
    step = max(1, len(X) // 10)
    for i, x in enumerate(X):
        F[i] = rr_features(x, fs)
        if progress and (i + 1) % step == 0:
            print(f"  rr features  {i + 1:,}/{len(X):,}", flush=True)
    return F


def _fingerprint(record: np.ndarray, offset: np.ndarray) -> str:
    """Identity of the window set, so a stale feature file is never reused."""
    h = hashlib.sha256()
    h.update("\n".join(map(str, record)).encode())
    h.update(np.asarray(offset, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def cached_rr_features(X: np.ndarray, record: np.ndarray, offset: np.ndarray,
                       path: Path, fs: int = TARGET_FS) -> np.ndarray:
    """Load RR features from `path`, or compute and save them.

    The file is reused only if it was built from exactly these windows (same
    records, same offsets, same order) under the same RR_VERSION. Anything
    else is recomputed, never silently trusted.
    """
    path = Path(path)
    key = _fingerprint(record, offset)
    if path.exists():
        d = np.load(path, allow_pickle=False)
        if (str(d["key"]) == key and int(d["version"]) == RR_VERSION
                and d["F"].shape == (len(X), len(RR_FEATURES))):
            print(f"  rr features  loaded {path.name}")
            return d["F"]
        print(f"  rr features  {path.name} is stale, rebuilding")

    F = rr_features_batch(X, fs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, F=F, key=key, version=RR_VERSION,
                        names=np.array(RR_FEATURES))
    print(f"  rr features  saved {path.name}")
    return F
'''


# ---------------------------------------------------------------- rhythm_net.py
M_INIT_OLD = r'''    def __init__(self, n_classes: int = len(RHYTHM_CLASSES),
                 widths=(16, 32, 64, 96, 128, 160), dropout: float = 0.1):
        super().__init__()
        dilations = (1, 2, 4, 8, 16, 32)'''
M_INIT_NEW = r'''    def __init__(self, n_classes: int = len(RHYTHM_CLASSES),
                 widths=(16, 32, 64, 96, 128, 160), dropout: float = 0.1,
                 n_rr: int = 0):
        super().__init__()
        dilations = (1, 2, 4, 8, 16, 32)'''

M_HEAD_OLD = r'''        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 3, 96),'''
M_HEAD_NEW = r'''        # Optional RR-interval features (munji_ai.features.rr), fused after
        # pooling. BatchNorm puts them on the embedding's scale using training
        # statistics only. n_rr = 0 builds exactly the original network, so
        # existing checkpoints still load.
        self.n_rr = n_rr
        self.rr_norm = nn.BatchNorm1d(n_rr) if n_rr else None
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 3 + n_rr, 96),'''

M_FWD_OLD = r'''    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.blocks(x)
        pooled = torch.cat(
            [h.mean(dim=-1), h.amax(dim=-1), h.std(dim=-1)], dim=1)
        return self.head(pooled)'''
M_FWD_NEW = r'''    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.blocks(x)
        pooled = torch.cat(
            [h.mean(dim=-1), h.amax(dim=-1), h.std(dim=-1)], dim=1)
        if self.n_rr:
            if rr is None:
                raise ValueError(
                    f"this model was built with {self.n_rr} RR features; pass rr")
            pooled = torch.cat([pooled, self.rr_norm(rr)], dim=1)
        return self.head(pooled)'''

M_PROBA_OLD = r'''    def predict_proba(self, x) -> torch.Tensor:'''
M_PROBA_NEW = r'''    def predict_proba(self, x, rr=None) -> torch.Tensor:'''

M_PROBA_RET_OLD = r'''        self.eval()
        return F.softmax(self(x), dim=-1)[:, AF_IDX]'''
M_PROBA_RET_NEW = r'''        self.eval()
        return F.softmax(self(x, rr), dim=-1)[:, AF_IDX]'''


# ------------------------------------------------------------ cross_validate.py
C_DEF_OLD = r'''def train_one_fold(stage, Xtr, ytr, Xva, device, args, classes):
    build, class_weights, Loss, penalty_kw = build_for(stage)
    cfg = STAGES[stage]
    idx = {c: i for i, c in enumerate(classes)}

    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.tensor([idx[s] for s in ytr], dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    model = build().to(device)'''
C_DEF_NEW = r'''def train_one_fold(stage, Xtr, ytr, Xva, device, args, classes,
                   Ftr=None, Fva=None):
    build, class_weights, Loss, penalty_kw = build_for(stage)
    cfg = STAGES[stage]
    idx = {c: i for i, c in enumerate(classes)}
    use_rr = Ftr is not None

    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.tensor([idx[s] for s in ytr], dtype=torch.long)
    tensors = (Xt, torch.from_numpy(Ftr.astype(np.float32)), yt) if use_rr \
        else (Xt, yt)
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    model = (build(n_rr=Ftr.shape[1]) if use_rr else build()).to(device)'''

C_VAL_OLD = r'''    Xv = torch.from_numpy(Xva.astype(np.float32))
    val_loader = DataLoader(TensorDataset(Xv, torch.zeros(len(Xv), dtype=torch.long)),
                            batch_size=256)'''
C_VAL_NEW = r'''    Xv = torch.from_numpy(Xva.astype(np.float32))
    extra = torch.from_numpy(Fva.astype(np.float32)) if use_rr \
        else torch.zeros(len(Xv), dtype=torch.long)
    val_loader = DataLoader(TensorDataset(Xv, extra), batch_size=256)'''

C_LOOP_OLD = r'''        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()'''
C_LOOP_NEW = r'''        for batch in loader:
            batch = [t.to(device) for t in batch]
            yb = batch[-1]
            opt.zero_grad()
            crit(model(*batch[:-1]), yb).backward()'''

C_PRED_OLD = r'''    for xb, _ in loader:
        out.append(torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy())'''
C_PRED_NEW = r'''    for xb, extra in loader:
        inputs = (xb.to(device), extra.to(device)) if getattr(model, "n_rr", 0) \
            else (xb.to(device),)
        out.append(torch.softmax(model(*inputs), dim=-1).cpu().numpy())'''

C_ARG_OLD = r'''    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")'''
C_ARG_NEW = r'''    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")
    ap.add_argument("--rr", action="store_true",
                    help="rhythm only: fuse RR-interval features into the model")'''

C_PARSE_OLD = r'''    a = ap.parse_args(argv)
'''
C_PARSE_NEW = r'''    a = ap.parse_args(argv)
    if a.rr and a.stage != "rhythm":
        ap.error("--rr applies to the rhythm stage only")
'''

C_LOAD_OLD = r'''    d = pooled_cache(a.stage, a.cache)
'''
C_LOAD_NEW = r'''    d = pooled_cache(a.stage, a.cache)
    F = None
    if a.rr:
        from ..features.rr import cached_rr_features
        F = cached_rr_features(d["X"], d["record"], d["offset"],
                               a.cache / f"{a.stage}_trainval_rr.npz")
'''

C_HDR_OLD = r'''    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")'''
C_HDR_NEW = r'''    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")
    print(f"rr features {'on, ' + str(F.shape[1]) if F is not None else 'off'}")'''

C_CALL_OLD = r'''        model, P = train_one_fold(a.stage, d["X"][tr], y[tr], d["X"][te],
                                  device, a, classes)'''
C_CALL_NEW = r'''        model, P = train_one_fold(a.stage, d["X"][tr], y[tr], d["X"][te],
                                  device, a, classes,
                                  *((F[tr], F[te]) if F is not None else ()))'''

C_CM_OLD = r'''    for i, c in enumerate(classes):
        print(f"  {c:<{width}}" + "".join(f"{v:>{width},}" for v in cm[i]))
'''
C_CM_NEW = r'''    for i, c in enumerate(classes):
        print(f"  {c:<{width}}" + "".join(f"{v:>{width},}" for v in cm[i]))
    f1s = []
    for i in range(n):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    print("\n  per-class F1  "
          + "   ".join(f"{c} {v:.3f}" for c, v in zip(classes, f1s)))
'''


PLAN = [
    (MODEL, M_INIT_OLD, M_INIT_NEW, "rhythm_net      n_rr argument"),
    (MODEL, M_HEAD_OLD, M_HEAD_NEW, "rhythm_net      fused head"),
    (MODEL, M_FWD_OLD, M_FWD_NEW, "rhythm_net      forward(x, rr)"),
    (MODEL, M_PROBA_OLD, M_PROBA_NEW, "rhythm_net      predict_proba(x, rr)"),
    (MODEL, M_PROBA_RET_OLD, M_PROBA_RET_NEW, "rhythm_net      predict_proba passes rr"),
    (CV, C_DEF_OLD, C_DEF_NEW, "cross_validate  train_one_fold takes features"),
    (CV, C_VAL_OLD, C_VAL_NEW, "cross_validate  validation loader"),
    (CV, C_LOOP_OLD, C_LOOP_NEW, "cross_validate  training loop"),
    (CV, C_PRED_OLD, C_PRED_NEW, "cross_validate  predict"),
    (CV, C_ARG_OLD, C_ARG_NEW, "cross_validate  --rr flag"),
    (CV, C_PARSE_OLD, C_PARSE_NEW, "cross_validate  --rr is rhythm only"),
    (CV, C_LOAD_OLD, C_LOAD_NEW, "cross_validate  loads features"),
    (CV, C_HDR_OLD, C_HDR_NEW, "cross_validate  header line"),
    (CV, C_CALL_OLD, C_CALL_NEW, "cross_validate  passes features"),
    (CV, C_CM_OLD, C_CM_NEW, "cross_validate  per-class F1"),
]


def read(path: Path):
    """Return (text with LF endings, original line ending)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    missing = [p for p in (MODEL, CV) if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING  {p}")
        print("run this from the repo root")
        return 1

    files = {p: read(p) for p in (MODEL, CV)}
    text = {p: t for p, (t, _) in files.items()}

    if "gamma=args.gamma" not in text[CV]:
        print("STOP  the focal-loss change (PR #1) is not on this branch.")
        print("      Branch from fix/rhythm-focal-loss, or from main after")
        print("      PR #1 is merged, then run this again.")
        return 1

    if RR.exists() and "n_rr" in text[MODEL] and "--rr" in text[CV]:
        print("already applied - nothing to do")
        return 0

    print("applying\n")
    ok = True
    for path, old, new, label in PLAN:
        if new in text[path]:
            print(f"  present  {label}")
        elif text[path].count(old) == 1:
            text[path] = text[path].replace(old, new, 1)
            print(f"  fixed    {label}")
        else:
            print(f"  FAILED   {label} - expected code not found")
            ok = False

    if not ok:
        print("\nnothing written - all files are untouched")
        return 1

    write(RR, RR_SOURCE, files[MODEL][1])
    print(f"  created  {RR.as_posix()}")
    for path, (_, eol) in files.items():
        write(path, text[path], eol)

    print("\napplied. config hash unchanged, no window cache invalidated.\n")
    print("now run:")
    print("  python tests/test_rr_features.py")
    print("then on Colab: the gamma 2 command plus --rr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
