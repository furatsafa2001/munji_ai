"""Add training-time augmentation and a held-out noisy evaluation.

Why
---
Both RhythmNet and ShockableNet train on clean recordings and are deployed on
a moving patient. In wearable defibrillators artifact is the main cause of
false alarms, and the ShockableNet target the client document states (fewer
than 1% false calls on a normal heart) is far from met (3.7%). augment.py has
loaded NSTDB for the quality gate since month 1; neither model trained on it.

One mechanism serves both models, which is why it comes first:

    noise    NSTDB electrode motion and muscle artifact, 3 to 20 dB SNR
    jitter   small white noise, 20 to 30 dB SNR
    cutout   up to 5% of the window set to baseline

Applied to training batches only, never to validation. Details and the
reasons for what was left out are in munji_ai/data/train_augment.py.

The CV numbers so far were measured on clean windows, which is not where the
benefit should show. So this also adds --noisy-eval: every held-out fold is
scored a second time with NSTDB noise mixed in, using only the last 30% of
each noise record (training uses the first 70%), so no model is scored on
artifact it trained on.

What changes
------------
    munji_ai/data/train_augment.py     NEW: noise bank, augmentation, noisy eval
    munji_ai/train/cross_validate.py   --augment, --noisy-eval, per-class F1 line

Without the new flags cross_validate behaves exactly as before (one extra
line with per-class F1). augment.py, the quality gate, config.py and every
cache are untouched.

How to run
----------
From the repo root, once:

    python fix_train_augmentation.py
    python tests/test_train_augment.py
    python -m munji_ai.data.train_augment --build

The last command needs NSTDB in data/raw (python -m munji_ai.data.download
nstdb) and writes data/cache/noise_bank.npz, about 4 MB.

Revert with:

    git checkout munji_ai/train/cross_validate.py
    and delete munji_ai/data/train_augment.py
"""

from pathlib import Path

MOD = Path("munji_ai/data/train_augment.py")
CV = Path("munji_ai/train/cross_validate.py")

MOD_SOURCE = r'''"""Training-time augmentation for the rhythm and shockable models.

    python -m munji_ai.data.train_augment --build     # once, needs NSTDB in data/raw

Why
---
Both models train on clean hospital and Holter recordings and are deployed on
an elderly patient walking, sleeping and praying with a chest patch. In
wearable defibrillators, artifact is the main cause of false alarms, so a model
that has never seen motion noise during training is being asked to generalise
to its most common input. augment.py already loads NSTDB for the quality gate;
nothing used it for these two models.

Three label-preserving augmentations, applied on the fly to training batches
only:

    noise    real electrode-motion (em) and muscle (ma) artifact from NSTDB,
             at 3 to 20 dB SNR. 3 dB is the quality gate's usable boundary, so
             the models practise on everything the gate can let through.
    jitter   small white noise, 20 to 30 dB SNR.
    cutout   a short stretch (up to 5% of the window) set to baseline,
             like a brief contact loss.

After mixing, each window is renormalised with the same median / IQR rule as
preprocess.normalize, because in deployment the noise is present before
normalisation, not after.

What is deliberately left out
-----------------------------
    time warping     changes the heart rate, and rate defines labels here
                     (VT is shockable only at 150 bpm or more)
    baseline wander  the 0.5 Hz high-pass already removes it; scaling the
                     residual up to a target SNR would invent a wander that
                     the device never sees
    synthetic beats  averaging rhythms makes traces that do not exist

Held-out noise for evaluation
-----------------------------
Each NSTDB record is split in time: the first 70% feeds training, the last 30%
is reserved for the noisy evaluation in cross_validate (--noisy-eval). A model
can therefore never be scored on an artifact waveform it trained on.

This module does not change augment.py or anything the quality gate uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..config import CACHE_DIR, RAW_DIR, TARGET_FS

NOISE_KINDS = ("em", "ma")
BANK_NAME = "noise_bank.npz"
TRAIN_SPLIT = 0.7

# Training-only knobs. Kept here rather than in config.py because nothing in
# deployment reads them and no cache depends on them.
P_NOISE = 0.5
NOISE_SNR_DB = (3.0, 20.0)
P_JITTER = 0.3
JITTER_SNR_DB = (20.0, 30.0)
P_CUTOUT = 0.2
CUTOUT_MAX_FRAC = 0.05


# ------------------------------------------------------------------ noise bank

def build_noise_bank(path: Path | None = None, root: Path = RAW_DIR) -> Path:
    """NSTDB em and ma, resampled to TARGET_FS and run through the model path."""
    from . import preprocess as pp
    from .augment import load_noise

    path = Path(path or Path(CACHE_DIR) / BANK_NAME)
    raw = load_noise(root, NOISE_KINDS)
    bank = {k: pp.model_path(v, TARGET_FS).astype(np.float32) for k, v in raw.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fs=TARGET_FS, **bank)
    for k, v in bank.items():
        print(f"  {k}  {len(v) / TARGET_FS / 60:.1f} min")
    print(f"saved {path}")
    return path


def load_noise_bank(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it once with\n"
            f"    python -m munji_ai.data.train_augment --build\n"
            f"(needs NSTDB: python -m munji_ai.data.download nstdb)")
    # Context manager so the file handle is released at once: Windows will
    # not delete or overwrite a file that is still open.
    with np.load(path, allow_pickle=False) as d:
        if int(d["fs"]) != TARGET_FS:
            raise ValueError(f"{path.name} was built at {int(d['fs'])} Hz, not {TARGET_FS}")
        return {k: np.array(d[k]) for k in NOISE_KINDS if k in d.files}


def split_bank(bank: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """(training part, evaluation part): first 70% and last 30% of each record."""
    train, held = {}, {}
    for k, v in bank.items():
        cut = int(len(v) * TRAIN_SPLIT)
        train[k], held[k] = v[:cut], v[cut:]
    return train, held


# ---------------------------------------------------------------- torch pieces

def renormalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise preprocess.normalize: (x - median) / (IQR / 1.349)."""
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device,
                                       dtype=x.dtype), dim=1)
    scale = ((q[2] - q[0]) / 1.349).unsqueeze(1)
    return (x - q[1].unsqueeze(1)) / (scale + eps)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2, dim=1) + 1e-12)


class NoiseMixer:
    """Adds NSTDB segments at a requested SNR, from a random record and offset."""

    def __init__(self, bank: dict[str, np.ndarray], device, seed: int = 0):
        self.device = torch.device(device)
        self.kinds = sorted(bank)
        self.bank = [torch.from_numpy(np.asarray(bank[k], dtype=np.float32)).to(self.device)
                     for k in self.kinds]
        self.gen = torch.Generator().manual_seed(seed)

    def rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def segments(self, n: int, length: int) -> torch.Tensor:
        kind = torch.randint(0, len(self.bank), (n,), generator=self.gen)
        out = torch.empty(n, length, device=self.device)
        ar = torch.arange(length, device=self.device)
        for i, b in enumerate(self.bank):
            m = kind == i
            k = int(m.sum())
            if k == 0:
                continue
            if b.numel() <= length:
                raise ValueError(f"noise record '{self.kinds[i]}' is shorter than a window")
            off = torch.randint(0, b.numel() - length, (k,), generator=self.gen).to(self.device)
            out[m.to(self.device)] = b[off.unsqueeze(1) + ar.unsqueeze(0)]
        return out - out.mean(dim=1, keepdim=True)

    def mix(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        seg = self.segments(x.shape[0], x.shape[1])
        target = _rms(x) / torch.pow(10.0, snr_db / 20.0)
        return x + seg * (target / _rms(seg)).unsqueeze(1)


class TrainAugment:
    """Callable on a (batch, length) tensor of normalised windows."""

    def __init__(self, bank_train: dict[str, np.ndarray], device, seed: int = 0):
        self.mixer = NoiseMixer(bank_train, device, seed)

    def _uniform(self, n: int, lo_hi: tuple[float, float]) -> torch.Tensor:
        lo, hi = lo_hi
        return lo + (hi - lo) * self.mixer.rand(n)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                     # (batch, 1, length)
            return self(x.squeeze(1)).unsqueeze(1)
        b, length = x.shape
        x = x.clone()

        m = self.mixer.rand(b) < P_NOISE
        if m.any():
            x[m] = self.mixer.mix(x[m], self._uniform(int(m.sum()), NOISE_SNR_DB))

        j = self.mixer.rand(b) < P_JITTER
        if j.any():
            w = torch.randn(int(j.sum()), length, generator=self.mixer.gen).to(x.device)
            snr = self._uniform(int(j.sum()), JITTER_SNR_DB)
            target = _rms(x[j]) / torch.pow(10.0, snr / 20.0)
            x[j] = x[j] + w * (target / _rms(w)).unsqueeze(1)

        c = self.mixer.rand(b) < P_CUTOUT
        if c.any():
            span = (self.mixer.rand(b) * CUTOUT_MAX_FRAC * length).long()
            start = (self.mixer.rand(b) * (length - span)).long()
            pos = torch.arange(length, device=x.device).unsqueeze(0)
            mask = (pos >= start.unsqueeze(1)) & (pos < (start + span).unsqueeze(1))
            x = x.masked_fill(mask & c.unsqueeze(1), 0.0)

        return renormalize(x)


@torch.no_grad()
def add_noise_fixed(X: np.ndarray, bank_eval: dict[str, np.ndarray], snr_db: float,
                    seed: int, chunk: int = 256) -> np.ndarray:
    """Deterministic noisy copy of X for evaluation, from held-out noise only."""
    mixer = NoiseMixer(bank_eval, "cpu", seed)
    out = []
    for i in range(0, len(X), chunk):
        x = torch.from_numpy(np.asarray(X[i:i + chunk], dtype=np.float32))
        y = mixer.mix(x, torch.full((len(x),), float(snr_db)))
        out.append(renormalize(y).numpy())
    return np.concatenate(out) if out else np.empty_like(X, dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build data/cache/noise_bank.npz")
    ap.add_argument("--root", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.build:
        ap.print_help()
        return 0
    build_noise_bank(a.out, a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


C_MODEL_OLD = r'''    model = build().to(device)
'''
C_MODEL_NEW = r'''    model = build().to(device)
    aug = None
    if getattr(args, "augment", False):
        from ..data.train_augment import TrainAugment
        aug = TrainAugment(args._bank_train, device, seed=args.seed)
'''

C_LOOP_OLD = r'''        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()'''
C_LOOP_NEW = r'''        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()
            crit(model(xb), yb).backward()'''

C_REPORT_OLD = r'''

def main(argv=None) -> int:
'''
C_REPORT_NEW = r'''

def _report_noisy(y, pred, score, classes, positive, snr_db) -> None:
    """Pooled out-of-fold metrics on the noisy copy of every held-out window."""
    s = binary_scores(y, pred, positive)
    print(f"\n  under NSTDB noise at {snr_db:g} dB (held-out noise, same windows)")
    print(f"    sensitivity   {s['sensitivity']:.3f}")
    print(f"    specificity   {s['specificity']:.3f}")
    print(f"    ROC AUC       {roc_auc(y == positive, score):.3f}")
    n = len(classes)
    ix = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[ix[t], ix[p]] += 1
    width = max(len(c) for c in classes) + 2
    print("    " + " " * width + "".join(f"{c:>{width}}" for c in classes))
    for i, c in enumerate(classes):
        print(f"    {c:<{width}}" + "".join(f"{v:>{width},}" for v in cm[i]))
    f1s = []
    for i in range(n):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    print("    per-class F1  "
          + "   ".join(f"{c} {v:.3f}" for c, v in zip(classes, f1s)))


def main(argv=None) -> int:
'''

C_ARG_OLD = r'''    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")'''
C_ARG_NEW = r'''    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")
    ap.add_argument("--augment", action="store_true",
                    help="NSTDB noise, jitter and cutout on training batches")
    ap.add_argument("--noisy-eval", type=float, default=None, metavar="SNR_DB",
                    help="also score each held-out fold with NSTDB noise at this SNR")'''

C_LOAD_OLD = r'''    d = pooled_cache(a.stage, a.cache)
'''
C_LOAD_NEW = r'''    d = pooled_cache(a.stage, a.cache)
    if a.augment or a.noisy_eval is not None:
        from ..data.train_augment import BANK_NAME, load_noise_bank, split_bank
        a._bank_train, a._bank_eval = split_bank(
            load_noise_bank(a.cache / BANK_NAME))
'''

C_HDR_OLD = r'''    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")'''
C_HDR_NEW = r'''    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")
    print(f"augment     {'on (noise, jitter, cutout)' if a.augment else 'off'}")
    if a.noisy_eval is not None:
        print(f"noisy eval  NSTDB at {a.noisy_eval:g} dB, held-out noise segments")'''

C_INIT_OLD = r'''    oof_pred = np.empty(len(y), dtype=object)
    oof_score = np.zeros(len(y))
'''
C_INIT_NEW = r'''    oof_pred = np.empty(len(y), dtype=object)
    oof_score = np.zeros(len(y))
    noisy_pred = np.empty(len(y), dtype=object)
    noisy_score = np.zeros(len(y))
'''

C_FOLD_OLD = r'''        oof_pred[te] = pred
        oof_score[te] = score
'''
C_FOLD_NEW = r'''        oof_pred[te] = pred
        oof_score[te] = score

        if a.noisy_eval is not None:
            from ..data.train_augment import add_noise_fixed
            Xn = add_noise_fixed(d["X"][te], a._bank_eval, a.noisy_eval,
                                 seed=1000 + f)
            Pn = predict(model, DataLoader(
                TensorDataset(torch.from_numpy(Xn),
                              torch.zeros(len(Xn), dtype=torch.long)),
                batch_size=256), device)
            noisy_pred[te] = np.array([classes[i] for i in Pn.argmax(1)])
            noisy_score[te] = Pn[:, classes.index(positive)]
'''

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

C_NOISY_OLD = r'''    print(f"    false alarms         {c['false_alarms']}")

    a.out.mkdir(parents=True, exist_ok=True)
'''
C_NOISY_NEW = r'''    print(f"    false alarms         {c['false_alarms']}")

    if a.noisy_eval is not None:
        _report_noisy(y, np.array([str(v) for v in noisy_pred]), noisy_score,
                      classes, positive, a.noisy_eval)

    a.out.mkdir(parents=True, exist_ok=True)
'''

C_SAVE_OLD = r'''    np.savez_compressed(oof_path, y=y, pred=oof, score=oof_score,
                        patient=patients, record=d["record"],
                        offset=d["offset"], fold=fold_of)'''
C_SAVE_NEW = r'''    extra = {} if a.noisy_eval is None else {
        "pred_noisy": np.array([str(v) for v in noisy_pred]),
        "score_noisy": noisy_score, "noisy_snr_db": a.noisy_eval}
    np.savez_compressed(oof_path, y=y, pred=oof, score=oof_score,
                        patient=patients, record=d["record"],
                        offset=d["offset"], fold=fold_of, **extra)'''


PLAN = [
    (C_MODEL_OLD, C_MODEL_NEW, "augmentation object per fold"),
    (C_LOOP_OLD, C_LOOP_NEW, "augmentation in the training loop"),
    (C_REPORT_OLD, C_REPORT_NEW, "noisy-evaluation report"),
    (C_ARG_OLD, C_ARG_NEW, "--augment and --noisy-eval flags"),
    (C_LOAD_OLD, C_LOAD_NEW, "noise bank loading"),
    (C_HDR_OLD, C_HDR_NEW, "header lines"),
    (C_INIT_OLD, C_INIT_NEW, "noisy prediction arrays"),
    (C_FOLD_OLD, C_FOLD_NEW, "noisy prediction per fold"),
    (C_CM_OLD, C_CM_NEW, "per-class F1"),
    (C_NOISY_OLD, C_NOISY_NEW, "noisy block in the report"),
    (C_SAVE_OLD, C_SAVE_NEW, "noisy predictions saved with OOF"),
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
    if not CV.exists():
        print(f"MISSING  {CV}\nrun this from the repo root")
        return 1

    text, eol = read(CV)

    if "gamma=args.gamma" not in text:
        print("STOP  the focal-loss change (PR #1) is not on this branch.")
        print("      Branch from fix/rhythm-focal-loss, or from main after")
        print("      PR #1 is merged, then run this again.")
        return 1
    if "n_rr" in text:
        print("STOP  this branch has the RR-features experiment, which was not")
        print("      adopted. Branch from fix/rhythm-focal-loss instead.")
        return 1
    if MOD.exists() and "--augment" in text:
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

    write(MOD, MOD_SOURCE, eol)
    print(f"  created  {MOD.as_posix()}")
    write(CV, text, eol)

    print("\napplied. config hash unchanged, no cache invalidated.\n")
    print("now run:")
    print("  python tests/test_train_augment.py")
    print("  python -m munji_ai.data.train_augment --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
