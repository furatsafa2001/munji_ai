"""VT detector: is this real ventricular tachycardia? Trained on VTaC alarms.

    python -m munji_ai.train.vt_detector --augment --out /content/drive/MyDrive/munji_vt
    python -m munji_ai.train.vt_detector --augment --init shockable --out ...

Why
---
The client needs VT from 100 bpm and wide-complex tachycardia. ShockableNet
covers VT from 150 bpm; the rate engine found only 14% of real VT. VTaC holds
confirmed VT and false VT alarms from the same ICU monitors, so a model that
separates them has learnt the rhythm, not the source. A frozen-feature
logistic model reached an event AUC of 0.785 on VTaC's test patients; this
trains the network itself to see whether it reaches the 0.85 needed to build
on.

How
---
    model      ShockableNet's architecture (length-agnostic), from scratch, or
               starting from ShockableNet's weights with --init shockable
    input      10 s of one ECG lead, filtered and normalised like every source
    training   random 10 s crops from 11 to 29 s of each 40 s slice (alarm at
               20 s), four per event per epoch; cross-entropy with label
               smoothing; optional NSTDB noise, jitter and cutout
    scoring    per event, the mean P(VT) of three crops starting at 12, 15
               and 18 s, as an alarm would use
    selection  none: a fixed number of epochs, so no held-out data is used
               to choose anything

Evaluation
----------
5 folds grouped by patient over VTaC's train and validation splits, then one
model trained on all of them and scored once on VTaC's own test split.
Results are split by the rate measured from the signal: VT under 150 bpm, the
gap ShockableNet leaves, and 150 and above.

Saves the final model to --out/vt_detector.pt.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CACHE_DIR, TARGET_FS
from ..data.preprocess import normalize
from ..data.windows import _episode_bpm
from ..models.shockable_net import build

FS = TARGET_FS
EVAL_STARTS_S = (12.0, 15.0, 18.0)
TRAIN_START_S = (11.0, 19.0)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float = 0.90) -> float:
    neg = np.sort(p[y == 0])
    if not len(neg) or not (y == 1).any():
        return float("nan")
    thr = neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)]
    return float((p[y == 1] > thr).mean())


def load(path: Path):
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["decision"].astype(int)
    g = d["record"].astype(str)
    s = d["split"].astype(str)
    bpm = np.array([(_episode_bpm(x[int(15 * FS):int(25 * FS)].astype(np.float64), FS) or np.nan)
                    for x in X], dtype=float)
    return X, y, g, s, bpm


def crops(X: np.ndarray, starts_s, win: int) -> np.ndarray:
    out = np.empty((len(X), len(starts_s), win), dtype=np.float32)
    for j, s0 in enumerate(starts_s):
        a = int(s0 * FS)
        for i in range(len(X)):
            out[i, j] = normalize(X[i, a:a + win])
    return out


def random_crops(X: np.ndarray, win: int, k: int, rng) -> np.ndarray:
    lo, hi = int(TRAIN_START_S[0] * FS), int(TRAIN_START_S[1] * FS)
    idx = np.repeat(np.arange(len(X)), k)
    st = rng.integers(lo, hi + 1, size=len(idx))
    return np.stack([normalize(X[i, s:s + win]) for i, s in zip(idx, st)]).astype(np.float32), idx


def train_model(X, y, a, device, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build().to(device)
    if a.init == "shockable":
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
    aug = None
    if a.augment:
        from ..data.train_augment import BANK_NAME, TrainAugment, load_noise_bank, split_bank
        bank_train, _ = split_bank(load_noise_bank(a.cache / BANK_NAME))
        aug = TrainAugment(bank_train, device, seed=seed)
    win = int(a.window * FS)
    steps = int(np.ceil(len(X) * a.crops / a.batch)) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    for ep in range(a.epochs):
        model.train()
        Xc, idx = random_crops(X, win, a.crops, rng)
        yc = torch.from_numpy(y[idx]).long()
        order = rng.permutation(len(Xc))
        for b in range(0, len(order), a.batch):
            sel = order[b:b + a.batch]
            xb = torch.from_numpy(Xc[sel]).to(device)
            if aug is not None:
                xb = aug(xb)
            loss = F.cross_entropy(model(xb), yc[sel].to(device), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def score_events(model, X, a, device) -> np.ndarray:
    C = crops(X, EVAL_STARTS_S, int(a.window * FS))
    flat = torch.from_numpy(C.reshape(-1, C.shape[-1]))
    p = []
    for i in range(0, len(flat), 256):
        p.append(torch.softmax(model(flat[i:i + 256].to(device)), -1)[:, 1].cpu().numpy())
    return np.concatenate(p).reshape(len(X), -1).mean(1)


def report(tag, y, p, bpm):
    slow = (y == 0) | ((y == 1) & (bpm < 150))
    fast = (y == 0) | ((y == 1) & (bpm >= 150))
    print(f"  {tag:<22}{auc(y, p):>9.3f}{auc(y[slow], p[slow]):>12.3f}"
          f"{auc(y[fast], p[fast]):>12.3f}{sens_at_spec(y, p):>16.0%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtac", type=Path, default=CACHE_DIR / "vtac_windows.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/shockable_final/shockable_net.pt"))
    ap.add_argument("--init", choices=["scratch", "shockable"], default="scratch")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--window", type=float, default=10.0, help="seconds")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--crops", type=int, default=4, help="random crops per event per epoch")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vt_detector"))
    a = ap.parse_args(argv)

    from sklearn.model_selection import GroupKFold

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if a.device == "auto" else a.device))
    X, y, g, s, bpm = load(a.vtac)
    dev, test = np.isin(s, ["train", "val"]), s == "test"
    print(f"events      dev {int(dev.sum()):,} ({int(y[dev].sum()):,} VT), "
          f"test {int(test.sum()):,} ({int(y[test].sum()):,} VT)")
    print(f"patients    dev {len(set(g[dev])):,}, test {len(set(g[test])):,}")
    print(f"model       ShockableNet architecture, init {a.init}, {a.window:.0f} s input, "
          f"augment {'on' if a.augment else 'off'}, {a.epochs} epochs, device {device}\n")

    Xd, yd, gd, bd = X[dev], y[dev], g[dev], bpm[dev]
    oof = np.full(len(Xd), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(a.folds).split(Xd, yd, gd)):
        m = train_model(Xd[tr], yd[tr], a, device, seed=k)
        oof[te] = score_events(m, Xd[te], a, device)
        print(f"  fold {k}   AUC {auc(yd[te], oof[te]):.3f}   {time.time() - t0:5.0f} s", flush=True)

    final = train_model(Xd, yd, a, device, seed=99)
    print(f"\nevent-level, true VT against false VT alarms")
    print(f"  {'':<22}{'AUC':>9}{'VT <150':>12}{'VT 150+':>12}{'sens @ spec 0.90':>16}")
    report("dev, out of fold", yd, oof, bd)
    best = float("nan")
    if test.any():
        pt = score_events(final, X[test], a, device)
        report("VTaC test split", y[test], pt, bpm[test])
        best = auc(y[test], pt)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "window_s": a.window, "init": a.init,
                "augment": a.augment, "epochs": a.epochs, "test_auc": best,
                "dev_auc": auc(yd, oof)}, a.out / "vt_detector.pt")
    print(f"\nsaved {a.out / 'vt_detector.pt'}")
    print(f"test AUC {best:.3f}: " + ("worth building on" if best >= 0.85 else "not enough yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
