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

Baseline rhythm (--baseline data/cache/vtac_baseline.npz)
----------------------------------------------------------
Trained on true against false alarms alone, the detector never saw a normal
heart; on healthy 24 h recordings it called ordinary rhythm VT, tens of times
a day. Published VTaC models share the limit: they confirm an alarm the
monitor has already raised, a second stage, never a continuous detector. With
--baseline, each patient's own rhythm from four minutes before the alarm joins
the negatives, so the detector learns "not VT" from normal and other rhythms
too, on the same monitor. Results are reported against false alarms and
against baseline rhythm separately.

Holter windows (--holter data/cache/holter_vt_windows.npz)
-----------------------------------------------------------
With VTaC alone the detector recognised ICU VT but only 14% of VT windows in
CUDB and VFDB, the ambulatory kind the patch will see. --holter adds 10 s
windows from those recordings with both sides present: VT as positives; VF
and every other rhythm as negatives. VF among the negatives is what lets the
detector name a ShockableNet alert: VT when it agrees, VF when it does not.
Holter windows are scored out of fold by patient like everything else, and
reported apart: VT against other rhythms, VT against VF (the naming
question), and how often each is called VT, with VT split at 150 bpm.

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


def load_baseline(path: Path):
    d = np.load(path)
    X = d["X"].astype(np.float32)
    bpm = np.array([(_episode_bpm(x[int(15 * FS):int(25 * FS)].astype(np.float64), FS) or np.nan)
                    for x in X], dtype=float)
    return X, np.zeros(len(X), int), d["record"].astype(str), d["split"].astype(str), bpm


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


def train_model(X, y, a, device, seed: int, W=None, yw=None):
    """X: 40 s alarm slices (random 10 s crops); W: fixed 10 s windows, normalised."""
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
    n_w = 0 if W is None else len(W)
    steps = int(np.ceil((len(X) * a.crops + n_w) / a.batch)) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    for ep in range(a.epochs):
        model.train()
        Xc, idx = random_crops(X, win, a.crops, rng)
        yc = y[idx]
        if n_w:
            Xc = np.concatenate([Xc, W])
            yc = np.concatenate([yc, yw])
        yc = torch.from_numpy(yc).long()
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
def score_windows(model, W: np.ndarray, device) -> np.ndarray:
    p = []
    for i in range(0, len(W), 256):
        p.append(torch.softmax(model(torch.from_numpy(W[i:i + 256]).to(device)), -1)[:, 1].cpu().numpy())
    return np.concatenate(p) if p else np.zeros(0, np.float32)


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
    ap.add_argument("--baseline", type=Path, default=None,
                    help="data/cache/vtac_baseline.npz: patients' own rhythm as negatives")
    ap.add_argument("--holter", type=Path, default=None,
                    help="data/cache/holter_vt_windows.npz: CUDB/VFDB VT, VF and other rhythms")
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
    kind = np.array(["alarm"] * len(X))
    if a.baseline is not None:
        Xb, yb, gb, sb, bb = load_baseline(a.baseline)
        X, y, g, s, bpm = (np.concatenate([X, Xb]), np.concatenate([y, yb]),
                           np.concatenate([g, gb]), np.concatenate([s, sb]),
                           np.concatenate([bpm, bb]))
        kind = np.concatenate([kind, np.array(["baseline"] * len(Xb))])
        print(f"baseline    {len(Xb):,} slices of patients' own rhythm as negatives")
    dev, test = np.isin(s, ["train", "val"]), s == "test"
    W = yw = gw = lw = bw = None
    if a.holter is not None:
        h = np.load(a.holter)
        lw = h["label"].astype(str)
        W = np.stack([normalize(x.astype(np.float64)) for x in h["X"]]).astype(np.float32)
        yw = (lw == "VT").astype(int)
        gw = np.array(["holter:" + gg for gg in h["group"].astype(str)])
        bw = h["bpm"].astype(float)
        print(f"holter      {len(W):,} windows: VT {int((lw == 'VT').sum()):,}, "
              f"VF {int((lw == 'VF').sum()):,}, other {int((lw == 'other').sum()):,}, "
              f"from {len(set(gw)):,} recordings")
    print(f"events      dev {int(dev.sum()):,} ({int(y[dev].sum()):,} VT), "
          f"test {int(test.sum()):,} ({int(y[test].sum()):,} VT)")
    print(f"patients    dev {len(set(g[dev])):,}, test {len(set(g[test])):,}")
    print(f"model       ShockableNet architecture, init {a.init}, {a.window:.0f} s input, "
          f"augment {'on' if a.augment else 'off'}, {a.epochs} epochs, device {device}\n")

    Xd, yd, gd, bd, kd = X[dev], y[dev], g[dev], bpm[dev], kind[dev]
    oof = np.full(len(Xd), np.nan)
    oof_w = None if W is None else np.full(len(W), np.nan)
    groups_all = gd if W is None else np.concatenate([gd, gw])
    n_e = len(Xd)
    t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(a.folds).split(groups_all, groups=groups_all)):
        tr_e, te_e = tr[tr < n_e], te[te < n_e]
        tr_w, te_w = tr[tr >= n_e] - n_e, te[te >= n_e] - n_e
        m = train_model(Xd[tr_e], yd[tr_e], a, device, seed=k,
                        W=None if W is None else W[tr_w], yw=None if W is None else yw[tr_w])
        if len(te_e):
            oof[te_e] = score_events(m, Xd[te_e], a, device)
        if W is not None and len(te_w):
            oof_w[te_w] = score_windows(m, W[te_w], device)
        msg = f"  fold {k}   alarms AUC {auc(yd[te_e], oof[te_e]):.3f}"
        if W is not None and len(te_w):
            msg += f"   holter AUC {auc(yw[te_w], oof_w[te_w]):.3f}"
        print(msg + f"   {time.time() - t0:5.0f} s", flush=True)

    final = train_model(Xd, yd, a, device, seed=99, W=W, yw=yw)
    pt = score_events(final, X[test], a, device) if test.any() else np.zeros(0)
    yt, bt, kt = y[test], bpm[test], kind[test]
    best = float("nan")
    for title, keep_d, keep_t in (
            ("true VT against false VT alarms", kd == "alarm", kt == "alarm"),
            ("true VT against the patients' baseline rhythm", (kd == "baseline") | (yd == 1),
             (kt == "baseline") | (yt == 1))):
        if not (keep_d & (yd == 0)).any():
            continue
        print(f"\nevent-level, {title}")
        print(f"  {'':<22}{'AUC':>9}{'VT <150':>12}{'VT 150+':>12}{'sens @ spec 0.90':>16}")
        report("dev, out of fold", yd[keep_d], oof[keep_d], bd[keep_d])
        if test.any() and (keep_t & (yt == 0)).any():
            report("VTaC test split", yt[keep_t], pt[keep_t], bt[keep_t])
            if title.startswith("true VT against false"):
                best = auc(yt[keep_t], pt[keep_t])
    if (kd == "baseline").any():
        for thr in (0.5, 0.7, 0.9):
            f = float((oof[kd == "baseline"] >= thr).mean())
            print(f"  baseline rhythm called VT at {thr:.1f}: {f:.0%} (dev, out of fold)")

    if W is not None:
        vt_, vf_, ot_ = lw == "VT", lw == "VF", lw == "other"
        slow = vt_ & (bw < 150)
        fast = vt_ & ~(bw < 150)
        print(f"\nholter windows, out of fold by patient (CUDB, VFDB, EDB, STDB, Fantasia)")
        print(f"  AUC, VT against other rhythms   {auc(np.r_[np.ones(vt_.sum()), np.zeros(ot_.sum())], np.r_[oof_w[vt_], oof_w[ot_]]):.3f}")
        print(f"  AUC, VT against VF (naming)     {auc(np.r_[np.ones(vt_.sum()), np.zeros(vf_.sum())], np.r_[oof_w[vt_], oof_w[vf_]]):.3f}")
        print(f"  {'called VT at':<14}{'VT <150':>9}{'VT 150+':>9}{'VF':>7}{'other':>8}")
        for thr in (0.5, 0.7, 0.9):
            f = lambda m: float((oof_w[m] >= thr).mean()) if m.any() else float("nan")
            print(f"  {thr:<14.1f}{f(slow):>9.0%}{f(fast):>9.0%}{f(vf_):>7.0%}{f(ot_):>8.0%}")

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "window_s": a.window, "init": a.init,
                "augment": a.augment, "epochs": a.epochs, "test_auc": best,
                "baseline": a.baseline is not None, "holter": a.holter is not None,
                "dev_auc": auc(yd, oof)}, a.out / "vt_detector.pt")
    print(f"\nsaved {a.out / 'vt_detector.pt'}")
    print(f"test AUC {best:.3f}: " + ("worth building on" if best >= 0.85 else "not enough yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
