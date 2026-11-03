"""How good is the VT detector, without testing it on what it learnt from?

    python -m munji_ai.eval.eval_vt_detector --info
    python -m munji_ai.eval.eval_vt_detector --part holter
    python -m munji_ai.eval.eval_vt_detector --part nsrdb --max-records 6
    python -m munji_ai.eval.eval_vt_detector --part holter --ckpt checkpoints/vt_detector_base/vt_detector.pt

Why
---
The beat model does not see long VT runs (eval_beats_model: sustained VT on
CUDB/VFDB alerted 0%, PVC share 12%), so neither the sustained-VT alert nor
VF/VT naming can come from it. The VT detector (train/vt_detector.py) was
built for exactly those questions: VT against other rhythms, and VT against
VF. This measures it, with one rule above all: never on data it trained on.

Leakage
-------
With --holter, train/vt_detector.py trains its FINAL model on every Holter
window (CUDB, VFDB, EDB, STDB, Fantasia). Scoring that model on CUDB/VFDB
would score it on its own training data and flatter it. So:

    --info          prints what each checkpoint was trained on
    --part holter   refuses a checkpoint trained with holter=True; its honest
                    numbers are the "holter windows, out of fold by patient"
                    lines its training run printed (or a re-run of that CV)
    --part nsrdb    always allowed: NSRDB is in none of its training sets, so
                    how often it calls healthy rhythm VT is an honest figure

Windows are scored as training reports them: 10 s, normalised, P(VT) =
softmax[:, 1]. A window takes a rhythm label when 80% of it lies in one
annotated episode; mixed windows are skipped. VT is split at 150 bpm with
the rate read off the signal (windows.split_vt_by_rate), as in training.
Test-split patients are skipped. --oracle scores 1 for VT windows and 0 for
the rest, to check the plumbing without PyTorch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .. import config as C
from ..data import preprocess as pp
from ..data import registry
from ..data.loader import _patient_id, list_records, load_record
from ..data.splits import assign
from ..data.windows import rhythm_intervals, split_vt_by_rate

FS = C.TARGET_FS
WIN_S = 10.0
STRIDE_S = 2.5            # ShockableNet's cadence
PURITY = 0.80
THRESHOLDS = (0.5, 0.7, 0.9)
CKPTS = (Path("checkpoints/vt_detector/vt_detector.pt"),
         Path("checkpoints/vt_detector_base/vt_detector.pt"))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Same rank AUC as train/vt_detector.py."""
    y, p = np.asarray(y), np.asarray(p, float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def eligible(ds_key: str, raw: Path, max_records: int = 0) -> list[str]:
    ds = registry.get(ds_key)
    try:
        names = list_records(ds_key, raw)
    except Exception as e:
        print(f"  {ds_key}: skipped ({e})")
        return []
    names = [n for n in names if ds.role != "val" and assign(_patient_id(ds, n)) != "test"]
    return names[:max_records] if max_records else names


class Scorer:
    def __init__(self, ckpt: Path | None):
        self.oracle = ckpt is None
        if self.oracle:
            return
        import torch

        from ..models.shockable_net import build
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = build()
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.torch = torch
        self.win = int(round(float(blob.get("window_s", WIN_S)) * FS))

    def __call__(self, X: np.ndarray, truth=None) -> np.ndarray:
        if self.oracle:
            return np.array([1.0 if t == "VT" else 0.0 for t in truth])
        out = []
        with self.torch.no_grad():
            for i in range(0, len(X), 256):
                xb = self.torch.from_numpy(X[i:i + 256].astype(np.float32))
                out.append(self.torch.softmax(self.model(xb), -1)[:, 1].numpy())
        return np.concatenate(out) if out else np.zeros(0)


def label_windows(rec, intervals, win: int, stride: int):
    """(start, label) for windows with one dominant episode, label in VT_SLOW/VT_FAST/VT/VF/other."""
    out = []
    for lo in range(0, len(rec.signal) - win + 1, stride):
        hi = lo + win
        best, share = None, 0.0
        for s, e, lab in intervals:
            ov = min(e, hi) - max(s, lo)
            if ov > 0 and ov / win > share:
                best, share = lab, ov / win
        if best is None or share < PURITY or best in ("NOISE", "UNKNOWN"):
            continue
        out.append((lo, best if best in ("VT_SLOW", "VT_FAST", "VT", "VF") else "other"))
    return out


def info(paths) -> None:
    import torch
    for p in paths:
        if not p.exists():
            print(f"{p}: not found")
            continue
        b = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n{p}")
        for k in ("window_s", "init", "augment", "epochs", "baseline", "holter", "dev_auc", "test_auc"):
            v = b.get(k, "-")
            print(f"  {k:<10}{v:.3f}" if isinstance(v, float) else f"  {k:<10}{v}")
        if b.get("holter"):
            print("  -> trained on CUDB/VFDB/EDB/STDB/Fantasia windows: do NOT score it on them.\n"
                  "     Honest Holter numbers: the 'holter windows, out of fold by patient' lines its\n"
                  "     training printed. NSRDB (--part nsrdb) stays a fair test.")
        else:
            print("  -> never saw Holter windows: --part holter is a fair test.")


def part_holter(score: Scorer, raw: Path, max_records: int) -> None:
    rows = []      # (patient, label, p)
    for ds_key in ("cudb", "vfdb"):
        for name in eligible(ds_key, raw, max_records):
            rec = load_record(ds_key, name, raw)
            win = getattr(score, "win", int(WIN_S * FS))
            lw = label_windows(rec, split_vt_by_rate(rec, rhythm_intervals(rec)), win, int(STRIDE_S * FS))
            if not lw:
                continue
            X = np.stack([pp.normalize(rec.signal[lo:lo + win]) for lo, _ in lw])
            truth = ["VT" if lab.startswith("VT") else lab for _, lab in lw]
            p = score(X, truth)
            rows += [(rec.patient, lab, float(q)) for (_, lab), q in zip(lw, p)]
            print(f"  {ds_key} {name}: {len(lw)} windows", flush=True)
    if not rows:
        print("no labelled windows")
        return
    lab = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    vt = np.char.startswith(lab, "VT")
    vf, ot = lab == "VF", lab == "other"
    slow, fast = lab == "VT_SLOW", lab == "VT_FAST"
    pats = lambda m: len({r[0] for r, k in zip(rows, m) if k})
    print(f"\nwindows: VT {vt.sum()} ({pats(vt)} patients; under 150: {slow.sum()}, 150+: {fast.sum()}), "
          f"VF {vf.sum()} ({pats(vf)}), other {ot.sum()} ({pats(ot)})")
    print(f"  AUC, VT against other rhythms   {auc(np.r_[np.ones(vt.sum()), np.zeros(ot.sum())], np.r_[p[vt], p[ot]]):.3f}")
    print(f"  AUC, VT against VF (naming)     {auc(np.r_[np.ones(vt.sum()), np.zeros(vf.sum())], np.r_[p[vt], p[vf]]):.3f}")
    print(f"  AUC, VT under 150 against other {auc(np.r_[np.ones(slow.sum()), np.zeros(ot.sum())], np.r_[p[slow], p[ot]]):.3f}")
    print(f"\n  {'called VT at':<14}{'VT <150':>9}{'VT 150+':>9}{'VF':>7}{'other':>8}")
    f = lambda m, t: f"{(p[m] >= t).mean():.0%}" if m.any() else "-"
    for t in THRESHOLDS:
        print(f"  {t:<14.1f}{f(slow, t):>9}{f(fast, t):>9}{f(vf, t):>7}{f(ot, t):>8}")
    print("\nSame windows and metrics as the training report, so the two can be compared directly.")


def part_nsrdb(score: Scorer, raw: Path, max_records: int) -> None:
    hours, calls = 0.0, {t: 0 for t in THRESHOLDS}
    n_w = 0
    for name in eligible("nsrdb", raw, max_records):
        rec = load_record("nsrdb", name, raw)
        if len(rec.beat_samples) < 2:
            continue
        lo, hi = int(rec.beat_samples.min()), int(rec.beat_samples.max())
        win = getattr(score, "win", int(WIN_S * FS))
        step = int(STRIDE_S * FS)
        starts = list(range(lo, hi - win + 1, step))
        for i in range(0, len(starts), 2048):                       # memory-bounded chunks
            chunk = starts[i:i + 2048]
            X = np.stack([pp.normalize(rec.signal[s:s + win]) for s in chunk])
            p = score(X, ["other"] * len(chunk))
            for t in THRESHOLDS:
                calls[t] += int((p >= t).sum())
            n_w += len(chunk)
        hours += (hi - lo) / FS / 3600.0
        print(f"  nsrdb {name}: {(hi - lo) / FS / 3600:.1f} h", flush=True)
    if not n_w:
        print("no NSRDB windows")
        return
    print(f"\nNSRDB (healthy, never in the detector's training): {hours:.0f} h, {n_w:,} windows of "
          f"{WIN_S:.0f} s every {STRIDE_S} s")
    print(f"  {'P(VT) at or above':<20}{'windows called VT':>18}{'share':>9}")
    for t in THRESHOLDS:
        print(f"  {t:<20.1f}{calls[t]:>18,}{calls[t] / n_w:>9.2%}")
    print("No quality gate here; an alert would also need its confirmation rule on top, so these"
          "\nare window rates, an upper bound on what a caregiver would see.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", action="store_true", help="what each checkpoint was trained on")
    ap.add_argument("--part", choices=("holter", "nsrdb"))
    ap.add_argument("--ckpt", type=Path, default=CKPTS[0])
    ap.add_argument("--oracle", action="store_true", help="score from the labels (plumbing check)")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0)
    a = ap.parse_args(argv)

    if a.info:
        info([a.ckpt] if a.ckpt not in CKPTS else list(CKPTS))
        return 0
    if not a.part:
        ap.error("give --info or --part")
    if not a.oracle:
        if not a.ckpt.exists():
            print(f"no checkpoint at {a.ckpt}")
            return 1
        import torch
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if a.part == "holter" and blob.get("holter"):
            print(f"{a.ckpt} was trained on the CUDB/VFDB windows (holter=True).\n"
                  "Scoring it on them would score it on its own training data. Its honest Holter\n"
                  "numbers are the 'holter windows, out of fold by patient' lines its training\n"
                  "printed. --part nsrdb is still a fair test.")
            return 2
        print(f"checkpoint  {a.ckpt}  (holter={blob.get('holter')}, baseline={blob.get('baseline')}, "
              f"test AUC {blob.get('test_auc', float('nan')):.3f})")
    score = Scorer(None if a.oracle else a.ckpt)
    t0 = time.time()
    (part_holter if a.part == "holter" else part_nsrdb)(score, a.raw, a.max_records)
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
