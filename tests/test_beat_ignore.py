"""Beats in no class (config.BEAT_IGNORE_ID) and checkpoint safety in train_beat.

    python tests/test_beat_ignore.py

Synthetic tensors only, no cache or PhysioNet needed. Checks that ignored
samples take no part in the loss, the class weights or any score, that data
without them trains exactly as before, and that the best checkpoint is written
when found and never silently over an existing one.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.models.beat_segmenter import SegmentationLoss, class_weights_from_masks
from munji_ai.train import train_beat as T

IG = C.BEAT_IGNORE_ID
NC = len(C.BEAT_CLASSES)


def old_loss(logits, target, w, dice_weight):
    """The loss as it was before ignore existed, for data without ignore."""
    ce = F.cross_entropy(logits, target, weight=w)
    probs = F.softmax(logits, dim=1)
    oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 2, 1).float()
    inter = (probs * oh).sum(dim=(0, 2))
    denom = probs.sum(dim=(0, 2)) + oh.sum(dim=(0, 2))
    dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6))
    present = oh.sum(dim=(0, 2)) > 0
    dice = dice[present].mean()
    return ce + dice_weight * dice


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    g = torch.Generator().manual_seed(0)
    w = torch.tensor([0.3, 1.0, 6.0, 8.0])
    crit = SegmentationLoss(w, dice_weight=0.5)

    print("1. loss")
    target = torch.randint(0, NC, (3, 400), generator=g)
    logits = torch.randn(3, NC, 400, generator=g)
    same = (crit(logits, target) - old_loss(logits, target, w, 0.5)).abs().item()
    check("without ignore, identical to the loss before", same < 1e-6, f"diff {same:.1e}")

    t_ig = target.clone()
    t_ig[:, 100:160] = IG
    a = logits.clone().requires_grad_(True)
    la = crit(a, t_ig)
    la.backward()
    b = logits.clone()
    b[:, :, 100:160] = torch.randn(3, NC, 60, generator=g) * 10
    lb = crit(b, t_ig)
    check("what the model says on ignored samples does not change the loss",
          abs(la.item() - lb.item()) < 1e-5, f"{la.item():.5f} vs {lb.item():.5f}")
    gmax = a.grad[:, :, 100:160].abs().max().item()
    check("ignored samples get no gradient", gmax == 0.0, f"max |grad| {gmax:.1e}")
    check("the other samples still do", a.grad[:, :, :100].abs().max().item() > 0)

    all_ig = torch.full((2, 300), IG)
    l0 = crit(torch.randn(2, NC, 300, requires_grad=True), all_ig)
    check("a batch of ignore only gives a finite zero loss",
          torch.isfinite(l0).item() and l0.item() == 0.0, f"{l0.item()}")

    print("\n2. class weights")
    m = np.random.default_rng(0).integers(0, NC, (20, 500)).astype(np.uint8)
    m_ig = m.copy()
    m_ig[:, :50] = IG
    m_ig2 = np.concatenate([m, np.full((20, 300), IG, np.uint8)], axis=1)
    w1, w2 = class_weights_from_masks(m), class_weights_from_masks(m_ig2)
    check("ignored samples do not shift the weights", torch.allclose(w1, w2),
          f"{w1.tolist()} vs {w2.tolist()}")

    print("\n3. scores")
    n, pvc = C.BEAT_CLASS_ID["N"], C.BEAT_CLASS_ID["PVC"]
    true = np.zeros(2000, np.uint8)
    true[480:520] = n
    true[1480:1520] = IG                       # a paced beat nobody classified
    pred = np.zeros(2000, np.uint8)
    pred[482:518] = n
    pred[1485:1515] = pvc                      # the model calls it a PVC
    check("an ignore region is not an event",
          [c for _, c in T.mask_to_events(true)] == ["N"])
    s = T.beat_scores(true[None], pred[None])
    check("a prediction on an ignored beat is not a false PVC",
          s["per_class"]["PVC"]["support"] == 0
          and np.isnan(s["per_class"]["PVC"]["precision"]),
          f"PVC {s['per_class']['PVC']}")
    check("... and the classified beat still scores",
          s["per_class"]["N"]["sensitivity"] == 1.0 and s["per_class"]["N"]["precision"] == 1.0)
    pred2 = pred.copy()
    pred2[1480:1520] = n
    f_a, f_b = T.sample_macro_f1(true[None], pred[None]), T.sample_macro_f1(true[None], pred2[None])
    check("per-sample F1 ignores what is said on ignored samples", f_a == f_b, f"{f_a:.4f} vs {f_b:.4f}")

    print("\n4. checkpoints")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "beat_segmenter.pt"
        T.save_checkpoint(p, {"x": torch.ones(2)}, {"epoch": 3, "val_beat_f1": 0.9})
        blob = torch.load(p, map_location="cpu", weights_only=False)
        check("the best model is written and loads back", blob["epoch"] == 3 and "state_dict" in blob)
        check("no temporary file is left behind", not any(Path(d).glob("*.tmp")))
        rc = T.main(["--out", d, "--cache", str(Path(d) / "no_cache_here")])
        check("a run refuses to overwrite an existing checkpoint", rc == 1)
        blob2 = torch.load(p, map_location="cpu", weights_only=False)
        check("... and leaves it untouched", blob2["epoch"] == 3)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
