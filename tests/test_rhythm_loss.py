"""Checks the focal term added to RhythmLoss by fix_rhythm_focal_loss.py.

No data and no training. Every check uses hand-built logits, so a failure
points at the loss and nothing else.

    python tests/test_rhythm_loss.py
"""

import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.config import RHYTHM_CLASSES
from munji_ai.models.rhythm_net import AF_IDX, RhythmLoss

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def old_loss(logits, target, w, penalty=1.3, smoothing=0.05):
    """RhythmLoss exactly as it was before the patch."""
    losses = F.cross_entropy(logits, target, weight=w,
                             label_smoothing=smoothing, reduction="none")
    pred = logits.argmax(dim=-1)
    missed = (target == AF_IDX) & (pred != AF_IDX)
    return (losses * torch.where(missed, penalty, 1.0)).mean()


def main() -> int:
    torch.manual_seed(0)
    n_cls = len(RHYTHM_CLASSES)
    nsr, af, other = (RHYTHM_CLASSES.index(c) for c in ("NSR", "AF", "OTHER"))
    w = torch.tensor([0.6, 1.1, 4.0])

    print("RhythmLoss focal term\n")

    check("gamma is a constructor argument",
          "gamma" in inspect.signature(RhythmLoss.__init__).parameters)
    check("default gamma is 2.0, matching the model card",
          RhythmLoss().gamma == 2.0)

    # 1. gamma 0 must be the old loss, bit for bit.
    logits = torch.randn(256, n_cls) * 3
    target = torch.randint(0, n_cls, (256,))
    new0 = RhythmLoss(w, gamma=0.0)(logits, target)
    ref = old_loss(logits, target, w)
    check("gamma 0 reproduces the previous loss exactly",
          torch.equal(new0, ref), f"{new0.item():.6f} vs {ref.item():.6f}")

    # 2. Focal never increases the per-window loss, since (1-p)^gamma <= 1.
    new2 = RhythmLoss(w, gamma=2.0)(logits, target)
    check("gamma 2 total loss is not above gamma 0",
          new2.item() <= new0.item() + 1e-7, f"{new2.item():.4f} <= {new0.item():.4f}")

    # 3. The point of the change: easy windows shrink far more than hard ones.
    easy = torch.tensor([[8.0, 0.0, 0.0]])     # confident and correct NSR
    hard = torch.tensor([[0.0, 0.0, 2.0]])     # true NSR, called OTHER
    t = torch.tensor([nsr])
    ratio_easy = (RhythmLoss(w, gamma=2.0)(easy, t) / RhythmLoss(w, gamma=0.0)(easy, t)).item()
    ratio_hard = (RhythmLoss(w, gamma=2.0)(hard, t) / RhythmLoss(w, gamma=0.0)(hard, t)).item()
    check("easy correct window is down-weighted hard",
          ratio_easy < 0.01, f"kept {ratio_easy:.2e} of its loss")
    check("misclassified window keeps most of its loss",
          ratio_hard > 0.5, f"kept {ratio_hard:.2f} of its loss")

    # 4. The AF miss penalty still applies on top of focal.
    missed = torch.tensor([[2.0, 0.0, 0.0]])   # true AF, called NSR
    t_af = torch.tensor([af])
    with_pen = RhythmLoss(w, af_miss_penalty=1.3, gamma=2.0)(missed, t_af)
    no_pen = RhythmLoss(w, af_miss_penalty=1.0, gamma=2.0)(missed, t_af)
    check("missed AF still costs 1.3x", torch.isclose(with_pen, no_pen * 1.3),
          f"{with_pen.item():.4f} vs {no_pen.item() * 1.3:.4f}")

    caught = torch.tensor([[0.0, 2.0, 0.0]])   # true AF, called AF
    a = RhythmLoss(w, af_miss_penalty=1.3, gamma=2.0)(caught, t_af)
    b = RhythmLoss(w, af_miss_penalty=1.0, gamma=2.0)(caught, t_af)
    check("penalty does not touch a caught AF window", torch.equal(a, b))

    # 5. Trains: gradient exists and is finite, including near-certain logits
    #    where (1 - p) is tiny and the clamp is what keeps it stable.
    x = torch.tensor([[40.0, -40.0, -40.0], [0.1, 0.2, 0.3]], requires_grad=True)
    loss = RhythmLoss(w, gamma=2.0)(x, torch.tensor([nsr, other]))
    loss.backward()
    check("gradient is finite at extreme confidence",
          torch.isfinite(x.grad).all().item())

    # 6. The CV script hands gamma to both stages' losses without error.
    from munji_ai.train.cross_validate import build_for
    for stage in ("rhythm", "shockable"):
        _, cw, Loss, kw = build_for(stage)
        try:
            crit = Loss(torch.ones(len(w) if stage == "rhythm" else 2),
                        **{kw: 1.0}, gamma=0.0)
            check(f"cross_validate builds the {stage} loss with gamma",
                  crit.gamma == 0.0)
        except TypeError as e:
            check(f"cross_validate builds the {stage} loss with gamma", False, str(e))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
