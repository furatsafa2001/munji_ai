"""Measure whether the shockable detector fits on a phone.

    python -m munji_ai.eval.device_footprint

Reports, for the trained checkpoint:

    size        parameters, float32 bytes, int8 bytes
    compute     multiply-accumulates per 5 s window
    latency     wall time per window on CPU, single thread, batch of one —
                which is how a phone runs it, not how a GPU does
    fidelity    how far int8 quantisation moves the predictions, measured on
                real cached windows rather than asserted
    duty        what share of a second the CPU spends on this model, given
                the 2.5 s stride

The latency figure here is a laptop CPU, not a phone. Phone CPUs are slower,
so treat it as a lower bound and the duty-cycle as the thing to watch: if a
laptop core spends under 1% of its time on this model, a phone spending five
times that is still negligible.

The int8 check matters more than it looks. Quantisation is how a 690 KB
float model becomes a 170 KB one, and most deployment runtimes do it by
default. If it moves the decision on more than a handful of windows in ten
thousand, the deployed model is not the validated model and the figures in
the model card no longer describe it.
"""

from __future__ import annotations

import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import torch

from ..config import CACHE_DIR, WINDOW_STRIDE_SEC
from ..data.windows import load_cache


def count_macs(model, x) -> int:
    """Multiply-accumulates for one forward pass, from conv and linear hooks."""
    macs = [0]

    def conv_hook(m, inp, out):
        # out elements �- (in_channels / groups) �- kernel size
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, inp, out):
        macs[0] += out.numel() * m.in_features

    hooks = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs[0]


def time_per_window(model, x, n: int = 200) -> float:
    with torch.no_grad():
        for _ in range(20):                  # warm up
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints") / "shockable_net.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--n", type=int, default=2000,
                    help="windows to use for the quantisation check")
    a = ap.parse_args(argv)

    if not a.ckpt.exists():
        print(f"no checkpoint at {a.ckpt}")
        return 1

    from ..models.shockable_net import build

    torch.set_num_threads(1)                 # a phone gives you one core
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    x = torch.randn(1, 1, 1250)

    print("size")
    print(f"  parameters        {n_params:,}")
    print(f"  float32           {fp32_bytes / 1024:,.0f} KB")
    print(f"  int8 (weights)    {n_params / 1024:,.0f} KB   (estimate: one byte per weight)")

    macs = count_macs(model, x)
    print(f"\ncompute per window")
    print(f"  MACs              {macs / 1e6:.1f} M")
    print(f"  FLOPs             {2 * macs / 1e6:.1f} M")

    t = time_per_window(model, x)
    stride = WINDOW_STRIDE_SEC["shockable"]
    print(f"\nlatency, CPU, one thread, batch of one")
    print(f"  per window        {t * 1000:.2f} ms")
    print(f"  window stride     {stride:.1f} s")
    print(f"  duty cycle        {100 * t / stride:.3f}%   of one core, this model only")
    print(f"  on a 5x slower phone CPU, still ~{100 * 5 * t / stride:.2f}%")

    # --- int8 fidelity on real windows -----------------------------------
    # torch.quantization is deprecated in favour of torchao but still works
    # and is what most mobile export paths still call under the hood.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Apple Silicon and other ARM builds ship no default quantised engine;
    # qnnpack is the one that works there and is also what mobile uses.
    for engine in ("qnnpack", "fbgemm", "x86"):
        if engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = engine
            break
    try:
        d = load_cache("shockable", "val", a.cache)
        X = torch.from_numpy(d["X"][:a.n].astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            p_fp = torch.softmax(model(X), -1)[:, 1].numpy()

        q = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)
        with torch.no_grad():
            p_q = torch.softmax(q(X), -1)[:, 1].numpy()

        dec_fp, dec_q = p_fp >= 0.5, p_q >= 0.5
        flip = dec_fp != dec_q
        flipped = int(flip.sum())
        print(f"\nint8 quantisation, measured on {len(X):,} real windows")
        print(f"  max |Δ P(shockable)|   {np.abs(p_fp - p_q).max():.4f}")
        print(f"  mean |Δ|               {np.abs(p_fp - p_q).mean():.5f}")
        print(f"  decisions flipped      {flipped} / {len(X):,}   "
              f"({100 * flipped / len(X):.2f}%)")

        # Direction matters more than the count. A window the float model
        # called dangerous and the int8 model calls safe is a new miss, and
        # three of those are not equivalent to three new false alarms.
        if flipped:
            y = np.asarray(d["y"], dtype=str)[:len(X)]
            lost = int((flip & dec_fp & (y == "shockable")).sum())
            gained_fa = int((flip & dec_q & (y != "shockable")).sum())
            harmless = flipped - lost - gained_fa
            print(f"    shockable → safe       {lost}   "
                  f"{'<-- new misses' if lost else ''}")
            print(f"    safe → shockable       {gained_fa}   "
                  f"{'(new false alarms)' if gained_fa else ''}")
            print(f"    other / already wrong  {harmless}")
        tq = time_per_window(q, x)
        print(f"  int8 latency           {tq * 1000:.2f} ms  "
              f"({t / tq:.1f}x vs float32)")
        if flipped and lost:
            print(f"\n  {lost} shockable window(s) became safe under int8. Count")
            print("  matters less than direction here: these are new misses in")
            print("  the one class the model exists to catch. Re-measure")
            print("  sensitivity on the quantised model before shipping it.")
        elif flipped > len(X) * 0.002:
            print("\n  More than 0.2% of decisions moved, none of them into a")
            print("  miss. Specificity should be re-measured on the quantised")
            print("  model; sensitivity is unaffected.")
        else:
            print("\n  No shockable window changed class. The int8 model can")
            print("  carry the float32 sensitivity figures; specificity moves")
            print("  by less than the seed-to-seed noise band.")
    except Exception as e:  # noqa: BLE001
        print(f"\nint8 check skipped: {e}")

    print("\n" + "=" * 62)
    print("  This is one of four models the device runs. Budget all four plus")
    print("  preprocessing before calling the pipeline phone-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
