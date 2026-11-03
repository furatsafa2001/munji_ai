"""Checks eval/vt_detect_feasibility.py on synthetic true and false VT alarms.

    python tests/test_vt_detect_feasibility.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import vt_detect_feasibility as vd
from munji_ai.models.shockable_net import build

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def vt(rng, bpm, n=10000, fs=250):
    x = rng.normal(0, 0.01, n)
    t = 0.3
    while t < n / fs - 0.3:
        c, w = int(t * fs), int(0.03 * fs)
        x[c - 4 * w:c + 4 * w] += np.exp(-0.5 * (np.arange(-4 * w, 4 * w) / w) ** 2)
        t += 60 / bpm
    return x


def artifact(rng, n=10000, fs=250):
    x = np.cumsum(rng.normal(0, 0.05, n))
    return x - np.convolve(x, np.ones(250) / 250, "same") + rng.normal(0, 0.3, n)


def main() -> int:
    rng = np.random.default_rng(0)
    X, rec, dec, split = [], [], [], []
    for p in range(40):
        sp = "test" if p >= 30 else ("val" if p >= 25 else "train")
        X.append(vt(rng, 110 + 3 * p))
        rec.append(f"p{p}"); dec.append(True); split.append(sp)
        X.append(artifact(rng))
        rec.append(f"p{p}"); dec.append(False); split.append(sp)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "vtac.npz", X=np.stack(X).astype(np.float32), record=np.array(rec),
                 event=np.array(rec), decision=np.array(dec), split=np.array(split),
                 lead=np.array(["V"] * len(X)))
        torch.manual_seed(0)
        torch.save({"state_dict": build().state_dict()}, tmp / "s.pt")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vd.main(["--vtac", str(tmp / "vtac.npz"), "--checkpoint", str(tmp / "s.pt")])
        text = buf.getvalue()
    check("runs end to end", rc == 0 and "event-level AUC" in text, text.splitlines()[-1])
    check("counts events and patients", "true VT 30 + 10 test, false alarms 30 + 10 test" in text,
          [l for l in text.splitlines() if l.startswith("events")])
    line = [l for l in text.splitlines() if l.strip().startswith("features, forest")]
    a = float(line[0].split()[3]) if line else 0.0
    check("separable synthetic VT and artifact score high on the test split", a > 0.9, f"{a:.3f}")
    check("rate subgroups reported", "under 150 bpm" in text)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
