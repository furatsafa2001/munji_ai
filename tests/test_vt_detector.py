"""Checks train/vt_detector.py on synthetic true and false VT alarms.

    python tests/test_vt_detector.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.train import vt_detector as vd

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


def artifact(rng, n=10000):
    x = np.cumsum(rng.normal(0, 0.05, n))
    return x - np.convolve(x, np.ones(250) / 250, "same") + rng.normal(0, 0.3, n)


def main() -> int:
    print("Helpers\n")
    y = np.array([0] * 20 + [1] * 20)
    check("AUC of a perfect score is 1", vd.auc(y, y.astype(float)) == 1.0)
    check("sensitivity at 90% specificity of a perfect score is 1",
          vd.sens_at_spec(y, y.astype(float)) == 1.0)
    X = np.random.default_rng(0).normal(size=(3, 10000)).astype(np.float32)
    c = vd.crops(X, (12.0, 15.0, 18.0), 2500)
    check("evaluation crops: 3 per event, 10 s each, normalised",
          c.shape == (3, 3, 2500) and abs(float(np.median(c))) < 0.1)

    print("\nEnd to end on synthetic alarms\n")
    rng = np.random.default_rng(0)
    X, rec, dec, split = [], [], [], []
    for p in range(30):
        sp = "test" if p >= 24 else "train"
        X.append(vt(rng, 110 + 4 * p)); rec.append(f"p{p}"); dec.append(True); split.append(sp)
        X.append(artifact(rng)); rec.append(f"p{p}"); dec.append(False); split.append(sp)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "vtac.npz", X=np.stack(X).astype(np.float32), record=np.array(rec),
                 event=np.array(rec), decision=np.array(dec), split=np.array(split),
                 lead=np.array(["V"] * len(X)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vd.main(["--vtac", str(tmp / "vtac.npz"), "--epochs", "3", "--folds", "3",
                          "--device", "cpu", "--out", str(tmp / "out")])
        text = buf.getvalue()
        saved = tmp / "out" / "vt_detector.pt"
        check("runs end to end and saves", rc == 0 and saved.exists(), text.splitlines()[-1])
        ck = torch.load(saved, map_location="cpu", weights_only=False)
        check("checkpoint records its settings", ck["window_s"] == 10.0 and ck["init"] == "scratch")
        line = [l for l in text.splitlines() if "VTaC test split" in l]
        a = float(line[0].split()[3]) if line else 0.0
        check("learns separable synthetic VT and artifact", a > 0.9, f"test AUC {a:.3f}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
