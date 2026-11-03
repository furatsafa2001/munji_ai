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

        print("\nWith baseline rhythm as negatives\n")
        Xb = []
        for p in range(30):
            x = np.random.default_rng(p).normal(0, 0.01, 10000)
            t = 0.3
            while t < 39.7:
                c, w = int(t * 250), 2
                x[c - 8:c + 8] += np.exp(-0.5 * (np.arange(-8, 8) / w) ** 2)
                t += 60 / 70
            Xb.append(x)
        np.savez(tmp / "base.npz", X=np.stack(Xb).astype(np.float32),
                 record=np.array([f"p{p}" for p in range(30)]),
                 split=np.array(["test" if p >= 24 else "train" for p in range(30)]))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vd.main(["--vtac", str(tmp / "vtac.npz"), "--baseline", str(tmp / "base.npz"),
                          "--epochs", "3", "--folds", "3", "--device", "cpu",
                          "--out", str(tmp / "out2")])
        text = buf.getvalue()
        check("runs with --baseline", rc == 0 and "baseline    30 slices" in text,
              text.splitlines()[0] if text else "")
        check("reports VT against baseline rhythm separately",
              "against the patients' baseline rhythm" in text and "baseline rhythm called VT" in text)

        print("\nWith Holter VT, VF and other windows\n")
        rng2 = np.random.default_rng(5)
        Wl, lab, grp, bpm = [], [], [], []
        for r in range(12):
            for j in range(6):
                kind = ("VT", "VF", "other")[j % 3]
                if kind == "VT":
                    w = vt(rng2, 120 + 5 * r, n=2500)
                elif kind == "VF":
                    t = np.arange(2500) / 250
                    w = np.sin(2 * np.pi * (4 + rng2.random()) * t) * rng2.random(2500)
                else:
                    w = vt(rng2, 70, n=2500) * 0.5
                Wl.append(w.astype(np.float16)); lab.append(kind)
                grp.append(f"cudb:r{r}"); bpm.append(130.0 if kind == "VT" else np.nan)
        np.savez(tmp / "holter.npz", X=np.stack(Wl), label=np.array(lab), group=np.array(grp),
                 dataset=np.array(["cudb"] * len(lab)), bpm=np.array(bpm))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vd.main(["--vtac", str(tmp / "vtac.npz"), "--holter", str(tmp / "holter.npz"),
                          "--epochs", "3", "--folds", "3", "--device", "cpu",
                          "--out", str(tmp / "out3")])
        text = buf.getvalue()
        check("runs with --holter", rc == 0 and "holter      72 windows" in text,
              [l for l in text.splitlines() if l.startswith("holter")])
        check("reports VT against VF (naming) on held-out patients",
              "AUC, VT against VF (naming)" in text and "holter AUC" in text)
        ck = torch.load(tmp / "out3" / "vt_detector.pt", map_location="cpu", weights_only=False)
        check("checkpoint records Holter training", ck.get("holter") is True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
