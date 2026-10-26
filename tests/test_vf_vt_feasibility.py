"""Checks eval/vf_vt_feasibility.py on synthetic VF and VT recordings.

    python tests/test_vf_vt_feasibility.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import vf_vt_feasibility as ff
from munji_ai.models.shockable_net import build

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    print("Building blocks\n")
    rng = np.random.default_rng(0)
    noise = rng.normal(size=1250)
    sine = np.sin(np.arange(1250) / 250 * 2 * np.pi * 3)
    check("Lempel-Ziv: noise is more complex than a sine",
          ff.lempel_ziv(noise) > ff.lempel_ziv(sine))
    check("permutation entropy: noise near 1, sine lower",
          ff.permutation_entropy(noise) > 0.95 > ff.permutation_entropy(sine))
    check("one value per feature name", len(ff.features(noise)) == len(ff.FEATURE_NAMES))
    y = np.array([0] * 50 + [1] * 50)
    check("AUC of a perfect score is 1", ff.auc(y, y.astype(float)) == 1.0)

    print("\nEnd to end on synthetic VF and VT\n")
    fs, dur = 250, 40
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, models = tmp / "raw", tmp / "cache", tmp / "models"
        for d in (raw / "vfdb", cache, models):
            d.mkdir(parents=True)
        names = []
        for k in range(8):
            kind = "VF" if k < 4 else "VT"
            name = str(418 + k)
            names.append(name)
            n = fs * dur
            if kind == "VF":
                tt = np.arange(n) / fs
                f = 5 + 1.5 * np.sin(2 * np.pi * 0.3 * tt + k)
                x = np.sin(2 * np.pi * np.cumsum(f) / fs) * (0.6 + 0.4 * rng.random(n))
                x += rng.normal(0, 0.2, n)
            else:
                x = rng.normal(0, 0.01, n)
                t = 0.3
                while t < dur - 0.3:
                    c, w = int(t * fs), int(0.03 * fs)
                    x[c - 4 * w:c + 4 * w] += np.exp(-0.5 * ((np.arange(-4 * w, 4 * w)) / w) ** 2)
                    t += 60 / (170 + 5 * k)
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                        fmt=["16"], write_dir=str(raw / "vfdb"))
            wfdb.wrann(name, "atr", np.array([5]), symbol=["+"], aux_note=[f"({kind}"],
                       write_dir=str(raw / "vfdb"))
        rec = np.array(names)
        for split, sl in (("train", slice(0, 4)), ("val", slice(4, 8))):
            np.savez(cache / f"shockable_{split}.npz", X=np.zeros((4, 1250), np.float16),
                     y=np.array(["shockable"] * 4), dataset=np.array(["vfdb"] * 4),
                     patient=rec[sl], record=rec[sl], offset=np.arange(4), cfg_hash=config_hash())
        np.savez(models / "shockable_oof.npz", record=rec, fold=np.arange(8) % 2,
                 y=np.array(["shockable"] * 8))
        torch.manual_seed(0)
        torch.save({"state_dict": build().state_dict()}, models / "final.pt")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ff.main(["--checkpoint", str(models / "final.pt"), "--models", str(models),
                          "--cache", str(cache), "--raw", str(raw), "--datasets", "vfdb"])
        text = buf.getvalue()
        check("runs end to end", rc == 0 and "by patient, out of fold" in text)
        line = [l for l in text.splitlines() if l.strip().startswith("features, forest")]
        a = float(line[0].split()[2]) if line else 0.0
        check("separable synthetic VF and VT score a high AUC", a > 0.9, f"AUC {a:.3f}")
        check("both classes counted", "recordings  with VF 4   with VT 4" in text)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
