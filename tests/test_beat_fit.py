"""beat_fit: the confusion matrix agrees with beat_scores, the interval behaves, end to end.

    python tests/test_beat_fit.py
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data.windows import config_hash, paint_ignore, paint_mask
from munji_ai.eval import beat_fit as BF
from munji_ai.train.train_beat import beat_scores

ID = C.BEAT_CLASS_ID
L = C.window_samples("beat")


def window(rng, ignore=False):
    peaks = np.arange(150, L - 150, int(rng.integers(150, 260)))
    cls = rng.choice(["N", "N", "N", "N", "PAC", "PVC"], len(peaks))
    y = paint_mask(L, peaks, cls)
    if ignore:
        y = paint_ignore(y, np.array([peaks[2]]))
    p = y.copy()
    for k in rng.choice(len(peaks), 4, replace=False):        # relabel, miss or shift
        pk, act = peaks[k], int(rng.integers(0, 4))
        p[pk - 20:pk + 21] = 0
        if act == 1:
            p[pk - 12:pk + 13] = rng.choice([ID["N"], ID["PAC"], ID["PVC"]])
        elif act == 2:
            p[pk + 60:pk + 85] = ID["PVC"]                     # shifted 70 ms: still matched
        elif act == 3:
            p[pk + 100:pk + 125] = ID["N"]                     # shifted 450 ms: a false beat
    q = int(rng.integers(L // 4, L // 2))
    p[q:q + 25] = ID["PVC"]                                    # an extra beat, maybe
    return y, p


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("1. the matrix reproduces the training script's scores")
    rng = np.random.default_rng(0)
    pairs = [window(rng, ignore=i % 3 == 0) for i in range(40)]
    Y = np.asarray([y for y, _ in pairs])
    P = np.asarray([p for _, p in pairs])
    ref = beat_scores(Y, P)
    got = BF.scores(BF.matrices(Y, P).sum(0))
    diffs = [abs(ref["per_class"][c][s] - got[c][s]) for c in list(BF.EVENT_CLASSES) + ["QRS (any beat)"]
             for s in ("sensitivity", "precision", "f1")]
    check("every sensitivity, precision and F1 equal", max(diffs) < 1e-12, f"largest difference {max(diffs):.1e}")
    check("macro F1 equal", abs(ref["macro_f1"] - got["macro_f1"]) < 1e-12,
          f"{ref['macro_f1']:.4f} and {got['macro_f1']:.4f}")
    M = BF.matrices(Y, P).sum(0)
    check("no count in the corner (no beat, missed)", M[BF.K, BF.K] == 0)

    print("\n2. resampling patients")
    one = BF.matrices(Y[:5], P[:5]).sum(0)
    same = np.stack([one] * 6)
    ci = BF.bootstrap(same, n=200)
    lo, hi = ci["macro F1"]
    check("identical patients give a zero-width interval", abs(hi - lo) < 1e-12, f"{lo:.4f} to {hi:.4f}")
    per = np.stack([BF.matrices(Y[i:i + 5], P[i:i + 5]).sum(0) for i in range(0, 40, 5)])
    ci = BF.bootstrap(per, n=300)
    v = BF.scores(per.sum(0))["macro_f1"]
    check("different patients give an interval around the value",
          ci["macro F1"][0] < v < ci["macro F1"][1], f"{v:.3f} in {ci['macro F1'][0]:.3f} to {ci['macro F1'][1]:.3f}")

    print("\n3. end to end on small caches, random-weight model")
    import torch
    from munji_ai.models.beat_segmenter import build
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        torch.manual_seed(0)
        (d / "v3").mkdir()
        torch.save({"state_dict": build().state_dict(), "epoch": 1, "val_beat_f1": 0.5},
                   d / "v3" / "beat_segmenter.pt")
        for split, n in (("train", 12), ("val", 8)):
            np.savez(d / f"beat_{split}.npz",
                     X=np.asarray([rng.normal(0, 1, L) for _ in range(n)], dtype=C.CACHE_DTYPE),
                     y=Y[:n].astype(np.uint8),
                     dataset=np.array((["icentia11k", "incartdb"] * n)[:n]),
                     patient=np.array([f"p{i % 4}" for i in range(n)]),
                     record=np.array(["r"] * n), offset=np.zeros(n, np.int64),
                     cfg_hash=np.asarray(config_hash("beat")), stage=np.asarray("beat"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = BF.main(["--ckpt", str(d / "v3" / "beat_segmenter.pt"), "--cache", str(d),
                          "--train-sample", "6"])
        out = buf.getvalue()
        check("runs through all three parts", rc == 0 and "1. TRAINING" in out
              and "2. CONFUSION" in out and "3. VALIDATION PATIENTS (4)" in out)
        check("scores the fixed training sample", "train  6 of 12 windows" in out)
        check("prints both matrices", out.count("rows are the expert's beats") == 2)
        if not ok:
            print(out)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
