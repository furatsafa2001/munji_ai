"""ectopy_patterns: the pattern definitions, counting once, end to end.

    python tests/test_ectopy_patterns.py
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data.windows import config_hash, paint_mask
from munji_ai.eval import ectopy_patterns as EP

ID = C.BEAT_CLASS_ID
L = C.window_samples("beat")
HALF = L // 2
n_, v_, a_ = ID["N"], ID["PVC"], ID["PAC"]


def kinds(seq):
    return [(k, vs) for k, vs, _, _ in EP.episodes(seq)]


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("1. the definitions")
    check("a couplet is exactly two PVCs", kinds([n_, v_, v_, n_]) == [("couplet", [1, 2])])
    check("three PVCs are a run, not a couplet", kinds([n_, v_, v_, v_, n_]) == [])
    check("bigeminy needs 3 cycles", kinds([n_, v_, n_, v_, n_, v_, n_]) == [("bigeminy", [1, 3, 5])]
          and kinds([n_, v_, n_, v_, n_]) == [])
    check("trigeminy needs 3 cycles", kinds([n_, n_, v_, n_, n_, v_, n_, n_, v_]) == [("trigeminy", [2, 5, 8])])
    check("a PAC breaks bigeminy, and a new one needs a normal beat before it",
          kinds([n_, v_, n_, v_, a_, v_, n_, v_, n_, v_, n_, v_]) == [("bigeminy", [7, 9, 11])])
    check("bigeminy must start after a normal beat", kinds([v_, n_, v_, n_, v_]) == [])
    check("a couplet inside alternation is not bigeminy",
          kinds([n_, v_, n_, v_, v_, n_, v_, n_]) == [("couplet", [3, 4])])

    print("\n2. one record, two windows overlapping by half")
    peaks = np.arange(100, 3 * HALF, 200)                 # 37 beats
    cls = np.array(["N"] * len(peaks), dtype="<U3")
    cls[[5, 6]] = "PVC"                                    # a couplet, first window only
    cls[[14, 16, 18, 20]] = "PVC"                          # bigeminy, 4 cycles, in both windows
    cls[[28, 31, 34]] = "PVC"                              # trigeminy, second window only
    Y, offs = [], (0, HALF)
    for o in offs:
        sel = (peaks >= o) & (peaks < o + L)
        Y.append(paint_mask(L, peaks[sel] - o, cls[sel]))
    Y = np.asarray(Y, np.uint8)
    P = Y.copy()
    for w, o in enumerate(offs):                           # the model calls one bigeminy PVC N
        p = peaks[16] - o
        P[w, p - 17:p + 18] = 0
        P[w, p - 12:p + 13] = n_
    p = peaks[10]                                          # and makes a couplet of its own
    P[0, p - 12:p + 13] = 0
    P[0, p - 17:p + 18] = v_
    P[0, peaks[11] - 12:peaks[11] + 13] = 0
    P[0, peaks[11] - 17:peaks[11] + 18] = v_
    beats, expert, model = EP.collect(Y, {"m": P}, ["ds"] * 2, ["p1"] * 2, ["r"] * 2, np.array(offs))
    got = sorted(r["kind"] for r in expert)
    check("each pattern once across the overlap", got == ["bigeminy", "couplet", "trigeminy"], str(got))
    by = {r["kind"]: r for r in expert}
    check("the couplet is caught", by["couplet"]["caught"]["m"])
    check("bigeminy with 3 of 4 PVCs left is not caught: the chain breaks at the missed one",
          not by["bigeminy"]["caught"]["m"])
    check("trigeminy is caught", by["trigeminy"]["caught"]["m"])
    falses = [r for r in model["m"] if not r["real"]]
    check("the model's own couplet is one false entry", len(falses) == 1 and falses[0]["kind"] == "couplet",
          str([(r["kind"], r["real"]) for r in model["m"]]))
    ctx = sorted(r["ctx"] for r in beats)
    check("each expert PVC once, with its pattern",
          ctx == sorted(["couplet"] * 2 + ["bigeminy"] * 4 + ["trigeminy"] * 3), str(ctx))

    print("\n3. end to end, random-weight models")
    import torch
    from munji_ai.models.beat_segmenter import build
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        torch.manual_seed(0)
        for name in ("a", "b"):
            torch.save({"state_dict": build().state_dict(), "epoch": 1}, d / f"{name}.pt")
        rng = np.random.default_rng(0)
        normal = paint_mask(L, peaks[peaks < L], np.array(["N"] * int((peaks < L).sum())))
        Yc = np.concatenate([Y, normal[None]]).astype(np.uint8)
        np.savez(d / "beat_val.npz",
                 X=np.asarray([rng.normal(0, 1, L) for _ in Yc], dtype=C.CACHE_DTYPE), y=Yc,
                 dataset=np.array(["ds"] * 3), patient=np.array(["p1"] * 3), record=np.array(["r", "r", "q"]),
                 offset=np.array([0, HALF, 0], np.int64), cfg_hash=np.asarray(config_hash("beat")),
                 stage=np.asarray("beat"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = EP.main(["--ckpt", f"a={d / 'a.pt'}", "--ckpt", f"b={d / 'b.pt'}", "--cache", str(d)])
        out = buf.getvalue()
        check("runs through", rc == 0 and "Patterns lying whole" in out and "By source" in out)
        check("only windows with a PVC are predicted", "with a PVC 2" in out)
        if not ok:
            print(out)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
