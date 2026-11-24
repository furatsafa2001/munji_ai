"""run_lengths: each in-run beat once, runs by length, end to end.

    python tests/test_run_lengths.py
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
from munji_ai.eval import run_lengths as RL

ID = C.BEAT_CLASS_ID
L = C.window_samples("beat")
HALF = L // 2


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("1. one record, two windows overlapping by half")
    # Beats every 200 samples over 30 s; a run of 4 at beats 18-21, inside both
    # windows, and a run of 12 at beats 4-15, whose last beats the second
    # window also sees, cut by its edge.
    peaks = np.arange(100, 3 * HALF, 200)
    cls = np.array(["N"] * len(peaks), dtype="<U3")
    cls[18:22] = "PVC"
    cls[4:16] = "PVC"
    Y, offs = [], (0, HALF)
    for o in offs:
        sel = (peaks >= o) & (peaks < o + L)
        Y.append(paint_mask(L, peaks[sel] - o, cls[sel]))
    Y = np.asarray(Y, np.uint8)
    P = Y.copy()
    # The model calls the run of 12 N in full, and one beat of the run of 4 PAC.
    for i in range(4, 16):
        p = peaks[i]
        P[0, p - 17:p + 18] = ID["N"]
    for w, o in enumerate(offs):
        p = peaks[19] - o
        P[w, p - 17:p + 18] = ID["PAC"]
    beats, runs = RL.collect(Y, {"m": P}, ["ds"] * 2, ["p1"] * 2, ["r"] * 2, np.array(offs))
    check("each in-run beat counted once across the overlap", len(beats) == 4 + 12, f"{len(beats)} beats")
    check("each whole run counted once", sorted(r["k"] for r in runs) == [4, 12],
          str(sorted(r["k"] for r in runs)))
    by = {RL.bin_name(r["k"]): r for r in runs}
    check("run lengths land in their bins", set(by) == {"3-4", "10-29"})
    check("a run broken by one PAC is not caught", RL.longest_pvc_run(by["3-4"]["pred"]["m"]) < 3
          and RL._caught([by["3-4"]], "m") == 0.0,
          "PVC PAC PVC PVC holds only 2 consecutive")
    check("a run called N is not caught", RL._caught([by["10-29"]], "m") == 0.0)
    b4 = [r for r in beats if r["k"] == 4]
    check("in-run sensitivity per bin", abs(RL._sens(b4, "m") - 0.75) < 1e-9, f"{RL._sens(b4, 'm')}")
    check("bins: nothing under 3, 30 and over open-ended", RL.bin_name(2) is None and RL.bin_name(300) == "30+")

    print("\n2. end to end on a small cache with a random-weight model")
    import torch
    from munji_ai.models.beat_segmenter import build
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        torch.manual_seed(0)
        torch.save({"state_dict": build().state_dict(), "epoch": 1}, d / "a.pt")
        rng = np.random.default_rng(0)
        Yc = np.concatenate([Y, np.asarray([paint_mask(L, peaks[peaks < L], np.array(["N"] * int((peaks < L).sum())))], np.uint8)])
        np.savez(d / "beat_val.npz",
                 X=np.asarray([rng.normal(0, 1, L) for _ in Yc], dtype=C.CACHE_DTYPE), y=Yc,
                 dataset=np.array(["ds"] * 3), patient=np.array(["p1"] * 3), record=np.array(["r", "r", "q"]),
                 offset=np.array([0, HALF, 0], np.int64), cfg_hash=np.asarray(config_hash("beat")),
                 stage=np.asarray("beat"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = RL.main(["--ckpt", f"a={d / 'a.pt'}", "--cache", str(d)])
        out = buf.getvalue()
        check("runs through", rc == 0 and "Runs lying whole" in out and "By source" in out)
        check("only windows with a run are predicted", "with a run of 3+ PVCs 2" in out)
        if not ok:
            print(out)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
