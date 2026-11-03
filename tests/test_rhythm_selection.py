"""Checks the RhythmNet selection metric fixed by fix_rhythm_selection.py.

    python tests/test_rhythm_selection.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval.metrics import macro_f1
from munji_ai.train.train_rhythm import scored_f1

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    y = np.array(["NSR"] * 100 + ["AF"] * 50 + ["OTHER"] * 20)
    p = y.copy()
    p[:40] = "OTHER"                       # 40% of NSR dumped into OTHER

    old = macro_f1(y, p, ["NSR", "AF"])
    new, other = scored_f1(y, p)
    print("Selection metric\n")
    check("the old subset metric is fooled (the bug)", old == 1.0, f"{old:.3f}")
    check("the new metric counts NSR sent to OTHER as misses", new < 0.9, f"{new:.3f}")
    check("OTHER F1 reflects the 40 wrong OTHER calls", other < 0.6, f"{other:.3f}")
    perfect, other_p = scored_f1(y, y.copy())
    check("a perfect prediction still scores 1.0", perfect == 1.0 and other_p == 1.0)
    p2 = y.copy()
    p2[100:125] = "NSR"                    # half the AF called NSR
    new2, _ = scored_f1(y, p2)
    check("missing AF still lowers the score", new2 < 0.9, f"{new2:.3f}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
