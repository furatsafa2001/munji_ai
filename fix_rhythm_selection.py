"""Fix the RhythmNet checkpoint-selection metric in train_rhythm.py.

What was wrong
--------------
train_rhythm.py picked its checkpoint with

    macro_f1(y_val, pred, ["NSR", "AF"])

and eval.metrics builds the confusion matrix only over the labels it is given.
Every window predicted OTHER therefore dropped out of the count: an NSR window
called OTHER was neither a hit nor a miss. A model that sends its hardest NSR
windows to OTHER scored as perfect. The final training run selected epoch 10
with a "macro F1 (NSR, AF)" of 0.987 while 345 of 1,596 NSR windows (22%) were
predicted OTHER and the real NSR F1 was 0.842. The per-epoch "OTHER F1" column
read 1.000 throughout for the same reason.

cross_validate.py was never affected: it uses its own macro_f1_of, which
scores each class against every prediction. So every cross-validated number
and every decision taken from them stands; only the final single-split
training picked the wrong epoch.

What changes
------------
    munji_ai/train/train_rhythm.py   scored_f1(): per-class F1 over the full
                                     label set, averaged over NSR and AF;
                                     OTHER F1 from the same table

eval/metrics.py is not changed: every other caller passes the full label set
(quality, shockable), where the restriction is harmless.

How to run
----------
From the repo root:

    python fix_rhythm_selection.py
    python tests/test_rhythm_selection.py

then retrain the final RhythmNet with the same command as before.

Revert with:

    git checkout munji_ai/train/train_rhythm.py
"""

from pathlib import Path

TR = Path("munji_ai/train/train_rhythm.py")

IMP_OLD = r'''from ..eval.metrics import macro_f1, report'''
IMP_NEW = r'''from ..eval.metrics import macro_f1, per_class, report'''

HELPER_OLD = r'''

def main(argv=None) -> int:
'''
HELPER_NEW = r'''

def scored_f1(y_true, y_pred) -> tuple[float, float]:
    """(macro F1 over SCORED_CLASSES, OTHER F1), every class scored against
    every prediction.

    macro_f1 with a subset of labels builds its confusion matrix over that
    subset only, so windows predicted OTHER vanish from NSR's and AF's counts
    and a model can look perfect by sending its hard windows to OTHER.
    """
    pc = per_class(y_true, y_pred, list(RHYTHM_CLASSES))

    def f(c):
        v = pc[c]["f1"]
        return 0.0 if np.isnan(v) else float(v)

    return float(np.mean([f(c) for c in SCORED_CLASSES])), f("OTHER")


def main(argv=None) -> int:
'''

SEL_OLD = r'''        f1 = macro_f1(y_va_str, pred, list(SCORED_CLASSES))
        other_f1 = macro_f1(y_va_str, pred, ["OTHER"])'''
SEL_NEW = r'''        f1, other_f1 = scored_f1(y_va_str, pred)'''

PLAN = [(IMP_OLD, IMP_NEW, "import per_class"),
        (HELPER_OLD, HELPER_NEW, "scored_f1 helper"),
        (SEL_OLD, SEL_NEW, "selection uses scored_f1")]


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    if not TR.exists():
        print(f"MISSING  {TR}\nrun this from the repo root")
        return 1
    text, eol = read(TR)
    if "def scored_f1" in text:
        print("already applied - nothing to do")
        return 0
    print("applying\n")
    ok = True
    for old, new, label in PLAN:
        if text.count(old) == 1:
            text = text.replace(old, new, 1)
            print(f"  fixed    train_rhythm  {label}")
        else:
            print(f"  FAILED   train_rhythm  {label} - expected code not found")
            ok = False
    if not ok:
        print("\nnothing written - the file is untouched")
        return 1
    write(TR, text, eol)
    print("\napplied. Cross-validation results are unaffected.\n")
    print("now run:")
    print("  python tests/test_rhythm_selection.py")
    print("then retrain the final RhythmNet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
