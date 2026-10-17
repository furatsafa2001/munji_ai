"""Give the shockable detector ordinary rhythm to learn "not dangerous" from.

Problem
-------
Cross-validation over 50 patients put specificity at 0.864 against sensitivity
0.923. Inspecting the failures showed the pattern is not spread evenly — it
concentrates on the patients who have the least danger in them:

    vfdb:614    0 shockable windows    40 false alarms
    vfdb:423    0 shockable windows    33 false alarms
    vfdb:605    0 shockable windows    10 false alarms
    vfdb:426    2 shockable windows    51 false alarms
    cudb:cu28   4 shockable windows    48 false alarms

Sensitivity on the same patients is 1.000 wherever an episode exists. So the
model is not confused about what fibrillation looks like. It has learned that
anything unfamiliar is dangerous, and a patient whose trace is entirely normal
but simply unlike the training set gets alarmed on continuously.

That is the expected behaviour of a model trained where 28% of windows are
dangerous. CUDB and VFDB are recordings of people who arrested; neither
contains much ordinary rhythm, so "not shockable" was learned almost entirely
from the quiet stretches of arrest recordings rather than from healthy hearts.

Lowering the miss penalty does not fix it — tried at 2.0 and specificity fell
to 0.864 with two more episodes missed. The penalty sets how the model leans;
the problem is what it has seen.

Fix
---
Register nsrdb for the shockable stage. Eighteen long recordings of subjects
screened for having no significant arrhythmia, which is exactly the class the
detector is short of. The loader already labels the whole record "(N" via
baseline_rhythm, so these arrive as clean negatives with no extra work.

Scope: this only widens the source list for one stage. It changes no
threshold, no window, no architecture, and does not touch the config hash, so
caches for beat, rhythm and quality stay valid. The shockable cache has to be
rebuilt because its source set changed.

Run once from the repo root:

    python fix_shockable_negatives.py
    python -m munji_ai.data.build_cache shockable --rebuild-splits
    python -m munji_ai.train.cross_validate --stage shockable

Expect specificity to rise and sensitivity to fall slightly. If sensitivity
drops below the 0.90 floor the trade is not worth it — revert and say so.
"""

from pathlib import Path
import sys

REGISTRY = Path("munji_ai/data/registry.py")

OLD = '''        stages=("quality", "rhythm"),
        baseline_rhythm="(N",'''

NEW = '''        # Also registered for shockable. CUDB and VFDB are arrest recordings
        # and carry almost no ordinary rhythm, so the detector learned
        # "not dangerous" from the quiet parts of dangerous recordings and
        # alarmed on any unfamiliar-looking normal trace. This supplies the
        # healthy hearts that set was missing.
        stages=("quality", "rhythm", "shockable"),
        baseline_rhythm="(N",'''


def main() -> int:
    if not REGISTRY.exists():
        print(f"MISSING  {REGISTRY} — run this from the repo root")
        return 1

    text = REGISTRY.read_text()
    if '"quality", "rhythm", "shockable"' in text:
        print("already applied — nothing to do")
        return 0

    for old, new in ((OLD, NEW),
                     (OLD.replace("\n", "\r\n"), NEW.replace("\n", "\r\n"))):
        if old in text:
            REGISTRY.write_text(text.replace(old, new, 1))
            print("applied — nsrdb now feeds the shockable stage\n")
            print("next:")
            print("  python -m munji_ai.data.build_cache shockable "
                  "--rebuild-splits")
            print("  python -m munji_ai.train.cross_validate --stage shockable")
            return 0

    print(f"FAILED   expected code not found in {REGISTRY} — nothing written")
    print("Check that fix_nsrdb_baseline.py was applied first: the nsrdb entry")
    print("should already carry baseline_rhythm=\"(N\".")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
