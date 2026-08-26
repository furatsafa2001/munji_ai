"""Label the stretch before CUDB's first VF episode instead of dropping it.

Problem
-------
CUDB carries no aux_note rhythm tokens — its episodes are marked only by the
WFDB event symbols '[' and ']'. So the first rhythm marker in a record is the
onset of fibrillation. Everything before it has no label at all, and the
stretch after ']' is an explicit unknown.

The result is that CUDB now yields positives and nothing else. That showed up
as a validation split holding 446 shockable windows against 1 non-shockable —
useless for measuring false alarms, which is the number that matters most in
an alerting product.

Fix
---
Insert a rhythm marker at sample 0 when a record's timeline starts with a
bracket. The stretch before a first fibrillation onset is, by construction,
not fibrillation, so it is valid negative material for the shockable stage.

Scope: cudb is registered for the shockable and quality stages only, not
rhythm, so this cannot leak a claimed sinus label into the rhythm classifier.
CUDB's own documentation notes every beat is marked normal even where it is
ectopic, so this label asserts "not a shockable episode" — not a clean trace.

Run once from the repo root:

    python fix_cudb_baseline.py

Idempotent. If the expected code is missing it stops without writing.
"""

from pathlib import Path
import sys

LOADER = Path("munji_ai/data/loader.py")

OLD = """                if len(rhy_s):
                    merged_s = np.concatenate([rhy_s, br_s])
                    merged_y = np.concatenate([rhy_y, br_y])
                    order = np.argsort(merged_s, kind="stable")
                    rhy_s, rhy_y = merged_s[order], merged_y[order]
                else:
                    rhy_s, rhy_y = br_s, br_y"""

NEW = """                if len(rhy_s):
                    merged_s = np.concatenate([rhy_s, br_s])
                    merged_y = np.concatenate([rhy_y, br_y])
                    order = np.argsort(merged_s, kind="stable")
                    rhy_s, rhy_y = merged_s[order], merged_y[order]
                else:
                    rhy_s, rhy_y = br_s, br_y

                # Without an aux_note timeline the first marker is a '[', so
                # everything before the first episode would carry no label and
                # be dropped — leaving cudb with positives only. The stretch
                # before a first fibrillation onset is by definition not
                # fibrillation, so it is valid negative material.
                if len(rhy_s) and rhy_s[0] > 0:
                    rhy_s = np.concatenate([[0], rhy_s])
                    rhy_y = np.concatenate([["(N"], rhy_y])"""


def main() -> int:
    if not LOADER.exists():
        print(f"MISSING  {LOADER} — run this from the repo root")
        return 1
    text = LOADER.read_text()
    if "before a first fibrillation onset" in text:
        print("already applied — nothing to do")
        return 0

    # tolerate either line ending
    for old, new in ((OLD, NEW), (OLD.replace("\n", "\r\n"), NEW.replace("\n", "\r\n"))):
        if old in text:
            LOADER.write_text(text.replace(old, new, 1))
            print("applied. now run:")
            print("  python -m munji_ai.data.build_cache shockable --rebuild-splits")
            return 0

    print(f"FAILED   expected code not found in {LOADER} — nothing written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
