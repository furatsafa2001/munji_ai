"""Register the European ST-T Database without touching config.py.

Why this version exists
-----------------------
EDB is ambulatory Holter from 79 subjects aged 30 to 84, all with diagnosed or
suspected ischaemia: older, cardiac, and moving. That is the cohort the sinus
specificity gap points at, and it is natively 250 Hz with real rhythm
annotations.

The obstacle is that most of its annotations are not rhythms. Counting the
first ten records:

    (N 54   (T0+ 28   (T1- 27   T0+) 27   (ST0+ 24   (AB 19   (SVTA 16

The ST and T entries mark the start and end of ST-segment and T-wave episodes,
which is what the database was built for. The loader reads anything starting
with "(" as a rhythm change.

The clean fix is to list those tokens as ambiguous in config.py. That changes
the config hash and invalidates every cache in the project, including the beat
and quality caches on Furat's machine. Not something to spend on a hypothesis.

So this version changes nothing in config. The ST and T tokens stay unknown to
the rhythm map, which means the loader treats each one as an unrecognised
rhythm and the windows it dominates get dropped rather than mislabelled.

What that costs
---------------
Coverage, not correctness. A two-hour record with an ST episode every few
minutes gets cut into shorter sinus stretches, and windows straddling a
boundary are discarded. We will harvest less from each record than the clean
version would.

Nothing is labelled wrongly, though — a dropped window is dropped, not called
sinus. So the result is a floor: whatever improvement shows up here, the
config version would show at least as much.

How to read it
--------------
Measured against three seeds on the current data, the noise band is:

    sensitivity   0.928 - 0.951     about +/- 0.012
    specificity   0.945 - 0.955     about +/- 0.005
    sinus spec    0.960             the number this is aimed at

An improvement smaller than that band is not an improvement. If sinus
specificity lands around 0.96 again, this did nothing and the config change
would not be worth asking for. If it clears 0.97, the full version is worth
discussing with Furat — and that discussion then starts from a measurement
rather than a hope.

Scope: one registry entry for the shockable stage. Config hash unchanged.

    python fix_add_edb_noconfig.py
    python -m munji_ai.data.build_cache shockable --rebuild-splits
    python -m munji_ai.train.cross_validate --stage shockable
    python -m munji_ai.eval.report_by_rhythm

    git checkout munji_ai/data/registry.py
    python fix_shockable_negatives.py
    python fix_add_fantasia.py
    python fix_add_stdb.py
"""

from pathlib import Path
import sys

REGISTRY = Path("munji_ai/data/registry.py")

ANCHOR = '    "nsrdb": Dataset('

ENTRY = '''    # Ambulatory Holter from 79 subjects aged 30 to 84, all with diagnosed or
    # suspected ischaemia. Registered for shockable as negative material: it
    # is older, cardiac and moving at once, which is the cohort this device
    # is for.
    #
    # V5 leads the preference list because it is the most common signal here
    # (51 of 90 records) and is the chest position CM5 is named after.
    #
    # Most annotations in this set mark ST-segment and T-wave episodes, not
    # rhythms: (ST0+, (ST1-, (T0+, (T1+ and their closers. They are not in
    # RHYTHM_MAP, so the loader treats each as an unrecognised rhythm and
    # drops the windows it dominates. That loses coverage but mislabels
    # nothing — a dropped window is dropped, not called sinus.
    #
    # Listing them as ambiguous in config would keep more of each record, but
    # AMBIGUOUS_RHYTHMS feeds the config hash, so it would invalidate every
    # cache in the project including Furat's. Worth asking for only if this
    # version shows a gain above the measured noise band.
    "edb": Dataset(
        key="edb", slug="edb", version="1.0.0",
        fs=250, channel=0, role="train", labels="rhythm",
        patient_re=r"^(e\\d+)$",
        stages=("shockable",),
        channel_names=("V5", "V4", "V3", "V2", "V1", "MLIII", "MLI", "D3"),
        note="European ST-T. 90 two-hour ambulatory records, 79 subjects aged "
             "30-84 with ischaemia. ST and T episode markers are left "
             "unrecognised, so affected windows drop rather than mislabel.",
    ),

'''


def main() -> int:
    if not REGISTRY.exists():
        print(f"MISSING  {REGISTRY} — run this from the repo root")
        return 1

    text = REGISTRY.read_text()

    if '"edb": Dataset(' in text:
        print("already applied — nothing to do")
        return 0

    for a, e in ((ANCHOR, ENTRY + ANCHOR),
                 (ANCHOR.replace("\n", "\r\n"),
                  (ENTRY + ANCHOR).replace("\n", "\r\n"))):
        if a in text:
            REGISTRY.write_text(text.replace(a, e, 1))
            print("applied — edb registered, config.py untouched\n")
            print("Expect the build to report fewer windows per record than")
            print("the record length suggests: ST-marked stretches are being")
            print("dropped, which is the cost of not changing the config.\n")
            print("  python -m munji_ai.data.build_cache shockable "
                  "--rebuild-splits")
            print("  python -m munji_ai.train.cross_validate --stage shockable")
            print("  python -m munji_ai.eval.report_by_rhythm")
            print("\nSinus specificity is 0.960 now, and the noise band across")
            print("three seeds is about 0.005. Below 0.97 means this did not")
            print("earn a config change. To revert:")
            print("  git checkout munji_ai/data/registry.py")
            print("  python fix_shockable_negatives.py")
            print("  python fix_add_fantasia.py")
            print("  python fix_add_stdb.py")
            return 0

    print("FAILED   no nsrdb entry to anchor against — nothing written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
