"""Register the MIT-BIH ST Change Database: sinus rhythm under exertion.

Why
---
Sinus specificity is 0.945 against an AHA target of 0.99. Ten attempts have
been measured; three helped. Every source of ordinary rhythm so far misses at
least one property of the people this device is for:

    NSRDB      healthy, 20-50, Holter          not elderly, not cardiac
    Fantasia   healthy, 20 of them elderly     recorded at rest
    CUDB/VFDB  cardiac, arrest recordings      not ordinary rhythm

stdb is 28 recordings made largely during exercise stress tests, from patients
showing transient ST depression. So: cardiac patients, in sinus rhythm, while
exerting themselves. All three at once, which nothing else here has.

Why it needs no config change
-----------------------------
EDB was the obvious candidate and was rejected because its ST episode markers
look like rhythm changes to the parser, which would mean extending
AMBIGUOUS_RHYTHMS — a config change, a new hash, and every cache in the
project rebuilt including Furat's.

stdb has no such markers. PhysioNet's own description says it carries beat
annotations but currently no ST change annotations, and reading the first ten
annotation files confirms it: aux_note comes back empty. Nothing to confuse
the parser, nothing to add to the config.

Native rate is 360 Hz, the same as mitdb, so the existing resampling path
handles it.

The assumption, stated plainly
------------------------------
With no rhythm annotations, baseline_rhythm declares the whole record sinus —
the same mechanism NSRDB and Fantasia use. For those two it was safe by
construction: their subjects were screened for having no significant
arrhythmia.

Here it is an assumption rather than a fact. These are cardiac patients being
deliberately stressed, and a stress test is a setting where arrhythmia can
appear.

Two things make it tolerable. The tests are supervised, with the patient wired
to a monitor and stopped at the first sign of trouble, so sustained
ventricular arrhythmia is very unlikely to be in the recordings. And the label
this feeds is binary: anything that is not shockable is a correct negative,
so ST depression, ectopy and supraventricular rhythms are all labelled right
even though the rhythm name would be wrong.

What is not tolerable is leaving it unsaid. If a ventricular arrhythmia is in
there, this teaches the detector that the shape of danger is safe — the worst
error available. The mitigation is measurement: if VF sensitivity falls, this
is the first thing to suspect, and the revert is one line.

Scope: one registry entry for the shockable stage. Config hash unchanged, no
cache of Furat's invalidated.

    python fix_add_stdb.py
    python -m munji_ai.data.build_cache shockable --rebuild-splits
    python -m munji_ai.train.cross_validate --stage shockable
    python -m munji_ai.eval.report_by_rhythm

Judge on sinus specificity against 0.945, and watch VF sensitivity against
0.949 — a fall there is the signal that the assumption above was wrong.

    git checkout munji_ai/data/registry.py
    python fix_shockable_negatives.py
    python fix_add_fantasia.py
"""

from pathlib import Path
import sys

REGISTRY = Path("munji_ai/data/registry.py")

ANCHOR = '    "nsrdb": Dataset('

ENTRY = '''    # 28 records, most from exercise stress tests, from patients with
    # transient ST depression. Registered for shockable as negative material:
    # it is the only source here that is cardiac, in sinus rhythm, and under
    # exertion at the same time — the combination the sinus specificity gap
    # points at.
    #
    # Chosen over the European ST-T Database, which fits the cohort equally
    # well but annotates ST episodes with tokens the rhythm parser would read
    # as rhythm changes. Extending AMBIGUOUS_RHYTHMS to cover them would
    # change the config hash and invalidate every cache in the project. stdb
    # carries beat annotations only — aux_note is empty across its files — so
    # it costs nothing outside this stage.
    #
    # baseline_rhythm declares the record non-shockable, as for NSRDB and
    # Fantasia. For those it was safe by construction; here it is an
    # assumption about supervised stress tests. If VF sensitivity drops after
    # adding this, suspect it first.
    "stdb": Dataset(
        key="stdb", slug="stdb", version="1.0.0",
        fs=360, channel=0, role="train", labels="beat",
        patient_re=r"^(\\d+)$",
        stages=("shockable",),
        baseline_rhythm="(N",
        note="MIT-BIH ST Change. 28 records, mostly exercise stress tests, "
             "showing transient ST depression. Beat annotations only, no ST "
             "or rhythm markers, so the whole record is declared sinus.",
    ),

'''


def main() -> int:
    if not REGISTRY.exists():
        print(f"MISSING  {REGISTRY} — run this from the repo root")
        return 1

    text = REGISTRY.read_text()

    if '"stdb": Dataset(' in text:
        print("already applied — nothing to do")
        return 0

    if "baseline_rhythm" not in text:
        print("FAILED   the registry has no baseline_rhythm field.")
        print("Run fix_nsrdb_baseline.py first — stdb needs it for the same")
        print("reason NSRDB does: beat labels, no rhythm markers.")
        return 1

    for a, e in ((ANCHOR, ENTRY + ANCHOR),
                 (ANCHOR.replace("\n", "\r\n"),
                  (ENTRY + ANCHOR).replace("\n", "\r\n"))):
        if a in text:
            REGISTRY.write_text(text.replace(a, e, 1))
            print("applied — stdb registered for the shockable stage\n")
            print("config hash unchanged, no cache of Furat's invalidated.\n")
            print("now run:")
            print("  python -m munji_ai.data.build_cache shockable "
                  "--rebuild-splits")
            print("  python -m munji_ai.train.cross_validate --stage shockable")
            print("  python -m munji_ai.eval.report_by_rhythm")
            print("\nJudge on sinus specificity against 0.945. Watch VF")
            print("sensitivity against 0.949 — a fall there means the")
            print("whole-record assumption was wrong. To revert:")
            print("  git checkout munji_ai/data/registry.py")
            print("  python fix_shockable_negatives.py")
            print("  python fix_add_fantasia.py")
            return 0

    print("FAILED   no nsrdb entry to anchor against — nothing written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
