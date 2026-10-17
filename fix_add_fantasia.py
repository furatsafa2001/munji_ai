"""Register Fantasia: healthy sinus rhythm from elderly subjects.

Why
---
Sinus rhythm specificity sits at 0.929 against an AHA target of 0.99, while
both shockable classes clear their own targets. The detector alarms on healthy
hearts, and the reason is in what it has seen: its only source of ordinary
rhythm is NSRDB, which is 18 subjects — 5 men aged 26 to 45 and 13 women aged
20 to 50. Young and healthy. The device is for elderly cardiac patients.

Three attempts at closing that gap have already failed, each measured rather
than guessed:

    miss penalty 2.5 -> 2.0     specificity fell, two more episodes missed
    SDDB in place of VFDB       positives dropped from 1,909 to 945; the
                                400-window per-patient cap takes 0.06% of a
                                19-hour recording, so sparse episodes vanish
    AFDB and MITDB as negatives sinus specificity fell to 0.910 — both are
                                arrhythmia corpora, so they added variety in
                                disease, not in normality

Fantasia is the one remaining open source of the missing class: 20 young and
20 elderly subjects, all healthy, all in sinus rhythm. The elderly half is
precisely what NSRDB lacks.

What to expect, honestly
------------------------
The recordings are made at rest. Munji is worn while walking, sleeping and
praying, so resting ECG from a healthy elderly subject is closer to the gap
than NSRDB but still not the deployment condition. Treat a modest improvement
as the realistic case and no improvement as entirely possible.

Technical notes from the headers
--------------------------------
    f1o01 2 250 1812903
    f1o01.dat ... RESP        <- channel 0
    f1o01.dat ... ECG         <- channel 1

250 Hz, so no resampling. Channel 0 is respiration, not ECG — taking the
default index 0 would feed the model a breathing trace. channel_names pins it
to the ECG lead by name, with channel=1 as the fallback.

Annotations live in .ecg files, not .atr. The loader already tries atr, ecg
then qrs, so that needs no change. They are beat annotations with no rhythm
markers, exactly like NSRDB — so baseline_rhythm carries the whole record as
sinus, which is what the database is.

Record names are f1o01..f1o10 and f1y01..f1y10 plus a second series: "o" for
old, "y" for young. patient_re takes the whole stem, since each recording is
one subject.

Scope: one new registry entry for the shockable stage. No threshold, window or
architecture changes; the config hash is unaffected, so every existing cache
stays valid.

Run from the repo root:

    python fix_add_fantasia.py
    python -m munji_ai.data.build_cache shockable --rebuild-splits
    python -m munji_ai.train.cross_validate --stage shockable
    python -m munji_ai.eval.report_by_rhythm

Judge on sinus specificity against 0.929. Revert if it does not move or if VF
sensitivity drops below 0.90:

    git checkout munji_ai/data/registry.py
    python fix_shockable_negatives.py
"""

from pathlib import Path
import sys

REGISTRY = Path("munji_ai/data/registry.py")

ANCHOR = '    "nsrdb": Dataset('

ENTRY = '''    # Healthy sinus rhythm from 20 young and 20 elderly subjects, two hours
    # each at rest. Registered for shockable only, as negative material: the
    # detector's sinus specificity was 0.929 against an AHA target of 0.99,
    # and NSRDB — its only ordinary-rhythm source — is 18 subjects aged 20 to
    # 50. The elderly half here is the cohort the device is actually for.
    #
    # Channel 0 is respiration. Taking the default index would feed the model
    # a breathing trace, so the ECG lead is pinned by name.
    #
    # Caveat worth carrying into the write-up: these are resting recordings.
    # Munji is worn while moving. Closer to the gap than NSRDB, still not the
    # deployment condition.
    "fantasia": Dataset(
        key="fantasia", slug="fantasia", version="1.0.0",
        fs=250, channel=1, role="train", labels="rhythm",
        patient_re=r"^(f\\d[oy]\\d+)$",
        stages=("shockable",),
        channel_names=("ECG",),
        baseline_rhythm="(N",
        note="Fantasia. 20 young and 20 elderly healthy subjects, sinus "
             "rhythm at rest, two hours each. Beat annotations in .ecg with "
             "no rhythm markers, so the whole record is sinus by declaration.",
    ),

'''


def main() -> int:
    if not REGISTRY.exists():
        print(f"MISSING  {REGISTRY} — run this from the repo root")
        return 1

    text = REGISTRY.read_text()

    if '"fantasia": Dataset(' in text:
        print("already applied — nothing to do")
        return 0

    if "baseline_rhythm" not in text:
        print("FAILED   the registry has no baseline_rhythm field yet.")
        print("Run fix_nsrdb_baseline.py first — Fantasia needs it for the")
        print("same reason NSRDB does: beat labels, no rhythm markers.")
        return 1

    for a, e in ((ANCHOR, ENTRY + ANCHOR),
                 (ANCHOR.replace("\n", "\r\n"), (ENTRY + ANCHOR).replace("\n", "\r\n"))):
        if a in text:
            REGISTRY.write_text(text.replace(a, e, 1))
            print("applied — fantasia registered for the shockable stage\n")
            print("now run:")
            print("  python -m munji_ai.data.build_cache shockable "
                  "--rebuild-splits")
            print("  python -m munji_ai.train.cross_validate --stage shockable")
            print("  python -m munji_ai.eval.report_by_rhythm")
            print("\nJudge on sinus specificity against 0.929. To revert:")
            print("  git checkout munji_ai/data/registry.py")
            print("  python fix_shockable_negatives.py")
            return 0

    print("FAILED   no nsrdb entry to anchor against — nothing written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
