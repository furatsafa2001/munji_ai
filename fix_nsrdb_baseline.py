"""Let whole-record datasets like NSRDB contribute rhythm windows.

Problem
-------
Adding nsrdb changed nothing: the rhythm cache came back at exactly 4192
training windows and 56 patients, the same as before it was downloaded. Its
15 training patients produced zero windows.

The reason is that NSRDB's annotation files carry beat labels only — there
are no '+' rhythm-change markers, because the rhythm never changes. Every
subject in that database was selected for having no significant arrhythmia,
so the entire recording is normal sinus rhythm. With no markers, the loader
builds no intervals, and nothing is extracted.

That left training at 3087 AF against 1068 NSR — 74% atrial fibrillation, a
base rate no home monitor will ever see. A model trained on it leans toward
calling AF, which in an alerting product means false alarms.

Fix
---
Add an optional `baseline_rhythm` to the registry: the rhythm that holds for
a whole record when the annotations declare no changes. Set it for nsrdb
only. The loader falls back to it when a record yields no rhythm markers at
all, which is exactly the whole-record case and never overrides real
annotations.

This does not touch the config hash, so existing caches for other stages stay
valid.

Run once from the repo root:

    python fix_nsrdb_baseline.py
"""

from pathlib import Path
import sys

REGISTRY = Path("munji_ai/data/registry.py")
LOADER = Path("munji_ai/data/loader.py")

OLD_FIELD = """    verified: bool = True      # False = record layout needs checking on download
    note: str = \"\""""

NEW_FIELD = """    verified: bool = True      # False = record layout needs checking on download
    # Rhythm that holds for an entire record when the annotations declare no
    # changes. NSRDB subjects were screened for having no significant
    # arrhythmia, so its files carry beat labels and no rhythm markers at all —
    # without this the loader finds no intervals and the set contributes
    # nothing. Leave empty for any set where the rhythm actually varies.
    baseline_rhythm: str = ""
    note: str = \"\""""

OLD_NSRDB = """        fs=128, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\\d+)$",
        stages=("quality", "rhythm"),
        note="Clean normal sinus rhythm. Primary carrier for synthesised quality \""""

NEW_NSRDB = """        fs=128, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\\d+)$",
        stages=("quality", "rhythm"),
        baseline_rhythm="(N",
        note="Clean normal sinus rhythm. Primary carrier for synthesised quality \""""

OLD_LOADER = """                else:
                    rhy_s, rhy_y = br_s, br_y

                # Without an aux_note timeline the first marker is a '[', so"""

NEW_LOADER = """                else:
                    rhy_s, rhy_y = br_s, br_y

                # Without an aux_note timeline the first marker is a '[', so"""

OLD_BREAK = """                if len(rhy_s) and rhy_s[0] > 0:
                    rhy_s = np.concatenate([[0], rhy_s])
                    rhy_y = np.concatenate([["(N"], rhy_y])
            break"""

NEW_BREAK = """                if len(rhy_s) and rhy_s[0] > 0:
                    rhy_s = np.concatenate([[0], rhy_s])
                    rhy_y = np.concatenate([["(N"], rhy_y])

            # No rhythm markers anywhere means the annotations declare no
            # change, not that the rhythm is unknown. For a set registered
            # with a baseline that is a positive statement about the whole
            # record; for every other set it stays empty and nothing happens.
            if not len(rhy_s) and ds.baseline_rhythm:
                rhy_s = np.array([0], dtype=np.int64)
                rhy_y = np.array([ds.baseline_rhythm], dtype=object)
            break"""


def patch(path: Path, old: str, new: str, label: str) -> bool:
    if not path.exists():
        print(f"  MISSING  {path} — run this from the repo root")
        return False
    text = path.read_text()
    if new.strip() and new.strip() in text:
        print(f"  already  {label}")
        return True
    for o, n in ((old, new), (old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))):
        if o in text:
            path.write_text(text.replace(o, n, 1))
            print(f"  fixed    {label}")
            return True
    print(f"  FAILED   {label} — expected code not found in {path}")
    return False


def main() -> int:
    print("applying fix\n")
    ok = [
        patch(REGISTRY, OLD_FIELD, NEW_FIELD, "1  baseline_rhythm field"),
        patch(REGISTRY, OLD_NSRDB, NEW_NSRDB, "2  nsrdb declares (N"),
        patch(LOADER, OLD_BREAK, NEW_BREAK, "3  loader falls back to it"),
    ]
    if all(ok):
        print("\napplied. now run:")
        print("  python -m munji_ai.data.build_cache rhythm")
        return 0
    print("\nsomething did not apply — no file was left half-written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
