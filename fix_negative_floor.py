"""Stop multi-segment datasets from being silently dropped.

Problem
-------
Downloading 50 icentia11k patients added 12 training windows and 2 patients.
The records were found, the patient ids parsed, no error was raised — and 32
of the 34 assigned patients produced nothing at all.

The cause is integer division, twice over.

build_cache shares a patient's window budget across their records:

    per_rec = max(1, min(room_pat, cap // n_rec))

icentia11k stores ~50 segments per patient. With the rhythm cap at 80 that
gives cap // n_rec == 1, so each segment is allowed one window. Fair enough.

But the extractors then size the negative allowance as `cap // 4`, and with
cap == 1 that floors to zero. Any record with no positive episode returns an
empty list. Most icentia11k segments are ordinary sinus rhythm with no AF, so
they returned nothing — and the only two patients that survived were the two
that happened to have AF episodes.

That inverts the reason for using icentia11k at all. It is the corpus meant
to supply realistic normal rhythm from a patch device; instead it was
contributing only its rarest positives.

Fix
---
Floor the negative allowance at one window wherever it is derived by
division. A record that is allowed any windows at all should be able to
contribute at least one, whether or not it contains an episode.

Applies to all three extractors, since they share the pattern.

Run once from the repo root:

    python fix_negative_floor.py
"""

from pathlib import Path
import sys

WINDOWS = Path("munji_ai/data/windows.py")

EDITS = [
    (
        "chosen = pos + neg[:n_neg] if pos else neg[: cap // 4]",
        "chosen = pos + neg[:n_neg] if pos else neg[: max(1, cap // 4)]",
        "beat",
    ),
    (
        "n_neg = min(len(neg), max(int(len(pos) * 1.5), cap // 4)) if pos else cap // 4",
        "n_neg = (min(len(neg), max(int(len(pos) * 1.5), max(1, cap // 4)))\n             if pos else max(1, cap // 4))",
        "rhythm",
    ),
    (
        "chosen = pos + neg[: max(len(pos), cap // 4)]",
        "chosen = pos + neg[: max(len(pos), 1, cap // 4)]",
        "shockable",
    ),
]


def main() -> int:
    if not WINDOWS.exists():
        print(f"MISSING  {WINDOWS} — run this from the repo root")
        return 1

    text = WINDOWS.read_text()
    done = 0

    for old, new, label in EDITS:
        if new.split("\n")[0].strip() in text:
            print(f"  already  {label}")
            done += 1
            continue
        hit = False
        for o, n in ((old, new), (old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))):
            if o in text:
                text = text.replace(o, n, 1)
                print(f"  fixed    {label}")
                done += 1
                hit = True
                break
        if not hit:
            print(f"  FAILED   {label} — expected code not found")

    if done != len(EDITS):
        print("\nnot all edits applied — nothing written")
        return 1

    WINDOWS.write_text(text)
    print("\napplied. now run:")
    print("  python -m munji_ai.data.build_cache rhythm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
