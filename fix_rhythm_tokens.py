"""Fix two rhythm-annotation parsing errors that mislabel training data.

What was wrong
--------------
1. "(VF" was not in RHYTHM_MAP. PhysioNet documents VFDB ventricular
   fibrillation as "(VF" or "(VFIB"; only "(VFIB" and "(VFL" were mapped.
   Unknown tokens fall back to OTHER, so every "(VF" episode (9 in vfdb,
   2 in cudb on a full download) was labelled not shockable. The detector was
   trained to call real VF safe, and every correct detection of it was
   scored as a false alarm.

2. EDB marks ST-segment and T-wave episodes with tokens that also start with
   "(": (ST0+, (ST1-, (T0+, (T1-, (st0-, (t1+ and so on. They were read as
   rhythm changes, so each one cut the real rhythm interval and relabelled
   the rest of it OTHER, including stretches of the 191 VT runs in EDB. The
   registry note assumed such windows were dropped; the fallback actually
   kept them, labelled OTHER.

What changes
------------
    munji_ai/data/loader.py      "(VF" maps to VF; ST/T change markers are
                                 skipped when building rhythm intervals;
                                 the tokens deliberately left as OTHER are
                                 named, so a test can tell them from a miss
    munji_ai/data/registry.py    the EDB note now describes what happens

Caches built before this fix carry the old labels and are not invalidated
automatically (the label map is not part of the config hash). Only the
shockable stage reads rhythm labels from vfdb, cudb and edb, so only it needs
rebuilding:

    python -m munji_ai.data.build_cache shockable

Unaffected: the quality stage (its labels come from injected noise at a known
SNR, not from rhythm annotations), the beat stage (beat symbols, not rhythm
tokens) and the rhythm stage (does not use these databases).

How to run
----------
From the repo root:

    python fix_rhythm_tokens.py
    python tests/test_rhythm_tokens.py

Revert with:

    git checkout munji_ai/data/loader.py munji_ai/data/registry.py
"""

from pathlib import Path

LOADER = Path("munji_ai/data/loader.py")
REGISTRY = Path("munji_ai/data/registry.py")

MAP_OLD = r'''    "(VT": "VT", "(VFL": "VF", "(VFIB": "VF",'''
MAP_NEW = r'''    # VFDB documents fibrillation as "(VF" or "(VFIB"; both mean VF.
    "(VT": "VT", "(VF": "VF", "(VFL": "VF", "(VFIB": "VF",'''

END_OLD = r'''    "(NOISE": "NOISE", "(UNKNOWN": "UNKNOWN",
}
'''
END_NEW = r'''    "(NOISE": "NOISE", "(UNKNOWN": "UNKNOWN",
}

# Rhythm tokens found in the downloaded databases that are deliberately left to
# the OTHER fallback. Every one is non-shockable and outside the rhythm
# classes, so OTHER is correct for both models. They are named so that
# tests/test_rhythm_tokens.py can tell a deliberate OTHER from a missing entry,
# which is how "(VF" went unnoticed.
OTHER_TOKENS = {
    "(ASYS",    # asystole (vfdb)
    "(BI",      # first-degree heart block (vfdb)
    "(BII",     # second-degree heart block (mitdb)
    "(HGEA",    # high-grade ventricular ectopic activity (vfdb)
    "(VER",     # ventricular escape rhythm (vfdb)
    "(PM",      # pacemaker rhythm (vfdb)
    "(PREX",    # pre-excitation (mitdb)
    "(J",       # junctional rhythm (afdb)
    "(AF",      # cudb; meaning not documented there, non-shockable either way
    "(SAB",     # sino-atrial block (edb)
    "(B3",      # edb, once; meaning not documented, non-shockable either way
}

# ST-segment and T-wave change markers (EDB): (ST0+, (ST1-, (T0+, (T1--,
# lower-case variants. They describe repolarisation, not rhythm, and must not
# cut a rhythm interval. "(T" alone is trigeminy and is not matched.
NON_RHYTHM = re.compile(r"^\((?:ST|st|T|t)[01]")
'''

RK_OLD = r'''                rk = np.array([a.startswith("(") for a in aux])'''
RK_NEW = r'''                # ST/T change markers start with "(" too but are not rhythm
                # changes; counting them would cut the real interval short.
                rk = np.array([a.startswith("(") and not NON_RHYTHM.match(a)
                               for a in aux])'''

REG_OLD = r'''    # Most annotations in this set mark ST-segment and T-wave episodes, not
    # rhythms: (ST0+, (ST1-, (T0+, (T1+ and their closers. They are not in
    # RHYTHM_MAP, so the loader treats each as an unrecognised rhythm and
    # drops the windows it dominates. That loses coverage but mislabels
    # nothing - a dropped window is dropped, not called sinus.
'''
REG_NEW = r'''    # Most annotations in this set mark ST-segment and T-wave episodes, not
    # rhythms: (ST0+, (ST1-, (T0+, (T1+ and their closers. The loader skips
    # them (loader.NON_RHYTHM) when building rhythm intervals. Before that
    # fix they were read as rhythm changes and fell back to OTHER, which cut
    # real intervals short, VT runs included.
'''

PLAN = [
    (LOADER, MAP_OLD, MAP_NEW, "(VF maps to VF"),
    (LOADER, END_OLD, END_NEW, "OTHER tokens named, ST/T markers defined"),
    (LOADER, RK_OLD, RK_NEW, "ST/T markers skipped in rhythm intervals"),
    (REGISTRY, REG_OLD, REG_NEW, "EDB note corrected"),
]


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    for p in (LOADER, REGISTRY):
        if not p.exists():
            print(f"MISSING  {p}\nrun this from the repo root")
            return 1
    files = {p: read(p) for p in (LOADER, REGISTRY)}
    text = {p: t for p, (t, _) in files.items()}
    if '"(VF": "VF"' in text[LOADER] and "NON_RHYTHM" in text[LOADER]:
        print("already applied - nothing to do")
        return 0
    print("applying\n")
    ok = True
    for path, old, new, label in PLAN:
        tag = f"{path.stem:<10}"
        if new in text[path]:
            print(f"  present  {tag}{label}")
        elif text[path].count(old) == 1:
            text[path] = text[path].replace(old, new, 1)
            print(f"  fixed    {tag}{label}")
        elif path == REGISTRY:
            print(f"  skipped  {tag}{label} - comment not found, code fix unaffected")
        else:
            print(f"  FAILED   {tag}{label} - expected code not found")
            ok = False
    if not ok:
        print("\nnothing written - both files are untouched")
        return 1
    for path, (_, eol) in files.items():
        write(path, text[path], eol)
    print("\napplied.\n")
    print("The shockable cache carries the old labels. Rebuild it:")
    print("  python -m munji_ai.data.build_cache shockable")
    print("(quality, beat and rhythm caches are unaffected)")
    print("\nnow run:")
    print("  python tests/test_rhythm_tokens.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
