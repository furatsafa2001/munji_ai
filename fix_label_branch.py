"""Move the per-record label branch out of the annotation-extension loop.

fix_add_challenge2017.py anchors its new branch on the baseline_rhythm line,
which sits inside

    for ext in ("atr", "ecg", "qrs"):
        ...
        break

That loop only runs its body when one of those three files opens. CinC 2017
ships none of them — its labels live in a CSV — so the branch placed beside
baseline_rhythm is never reached, and every record comes back with no rhythm
intervals at all.

The symptom is quiet: the build succeeds, CinC 2017 appears in the source
list, and OTHER stays at 27 windows instead of 2,141. Nothing errors.

This lifts the branch to function level, just before the Record is built, so
it runs whether or not an annotation file was found. baseline_rhythm stays
where it is: that one is only reached for datasets that do have .atr files.

    python fix_label_branch.py
    python -c "from munji_ai.data.loader import load_record; from munji_ai.data.windows import rhythm_intervals; print(rhythm_intervals(load_record('challenge2017','A00001')))"

The check should print [(0, 7500, 'NSR')]. Then rebuild:

    python -m munji_ai.data.build_cache rhythm

and OTHER should be in the thousands.
"""

import re
from pathlib import Path

LOADER = Path("munji_ai/data/loader.py")

BRANCH = re.compile(
    r"[ \t]*if not len\(rhy_s\) and ds\.label_file:[ \t]*\r?\n"
    r"(?:[ \t]*#[^\r\n]*\r?\n)*"
    r"[ \t]*token = _record_label\(ds, name, root\)[ \t]*\r?\n"
    r"[ \t]*if token:[ \t]*\r?\n"
    r"[ \t]*rhy_s = np\.array\(\[0\], dtype=np\.int64\)[ \t]*\r?\n"
    r"[ \t]*rhy_y = np\.array\(\[token\], dtype=object\)[ \t]*\r?\n"
)


def main() -> int:
    if not LOADER.exists():
        print(f"MISSING  {LOADER} - run this from the repo root")
        return 1

    s = LOADER.read_text(encoding="utf-8")

    if "# Must sit outside the annotation-extension loop" in s:
        print("already applied - nothing to do")
        return 0

    m = BRANCH.search(s)
    if not m:
        print("FAILED   could not find the label_file branch.")
        print("         Has fix_add_challenge2017.py been run?")
        return 1

    indent = len(m.group(0)) - len(m.group(0).lstrip(" \t"))
    print(f"found the branch at indent {indent}"
          f"{' - inside the loop' if indent > 4 else ' - already at function level'}")

    s = s[:m.start()] + s[m.end():]

    nl = "\r\n" if "\r\n" in s else "\n"
    anchor = nl + "    return Record(" + nl
    if anchor not in s:
        print("FAILED   could not find the Record construction - nothing written")
        return 1

    block = nl.join([
        "    # Per-record labels from a CSV, for sets that ship no .atr files.",
        "    # Must sit outside the annotation-extension loop above: that loop",
        "    # only runs its body when one of atr/ecg/qrs opens, and CinC 2017",
        "    # has none of them, so a branch placed inside it never fires and",
        "    # every record comes back with no rhythm intervals at all.",
        "    if not len(rhy_s) and ds.label_file:",
        "        token = _record_label(ds, name, root)",
        "        if token:",
        "            rhy_s = np.array([0], dtype=np.int64)",
        "            rhy_y = np.array([token], dtype=object)",
        "", ""])

    s = s.replace(anchor, nl + block + "    return Record(" + nl, 1)
    LOADER.write_text(s, encoding="utf-8")

    print("moved to function level\n")
    print("check it:")
    print('  python -c "from munji_ai.data.loader import load_record; '
          'from munji_ai.data.windows import rhythm_intervals; '
          "print(rhythm_intervals(load_record('challenge2017','A00001')))\"")
    print("\nexpected:  [(0, 7500, 'NSR')]\n")
    print("then rebuild and watch OTHER go from 27 into the thousands:")
    print("  python -m munji_ai.data.build_cache rhythm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
