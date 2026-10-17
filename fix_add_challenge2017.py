"""Register the PhysioNet/CinC Challenge 2017 set for the rhythm stage.

Why
---
The rhythm classifier has three classes and training data for two. OTHER —
every rhythm that is neither sinus nor atrial fibrillation — has 34 training
windows against 190 in test. Raising icentia from 145 patients to 235 moved it
from 26 to 34, so the shortage is in what the sources contain, not how much of
them we read: ltafdb, icentia and nsrdb are all effectively binary.

The CinC 2017 challenge was built for exactly this problem. Its task is to sort
single-lead recordings into normal sinus rhythm, atrial fibrillation, an
alternative rhythm, or too noisy to classify — the same four-way split, with
"other" defined the same way: any identifiable rhythm that is neither AF nor
normal.

And it has the data we lack:

    N  5,076      A  758      O  2,415      ~  279

2,415 recordings of other rhythms against our 34 windows.

It also matches this stage exactly: single lead, 300 Hz, and every recording is
9,000 samples — 30.0 seconds, which is the rhythm window length. One recording
becomes one window.

What this does not promise
--------------------------
OTHER is the hardest class in this task even for the teams that had all 2,415
recordings. One challenge entry reported that the class "is mistaken for AF and
even more for Normal", and that the network "struggles to generalise the Other
class, maybe the intra-class variability is too impacting". The winning overall
score across the four classes was 82.6%.

So the realistic outcome is a usable OTHER, not a strong one. If it moves from
unusable to roughly the published level, that is the whole of what is on offer.

The mechanism this needs
------------------------
Labels here live in a CSV next to the records, not in .atr files, and each
recording has its own label. baseline_rhythm cannot express that: it is one
fixed rhythm for an entire dataset.

So this adds `label_file` and `label_map` to the registry, and six lines in the
loader that read the CSV when a dataset declares one. The branch only fires
for datasets that set `label_file`, which is this one and nothing else, so no
existing dataset changes behaviour.

RHYTHM_MAP also gains "(OTHER" -> "OTHER". Using an existing token such as
"(AB" would have avoided the edit, but (AB means atrial bigeminy, and labelling
2,415 mixed recordings as bigeminy to save one line is the kind of shortcut
that shows up later as a result nobody can explain.

Cost to the rest of the project
-------------------------------
None. config_hash covers preprocessing, windows, purity thresholds and the
ambiguous set; it does not read the registry or RHYTHM_MAP. The hash is
unchanged, so every existing cache stays valid and nothing of Furat's needs
rebuilding. The rhythm cache has to be rebuilt because its source set changed.

Two files are shared, though — loader.py and registry.py — so tell her the
change exists even though it cannot affect her stages.

Before running, delete the duplicated validation folder:

    rm -rf data/raw/challenge2017/validation

Those 300 records are a subset of training under the same names; keeping both
would feed each of them twice.

    python fix_add_challenge2017.py
    python -m munji_ai.data.build_cache rhythm --rebuild-splits
    python -m munji_ai.train.train_rhythm

Judge it on the OTHER column. Revert with:

    git checkout munji_ai/data/loader.py munji_ai/data/registry.py
"""

from pathlib import Path
import sys

LOADER = Path("munji_ai/data/loader.py")
REGISTRY = Path("munji_ai/data/registry.py")

# ---------------------------------------------------------------- registry

REG_FIELD_OLD = '''    baseline_rhythm: str = ""'''

REG_FIELD_NEW = '''    baseline_rhythm: str = ""
    # Per-record labels held in a CSV beside the data, rather than in .atr
    # files. CinC 2017 ships one label per recording this way, and
    # baseline_rhythm cannot express it because that is a single rhythm for a
    # whole dataset. Path is relative to the dataset directory; the first CSV
    # column is the record (any directory prefix is ignored) and the second is
    # the label. Empty for every set that annotates normally.
    label_file: str = ""
    # Maps the label strings in that CSV onto WFDB rhythm tokens, as pairs
    # because a frozen dataclass cannot hold a dict.
    label_map: tuple = field(default_factory=tuple)'''

REG_ANCHOR = '    "icentia11k": Dataset('

REG_ENTRY = '''    # The CinC 2017 challenge set: 8,528 single-lead recordings, 30 s each at
    # 300 Hz, sorted into normal / AF / other / too noisy. Registered for the
    # rhythm stage to supply the OTHER class, which the other sources do not
    # carry — ltafdb, icentia and nsrdb are all effectively binary, and OTHER
    # sat at 34 training windows against 190 in test.
    #
    # 2,415 of these recordings are labelled "other rhythm", which is the
    # whole reason for adding it.
    #
    # Labels are in training/REFERENCE.csv, one per recording, so this is the
    # first dataset to use label_file. Noisy recordings map to "(NOISE" and
    # are dropped downstream rather than assigned to a class.
    #
    # Each recording is treated as its own patient. The challenge provides no
    # subject identifiers, so recordings from the same person cannot be
    # grouped; the challenge itself scored them independently. Worth stating
    # in the write-up as a limitation of this source.
    "challenge2017": Dataset(
        key="challenge2017", slug="challenge-2017", version="1.0.0",
        fs=300, channel=0, role="train", labels="rhythm",
        patient_re=r"^(A\\d+)$",
        stages=("rhythm",),
        label_file="training/REFERENCE.csv",
        label_map=(("N", "(N"), ("A", "(AFIB"),
                   ("O", "(OTHER"), ("~", "(NOISE")),
        note="PhysioNet/CinC Challenge 2017. 8,528 recordings, 30 s, single "
             "lead, 300 Hz, AliveCor handheld. N 5,076 / A 758 / O 2,415 / "
             "noisy 279. Added for the OTHER class.",
    ),

'''

# ------------------------------------------------------------------ loader

MAP_OLD = '''    "(P": "PACED", "(AB": "OTHER", "(IVR": "OTHER", "(NOD": "OTHER",'''
MAP_NEW = '''    "(P": "PACED", "(AB": "OTHER", "(IVR": "OTHER", "(NOD": "OTHER",
    # Generic marker for a set that labels "some other rhythm" without naming
    # it, as CinC 2017 does. Mapping those onto a named token such as (AB
    # would assert a diagnosis the source never made.
    "(OTHER": "OTHER",'''

READER = '''

# Per-record labels read from a CSV, cached per dataset so the file is parsed
# once rather than for every record in it.
_LABEL_CACHE: dict[tuple, dict] = {}


def _record_label(ds: Dataset, name: str, root: Path) -> str:
    """Rhythm token for one record, from the dataset's label CSV.

    The first column may carry a directory prefix (CinC 2017 writes
    "A00/A00001"), so records are indexed by their basename. An unmapped or
    missing label returns "", and the caller leaves the record without rhythm
    intervals — it then contributes nothing, which is the right outcome for a
    label we cannot interpret.
    """
    key = (ds.key, str(root))
    if key not in _LABEL_CACHE:
        path = dataset_dir(ds, root) / ds.label_file
        table: dict[str, str] = {}
        if path.exists():
            mapping = dict(ds.label_map)
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if len(row) < 2:
                        continue
                    rec = row[0].strip().split("/")[-1]
                    token = mapping.get(row[1].strip(), "")
                    if token:
                        table[rec] = token
        _LABEL_CACHE[key] = table
    return _LABEL_CACHE[key].get(name, "")
'''

# The baseline branch sits at whatever indentation its enclosing block has,
# which differs between versions of this file, so it is matched by regex and
# the new branch is written at the indentation actually found.
LOAD_RE = r"^([ \t]*)if not len\(rhy_s\) and ds\.baseline_rhythm:"

LOAD_TEMPLATE = (
    "{i}if not len(rhy_s) and ds.label_file:\n"
    "{i}    # One label for the whole recording, from the dataset's CSV.\n"
    "{i}    token = _record_label(ds, name, root)\n"
    "{i}    if token:\n"
    "{i}        rhy_s = np.array([0], dtype=np.int64)\n"
    "{i}        rhy_y = np.array([token], dtype=object)\n"
    "{i}if not len(rhy_s) and ds.baseline_rhythm:"
)


def patch(text, old, new, label):
    for o, n in ((old, new),
                 (old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))):
        if o in text:
            print(f"  fixed    {label}")
            return text.replace(o, n, 1), True
    print(f"  FAILED   {label} — expected code not found")
    return text, False


def main() -> int:
    for p in (LOADER, REGISTRY):
        if not p.exists():
            print(f"MISSING  {p} — run this from the repo root")
            return 1

    loader = LOADER.read_text()
    registry = REGISTRY.read_text()

    if '"challenge2017": Dataset(' in registry:
        print("already applied — nothing to do")
        return 0

    if "baseline_rhythm" not in loader:
        print("FAILED   loader.py has no baseline_rhythm branch to anchor on.")
        print("Run fix_nsrdb_baseline.py first.")
        return 1

    print("applying\n")
    ok = True

    registry, h = patch(registry, REG_FIELD_OLD, REG_FIELD_NEW,
                        "1  label_file / label_map fields")
    ok = ok and h

    for a, e in ((REG_ANCHOR, REG_ENTRY + REG_ANCHOR),
                 (REG_ANCHOR.replace("\n", "\r\n"),
                  (REG_ENTRY + REG_ANCHOR).replace("\n", "\r\n"))):
        if a in registry:
            registry = registry.replace(a, e, 1)
            print("  fixed    2  challenge2017 registered")
            break
    else:
        print("  FAILED   2  no icentia11k entry to anchor against")
        ok = False

    if "import csv" not in loader:
        for old, new in (("import re\n", "import csv\nimport re\n"),
                         ("import re\r\n", "import csv\r\nimport re\r\n")):
            if old in loader:
                loader = loader.replace(old, new, 1)
                print("  fixed    3  csv import")
                break
        else:
            print("  FAILED   3  could not place the csv import")
            ok = False
    else:
        print("  already  3  csv import")

    loader, h = patch(loader, MAP_OLD, MAP_NEW, "4  (OTHER token in RHYTHM_MAP")
    ok = ok and h

    # the reader goes after dataset_dir, which it calls
    for old, new in (("def _pick_channel(", READER.lstrip("\n") + "\n\ndef _pick_channel("),
                     ("def _pick_channel(", READER.lstrip("\n") + "\r\n\r\ndef _pick_channel(")):
        if old in loader:
            loader = loader.replace(old, new, 1)
            print("  fixed    5  _record_label reader")
            break
    else:
        print("  FAILED   5  could not place the reader")
        ok = False

    import re as _re
    m = _re.search(LOAD_RE, loader, _re.M)
    if m:
        indent = m.group(1)
        block = LOAD_TEMPLATE.format(i=indent)
        if "\r\n" in loader:
            block = block.replace("\n", "\r\n")
        loader = loader[:m.start()] + block + loader[m.end():]
        print("  fixed    6  label_file branch in load_record")
    else:
        print("  FAILED   6  baseline_rhythm branch not found")
        ok = False

    if not ok:
        print("\nnothing written — both files are untouched")
        return 1

    if "field" not in registry.split("\n")[0:40][0] and "from dataclasses import" in registry:
        import re as _re
        m = _re.search(r"from dataclasses import ([^\n]+)", registry)
        if m and "field" not in m.group(1):
            registry = registry.replace(
                m.group(0), f"from dataclasses import {m.group(1).strip()}, field", 1)
            print("  fixed    7  dataclasses.field import")

    LOADER.write_text(loader)
    REGISTRY.write_text(registry)

    print("\napplied. config hash unchanged, no cache of Furat's invalidated.\n")
    print("first, remove the duplicated validation copies:")
    print("  rm -rf data/raw/challenge2017/validation\n")
    print("then:")
    print("  python -m munji_ai.data.build_cache rhythm --rebuild-splits")
    print("  python -m munji_ai.train.train_rhythm\n")
    print("Watch the OTHER count in the build output — it should go from 34")
    print("into the thousands. Revert with:")
    print("  git checkout munji_ai/data/loader.py munji_ai/data/registry.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
