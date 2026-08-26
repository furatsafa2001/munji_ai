"""Cap how many records are read per patient when building caches.

    python apply_record_cap.py

Safe to run twice. Delete once it has run.


WHY
---
Icentia stores roughly 50 segments per patient, each about 70 minutes. The
window cap is 150 per patient, so the builder reads all 50 segments to take
about 3 windows from each. Reading and filtering a segment is the expensive
part; extracting 3 windows from it is nearly free. At 237 training patients
that is 11,649 records read to produce 35,550 windows — hours of work for a
cache that fits in half a gigabyte.

Reading 10 segments and taking 15 windows from each gives the same 150 windows
for a fifth of the time.

The cost is temporal diversity: 10 stretches of a patient's day instead of 50.
That is a real loss and worth naming. It is accepted here because 10 segments
still span roughly 12 hours per patient, and because the cache gets rebuilt
several times during development — a 4-hour build that has to be repeated is a
larger cost than slightly narrower sampling.

Records are chosen by a deterministic hash of the record name, not by taking
the first N. The first N would be consecutive segments from the start of the
recording, so every patient would be sampled from the same part of their day.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
applied, skipped, failed = [], [], []


def patch(rel: str, name: str, old: str, new: str, marker: str) -> None:
    p = ROOT / rel
    if not p.exists():
        failed.append(f"{name}: {rel} not found")
        return
    with open(p, encoding="utf-8", newline=None) as f:
        text = f.read()
    if marker in text:
        skipped.append(name)
        return
    if old not in text:
        failed.append(f"{name}: anchor not found in {rel}")
        return
    with open(p, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text.replace(old, new, 1))
    applied.append(name)


patch(
    "munji_ai/config.py", "config: records per patient",
    old='MAX_WINDOWS_PER_PATIENT = {"quality": 60, "beat": 150, "rhythm": 80, "shockable": 400}',
    new='''MAX_WINDOWS_PER_PATIENT = {"quality": 60, "beat": 150, "rhythm": 80, "shockable": 400}

# How many of a patient's records to read at all. Icentia holds ~50 segments per
# patient; reading every one to take 3 windows from each wastes almost all of
# the time in loading and filtering. Ten segments still span roughly 12 hours,
# and the build runs five times faster.
#
# Datasets with a single record per patient are unaffected.
MAX_RECORDS_PER_PATIENT = {"quality": 6, "beat": 10, "rhythm": 8, "shockable": 50}''',
    marker="MAX_RECORDS_PER_PATIENT",
)

patch(
    "munji_ai/data/windows.py", "windows: subsample records per patient",
    old='''            names = [e["record"] for e in items]
            lookup = {e["record"]: e for e in items}''',
    new='''            # Keep only a few records per patient. Chosen by a stable hash of
            # the record name rather than the first N, so the sample is spread
            # across the recording instead of clustering at its start — every
            # patient would otherwise be sampled from the same part of their day.
            rec_cap = C.MAX_RECORDS_PER_PATIENT.get(stage)
            if rec_cap:
                grouped: dict[str, list] = {}
                for e in items:
                    grouped.setdefault(e["patient"], []).append(e)
                items = []
                for pid, recs in grouped.items():
                    recs.sort(key=lambda e: hashlib.sha256(
                        f"{C.SPLIT_SEED}:{e['record']}".encode()).hexdigest())
                    items.extend(recs[:rec_cap])

            names = [e["record"] for e in items]
            lookup = {e["record"]: e for e in items}''',
    marker="rec_cap = C.MAX_RECORDS_PER_PATIENT",
)

patch(
    "munji_ai/data/windows.py", "windows: hash covers the record cap",
    old='''        "quality_classes": sorted(C.QUALITY_CLASSES),''',
    new='''        "quality_classes": sorted(C.QUALITY_CLASSES),
        "records_per_patient": C.MAX_RECORDS_PER_PATIENT,''',
    marker='"records_per_patient"',
)


def verify() -> bool:
    sys.path.insert(0, str(ROOT))
    for m in [m for m in sys.modules if m.startswith("munji_ai")]:
        del sys.modules[m]
    try:
        from munji_ai import config as C
        from munji_ai.data import windows as W
    except Exception as e:
        print(f"  import failed: {type(e).__name__}: {e}")
        return False

    checks = [
        ("record cap defined", set(C.MAX_RECORDS_PER_PATIENT) ==
         {"quality", "beat", "rhythm", "shockable"}),
        ("beat reads 10 records", C.MAX_RECORDS_PER_PATIENT["beat"] == 10),
        ("hash covers it", "records_per_patient" in
         open(ROOT / "munji_ai/data/windows.py", encoding="utf-8").read()),
    ]
    ok = True
    for n, c in checks:
        print(f"  {'PASS' if c else 'FAIL'}  {n}")
        ok &= bool(c)

    cap = C.MAX_WINDOWS_PER_PATIENT["beat"]
    r = C.MAX_RECORDS_PER_PATIENT["beat"]
    print(f"\n  beat: was 46 records x {cap // 46} windows, "
          f"now {r} records x {cap // r} windows")
    print(f"  records read for 237 train patients: 11,649 -> {237 * r:,}")
    print(f"\n  config hash: {W.config_hash()}")
    return ok


print(__doc__.split("WHY")[0].strip())
print("\n" + "-" * 62)
for n in applied:
    print(f"  applied   {n}")
for n in skipped:
    print(f"  already   {n}")
for n in failed:
    print(f"  FAILED    {n}")
if failed:
    raise SystemExit(1)

print("-" * 62)
print("\nverifying\n")
ok = verify()
print("\n" + "-" * 62)
print("""
Next:
    python tests/test_windows.py
    python -m munji_ai.data.build_cache beat --raw E:\\munji\\raw --cache data\\cache
    python -m munji_ai.data.build_cache quality --raw E:\\munji\\raw --cache data\\cache

Splits are already built and unchanged, so --rebuild-splits is not needed.
Then delete this file.
""".strip() if ok else "Verification failed.")
raise SystemExit(0 if ok else 1)