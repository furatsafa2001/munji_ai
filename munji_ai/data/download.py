"""Fetch registered datasets from PhysioNet.

Run locally — PhysioNet is not reachable from every environment.

    python -m munji_ai.data.download --list
    python -m munji_ai.data.download nsrdb nstdb cudb vfdb
    python -m munji_ai.data.download icentia11k          # capped by default

Downloads are idempotent: an existing directory is skipped unless --force.

Datasets with a SAMPLE_PATIENTS entry in config fetch only that many patients.
Icentia has 11k but the pipeline samples 2000, so pulling all of them spends
hours transferring data that is never read. Use --all-patients to override, or
--patients N to change the count.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ..config import RAW_DIR, SAMPLE_PATIENTS
from .registry import REGISTRY, get


def patient_of(ds, record_name: str) -> str:
    """Patient id for a record path as listed by PhysioNet.

    Listings may be bare names ('16265') or nested paths ('p00/p00001_s00').
    Try the registry pattern on the stem first, then fall back to the leading
    directory, which is how the large patient-partitioned sets are organised.
    """
    stem = Path(record_name).name
    m = re.match(ds.patient_re, stem)
    if m:
        return m.group(1)
    parts = Path(record_name).parts
    return parts[0] if len(parts) > 1 else stem


def limited_records(ds, limit: int) -> list[str] | None:
    """Records belonging to the first `limit` patients, or None to fetch all.

    Patients are taken in sorted order, not sampled randomly: the download has
    to be resumable, and a random subset would pick a different set on every
    retry, leaving a partial mess on disk.
    """
    import wfdb

    try:
        names = wfdb.get_record_list(ds.slug)
    except Exception as e:
        print(f"[warn] could not list {ds.slug} ({type(e).__name__}: {e}) — "
              f"falling back to a full download")
        return None

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(patient_of(ds, n), []).append(n)

    if len(groups) <= limit:
        print(f"[note] {ds.key}: {len(groups):,} patients, at or under the "
              f"cap of {limit:,} — fetching all")
        return None

    keep = sorted(groups)[:limit]
    out = [r for p in keep for r in groups[p]]
    print(f"[note] {ds.key}: {len(groups):,} patients available, fetching "
          f"{len(keep):,} ({len(out):,} records). Use --all-patients for the rest.")
    return out


def download(key: str, root: Path = RAW_DIR, force: bool = False,
             patients: int | None = None, all_patients: bool = False) -> Path:
    import wfdb

    ds = get(key)
    dest = Path(root) / ds.key
    if dest.exists() and any(dest.rglob("*.hea")) and not force:
        print(f"[skip] {key} already present at {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {key} -> {dest}  (physionet: {ds.slug} v{ds.version})")

    cap = None if all_patients else (patients or SAMPLE_PATIENTS.get(key))
    records = limited_records(ds, cap) if cap else None

    if records:
        wfdb.dl_database(ds.slug, dl_dir=str(dest), records=records)
    else:
        wfdb.dl_database(ds.slug, dl_dir=str(dest))

    n = len(list(dest.rglob("*.hea")))
    print(f"[done] {key}: {n} records")
    if not ds.verified:
        print(f"[note] {key} record layout is unverified — confirm patient_re "
              f"({ds.patient_re}) matches the actual filenames before training")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="*", help="registry keys, or 'all'")
    ap.add_argument("--root", type=Path, default=RAW_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--patients", type=int, default=None,
                    help="patient cap for this run, overriding config")
    ap.add_argument("--all-patients", action="store_true",
                    help="ignore the cap and fetch every patient")
    a = ap.parse_args(argv)

    if a.list or not a.datasets:
        w = max(len(k) for k in REGISTRY)
        for k, d in REGISTRY.items():
            cap = SAMPLE_PATIENTS.get(k)
            tag = f"   [cap {cap:,} patients]" if cap else ""
            print(f"{k:<{w}}  {d.role:<5}  {d.fs:>4} Hz  "
                  f"{d.note.splitlines()[0][:58]}{tag}")
        return 0

    keys = list(REGISTRY) if a.datasets == ["all"] else a.datasets
    for k in keys:
        try:
            download(k, a.root, a.force, a.patients, a.all_patients)
        except Exception as e:
            print(f"[fail] {k}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())