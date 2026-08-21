"""Fetch registered datasets from PhysioNet.

Run locally — PhysioNet is not reachable from every environment.

    python -m munji_ai.data.download --list
    python -m munji_ai.data.download mitdb afdb nstdb cudb vfdb
    python -m munji_ai.data.download icentia11k          # large, hours

Downloads are idempotent: an existing directory is skipped unless --force.
Start icentia11k first — it is the long pole and runs unattended.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import RAW_DIR
from .registry import REGISTRY, get


def download(key: str, root: Path = RAW_DIR, force: bool = False) -> Path:
    import wfdb

    ds = get(key)
    dest = Path(root) / ds.key
    if dest.exists() and any(dest.rglob("*.hea")) and not force:
        print(f"[skip] {key} already present at {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {key} -> {dest}  (physionet: {ds.slug} v{ds.version})")
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
    a = ap.parse_args(argv)

    if a.list or not a.datasets:
        w = max(len(k) for k in REGISTRY)
        for k, d in REGISTRY.items():
            print(f"{k:<{w}}  {d.role:<5}  {d.fs:>4} Hz  {d.note.splitlines()[0][:70]}")
        return 0

    keys = list(REGISTRY) if a.datasets == ["all"] else a.datasets
    for k in keys:
        try:
            download(k, a.root, a.force)
        except Exception as e:
            print(f"[fail] {k}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
