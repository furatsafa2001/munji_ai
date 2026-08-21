"""Build patient-level splits and stage caches.

    python -m munji_ai.data.build_cache --splits-only
    python -m munji_ai.data.build_cache quality shockable
    python -m munji_ai.data.build_cache all

Splits are written once and reused. Deleting splits.json reshuffles every
assignment and invalidates any comparison with earlier results — don't.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config as C
from . import splits as S
from .registry import REGISTRY, by_stage
from .windows import build_stage, config_hash

STAGES = ("quality", "beat", "rhythm", "shockable")


def available(root: Path) -> list[str]:
    return [k for k in REGISTRY if (Path(root) / k).exists()
            and any((Path(root) / k).rglob("*.hea"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="*", help=f"{', '.join(STAGES)} or 'all'")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--splits-only", action="store_true")
    ap.add_argument("--rebuild-splits", action="store_true")
    a = ap.parse_args(argv)

    have = available(a.raw)
    if not have:
        print(f"no datasets under {a.raw} — run download.py first")
        return 1
    print(f"config hash : {config_hash()}")
    print(f"downloaded  : {', '.join(have)}\n")

    split_file = Path(C.SPLIT_DIR) / "splits.json"
    if split_file.exists() and not a.rebuild_splits:
        manifest = S.load(split_file)
        print(f"splits      : reusing {split_file}")
    else:
        manifest = S.build(have, root=a.raw)
        S.save(manifest, split_file)
        print(f"splits      : built {split_file}")

    for ds, cells in sorted(S.summarize(manifest).items()):
        parts = " ".join(f"{s}={v['records']}r/{v['patients']}p"
                         for s, v in sorted(cells.items()))
        print(f"  {ds:<12} {parts}")

    if a.splits_only:
        return 0

    stages = STAGES if (not a.stages or a.stages == ["all"]) else a.stages
    for stage in stages:
        need = [d.key for d in by_stage(stage)]
        if not set(need) & set(have):
            print(f"\n[{stage}] skipped — needs any of {need}")
            continue
        print(f"\n[{stage}] building (sources: {', '.join(need)})")
        build_stage(stage, manifest, out_dir=a.cache)

    print(f"\ndone. caches in {a.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
