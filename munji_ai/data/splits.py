"""Patient-level train / val / test splits.

Splitting on records instead of patients is the most common way to produce an
accuracy number that collapses in the field: the same subject's beats appear in
both train and test, and the model scores well by recognising the person rather
than the arrhythmia. Every split here is keyed on patient id.

Assignment is a deterministic hash of the patient id, not a shuffle. Adding a
new dataset or new patients therefore leaves every existing assignment
untouched, so results stay comparable across runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..config import SPLIT_DIR, SPLIT_FRACTIONS, SPLIT_SEED
from .loader import list_records, _patient_id
from .registry import get


def _bucket(patient: str, seed: int = SPLIT_SEED) -> float:
    h = hashlib.sha256(f"{seed}:{patient}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign(patient: str, fractions: dict | None = None) -> str:
    f = fractions or SPLIT_FRACTIONS
    u = _bucket(patient)
    acc = 0.0
    for name in ("train", "val", "test"):
        acc += f[name]
        if u < acc:
            return name
    return "test"


def build(datasets: list[str], root: Path | None = None) -> dict:
    """Map every record in the given datasets to a split.

    Datasets registered with role='val' are forced entirely into test — they
    are benchmarks, and training on any part of them destroys comparability
    with published results.
    """
    manifest: dict[str, dict] = {}
    for key in datasets:
        ds = get(key)
        recs = list_records(key) if root is None else list_records(key, root)
        for name in recs:
            pid = _patient_id(ds, name)
            split = "test" if ds.role == "val" else assign(pid)
            manifest[f"{key}/{name}"] = {"dataset": key, "record": name,
                                         "patient": pid, "split": split}
    _assert_disjoint(manifest)
    return manifest


def _assert_disjoint(manifest: dict) -> None:
    seen: dict[str, str] = {}
    for entry in manifest.values():
        pid, split = entry["patient"], entry["split"]
        if pid in seen and seen[pid] != split:
            raise AssertionError(
                f"patient {pid} appears in both {seen[pid]} and {split} — leak"
            )
        seen[pid] = split


def summarize(manifest: dict) -> dict:
    out: dict = defaultdict(lambda: defaultdict(lambda: {"records": 0, "patients": set()}))
    for e in manifest.values():
        cell = out[e["dataset"]][e["split"]]
        cell["records"] += 1
        cell["patients"].add(e["patient"])
    return {
        d: {s: {"records": v["records"], "patients": len(v["patients"])}
            for s, v in splits.items()}
        for d, splits in out.items()
    }


def save(manifest: dict, path: Path | None = None) -> Path:
    p = Path(path or SPLIT_DIR / "splits.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return p


def load(path: Path | None = None) -> dict:
    p = Path(path or SPLIT_DIR / "splits.json")
    manifest = json.loads(p.read_text())
    _assert_disjoint(manifest)
    return manifest


def records_in(manifest: dict, split: str, dataset: str | None = None) -> list[str]:
    return sorted(
        e["record"] for e in manifest.values()
        if e["split"] == split and (dataset is None or e["dataset"] == dataset)
    )
