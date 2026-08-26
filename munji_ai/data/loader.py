"""Unified record loader.

Every dataset is read through this one interface, so the rest of the pipeline
never learns which corpus a signal came from. Swapping the training set is a
change to `registry.py` alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..config import RAW_DIR
from . import preprocess as pp
from .registry import SAME_SUBJECT, Dataset, get


@dataclass
class Record:
    dataset: str
    name: str
    patient: str
    signal: np.ndarray          # preprocessed, at TARGET_FS
    fs: int
    beat_samples: np.ndarray    # sample index of each annotated beat
    beat_symbols: np.ndarray    # WFDB beat symbol per annotation
    rhythm_starts: np.ndarray   # sample index where each rhythm segment begins
    rhythm_labels: np.ndarray   # rhythm token, e.g. '(AFIB'

    @property
    def duration_sec(self) -> float:
        return len(self.signal) / self.fs


# WFDB beat symbols -> MUNJI beat classes. Anything unmapped is dropped rather
# than folded into 'normal', so unknown symbols never silently pollute a class.
BEAT_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # normal / bundle branch
    "A": "PAC", "a": "PAC", "J": "PAC", "S": "PAC",     # supraventricular ectopic
    "V": "PVC", "E": "PVC",                              # ventricular ectopic
}

# WFDB rhythm tokens -> MUNJI rhythm classes.
RHYTHM_MAP = {
    "(N": "NSR", "(NSR": "NSR",
    "(AFIB": "AF",
    "(AFL": "AFL",
    "(SVTA": "SVT", "(SBR": "BRADY",
    "(VT": "VT", "(VFL": "VF", "(VFIB": "VF",
    "(B": "BIGEMINY", "(T": "TRIGEMINY",
    "(P": "PACED", "(AB": "OTHER", "(IVR": "OTHER", "(NOD": "OTHER",
    # Ambiguity markers, not rhythms. Kept as intervals so they occupy their
    # own span; windows dominated by them are dropped downstream.
    "(NOISE": "NOISE", "(UNKNOWN": "UNKNOWN",
}


def _patient_id(ds: Dataset, record: str) -> str:
    canonical = SAME_SUBJECT.get((ds.key, record), record)
    m = re.match(ds.patient_re, canonical)
    if not m:
        raise ValueError(
            f"{ds.key}: record {record!r} does not match patient pattern "
            f"{ds.patient_re!r} — registry needs updating for this layout"
        )
    return f"{ds.key}:{m.group(1)}"


def dataset_dir(ds: Dataset, root: Path = RAW_DIR) -> Path:
    return Path(root) / ds.key


def _pick_channel(ds: Dataset, rec) -> int:
    """Prefer a named lead; fall back to the configured index.

    On multi-lead sets the index is a poor default — index 0 is usually a limb
    lead, while MUNJI's CM5 sits at a V5-like chest position. Matching by name
    keeps the training domain honest.
    """
    n = rec.p_signal.shape[1]
    names = [str(s).strip() for s in (getattr(rec, "sig_name", None) or [])]
    for want in ds.channel_names:
        for i, got in enumerate(names):
            if got.lower() == want.lower():
                return i
    return min(ds.channel, n - 1)


def challenge2011_labels(root: Path = RAW_DIR) -> dict[str, str]:
    """Quality labels for the Challenge 2011 benchmark.

    Labels ship as RECORDS-acceptable / RECORDS-unacceptable listings rather
    than WFDB annotations, so they need reading separately.
    """
    d = dataset_dir(get("challenge2011"), root)
    out: dict[str, str] = {}
    for fname, label in (("RECORDS-acceptable", "acceptable"),
                         ("RECORDS-unacceptable", "unacceptable")):
        for f in d.rglob(fname):
            for line in f.read_text().split():
                if line.strip():
                    out[Path(line.strip()).stem] = label
    if not out:
        raise FileNotFoundError(
            f"no RECORDS-acceptable/unacceptable under {d}. Check the download "
            f"layout — labels may sit under set-a/."
        )
    return out


def list_records(dataset: str, root: Path = RAW_DIR) -> list[str]:
    """Discover record names by scanning for .hea headers."""
    ds = get(dataset)
    d = dataset_dir(ds, root)
    if not d.exists():
        raise FileNotFoundError(f"{d} not found — run download.py for {dataset!r} first")
    return sorted(p.stem for p in d.rglob("*.hea"))


def load_record(
    dataset: str,
    name: str,
    root: Path = RAW_DIR,
    path: str = "model",
    with_annotations: bool = True,
) -> Record:
    """Read one record, resample to TARGET_FS, and rescale annotation indices."""
    import wfdb

    ds = get(dataset)
    d = dataset_dir(ds, root)
    matches = list(d.rglob(f"{name}.hea"))
    if not matches:
        raise FileNotFoundError(f"{dataset}/{name} not found under {d}")
    stem = str(matches[0].with_suffix(""))

    rec = wfdb.rdrecord(stem)
    fs_in = int(round(rec.fs))
    ch = _pick_channel(ds, rec)
    raw = np.asarray(rec.p_signal[:, ch], dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    fn = pp.diagnostic_path if path == "diagnostic" else pp.model_path
    sigout = fn(raw, fs_in)

    # Annotation indices are in native-rate samples; rescale to the new rate.
    ratio = len(sigout) / max(len(raw), 1)
    beats_s = np.array([], dtype=np.int64)
    beats_y = np.array([], dtype="<U2")
    rhy_s = np.array([], dtype=np.int64)
    rhy_y = np.array([], dtype=object)

    if with_annotations:
        for ext in ("atr", "ecg", "qrs"):
            try:
                ann = wfdb.rdann(stem, ext)
            except Exception:
                continue
            idx = np.round(np.asarray(ann.sample) * ratio).astype(np.int64)
            sym = np.asarray(ann.symbol, dtype="<U2")

            keep = np.isin(sym, list(BEAT_MAP))
            if keep.any():
                beats_s, beats_y = idx[keep], sym[keep]

            aux = getattr(ann, "aux_note", None)
            if aux:
                aux = np.asarray([str(a).strip("\x00").strip() for a in aux], dtype=object)
                # (NOISE is kept, not filtered. Intervals run until the
                # next marker, so dropping it here would make the preceding
                # rhythm extend across unreadable signal and mislabel it.
                rk = np.array([a.startswith("(") for a in aux])
                if rk.any():
                    rhy_s, rhy_y = idx[rk], aux[rk]

            br = np.isin(sym, ["[", "]"])
            if br.any():
                br_s = idx[br]
                # '[' opens a ventricular fibrillation / flutter episode.
                # ']' closes it, but says nothing about what follows — often
                # asystole, VT, or defibrillation artifact. Calling it sinus
                # rhythm would label a dangerous stretch as safe, so it becomes
                # an explicit unknown and those windows are dropped.
                br_y = np.array(
                    ["(VFIB" if s == "[" else "(UNKNOWN" for s in sym[br]],
                    dtype=object,
                )
                if len(rhy_s):
                    merged_s = np.concatenate([rhy_s, br_s])
                    merged_y = np.concatenate([rhy_y, br_y])
                    order = np.argsort(merged_s, kind="stable")
                    rhy_s, rhy_y = merged_s[order], merged_y[order]
                else:
                    rhy_s, rhy_y = br_s, br_y

                # Without an aux_note timeline the first marker is a '[', so
                # everything before the first episode would carry no label and
                # be dropped — leaving cudb with positives only. The stretch
                # before a first fibrillation onset is by definition not
                # fibrillation, so it is valid negative material.
                if len(rhy_s) and rhy_s[0] > 0:
                    rhy_s = np.concatenate([[0], rhy_s])
                    rhy_y = np.concatenate([["(N"], rhy_y])

            # No rhythm markers anywhere means the annotations declare no
            # change, not that the rhythm is unknown. For a set registered
            # with a baseline that is a positive statement about the whole
            # record; for every other set it stays empty and nothing happens.
            if not len(rhy_s) and ds.baseline_rhythm:
                rhy_s = np.array([0], dtype=np.int64)
                rhy_y = np.array([ds.baseline_rhythm], dtype=object)
            break

    return Record(
        dataset=ds.key,
        name=name,
        patient=_patient_id(ds, name),
        signal=sigout,
        fs=int(pp.TARGET_FS),
        beat_samples=beats_s,
        beat_symbols=beats_y,
        rhythm_starts=rhy_s,
        rhythm_labels=rhy_y,
    )


def iter_records(
    dataset: str,
    root: Path = RAW_DIR,
    names: Optional[list[str]] = None,
    **kw,
) -> Iterator[Record]:
    for n in names if names is not None else list_records(dataset, root):
        try:
            yield load_record(dataset, n, root=root, **kw)
        except Exception as e:  # a corrupt record must not kill a training run
            print(f"[skip] {dataset}/{n}: {type(e).__name__}: {e}")


def beat_classes(rec: Record) -> np.ndarray:
    return np.array([BEAT_MAP.get(s, "?") for s in rec.beat_symbols], dtype="<U8")


def rhythm_classes(rec: Record) -> np.ndarray:
    return np.array(
        [RHYTHM_MAP.get(str(a).split("\x00")[0].strip(), "OTHER") for a in rec.rhythm_labels],
        dtype="<U10",
    )
