"""The ECG Fragment Database, found in our own VFDB copy by its waveform.

    from munji_ai.data.fragments import locate
    found, skipped, best_r, scales = locate(raw / "ecg_fragments", raw)
    for f in found:            # {"record", "label", "centre_s", "r"}
        ...

PhysioNet's ECG Fragment Database (ecg-fragment-high-risk-label) re-labels
VFDB in short fragments, reviewed by cardiologists of the Almazov Centre, and
splits VT into high rate (VTHR), low rate (VTLR) and torsade de pointes
(VTTdP), beside flutter (VFL) and fibrillation (VF). Each fragment's file name
gives its VFDB record and a second, but on a 360 Hz time base although VFDB is
250 Hz: the true time is 1.44 times the name's (shockable_labels_check.py
found 472 of 474 fragments this way, median correlation 0.99). So each
fragment is looked for near every time its name could mean (ALIGN_SCALES),
in both VFDB channels, by normalised cross-correlation of the 1-30 Hz band,
and kept only when the best match reaches ALIGN_MIN_R. centre_s is then the
fragment's true centre in the VFDB recording, in seconds: the same time in
our resampled copy (load_record), so windows are cut from our own signal and
nothing is read at the fragment's own sampling rate.

Test-split records are skipped, as everywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np

FRAGMENT = re.compile(r"^(\d+)_C_([A-Za-z]+)_(\d+)s_frag$", re.IGNORECASE)
ALIGN_SCALES = (1.0, 360.0 / 250.0, 250.0 / 360.0)   # the name's seconds as given, or read at the other rate
ALIGN_SEARCH_SEC = 10.0
ALIGN_MIN_R = 0.8


def read_fragment(path: Path):
    """(samples, sampling rate if the file states one) of one fragment file."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".hea":
            import wfdb
            r = wfdb.rdrecord(str(path.with_suffix("")))
            return np.asarray(r.p_signal[:, 0], np.float64), float(r.fs)
        if suffix == ".mat":
            from scipy.io import loadmat
            arrays = [np.asarray(v, np.float64).ravel() for k, v in loadmat(str(path)).items()
                      if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)]
            return (max(arrays, key=len), None) if arrays else (None, None)
        vals = []
        for line in path.read_text(errors="ignore").split():
            try:
                vals.append(float(line.replace(",", ".")))
            except ValueError:
                continue
        return (np.asarray(vals, np.float64), None) if vals else (None, None)
    except Exception:
        return None, None


def bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """1-30 Hz, zero phase: the band the matching runs in."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, [1.0 / (fs / 2), 30.0 / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def best_match(seg: np.ndarray, tpl: np.ndarray):
    """Highest Pearson correlation of tpl against every position in seg, and where."""
    n = len(tpl)
    if len(seg) < n or n < 10:
        return -1.0, 0
    t = tpl - tpl.mean()
    t = t / (np.sqrt((t * t).sum()) + 1e-12)
    c = np.correlate(seg, t, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
    s1, s2 = cs[n:] - cs[:-n], cs2[n:] - cs2[:-n]
    r = c / np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-12))
    k = int(np.argmax(r))
    return float(r[k]), k


def fragment_files(root: Path) -> dict:
    """(record, LABEL, second) -> file, the frag folders preferred."""
    files = {}
    for path in sorted(Path(root).rglob("*")):
        m = FRAGMENT.match(path.stem)
        if m and path.is_file() and path.suffix.lower() != ".dat":
            key = (m.group(1), m.group(2).upper(), int(m.group(3)))
            if key not in files or path.parent.name.lower() == "frag":
                files[key] = path
    return files


def locate(root: Path, raw: Path):
    """Every fragment found in our VFDB copy.

    Returns (found, skipped, best_r, scales): found is a list of
    {"record", "label", "centre_s", "r"}; skipped counts the fragments left
    out, by reason; best_r is every fragment's best correlation; scales
    counts how the name's seconds mapped to VFDB time (scale, rate, channel).
    """
    import wfdb
    from scipy.signal import resample_poly

    from .loader import _patient_id, dataset_dir
    from .registry import get
    from .splits import assign

    ds = get("vfdb")
    chans_of, found, skipped, best_r, scales = {}, [], Counter(), [], Counter()
    for (rec_name, lab, sec), path in sorted(fragment_files(root).items()):
        try:
            if assign(_patient_id(ds, rec_name)) == "test":
                skipped["test split"] += 1
                continue
        except ValueError:
            skipped["not a VFDB record"] += 1
            continue
        if rec_name not in chans_of:
            try:
                hit = next(iter(dataset_dir(ds, raw).rglob(f"{rec_name}.hea")))
                r = wfdb.rdrecord(str(hit.with_suffix("")))
                chans_of[rec_name] = ([bandpass(np.nan_to_num(np.asarray(r.p_signal[:, c], np.float64)), r.fs)
                                       for c in range(r.p_signal.shape[1])], float(r.fs))
            except Exception:
                chans_of[rec_name] = None
        if chans_of[rec_name] is None:
            skipped["VFDB record not found"] += 1
            continue
        chans, fs_raw = chans_of[rec_name]
        vals, fs_file = read_fragment(path)
        if vals is None or len(vals) < 100:
            skipped["fragment unreadable"] += 1
            continue
        best = (-1.0, None)
        for fs_frag in ([fs_file] if fs_file else [360.0, 250.0]):
            up, down = int(fs_raw), int(fs_frag)
            g = gcd(up, down)
            tpl = bandpass(resample_poly(vals, up // g, down // g), fs_raw)
            for k in ALIGN_SCALES:
                a = int(max(0.0, (sec * k - ALIGN_SEARCH_SEC) * fs_raw))
                b = int((sec * k + ALIGN_SEARCH_SEC) * fs_raw) + len(tpl)
                for ch, xc in enumerate(chans):
                    rr, pos = best_match(xc[a:b], tpl)
                    if rr > best[0]:
                        best = (rr, (k, fs_frag, ch, (a + pos + len(tpl) / 2) / fs_raw))
        best_r.append(best[0])
        if best[0] < ALIGN_MIN_R:
            skipped[f"not found in VFDB (best correlation under {ALIGN_MIN_R})"] += 1
            continue
        k, fs_frag, ch, centre = best[1]
        scales[(k, fs_frag, ch)] += 1
        found.append({"record": rec_name, "label": lab, "centre_s": float(centre), "r": best[0]})
    return found, skipped, best_r, scales
