"""Window extraction.

Turns loaded records into training-ready arrays and writes one cache per
(stage, split). This is the layer that sits between the loader and the models.

Four stages, four different label shapes:

    quality    (n, 1250)  ->  class per window        NSRDB + NSTDB
    beat       (n, 2000)  ->  mask per sample         Icentia + SVDB
    rhythm     (n, 7500)  ->  class per window        Icentia + LTAFDB
    shockable  (n, 1250)  ->  binary per window       CUDB + VFDB

Caches are kept separate rather than merged: window lengths differ, label
shapes differ, and merging would force loading 40 GB to train a gate that
needs 2 GB.

Every cache records the config hash it was built under. Change a filter cutoff
or a sampling rate and the loader refuses stale caches instead of silently
training on mismatched data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .. import config as C
from . import augment, preprocess as pp
from .loader import Record, beat_classes, iter_records, list_records
from .registry import get

# --------------------------------------------------------------------- hashing


def config_hash() -> str:
    """Fingerprint of every config value a cache depends on."""
    payload = json.dumps({
        "fs": C.TARGET_FS,
        "model_band": C.MODEL_BAND,
        "mains": C.MAINS_HZ,
        "order": C.FILTER_ORDER,
        "windows": C.WINDOW_SEC,
        "strides": C.WINDOW_STRIDE_SEC,
        "qrs": C.QRS_HALFWIDTH_MS,
        "beat_classes": C.BEAT_CLASSES,
        "purity": [C.RHYTHM_PURITY, C.SHOCKABLE_PURITY],
        "shockable_rhythms": sorted(C.SHOCKABLE_RHYTHMS),
        "vt_fast_bpm": C.VT_FAST_BPM,
        "ambiguous": sorted(C.AMBIGUOUS_RHYTHMS),
        "shockable_excluded": sorted(C.SHOCKABLE_EXCLUDED),
        "quality_classes": sorted(C.QUALITY_CLASSES),
        "quality_bands": augment.QUALITY_BANDS,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ----------------------------------------------------------------- mask paint


def paint_mask(length: int, beat_idx: np.ndarray, beat_cls: np.ndarray,
               fs: int = C.TARGET_FS) -> np.ndarray:
    """Per-sample class mask for one window.

    0 = background, then one id per beat class. Overlapping regions resolve to
    the ectopic class, since ectopics are the rare and clinically interesting
    label and must not be erased by an adjacent normal beat.
    """
    mask = np.zeros(length, dtype=np.uint8)
    for idx, cls in zip(beat_idx, beat_cls):
        if cls not in C.QRS_HALFWIDTH_MS:
            continue
        half = int(round(C.QRS_HALFWIDTH_MS[cls] * fs / 1000.0))
        lo, hi = max(int(idx) - half, 0), min(int(idx) + half + 1, length)
        if hi <= lo:
            continue
        cid = C.BEAT_CLASS_ID[cls]
        region = mask[lo:hi]
        mask[lo:hi] = np.where(region == 0, cid, np.maximum(region, cid))
    return mask


# ------------------------------------------------------------ rhythm intervals


def rhythm_intervals(rec: Record) -> list[tuple[int, int, str]]:
    """Rhythm annotations are sparse markers that persist until the next one."""
    from .loader import rhythm_classes

    if len(rec.rhythm_starts) == 0:
        return []
    labels = rhythm_classes(rec)
    starts = np.asarray(rec.rhythm_starts, dtype=np.int64)
    order = np.argsort(starts)
    starts, labels = starts[order], labels[order]
    ends = np.append(starts[1:], len(rec.signal))
    return [(int(s), int(e), str(l)) for s, e, l in zip(starts, ends, labels) if e > s]


def dominant_label(intervals, lo: int, hi: int, purity: float,
                   excluded=()) -> str | None:
    """Label covering at least `purity` of [lo, hi); otherwise None.

    `excluded` names labels that mean "we do not know", not a rhythm. A window
    dominated by one of them returns None and is dropped rather than assigned
    to a class. Assigning it would be worse than discarding it: in an alerting
    system, ambiguity labelled 'safe' is a miss waiting to happen.
    """
    if not intervals:
        return None
    span = hi - lo
    cover: Counter = Counter()
    for s, e, lab in intervals:
        overlap = min(hi, e) - max(lo, s)
        if overlap > 0:
            cover[lab] += overlap
    if not cover:
        return None
    lab, amount = cover.most_common(1)[0]
    if lab in excluded:
        return None
    return lab if amount / span >= purity else None


# ------------------------------------------------------------------ extractors


def _frames(rec: Record, stage: str):
    win, stride = C.window_samples(stage), C.stride_samples(stage)
    n = 1 + (len(rec.signal) - win) // stride if len(rec.signal) >= win else 0
    for i in range(n):
        lo = i * stride
        yield lo, lo + win


def extract_beat(rec: Record, rng, cap: int):
    """Windows with per-sample masks. Ectopic-bearing windows are prioritised."""
    cls = beat_classes(rec)
    keep = cls != "?"
    idx, cls = rec.beat_samples[keep], cls[keep]
    if len(idx) == 0:
        return [], [], []

    pos, neg = [], []
    for lo, hi in _frames(rec, "beat"):
        sel = (idx >= lo) & (idx < hi)
        if not sel.any():
            continue
        w_cls = cls[sel]
        item = (lo, hi, idx[sel] - lo, w_cls)
        (pos if np.isin(w_cls, ("PAC", "PVC")).any() else neg).append(item)

    rng.shuffle(neg)
    n_neg = min(len(neg), int(len(pos) * C.NEGATIVE_KEEP_RATE["beat"] /
                             max(1 - C.NEGATIVE_KEEP_RATE["beat"], 1e-6)))
    chosen = pos + neg[:n_neg] if pos else neg[: max(1, cap // 4)]
    rng.shuffle(chosen)
    chosen = chosen[:cap]

    X, Y, M = [], [], []
    for lo, hi, rel, wcls in chosen:
        X.append(pp.normalize(rec.signal[lo:hi]))
        Y.append(paint_mask(hi - lo, rel, wcls))
        M.append(lo)
    return X, Y, M


def extract_rhythm(rec: Record, rng, cap: int):
    iv = rhythm_intervals(rec)
    if not iv:
        return [], [], []
    pos, neg = [], []
    for lo, hi in _frames(rec, "rhythm"):
        lab = dominant_label(iv, lo, hi, C.RHYTHM_PURITY,
                             excluded=C.AMBIGUOUS_RHYTHMS)
        if lab is None:
            continue
        cls = lab if lab in C.RHYTHM_CLASSES else ("AF" if lab == "AFL" else "OTHER")
        (pos if cls == "AF" else neg).append((lo, hi, cls))

    rng.shuffle(neg)
    n_neg = (min(len(neg), max(int(len(pos) * 1.5), max(1, cap // 4)))
             if pos else max(1, cap // 4))
    chosen = pos + neg[:n_neg]
    rng.shuffle(chosen)
    chosen = chosen[:cap]

    X = [pp.normalize(rec.signal[lo:hi]) for lo, hi, _ in chosen]
    return X, [c for _, _, c in chosen], [lo for lo, _, _ in chosen]


SHOCKABLE = set(C.SHOCKABLE_RHYTHMS)


def _episode_bpm(sig, fs: int):
    """Median heart rate inside one episode, or None if unmeasurable.

    VFDB carries no beat annotations at all, so the rate cannot come from
    the label file — it has to be measured off the signal. Peaks are taken
    on the rectified trace because ventricular complexes are often
    predominantly negative on a single lead.
    """
    from scipy.signal import find_peaks

    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < fs:
        return None
    x = np.abs(sig - np.median(sig))
    scale = float(np.median(x)) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        return None
    peaks, _ = find_peaks(x, distance=max(1, int(0.2 * fs)), prominence=2.0 * scale)
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / float(fs)
    rr = rr[rr > 0]
    if not len(rr):
        return None
    return float(60.0 / np.median(rr))


def split_vt_by_rate(rec: Record, intervals):
    """Relabel VT episodes as VT_FAST or VT_SLOW using the measured rate."""
    out = []
    for s, e, lab in intervals:
        if lab != "VT":
            out.append((s, e, lab))
            continue
        bpm = _episode_bpm(rec.signal[s:e], rec.fs)
        if bpm is None:
            out.append((s, e, "VT"))
        elif bpm >= C.VT_FAST_BPM:
            out.append((s, e, "VT_FAST"))
        else:
            out.append((s, e, "VT_SLOW"))
    return out


def extract_shockable(rec: Record, rng, cap: int):
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    if not iv:
        return [], [], []
    pos, neg = [], []
    for lo, hi in _frames(rec, "shockable"):
        lab = dominant_label(iv, lo, hi, C.SHOCKABLE_PURITY,
                             excluded=C.SHOCKABLE_EXCLUDED)
        if lab is None:
            continue
        cls = "shockable" if lab in SHOCKABLE else "not_shockable"
        (pos if cls == "shockable" else neg).append((lo, hi, cls))

    rng.shuffle(neg)
    chosen = pos + neg[: max(len(pos), 1, cap // 4)]
    rng.shuffle(chosen)
    chosen = chosen[:cap]

    X = [pp.normalize(rec.signal[lo:hi]) for lo, hi, _ in chosen]
    return X, [c for _, _, c in chosen], [lo for lo, _, _ in chosen]


def extract_quality(rec: Record, rng, cap: int, noises: dict):
    """Clean windows plus noise-injected copies at known SNR.

    SNR is chosen, so the label is exact rather than a human judgement call.
    """
    win, stride = C.window_samples("quality"), C.stride_samples("quality")
    frames = pp.sliding_windows(rec.signal, win, stride)
    if len(frames) == 0:
        return [], [], []
    take = rng.permutation(len(frames))[:cap]
    X, Y, M = [], [], []
    for i in take:
        w = frames[i]
        if rng.random() < 0.30:
            X.append(pp.normalize(w))
            Y.append("usable")
        else:
            noisy, _, lab = augment.augment_window(w, noises, rng)
            X.append(pp.normalize(noisy))
            Y.append(lab)
        M.append(int(i) * stride)
    return X, Y, M


# ---------------------------------------------------------------------- build

EXTRACTORS = {"beat": extract_beat, "rhythm": extract_rhythm,
              "shockable": extract_shockable, "quality": extract_quality}


def build_stage(stage: str, manifest: dict, out_dir: Path | None = None,
                raw_dir: Path | None = None, seed: int = 0,
                verbose: bool = True) -> dict[str, Path]:
    """Build train/val/test caches for one stage.

    `raw_dir` must be threaded through to both the noise loader and the record
    iterator. Defaulting either of them to config silently reads from the wrong
    disk when raw data lives on external storage.
    """
    from .registry import by_stage

    out_dir = Path(out_dir or C.CACHE_DIR)
    raw_dir = Path(raw_dir or C.RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only build from datasets that are actually downloaded — a registered but
    # absent benchmark (challenge2011) should not abort the whole build.
    present = {e["dataset"] for e in manifest.values()}
    datasets = [d.key for d in by_stage(stage) if d.key in present]
    if not datasets:
        raise FileNotFoundError(
            f"no downloaded datasets feed stage {stage!r}. Expected any of "
            f"{[d.key for d in by_stage(stage)]}"
        )
    noises = augment.load_noise(raw_dir) if stage == "quality" else None
    cap = C.MAX_WINDOWS_PER_PATIENT[stage]
    written: dict[str, Path] = {}

    for split in ("train", "val", "test"):
        entries = [e for e in manifest.values()
                   if e["split"] == split and e["dataset"] in datasets]
        by_ds: dict[str, list] = {}
        for e in entries:
            by_ds.setdefault(e["dataset"], []).append(e)

        X, Y, meta_ds, meta_pat, meta_rec, meta_off = [], [], [], [], [], []
        budget = C.MAX_WINDOWS_PER_STAGE[stage]

        for ds_key, items in by_ds.items():
            limit = C.SAMPLE_PATIENTS.get(ds_key)
            if limit:
                pats = sorted({e["patient"] for e in items})
                rng0 = np.random.default_rng(C.SPLIT_SEED)
                keep = set(rng0.permutation(pats)[:limit])
                items = [e for e in items if e["patient"] in keep]

            names = [e["record"] for e in items]
            lookup = {e["record"]: e for e in items}
            # Share each patient's budget across their records. Giving every
            # record the full cap lets a patient with 50 segments contribute
            # 50x what a single-record patient does; spending it all on the
            # first record would take every window from one stretch of signal.
            per_patient = Counter(e["patient"] for e in items)
            spent: Counter = Counter()
            for rec in iter_records(ds_key, root=raw_dir, names=names):
                if len(X) >= budget:
                    break
                n_rec = max(per_patient[rec.patient], 1)
                room_pat = cap - spent[rec.patient]
                if room_pat <= 0:
                    continue
                per_rec = max(1, min(room_pat, cap // n_rec))
                rng = np.random.default_rng(abs(hash((seed, rec.patient))) % 2**32)
                fn = EXTRACTORS[stage]
                out = (fn(rec, rng, per_rec, noises) if stage == "quality"
                       else fn(rec, rng, per_rec))
                xs, ys, offs = out
                room = budget - len(X)
                xs, ys, offs = xs[:room], ys[:room], offs[:room]
                X.extend(xs)
                Y.extend(ys)
                spent[rec.patient] += len(xs)
                e = lookup[rec.name]
                meta_ds += [ds_key] * len(xs)
                meta_pat += [e["patient"]] * len(xs)
                meta_rec += [rec.name] * len(xs)
                meta_off += offs

        if not X:
            if verbose:
                print(f"  [{stage}/{split}] no windows — datasets missing?")
            continue

        Xa = np.asarray(X, dtype=C.CACHE_DTYPE)
        Ya = (np.asarray(Y, dtype=np.uint8) if stage == "beat"
              else np.asarray(Y, dtype="<U13"))
        path = out_dir / f"{stage}_{split}.npz"
        np.savez(
            path, X=Xa, y=Ya,
            dataset=np.asarray(meta_ds), patient=np.asarray(meta_pat),
            record=np.asarray(meta_rec), offset=np.asarray(meta_off, dtype=np.int64),
            cfg_hash=np.asarray(config_hash()), stage=np.asarray(stage),
        )
        written[split] = path
        if verbose:
            mb = path.stat().st_size / 1e6
            dist = (dict(Counter(Ya.ravel().tolist())) if stage == "beat"
                    else dict(Counter(Ya.tolist())))
            print(f"  [{stage}/{split}] {Xa.shape} {mb:.0f} MB "
                  f"patients={len(set(meta_pat))} {dist}")
    return written


def load_cache(stage: str, split: str, cache_dir: Path | None = None,
               check_hash: bool = True) -> dict:
    p = Path(cache_dir or C.CACHE_DIR) / f"{stage}_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run build_stage({stage!r}) first")
    d = np.load(p, allow_pickle=False)
    if check_hash and str(d["cfg_hash"]) != config_hash():
        raise RuntimeError(
            f"{p.name} was built under config {str(d['cfg_hash'])}, current is "
            f"{config_hash()}. Rebuild the cache — preprocessing changed."
        )
    return {k: d[k] for k in d.files}