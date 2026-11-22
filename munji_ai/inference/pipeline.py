"""Raw ECG in, structured findings out.

    from munji_ai.inference.pipeline import BeatPipeline

    pipe = BeatPipeline()                  # loads both checkpoints
    result = pipe.process(signal, fs=256)  # device rate

    result.quality      per 5 s window: P(usable) and the pass decision
    result.beats        (sample, class) for every beat in a passed window
    result.burden       PVC share over the configured window, with its tier
    result.rejected     how many windows the gate refused, and why

Or from the command line, against a cached window:

    python -m munji_ai.inference.pipeline --cache data/cache --split val --n 20


WHAT THIS IS FOR
----------------
The models are trained and the evaluation tools report on cached windows, but
nothing took a raw signal end to end. The app and the backend need one entry
point with a fixed contract, and the decision layer needs model output in a
shape it can consume.

This is that entry point for the two models in this half of the project: the
quality gate and the beat segmenter. It returns findings, not alerts. Deciding
what to tell a caregiver means weighing these against the rhythm classifier,
the shockable detector and the rate rules, and that arbitration belongs in one
place, not four.


THE ORDER IS THE ARCHITECTURE
-----------------------------
    resample 256 -> 250
    filter 0.5 - 40 Hz, notch 50 Hz
    normalise (median and IQR)
    window into 5 s for the gate
    gate every window
    run the beat segmenter only on 20 s spans the gate passed
    read beats off the mask, drop regions under 60 ms
    accumulate PVC burden

The gate runs first and the segmenter never sees what it refused. Evaluating
the segmenter on windows the gate would have dropped charges its mistakes to
the wrong model, which is why gated_beat.py exists and why this follows the
same order.

Sub-window decisions combine by config.GATE_COMBINER, which is "mean" because
it was measured that way: under "min" the rejected windows scored better than
the kept ones, meaning the gate was discarding windows the segmenter handled.


A KNOWN MISMATCH, MEASURED HERE RATHER THAN ASSUMED
---------------------------------------------------
Training used filtfilt, which runs the filter forwards and backwards and so
has no phase delay. Measured R-peak drift against annotations was 0 samples.

A device cannot do that. Filtering backwards needs the future, so on-device
filtering is causal and shifts peaks. How far has never been measured, and it
matters: the beat masks are drawn around annotated R-peaks, so a systematic
shift moves every predicted beat relative to where it was trained to be.

`causal=True` uses a causal filter. `measure_filter_drift()` reports the shift
in milliseconds on real signal, so the gap stops being an unknown.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sig

from ..config import (BEAT_CLASSES, CACHE_DIR, DEVICE_FS, GATE_COMBINER,
                      GATE_MIN_P, MAINS_HZ, MODEL_BAND, PVC_BURDEN,
                      PVC_BURDEN_MIN_BEATS, PVC_BURDEN_WINDOW_SEC,
                      QUALITY_CLASSES, TARGET_FS, window_samples)
from ..data import preprocess as pp


@dataclass
class QualityWindow:
    start: int          # sample index into the resampled signal
    p_usable: float
    passed: bool


@dataclass
class Beat:
    sample: int         # centre of the mask region
    cls: str            # "N" | "PAC" | "PVC"
    second: float       # position in seconds, for readability


@dataclass
class Burden:
    pvc: int
    total: int
    fraction: float
    tier: str           # "" | "info" | "warning" | "critical"
    reliable: bool      # False when too few beats to divide by


@dataclass
class PipelineResult:
    fs: int
    duration_sec: float
    quality: list[QualityWindow] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)
    burden: Burden | None = None
    rejected_windows: int = 0
    analysed_spans: int = 0
    skipped_spans: int = 0

    @property
    def coverage(self) -> float:
        """Share of 5 s windows the gate passed. The blind spot is 1 - this."""
        return (sum(q.passed for q in self.quality) / len(self.quality)
                if self.quality else 0.0)

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in BEAT_CLASSES if c != "background"}
        for b in self.beats:
            out[b.cls] = out.get(b.cls, 0) + 1
        return out


# --------------------------------------------------------------- filtering


def causal_model_path(x: np.ndarray, fs_in: int) -> np.ndarray:
    """The filter a device can actually run: forward only.

    `pp.model_path` uses filtfilt, which is zero-phase and needs the whole
    signal. Real-time filtering sees only the past, so it introduces a delay
    that shifts every peak later in time.
    """
    x = pp.resample_to(x, fs_in)
    nyq = TARGET_FS / 2.0
    lo, hi = MODEL_BAND
    sos = sig.butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq],
                     btype="band", output="sos")
    y = sig.sosfilt(sos, x)
    if MAINS_HZ < nyq:
        b, a = sig.iirnotch(MAINS_HZ, 30.0, TARGET_FS)
        y = sig.lfilter(b, a, y)
    return y


def measure_filter_drift(x: np.ndarray, fs_in: int = DEVICE_FS) -> dict:
    """How far the causal filter moves R peaks relative to the trained one.

    Both filters run on the same signal and their peak positions are matched
    one to one. The median shift is the systematic delay; the spread tells you
    whether it is a constant offset that could be corrected or a smear that
    could not.
    """
    from ..features.sqi import detect_r_pantompkins

    zero = pp.model_path(x, fs_in)
    caus = causal_model_path(x, fs_in)
    pz = detect_r_pantompkins(zero)
    pc = detect_r_pantompkins(caus)
    if len(pz) < 3 or len(pc) < 3:
        return {"n": 0}

    used = np.zeros(len(pc), dtype=bool)
    shifts = []
    tol = int(0.20 * TARGET_FS)
    for p in pz:
        d = np.abs(pc.astype(float) - float(p))
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            shifts.append(pc[j] - p)
    if not shifts:
        return {"n": 0}

    s = np.array(shifts, dtype=float)
    to_ms = 1000.0 / TARGET_FS
    return {
        "n": len(s),
        "median_ms": float(np.median(s) * to_ms),
        "p10_ms": float(np.percentile(s, 10) * to_ms),
        "p90_ms": float(np.percentile(s, 90) * to_ms),
        "spread_ms": float((np.percentile(s, 90) - np.percentile(s, 10)) * to_ms),
    }


# ---------------------------------------------------------------- pipeline


class BeatPipeline:
    """Quality gate and beat segmenter behind one call.

    Both checkpoints load once. `process` is stateless, so the same instance
    serves many signals.
    """

    def __init__(self, gate_ckpt: Path = Path("checkpoints/quality_gate.pt"),
                 beat_ckpt: Path = Path("checkpoints/beat_segmenter.pt"),
                 device: str = "cpu", causal: bool = False):
        from ..models.beat_segmenter import build as build_beat
        from ..models.quality_gate import build as build_gate

        self.device = torch.device(device)
        self.causal = causal
        self.qlen = window_samples("quality")
        self.blen = window_samples("beat")
        self.usable_idx = QUALITY_CLASSES.index("usable")

        for p, what in ((gate_ckpt, "quality gate"), (beat_ckpt, "beat segmenter")):
            if not Path(p).exists():
                raise FileNotFoundError(f"no {what} checkpoint at {p}")

        g = torch.load(gate_ckpt, map_location=self.device, weights_only=False)
        self.gate = build_gate().to(self.device)
        self.gate.load_state_dict(g["state_dict"])
        self.gate.eval()

        b = torch.load(beat_ckpt, map_location=self.device, weights_only=False)
        self.beat = build_beat().to(self.device)
        self.beat.load_state_dict(b["state_dict"])
        self.beat.eval()

        self.gate_epoch = g.get("epoch")
        self.beat_epoch = b.get("epoch")

    # ------------------------------------------------------------- stages
    def _prepare(self, x: np.ndarray, fs: int) -> np.ndarray:
        fn = causal_model_path if self.causal else pp.model_path
        return fn(np.asarray(x, dtype=np.float64), fs)

    @torch.no_grad()
    def _gate(self, sigal: np.ndarray) -> list[QualityWindow]:
        n = len(sigal) // self.qlen
        if n == 0:
            return []
        frames = sigal[: n * self.qlen].reshape(n, self.qlen)
        frames = np.stack([pp.normalize(f) for f in frames])
        xb = torch.from_numpy(frames.astype(np.float32)).to(self.device)

        probs = []
        for i in range(0, len(xb), 256):
            p = torch.softmax(self.gate(xb[i:i + 256]), dim=-1)[:, self.usable_idx]
            probs.append(p.cpu().numpy())
        p_usable = np.concatenate(probs)

        thr = GATE_MIN_P["morphology"]
        return [QualityWindow(i * self.qlen, float(p), bool(p >= thr))
                for i, p in enumerate(p_usable)]

    def _span_passes(self, quality: list[QualityWindow], start: int) -> bool:
        """Does the gate pass the 20 s span beginning at `start`?

        A beat span covers several gate windows. They combine by
        config.GATE_COMBINER rather than by whichever rule seems reasonable at
        the call site, so evaluation and deployment agree.
        """
        n_sub = self.blen // self.qlen
        idx = start // self.qlen
        sub = quality[idx: idx + n_sub]
        if len(sub) < n_sub:
            return False
        ps = np.array([q.p_usable for q in sub])
        thr = GATE_MIN_P["morphology"]
        return bool((ps.mean() if GATE_COMBINER == "mean" else ps.min()) >= thr)

    @torch.no_grad()
    def _beats(self, sigal: np.ndarray, quality: list[QualityWindow],
               result: PipelineResult) -> list[Beat]:
        from ..models.beat_segmenter import clean_mask
        from ..train.train_beat import mask_to_events

        beats = []
        for start in range(0, len(sigal) - self.blen + 1, self.blen):
            if not self._span_passes(quality, start):
                result.skipped_spans += 1
                continue
            result.analysed_spans += 1

            w = pp.normalize(sigal[start:start + self.blen])
            xb = torch.from_numpy(w.astype(np.float32)[None, :]).to(self.device)
            mask = clean_mask(self.beat(xb).argmax(dim=1).cpu().numpy())[0]

            for pos, cls in mask_to_events(mask):
                s = start + pos
                beats.append(Beat(s, cls, round(s / TARGET_FS, 2)))
        return beats

    def _burden(self, beats: list[Beat], duration: float) -> Burden:
        """PVC share, with a floor on how few beats the fraction may rest on.

        Burden is defined clinically over 24 hours. Over a short span the same
        percentage means something else entirely: twenty seconds of bigeminy
        is 50% and is not a finding. The reliability flag carries that, and
        the tier is withheld when it is not met.
        """
        total = len(beats)
        pvc = sum(1 for b in beats if b.cls == "PVC")
        frac = pvc / total if total else 0.0
        reliable = total >= PVC_BURDEN_MIN_BEATS and duration >= 60.0

        tier = ""
        if reliable:
            for name in ("critical", "warning", "info"):
                if frac >= PVC_BURDEN[name]:
                    tier = name
                    break
        return Burden(pvc, total, float(frac), tier, reliable)

    # -------------------------------------------------------------- public
    def process(self, x: np.ndarray, fs: int = DEVICE_FS) -> PipelineResult:
        sigal = self._prepare(x, fs)
        res = PipelineResult(fs=TARGET_FS,
                             duration_sec=len(sigal) / TARGET_FS)
        res.quality = self._gate(sigal)
        res.rejected_windows = sum(1 for q in res.quality if not q.passed)
        res.beats = self._beats(sigal, res.quality, res)
        res.burden = self._burden(res.beats, res.duration_sec)
        return res


# ------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--n", type=int, default=20,
                    help="how many cached windows to stitch into one signal")
    ap.add_argument("--causal", action="store_true",
                    help="use the filter a device can run")
    ap.add_argument("--gate-ckpt", type=Path,
                    default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--beat-ckpt", type=Path,
                    default=Path("checkpoints/beat_segmenter.pt"))
    a = ap.parse_args(argv)

    from ..data.windows import load_cache

    d = load_cache("beat", a.split, a.cache)
    X = np.asarray(d["X"][: a.n], dtype=np.float64)
    raw = X.reshape(-1)

    pipe = BeatPipeline(a.gate_ckpt, a.beat_ckpt, causal=a.causal)
    print(f"gate ckpt epoch {pipe.gate_epoch} | beat ckpt epoch {pipe.beat_epoch}")
    print(f"filter    {'causal (device)' if a.causal else 'zero-phase (training)'}")

    res = pipe.process(raw, fs=TARGET_FS)

    print(f"\nsignal    {res.duration_sec:.0f} s at {res.fs} Hz")
    print(f"quality   {len(res.quality)} windows, "
          f"{res.rejected_windows} rejected, coverage {res.coverage:.1%}")
    print(f"spans     {res.analysed_spans} analysed, {res.skipped_spans} skipped")

    c = res.counts()
    print(f"\nbeats     {len(res.beats):,}   " +
          "  ".join(f"{k} {v:,}" for k, v in c.items()))

    b = res.burden
    print(f"burden    {b.pvc}/{b.total} = {b.fraction:.1%}"
          f"   tier: {b.tier or 'none'}"
          f"   {'' if b.reliable else '(too few beats to be reliable)'}")

    print("\n" + "=" * 62)
    print("CAUSAL FILTER DRIFT")
    print("=" * 62)
    drift = measure_filter_drift(raw, TARGET_FS)
    if drift["n"]:
        print(f"  matched peaks   {drift['n']:,}")
        print(f"  median shift    {drift['median_ms']:+.1f} ms")
        print(f"  p10 to p90      {drift['p10_ms']:+.1f} to {drift['p90_ms']:+.1f} ms"
              f"   (spread {drift['spread_ms']:.1f} ms)")
        print(f"""
  The device filters causally and cannot do what training did. A constant
  median shift is correctable by offsetting the mask; a wide spread is not,
  because then the delay depends on the beat and no single offset fixes it.

  Mask half-widths are 50 ms for N and PAC, 70 ms for PVC, so a shift beyond
  roughly 50 ms moves a predicted beat clear of the region it was trained on.
""".strip())
    else:
        print("  not enough matched peaks to measure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
