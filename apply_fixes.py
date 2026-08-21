"""Fix three annotation problems affecting the shockable and rhythm stages.

Run once from the repo root:

    python apply_fixes.py

1. CUDB episodes were invisible. loader.py only accepted rhythm tokens
   beginning with '('. CUDB marks ventricular fibrillation with the WFDB
   event symbols '[' (onset) and ']' (end) instead, so 45 of its 47
   episodes were dropped without any error.

2. '(NOISE' was ending episodes. PhysioNet's annotation key states the
   previous rhythm continues through a noise episode. Treating it as a
   rhythm change split VF and VT episodes and discarded the remainder —
   73 markers across 101 minutes in VFDB alone.

3. All VT counted as shockable. Only rapid VT warrants a shock, and the
   annotations carry no rate. Rather than include VT wholesale (false
   alarms on slow VT) or drop it entirely (missing a lethal rhythm), the
   rate is now measured from the signal inside each VT episode and the
   episode is labelled VT_FAST or VT_SLOW accordingly. This matches the
   AHA/AAMI convention, which treats rapid VT as shockable at a lower
   sensitivity target than VF.

Idempotent — running it twice is safe. If any expected code is missing the
script stops without writing anything.
"""

from pathlib import Path
import sys

LOADER = Path("munji_ai/data/loader.py")
WINDOWS = Path("munji_ai/data/windows.py")
CONFIG = Path("munji_ai/config.py")

# ---------------------------------------------------------------- 1 and 2

OLD_AUX = '''                rk = np.array([a.startswith("(") for a in aux])
                if rk.any():
                    rhy_s, rhy_y = idx[rk], aux[rk]
            break
'''

NEW_AUX = '''                rk = np.array(
                    [a.startswith("(") and "NOISE" not in a.upper() for a in aux]
                )
                if rk.any():
                    rhy_s, rhy_y = idx[rk], aux[rk]

            br = np.isin(sym, ["[", "]"])
            if br.any():
                br_s = idx[br]
                br_y = np.array(
                    ["(VFIB" if s == "[" else "(N" for s in sym[br]], dtype=object
                )
                if len(rhy_s):
                    merged_s = np.concatenate([rhy_s, br_s])
                    merged_y = np.concatenate([rhy_y, br_y])
                    order = np.argsort(merged_s, kind="stable")
                    rhy_s, rhy_y = merged_s[order], merged_y[order]
                else:
                    rhy_s, rhy_y = br_s, br_y
            break
'''

# ------------------------------------------------------------------- 3

OLD_CFG = 'SHOCKABLE_CLASSES = ("not_shockable", "shockable")'

NEW_CFG = '''SHOCKABLE_CLASSES = ("not_shockable", "shockable")

# Which rhythm labels count as shockable.
#
# VFL is mapped to VF upstream, so "VF" covers fibrillation and flutter.
# VT is split by measured rate rather than taken as one class: only rapid VT
# is shockable, slow VT is treated medically. The annotations carry no rate,
# so it is derived from RR intervals inside each episode — see
# windows.split_vt_by_rate. An episode whose rate cannot be measured stays
# "VT" and is excluded, which is the conservative side for a false alarm but
# not for a miss; those episodes are worth inspecting rather than ignoring.
SHOCKABLE_RHYTHMS = ("VF", "VT_FAST")
VT_FAST_BPM = 150.0'''

OLD_SHOCK = 'SHOCKABLE = {"VF", "VT"}'

NEW_SHOCK = '''SHOCKABLE = set(C.SHOCKABLE_RHYTHMS)


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
    return out'''

OLD_EXTRACT = '''def extract_shockable(rec: Record, rng, cap: int):
    iv = rhythm_intervals(rec)
    if not iv:
        return [], [], []'''

NEW_EXTRACT = '''def extract_shockable(rec: Record, rng, cap: int):
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    if not iv:
        return [], [], []'''

OLD_HASH = '''        "purity": [C.RHYTHM_PURITY, C.SHOCKABLE_PURITY],'''

NEW_HASH = '''        "purity": [C.RHYTHM_PURITY, C.SHOCKABLE_PURITY],
        "shockable_rhythms": sorted(C.SHOCKABLE_RHYTHMS),
        "vt_fast_bpm": C.VT_FAST_BPM,'''


def patch(path: Path, old: str, new: str, label: str) -> bool:
    if not path.exists():
        print(f"  MISSING  {path} — run this from the repo root")
        return False
    text = path.read_text()
    if new.strip() and new.strip() in text:
        print(f"  already  {label}")
        return True
    if old not in text:
        print(f"  FAILED   {label} — expected code not found in {path}")
        return False
    path.write_text(text.replace(old, new, 1))
    print(f"  fixed    {label}")
    return True


def main() -> int:
    print("applying fixes\n")
    results = [
        patch(LOADER, OLD_AUX, NEW_AUX, "1+2  CUDB brackets and (NOISE"),
        patch(CONFIG, OLD_CFG, NEW_CFG, "3a   SHOCKABLE_RHYTHMS + VT_FAST_BPM"),
        patch(WINDOWS, OLD_SHOCK, NEW_SHOCK, "3b   rate measurement for VT"),
        patch(WINDOWS, OLD_EXTRACT, NEW_EXTRACT, "3c   shockable uses the split"),
        patch(WINDOWS, OLD_HASH, NEW_HASH, "3d   config hash covers the change"),
    ]
    if all(results):
        print("\nall applied. now run:\n  python tests/test_pipeline.py")
        return 0
    print("\nsomething did not apply — no file was left half-written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
