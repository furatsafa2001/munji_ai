"""Central configuration for the MUNJI ECG pipeline.

Every tunable constant lives here. Nothing downstream should hardcode a
sampling rate, window length, or filter cutoff.
"""

from pathlib import Path

# --------------------------------------------------------------------- paths
DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"          # untouched PhysioNet downloads
CACHE_DIR = DATA_ROOT / "cache"      # resampled / filtered numpy arrays
SPLIT_DIR = DATA_ROOT / "splits"     # patient-level split manifests

# ------------------------------------------------------------------ sampling
# Training runs at 250 Hz: MITDB/AFDB/CUDB/VFDB/Icentia are natively 250 or
# resample to it cleanly. The MAX30003 runs at 256 Hz and is resampled to 250
# at inference, so train and deploy see identical signal characteristics.
TARGET_FS = 250
DEVICE_FS = 256

# -------------------------------------------------------------------- filters
# Two independent paths, deliberately.
#
# DIAGNOSTIC preserves the ST segment (0.05 Hz high-pass, per AHA/AAMI
# diagnostic-bandwidth convention). Used for storage and future ST work.
# Raising this cutoff destroys ST fidelity irreversibly — do not "clean up"
# the trace by moving it to 0.5 Hz.
#
# MODEL is the standard monitoring band. Model inputs use this path.
DIAGNOSTIC_BAND = (0.05, 40.0)
MODEL_BAND = (0.5, 40.0)

MAINS_HZ = 50.0        # Saudi grid frequency
NOTCH_Q = 30.0
FILTER_ORDER = 4

# -------------------------------------------------------------------- windows
# Length in seconds per pipeline stage. Rationale in the detection matrix.
WINDOW_SEC = {
    "quality": 5.0,     # signal quality gate
    "beat": 8.0,        # beat classifier (PVC / PAC)
    "rhythm": 30.0,     # AF — needs enough RR intervals to judge irregularity
    "shockable": 5.0,   # VF / wide-complex, per EC57 convention
}
WINDOW_STRIDE_SEC = {
    "quality": 5.0,
    "beat": 4.0,        # 50% overlap — beats near edges get a second chance
    "rhythm": 15.0,
    "shockable": 2.5,
}


def window_samples(stage: str) -> int:
    return int(round(WINDOW_SEC[stage] * TARGET_FS))


def stride_samples(stage: str) -> int:
    return int(round(WINDOW_STRIDE_SEC[stage] * TARGET_FS))


# --------------------------------------------------------------------- splits
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 20260817

# ---------------------------------------------------- beat segmentation masks
# The beat classifier predicts a per-sample mask rather than one label per
# window, so it needs no R-peak detector at inference. Annotations give only
# the R-peak index, so a region is painted around each one.
#
# Widths are deliberately class-dependent and physiological: a normal QRS is
# under 120 ms, a ventricular ectopic beat is wider. The model learning to
# paint wider regions for PVCs is the intended behaviour — width is the
# discriminating feature on a single lead.
BEAT_CLASSES = ("background", "N", "PAC", "PVC")
BEAT_CLASS_ID = {c: i for i, c in enumerate(BEAT_CLASSES)}
QRS_HALFWIDTH_MS = {"N": 50.0, "PAC": 50.0, "PVC": 70.0}

# --------------------------------------------------------------- window labels
# Fraction of a window that must fall inside a rhythm episode for the window to
# carry that label. Below this the window is ambiguous and is dropped rather
# than assigned to either class.
RHYTHM_PURITY = 0.7
SHOCKABLE_PURITY = 0.5

RHYTHM_CLASSES = ("NSR", "AF", "OTHER")
SHOCKABLE_CLASSES = ("not_shockable", "shockable")
QUALITY_CLASSES = ("good", "qrs_only", "unusable")

# ---------------------------------------------------------------- cache limits
# Icentia has ~11k patients of multi-day recording — orders of magnitude more
# windows than are useful. Roughly 95% of beats are normal, so the constraint
# is class balance, not volume.
SAMPLE_PATIENTS = {"icentia11k": 2000}
MAX_WINDOWS_PER_PATIENT = {"quality": 60, "beat": 150, "rhythm": 80, "shockable": 400}
MAX_WINDOWS_PER_STAGE = {"quality": 120_000, "beat": 300_000,
                         "rhythm": 150_000, "shockable": 120_000}
# Share of retained windows allowed to contain no ectopic beat / no episode.
NEGATIVE_KEEP_RATE = {"beat": 0.15, "rhythm": 0.35, "shockable": 0.5}

CACHE_DTYPE = "float16"

# --------------------------------------------------------------- augmentation
# SNR range for NSTDB noise injection. 20 dB is barely perceptible; below 0 dB
# noise dominates the signal entirely.
#
# Widened from (0, 24) after measuring the resulting class balance. With a
# uniform draw over that range and the bands below, only ~8% of windows landed
# in 'unusable' — too few for the classifier to learn the reject decision at
# all. This range gives roughly 50/25/25, which is still 'good'-heavy, and it
# should be: most real windows genuinely are usable.
#
# Fixing the data beats compensating in the loss. Class weights can rebalance a
# skewed set but cannot invent examples that were never generated.
NOISE_SNR_DB_RANGE = (-6.0, 20.0)
AUGMENT_PROB = 0.5

# ------------------------------------------------------------------ rule tiers
# Engineering proposals. Clinical thresholds are not settled and any value
# here should be treated as provisional.
HR_BRADY = {"info": 59, "warning": 49, "critical": 39}   # HR at or below
HR_TACHY = {"info": 101, "warning": 121, "critical": 151}  # HR at or above
PAUSE_WARNING_SEC = 3.0
ASYSTOLE_CRITICAL_SEC = 4.0
NARROW_QRS_MS = 120
FAST_NARROW_BPM = 150

# Confirmation windows — N of M consecutive positives before an alert fires.
# A single window must never trigger a phone call.
CONFIRMATION = {
    "vf": (2, 3),
    "asystole": (2, 3),
    "wide_complex": (3, 4),
    "extreme_brady": (3, 4),
    "extreme_tachy": (3, 4),
    "pause": (2, 3),
    "af": (2, 3),
    "fast_narrow": (3, 4),
}
