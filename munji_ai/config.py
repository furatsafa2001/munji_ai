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
# resample to it cleanly. The patch firmware sets the MAX30003 to exactly
# 250 samples/s (CNFG_GEN FMSTR=01, CNFG_ECG RATE=01), the same rate the app
# and the .ecg file use, so inference needs no resampling.
# If the firmware rate ever changes, DEVICE_FS changes with it and inference
# must resample DEVICE_FS -> TARGET_FS.
TARGET_FS = 250
DEVICE_FS = 250

# -------------------------------------------------------------------- filters
# Two independent paths, deliberately.
#
# DIAGNOSTIC (0.05 Hz high-pass, per AHA/AAMI diagnostic-bandwidth
# convention) applies to the public databases only. The patch cannot give
# it: the MAX30003 applies its own 0.5 Hz high-pass and 40 Hz low-pass on
# chip (CNFG_ECG DHPF=1, DLPF=01), so every signal from the device,
# saved recordings included, is already in the monitoring band. ST
# analysis is out of scope (single lead).
#
# MODEL is the standard monitoring band, the same band the chip applies.
# Model inputs use this path.
DIAGNOSTIC_BAND = (0.05, 40.0)
MODEL_BAND = (0.5, 40.0)

MAINS_HZ = 50.0        # Saudi grid frequency
NOTCH_Q = 30.0
FILTER_ORDER = 4

# -------------------------------------------------------------------- windows
# Length in seconds per pipeline stage. Rationale in the detection matrix.
WINDOW_SEC = {
    "quality": 5.0,     # signal quality gate
    # 20 s, matching Sensors 2026 (MDPI 26/2/513), whose 1D U-Net reports
    # QRS sensitivity and precision up to 0.999 on this window length. Judging
    # a beat "premature" needs the surrounding rhythm, not just the complex, so
    # a longer window carries information a shorter one cannot. Segmentation
    # also wastes less at the edges: one boundary per 20 s instead of per 8 s.
    "beat": 20.0,       # beat classifier (PVC / PAC), per the reference paper
    "rhythm": 30.0,     # AF — needs enough RR intervals to judge irregularity
    "shockable": 5.0,   # VF / wide-complex, per EC57 convention
}
WINDOW_STRIDE_SEC = {
    "quality": 5.0,
    "beat": 10.0,       # 50% overlap — beats near edges get a second chance
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

# MITDB, by the inter-patient division of de Chazal et al. (2004): DS1 to
# develop on, DS2 held out. Published inter-patient results report on DS2, so
# training on DS1 keeps every comparison valid. Agreed with the decision-layer
# side (Nov 2026) on three conditions:
#   DS1 is for the beat model only: no other stage reads it.
#   202 is the same subject as 201 (PhysioNet) but sits in DS2; it moves to
#   DS1 with 201, and DS2 without it is the end-to-end test.
#   The four paced records (102, 104, 107, 217) are in neither half, as AAMI
#   EC57 excludes them; they are used nowhere.
MITDB_DS1 = ("101", "106", "108", "109", "112", "114", "115", "116", "118",
             "119", "122", "124", "201", "203", "205", "207", "208", "209",
             "215", "220", "223", "230")
MITDB_DS2 = ("100", "103", "105", "111", "113", "117", "121", "123", "200",
             "202", "210", "212", "213", "214", "219", "221", "222", "228",
             "231", "232", "233", "234")
MITDB_PACED = ("102", "104", "107", "217")
MITDB_MOVED_TO_DS1 = ("202",)
# DS1 records that validate the beat model rather than train it. Hashing by
# patient put all four bundle branch block records in train, leaving nothing
# to check BBB on before the final test. Rule, fixed before any training: for
# each block type, the record that also carries ventricular runs trains (207,
# 105-beat run; 124, 18-beat run) and the other validates (109 LBBB, 118 RBBB).
# 101, 115 and 122 are where the hash put them (val, or test, which DS1 has none
# of). Beat counts per record: tools run on 23 Nov 2026 over the annotations.
MITDB_DS1_VAL = ("101", "109", "115", "118", "122")

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

# Real QRS complexes that belong to none of the three classes. AAMI EC57 puts
# fusion beats (F) in class F, and paced beats (/), fusions of paced and normal
# (f) and unclassifiable beats (Q) in class Q; neither class is one MUNJI
# reports. The rest are WFDB beat codes outside the AAMI mapping (? not
# classified, B bundle branch block unspecified, r R-on-T, n supraventricular
# escape), left unmapped until a source settles their class.
#
# Their samples were painted background, which taught the model that a real
# complex is "no beat" (MITDB 208 alone carries 373 fusion beats). They are now
# painted BEAT_IGNORE_ID: left out of the loss, the class weights and every
# score, so the model is neither rewarded nor punished for what it says there.
# The half-width is the widest class's: these beats are wide.
BEAT_IGNORE_ID = 255
BEAT_IGNORE_SYMBOLS = ("F", "/", "f", "Q", "?", "B", "r", "n")
BEAT_IGNORE_HALFWIDTH_MS = 70.0

# Predicted mask regions narrower than this are dropped before beats are read
# off the mask. A real QRS spans about 100 ms; anything much narrower is a
# patch of uncertainty, not a complex.
#
# Measured on the full validation split: 60 ms lifts beat F1 from 0.793 to
# 0.846 and PVC precision from 0.798 to 0.934, costing 0.005 of QRS
# sensitivity. 80 ms gives the same F1 while losing twice as much sensitivity.
BEAT_MIN_REGION_MS = 60.0

# ------------------------------------------------------------------ PVC burden
# Share of beats in a period that are ventricular. The thresholds come from
# published consensus, not from this project:
#
#   2017 AHA/ACC/HRS expert consensus   >= 15% of daily heartbeats
#   2014 EHRA/HRS/APHRS expert consensus >= 10,000 PVCs per 24 h
#   Reviewed series                      no cardiomyopathy below 10%
#
# A commercial insertable monitor using the 10% threshold reported 84%
# patient-level sensitivity at 100% PPV against 24-hour Holter.
PVC_BURDEN = {"info": 0.05, "warning": 0.10, "critical": 0.15}

# The window burden is measured over. Burden is defined clinically over 24
# hours, and a short window means something different: 20 seconds of bigeminy
# is 50% burden and is not a finding. One hour is the shortest period that is
# not dominated by a single run; the daily figure is the one that matches the
# literature and the one a clinician will ask for.
PVC_BURDEN_WINDOW_SEC = 3600.0
PVC_BURDEN_DAILY_SEC = 86400.0

# Below this many beats the percentage is noise. Ten PVCs out of thirty beats
# is 33% and tells you nothing.
PVC_BURDEN_MIN_BEATS = 200

# ------------------------------------------------- ventricular rhythms from beats
# Read off the beat model's labels by the decision layer (on_beats).
#
# VT, by definition. 2017 AHA/ACC/HRS ventricular arrhythmia guideline,
# definitions table: VT is 3 or more consecutive complexes of ventricular
# origin at more than 100 bpm (cycle length under 600 ms); sustained when it
# lasts more than 30 s, nonsustained when it ends on its own before.
#
# Sustained VT alerts as emergent: in the 26,438-patient MCOT series above,
# wide-complex tachycardia was among the potentially emergent events.
VT_MIN_BEATS = 3
VT_MIN_BPM = 100.0              # strictly above
VT_SUSTAINED_SEC = 30.0

# Sustained VT is read as "at least this share of the last 30 s of beats is
# ventricular", not "every beat". Engineering, not clinical: the beat model
# misses some PVCs, and one missed PVC splits a strict consecutive run into
# short pieces that never reach 30 s, so a long VT never alerts.
# eval/vt_run_simulation.py measured that effect; 0.80 tolerates about one
# missed PVC in five.
VT_SUSTAINED_PVC_FRACTION = 0.80

# Bigeminy: a PVC alternating with a normal beat; trigeminy: every third beat a
# PVC (StatPearls, "Premature Ventricular Contraction", NCBI Bookshelf). The
# minimum of three repeated cycles is a detection convention used by
# arrhythmia-analysis software (e.g. US patent 12268527: "at least three
# consecutive pairs ... are required"), not a guideline value.
ECTOPY_PATTERN_MIN_CYCLES = 3

# Beat windows overlap (20 s every 10 s), so a beat arrives twice with its time
# shifted by a few ms. Two beats closer than this are one. Engineering: under
# the 200 ms refractory period QRS detectors use (Pan and Tompkins 1985),
# over the jitter of a mask centre between windows.
BEAT_DEDUP_SEC = 0.15

# ECG saved around an emergency alert (VF/VT, asystole, sustained VT): the
# decision layer asks, the app saves. Team decision to auto-save a short strip
# around critical events so the doctor sees what triggered the alert.
ECG_SAVE_BEFORE_SEC = 30.0
ECG_SAVE_AFTER_SEC = 30.0

# Naming a ShockableNet alert VF or VT from the beat labels: VT when the beat
# model sees organised PVCs in a row, VF when it sees no beats at all. Off
# until validated on new patients (ECG Fragment Database, CUDB, VFDB); while
# off the alert stays "VF/VT" and the guess is saved with the recording only
# (shadow mode), to be compared with the doctor's reading later.
VFVT_NAMING_ENABLED = False
VFVT_NAMING_LOOKBACK_SEC = 10.0  # engineering: the VT detector's 10 s input

# --------------------------------------------------------------- window labels
# Fraction of a window that must fall inside a rhythm episode for the window to
# carry that label. Below this the window is ambiguous and is dropped rather
# than assigned to either class.
RHYTHM_PURITY = 0.7
SHOCKABLE_PURITY = 0.5

RHYTHM_CLASSES = ("NSR", "AF", "OTHER")
SHOCKABLE_CLASSES = ("not_shockable", "shockable")

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
VT_FAST_BPM = 150.0

# Labels that mark ambiguity rather than a rhythm. A window dominated by one of
# these is dropped, never assigned to a class.
#
# NOISE   the original annotators marked this stretch unreadable. Deleting the
#         annotation is worse than keeping it: intervals run until the next
#         marker, so removing NOISE makes the preceding rhythm extend across it
#         and unreadable signal gets labelled as that rhythm.
# UNKNOWN whatever follows a ventricular fibrillation episode. Often asystole,
#         VT, or defibrillation artifact — asserting sinus rhythm there teaches
#         the model that a still-dangerous rhythm is safe.
AMBIGUOUS_RHYTHMS = ("NOISE", "UNKNOWN")

# Excluded from the shockable stage on top of the above: VT whose rate could
# not be measured. Neither shockable nor safe — unknown. Labelling it
# not_shockable is the wrong side of the miss/false-alarm trade.
SHOCKABLE_EXCLUDED = AMBIGUOUS_RHYTHMS + ("VT",)

# Shockable labels, one per window (data/shockable_labels.py): the AHA
# framework for shock advisory algorithms as published work applies it to
# CUDB and VFDB. Used by the shockable cache and by the shockable evaluation
# (false_alarm_rate.truth_flags), so training and scoring read the same labels.
#
# AHA (Kerber et al., Circulation 1997;95:1677) sorts rhythms into shockable
# (coarse VF, rapid VT), nonshockable, and intermediate (fine VF, other VT);
# intermediate rhythms have no performance goal and are reported only.
# Figuera et al. (PLoS ONE 2016;11:e0159654), on these same databases: "slow
# VT (rate under 150 bpm) and fine VF were excluded", fine VF being under
# 200 uV peak to peak, and "segments with rhythm transitions were excluded".
# Their labels were revised by hand; here each window is measured:
#   VT rate   autocorrelation of the window. The episode rate used before
#             (windows._episode_bpm) read smooth or small VT far too slow:
#             47% of the windows it called VT_SLOW, trained as NOT shockable,
#             were 150 bpm or more (experiments/vt_rate_check.py).
#   fine VF   peak to peak of the window, model path, mV.
# An irregular VT window has no rate to compare with 150: it does not repeat,
# which is polymorphic VT. It is shockable. AHA's shockable VT category covers
# polymorphic VT and flutter (performance table, Kerber 1997, as AED makers
# report it), and AHA 2025 ACLS shocks polymorphic VT unsynchronised, like VF.
# Checked against cardiologist labels (ECG Fragment Database: VFDB re-labelled
# at LETI with Almazov Centre cardiologists; experiments/shockable_labels_check.py):
# every one of the 31 irregular VT windows there was VF or torsade de pointes.
#
# SHOCKABLE_PURITY and SHOCKABLE_EXCLUDED above still describe the episode
# labels of windows.split_vt_by_rate, which the VT experiments read. They stay
# unchanged in every cache's fingerprint; this block is in the shockable
# cache's alone, so changing it invalidates that cache and no other.
SHOCKABLE_LABELS = "aha-window-v2"   # v2: irregular VT shockable, not left out
SHOCKABLE_WINDOW_PURITY = 1.0        # one rhythm over the whole window
FINE_VF_MAX_PTP_MV = 0.2             # VF under this, peak to peak: fine VF
VT_RATE_LAG_SEC = (0.2, 1.2)         # autocorrelation search: 300 to 50 bpm
VT_RATE_NEAR_TOP = 0.85              # first peak this close to the highest: not a multiple
VT_RATE_MIN_REGULARITY = 0.5         # highest peak under this: irregular, no rate
SHOCKABLE_WINDOW_RHYTHMS = ("VF", "VT_FAST", "VT_IRREGULAR")
SHOCKABLE_WINDOW_EXCLUDED = AMBIGUOUS_RHYTHMS + ("VT_SLOW", "VF_FINE")
# Two classes, not three. The old good / qrs_only / unusable split put an
# arbitrary 12 dB line through a continuum; merging the first two raised
# accuracy from 0.66 to 0.92 on identical data. The remaining boundary is
# physical: is there recoverable QRS structure or not.
QUALITY_CLASSES = ("usable", "unusable")

# The detail qrs_only used to carry comes back as a continuous score. Different
# stages need different amounts of signal quality, so they threshold the same
# P(usable) at different points rather than sharing one hard label.
#
#   beat classifier needs clean morphology       -> strict
#   rule engine needs only R-peak timing         -> permissive
GATE_MIN_P = {"morphology": 0.90, "rhythm": 0.60}

# How several gate decisions over one longer window combine into one.
#
# The gate judges 5 s windows; beat windows are 20 s and rhythm windows 30 s,
# so each carries four or six gate decisions that can disagree.
#
#   "mean"  average P(usable) across the sub-windows
#   "min"   the worst sub-window decides
#
# Measured on 3,906 beat validation windows with the trained gate:
#
#   mean @ 0.90   keeps 97.6%   passed F1 0.846   rejected F1 0.820
#   min  @ 0.90   keeps 87.5%   passed F1 0.843   rejected F1 0.859
#
# The rejected column settles it. Under "mean" the dropped windows are worse
# than the kept ones, which is what a gate is for. Under "min" they are
# better, so it discards windows the classifier was handling, and costs 12.5%
# of coverage to do it.
#
# "min" fails structurally, not by tuning: one bad second disqualifies
# nineteen good ones, and the segmenter labels each sample independently.
GATE_COMBINER = "mean"

# Caveat on the above. P(usable) over the beat cache has a median of 0.922 and
# a 10th-to-90th range of 0.906 to 0.938 — a very narrow spread, because
# Icentia and SVDB are comparatively clean. The measurement shows the gate
# costs nothing on clean data; it does not show the gate is unnecessary. A
# moving patient will produce a far wider spread.

# ---------------------------------------------------------------- cache limits
# Icentia has ~11k patients of multi-day recording — orders of magnitude more
# windows than are useful. Roughly 95% of beats are normal, so the constraint
# is class balance, not volume.
SAMPLE_PATIENTS = {"icentia11k": 2000}
MAX_WINDOWS_PER_PATIENT = {"quality": 60, "beat": 150, "rhythm": 80, "shockable": 400}

# How many of a patient's records to read at all. Icentia holds ~50 segments per
# patient; reading every one to take 3 windows from each wastes almost all of
# the time in loading and filtering. Ten segments still span roughly 12 hours,
# and the build runs five times faster.
#
# Datasets with a single record per patient are unaffected.
MAX_RECORDS_PER_PATIENT = {"quality": 6, "beat": 10, "rhythm": 8, "shockable": 50}
# Ceiling per stage, sized so the cache stays loadable in memory. Beat windows
# are 20 s rather than 5 s, so the same count costs four times as much — 300k
# would be 4.5 GB. These are guardrails, not targets: 235 patients at 150
# windows each comes to ~35k, well under.
MAX_WINDOWS_PER_STAGE = {"quality": 120_000, "beat": 150_000,
                         "rhythm": 150_000, "shockable": 120_000}
# Share of retained windows allowed to contain no ectopic beat / no episode.
NEGATIVE_KEEP_RATE = {"beat": 0.15, "rhythm": 0.35, "shockable": 0.5}

# For the beat stage the per-patient cap applies to NORMAL windows only.
# Windows carrying at least one PAC or PVC are all kept, because they are the
# scarce thing: each ectopic class is under 1% of mask samples, and the
# previous code shuffled ectopic and normal windows together before
# truncating, so it discarded ectopic windows at random.
#
# The reference paper keeps every segment containing an ectopic beat and
# samples 1% of the purely normal ones. This is the same idea with a ceiling,
# so one patient with very frequent ectopy cannot fill the cache alone.
BEAT_KEEP_ALL_ECTOPIC = True
BEAT_MAX_ECTOPIC_PER_PATIENT = 600

# The ceiling holds across all of a patient's records (it used to apply per
# record), and ectopic windows no longer use up the normal-window cap above.
# They did: a multi-record patient with frequent ectopy spent the whole cap on
# the first record read and every other record was skipped, so an INCART
# patient whose VT sits in their third half-hour could lose it.
#
# Windows holding a run of at least this many consecutive ventricular beats are
# always kept and do not count toward the ceiling, which then bounds the other
# ectopic windows only; 0 turns this off. A long-term record can
# carry thousands of ectopic windows, and the ceiling used to drop runs at
# random with the rest. Runs are where the beat model fails (PVC sensitivity
# 0.93 isolated, about 0.84 in runs of 3 to 4, about 0.05 inside sustained VT),
# three consecutive ventricular beats at 100 bpm or more is the definition of
# VT, and runs are rare enough that keeping all of them costs little.
BEAT_ALWAYS_KEEP_RUNS = 3

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
# Every clinical number here has a published source, written beside it with
# its comparison. The measured rate is continuous, so "below 60" and "at or
# below 59" are different rules; the second left 59-60 unclassified.
#
# HR_BRADY  rate strictly BELOW the value
#   info      60  conventional definition of bradycardia
#   warning   50  2018 ACC/AHA/HRS bradycardia guideline: sinus bradycardia
#                 is a rate below 50 while awake
#   critical  40  2018 ACC/AHA/HRS: sinus node dysfunction is diagnosed on
#                 symptoms with a rate below 40; NEWS2 (Royal College of
#                 Physicians 2017) gives 40 or less its most abnormal score
#
# HR_TACHY  info and warning strictly ABOVE; critical AT OR ABOVE
#   info     100  conventional definition of tachycardia
#   warning  110  NEWS2: 111-130 scores 2 (91-110 scores 1)
#   critical 150  AHA adult tachyarrhythmia algorithm: rate typically >= 150
#
# These tiers grade findings for the doctor's report. Who is NOTIFIED is a
# separate decision, below under "notification".
HR_BRADY = {"info": 60, "warning": 50, "critical": 40}     # strictly below
HR_TACHY = {"info": 100, "warning": 110, "critical": 150}  # see semantics above

# A rate finding needs the rate held over this many consecutive beats.
# PhysioNet/CinC Challenge 2015 alarm definitions
# (Clifford et al., Physiol Meas 2016): extreme bradycardia, HR < 40 for 5
# consecutive beats; extreme tachycardia, HR > 140 for 17 consecutive beats.
# No source gives a count for the milder tiers, so each tier uses its family's
# count. These counts replace N-of-M window confirmation for the rate rules.
RATE_RUN_BEATS = {"brady": 5, "tachy": 17}
# The rate of a run is 60 / the mean RR after dropping this fraction of the
# longest and of the shortest intervals, the way bedside monitors average
# heart rate (the 3 longest and 3 shortest of 16 RR dropped, the residual 10
# averaged; EC13 requires monitors to disclose their averaging). One missed
# beat or one pause cannot make or break a run; an irregular fast rhythm
# (AF at 160) still counts; a burst shorter than half the run does not.
RATE_RUN_TRIM = 3 / 16

# Pause (warning): longest beat-to-beat gap strictly longer than this.
# 2018 ACC/AHA/HRS: pauses longer than 3 s in sinus node dysfunction;
# 2018 ESC syncope guidelines: a symptomatic pause longer than 3 s.
PAUSE_WARNING_SEC = 3.0
# Asystole (critical): gap strictly longer than this. 2018 ESC syncope
# guidelines: an asymptomatic pause longer than 6 s is significant on its own.
# (CinC 2015 used 4 s for ICU alarms; the clinical guideline is preferred.)
ASYSTOLE_CRITICAL_SEC = 6.0
# A gap is read as pause or asystole only if it is quiet: nothing inside it
# reaches this, peak to valley. Monitors meeting EC13 / IEC 60601-2-27 must
# detect QRS complexes from 0.5 to 5 mV, so a deflection of 0.5 mV inside a
# "gap" is either a beat the detector missed or artifact: the gap is
# unreadable, not empty. Asystole is the absence of electrical activity; sensor
# noise and the P waves of ventricular standstill (about 0.1-0.25 mV) stay
# below it. Measured need: NSRDB 16272, an hour of motion artifact up to 10 mV
# read as repeated asystole when beats under it were missed.
ABSENCE_MAX_PTP_MV = 0.5

# Wide QRS at or above this. AHA adult tachyarrhythmia algorithm (QRS >= 0.12 s);
# wide-complex tachycardia is defined as QRS >= 120 ms.
NARROW_QRS_MS = 120
# The fast narrow-complex label uses the same line as the critical tachy tier.
FAST_NARROW_BPM = 150

# A detected peak counts as a beat only if its QRS is larger than this, peak
# to valley, in millivolts. ANSI/AAMI EC13: a heart-rate meter must not respond
# to a QRS of 0.15 mV or less, so that P waves during ventricular standstill
# are not counted (Drew et al., PLoS ONE 2014). It is also what lets the engine
# see a real asystole: the R-peak detector's threshold is relative, so on a
# quiet trace it always "finds" peaks in the noise.
# The rate engine's input must therefore be in millivolts, never normalised.
QRS_MIN_AMP_MV = 0.15
# Amplitude is read over +/- this around each peak: a QRS up to 120 ms, the
# upper limit of a normal complex (same 120 ms line as above).
QRS_AMP_HALF_MS = 60
# Width is searched over +/- this. It must be able to measure the widest
# complexes reported clinically: 69% of VT exceeds 140 ms (Wellens 1978) and
# LBBB-pattern VT exceeds 160 ms. The old 60 ms capped every reading at 124 ms.
QRS_SEARCH_HALF_MS = 100
# Half-width of the band around NARROW_QRS_MS where width is reported as
# "uncertain" instead of narrow or wide. Set it from the measured error in
# eval_qrs_width_qtdb (acceptance: IEC 60601-2-25, mean difference within
# +/-10 ms, SD 10 ms against expert measurement). None until then: no band.
QRS_UNCERTAIN_MS = None

# ------------------------------------------------------------------ notification
# Which findings reach the caregiver. Everything else goes to the report.
#
# Base: ambulatory cardiac telemetry practice (MCOT), the same class of device
# as this patch. Physician-notification criteria of a commercial MCOT service
# (Philips BioTel MCOT protocol, as published in J Cardiol Curr Res,
# "Outpatient cardiac telemetry monitoring for early patient discharge"):
# pause of 3 s or more; SVT or AF at 180 bpm or more for 30 s or more;
# bradycardia at 30 bpm or less for 30 s or more; wide-complex tachycardia at
# 150 bpm or more for 6 s or more. In a retrospective series of 26,438 MCOT
# patients (cited in payer medical-policy reviews of MCOT) the potentially
# emergent events were wide-complex tachycardia, pauses of 6 s or longer, and
# sustained bradycardia below 30 bpm.
#
# Bradycardia below 40 held 30 s is also sent (team decision for elderly users
# at home, checked against the literature): the ARIC study's alert criteria
# for the Zio single-lead patch in older adults include "bradycardia < 40 bpm
# sustained for > 30 seconds"; pacing guidelines treat an awake rate below 40
# as significant even without clear symptoms (ACC/AHA; ESC 2008/2021). Caveat:
# the guidelines say awake, and the patch cannot tell sleep from waking.
# Tachycardia stays at the monitoring line, 180: MCOT (>= 180 for 30 s) and
# ARIC (narrow QRS > 180 for 60 s). The AHA 150 line is a bedside rule for
# whether a fast rate explains a patient's instability, not a monitoring alert
# threshold, so 150-180 goes to the report.
#
#   sustained bradycardia  rate below 40, held 30 s          urgent    (ARIC)
#   profound bradycardia   rate at or below 30, held 30 s    emergent  (MCOT)
#   sustained tachycardia  rate at or above 180, held 30 s   urgent    (MCOT, ARIC)
#   pause                  gap longer than PAUSE_WARNING_SEC     urgent
#   asystole               gap longer than ASYSTOLE_CRITICAL_SEC emergent
#   wide-complex rhythm    ventricular-beat detector, not here (ARIC: wide QRS
#                          > 120 bpm for 30 s; MCOT: >= 150 for 6 s)
NOTIFY_SUSTAIN_SEC = 30.0
NOTIFY_BRADY_URGENT_BPM = 40       # strictly below, held NOTIFY_SUSTAIN_SEC
NOTIFY_BRADY_EMERGENT_BPM = 30     # at or below, held NOTIFY_SUSTAIN_SEC
NOTIFY_TACHY_URGENT_BPM = 180      # at or above, held NOTIFY_SUSTAIN_SEC

# The "fast narrow" (SVT) label rests on the QRS width estimate, which failed
# against hand-measured QRS on the QT Database: mean -50 ms, SD 35 ms against
# IEC 60601-2-25 limits of +/-10 ms and 10 ms, correlation 0.25. Off until a
# width source passes. A fast rhythm is still reported and notified by rate.
FAST_NARROW_LABEL = False

# Windows the rate engine runs on, derived from the definitions above, not
# chosen. 40 s holds a 30 s sustained run whole (40 - 2.5 > 30), and so also a
# gap longer than 6 s and a 5-beat run at any rate a heart can hold. Latency is
# the definition itself (30 s, or the gap) plus at most one stride.
RULES_WINDOW_SEC = 40.0
RULES_STRIDE_SEC = 2.5

# Confirmation windows — N of M consecutive positives before an alert fires.
# A single window must never trigger a phone call.
# The rate rules carry their own persistence (beats held, or gap length), so
# one window with a finding is already confirmed: (1, 1). "asystole" covers
# the whole absence family (pause and asystole) in the decision layer.
CONFIRMATION = {
    "vf": (4, 6),
    "asystole": (1, 1),
    "extreme_brady": (1, 1),
    "extreme_tachy": (1, 1),
    "af": (4, 6),
    "fast_narrow": (1, 1),
}

# ShockableNet's operating threshold on P(shockable); an alert also needs
# CONFIRMATION["vf"]. Chosen on full-day cross-validation with the per-window
# AHA labels (eval/shockable_report.py, 5 folds by patient): every threshold
# from 0.54 to 0.58 caught all 83 sustained VF/VT episodes (30 s or more) with
# no more than 0.29 false alarms per patient-day, the wearable defibrillator
# study goal (Poole et al., J Cardiovasc Electrophysiol 2022); 0.53 gave 0.43
# and 0.59 missed a sustained episode. 0.56 is the middle of that range (0.14
# per patient-day), so a small shift in the final model's scores keeps both.
# Training records it in the checkpoint and the TFLite export carries it.
SHOCKABLE_THRESHOLD = 0.56
