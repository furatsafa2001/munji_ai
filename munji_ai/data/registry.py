"""Dataset registry.

Every dataset the pipeline can consume is described here and nowhere else.
Adding, removing, or swapping a corpus is an edit to this file — no loader,
trainer, or evaluation code changes.

`patient_re` extracts a patient identifier from a record name so splits can be
made at patient level rather than record level. Several PhysioNet sets contain
multiple records per subject; splitting on records leaks a patient across
train and test and inflates every metric.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["train", "val", "noise"]
Labels = Literal["beat", "rhythm", "both", "none"]


@dataclass(frozen=True)
class Dataset:
    key: str
    slug: str                  # PhysioNet content directory
    version: str
    fs: int                    # native sampling rate (Hz)
    channel: int               # fallback channel index
    role: Role
    labels: Labels
    patient_re: str            # regex, group 1 = patient id
    stages: tuple = field(default_factory=tuple)
    # Preferred channel by signal name, tried in order before falling back to
    # `channel`. Multi-lead sets need the lead closest to CM5 (a V5-like chest
    # position); picking index 0 would silently give a limb lead instead.
    channel_names: tuple = field(default_factory=tuple)
    verified: bool = True      # False = record layout needs checking on download
    # Rhythm that holds for an entire record when the annotations declare no
    # changes. NSRDB subjects were screened for having no significant
    # arrhythmia, so its files carry beat labels and no rhythm markers at all —
    # without this the loader finds no intervals and the set contributes
    # nothing. Leave empty for any set where the rhythm actually varies.
    baseline_rhythm: str = ""
    # Per-record labels held in a CSV beside the data, rather than in .atr
    # files. CinC 2017 ships one label per recording this way, and
    # baseline_rhythm cannot express it because that is a single rhythm for a
    # whole dataset. Path is relative to the dataset directory; the first CSV
    # column is the record (any directory prefix is ignored) and the second is
    # the label. Empty for every set that annotates normally.
    label_file: str = ""
    # Maps the label strings in that CSV onto WFDB rhythm tokens, as pairs
    # because a frozen dataclass cannot hold a dict.
    label_map: tuple = field(default_factory=tuple)
    note: str = ""


REGISTRY: dict[str, Dataset] = {
    # ---------------------------------------------------------------- training
    # The CinC 2017 challenge set: 8,528 single-lead recordings, 30 s each at
    # 300 Hz, sorted into normal / AF / other / too noisy. Registered for the
    # rhythm stage to supply the OTHER class, which the other sources do not
    # carry - ltafdb, icentia and nsrdb are all effectively binary, and OTHER
    # sat at 34 training windows against 190 in test.
    #
    # 2,415 of these recordings are labelled "other rhythm", which is the
    # whole reason for adding it.
    #
    # Labels are in training/REFERENCE.csv, one per recording, so this is the
    # first dataset to use label_file. Noisy recordings map to "(NOISE" and
    # are dropped downstream rather than assigned to a class.
    #
    # Each recording is treated as its own patient. The challenge provides no
    # subject identifiers, so recordings from the same person cannot be
    # grouped; the challenge itself scored them independently. Worth stating
    # in the write-up as a limitation of this source.
    "challenge2017": Dataset(
        key="challenge2017", slug="challenge-2017", version="1.0.0",
        fs=300, channel=0, role="train", labels="rhythm",
        patient_re=r"^(A\d+)$",
        stages=("rhythm",),
        label_file="training/REFERENCE.csv",
        label_map=(("N", "(N"), ("A", "(AFIB"),
                   ("O", "(OTHER"), ("~", "(NOISE")),
        note="PhysioNet/CinC Challenge 2017. 8,528 recordings, 30 s, single "
             "lead, 300 Hz, AliveCor handheld. N 5,076 / A 758 / O 2,415 / "
             "noisy 279. Added for the OTHER class.",
    ),

    "icentia11k": Dataset(
        key="icentia11k", slug="icentia11k-continuous-ecg", version="1.0",
        fs=250, channel=0, role="train", labels="both",
        patient_re=r"^(p\d+)_s\d+$",
        stages=("beat", "rhythm"),
        verified=False,
        note="Primary. Hydrogel patch, modified lead I, ambulatory, elderly-skewing. "
             "CC BY-NC-SA 4.0 — research/prototype use only. Segment layout should be "
             "confirmed against the actual download before first training run.",
    ),
    "cudb": Dataset(
        key="cudb", slug="cudb", version="1.0.0",
        fs=250, channel=0, role="train", labels="rhythm",
        patient_re=r"^(cu\d+)$",
        stages=("shockable", "quality"),
        note="VT, VF, ventricular flutter. 35 records of 8 minutes. Also a quality "
             "carrier: VF has no organised QRS and a gate trained only on sinus "
             "rhythm will call it noise — the most dangerous possible failure.",
    ),
    "vfdb": Dataset(
        key="vfdb", slug="vfdb", version="1.0.0",
        fs=250, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\d+)$",
        stages=("shockable", "quality"),
        note="Malignant ventricular arrhythmia. 22 records of 35 minutes. Also a "
             "quality carrier, same reason as CUDB.",
    ),
    # Healthy sinus rhythm from 20 young and 20 elderly subjects, two hours
    # each at rest. Registered for shockable only, as negative material: the
    # detector's sinus specificity was 0.929 against an AHA target of 0.99,
    # and NSRDB - its only ordinary-rhythm source - is 18 subjects aged 20 to
    # 50. The elderly half here is the cohort the device is actually for.
    #
    # Channel 0 is respiration. Taking the default index would feed the model
    # a breathing trace, so the ECG lead is pinned by name.
    #
    # Caveat worth carrying into the write-up: these are resting recordings.
    # Munji is worn while moving. Closer to the gap than NSRDB, still not the
    # deployment condition.
    "fantasia": Dataset(
        key="fantasia", slug="fantasia", version="1.0.0",
        fs=250, channel=1, role="train", labels="rhythm",
        patient_re=r"^(f\d[oy]\d+)$",
        stages=("shockable",),
        channel_names=("ECG",),
        baseline_rhythm="(N",
        note="Fantasia. 20 young and 20 elderly healthy subjects, sinus "
             "rhythm at rest, two hours each. Beat annotations in .ecg with "
             "no rhythm markers, so the whole record is sinus by declaration.",
    ),

    # 28 records, most from exercise stress tests, from patients with
    # transient ST depression. Registered for shockable as negative material:
    # it is the only source here that is cardiac, in sinus rhythm, and under
    # exertion at the same time - the combination the sinus specificity gap
    # points at.
    #
    # Chosen over the European ST-T Database, which fits the cohort equally
    # well but annotates ST episodes with tokens the rhythm parser would read
    # as rhythm changes. Extending AMBIGUOUS_RHYTHMS to cover them would
    # change the config hash and invalidate every cache in the project. stdb
    # carries beat annotations only - aux_note is empty across its files - so
    # it costs nothing outside this stage.
    #
    # baseline_rhythm declares the record non-shockable, as for NSRDB and
    # Fantasia. For those it was safe by construction; here it is an
    # assumption about supervised stress tests. If VF sensitivity drops after
    # adding this, suspect it first.
    "stdb": Dataset(
        key="stdb", slug="stdb", version="1.0.0",
        fs=360, channel=0, role="train", labels="beat",
        patient_re=r"^(\d+)$",
        stages=("shockable",),
        baseline_rhythm="(N",
        note="MIT-BIH ST Change. 28 records, mostly exercise stress tests, "
             "showing transient ST depression. Beat annotations only, no ST "
             "or rhythm markers, so the whole record is declared sinus.",
    ),

    # Ambulatory Holter from 79 subjects aged 30 to 84, all with diagnosed or
    # suspected ischaemia. Registered for shockable as negative material: it
    # is older, cardiac and moving at once, which is the cohort this device
    # is for.
    #
    # V5 leads the preference list because it is the most common signal here
    # (51 of 90 records) and is the chest position CM5 is named after.
    #
    # Most annotations in this set mark ST-segment and T-wave episodes, not
    # rhythms: (ST0+, (ST1-, (T0+, (T1+ and their closers. The loader skips
    # them (loader.NON_RHYTHM) when building rhythm intervals. Before that
    # fix they were read as rhythm changes and fell back to OTHER, which cut
    # real intervals short, VT runs included.
    #
    # Listing them as ambiguous in config would keep more of each record, but
    # AMBIGUOUS_RHYTHMS feeds the config hash, so it would invalidate every
    # cache in the project including Furat's. Worth asking for only if this
    # version shows a gain above the measured noise band.
    "edb": Dataset(
        key="edb", slug="edb", version="1.0.0",
        fs=250, channel=0, role="train", labels="rhythm",
        patient_re=r"^(e\d+)$",
        stages=("shockable",),
        channel_names=("V5", "V4", "V3", "V2", "V1", "MLIII", "MLI", "D3"),
        note="European ST-T. 90 two-hour ambulatory records, 79 subjects aged "
             "30-84 with ischaemia. ST and T episode markers are left "
             "unrecognised, so affected windows drop rather than mislabel.",
    ),

    "nsrdb": Dataset(
        key="nsrdb", slug="nsrdb", version="1.0.0",
        fs=128, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\d+)$",
        # Also registered for shockable. CUDB and VFDB are arrest recordings
        # and carry almost no ordinary rhythm, so the detector learned
        # "not dangerous" from the quiet parts of dangerous recordings and
        # alarmed on any unfamiliar-looking normal trace. This supplies the
        # healthy hearts that set was missing.
        stages=("quality", "rhythm", "shockable"),
        baseline_rhythm="(N",
        note="Clean normal sinus rhythm. Primary carrier for synthesised quality "
             "labels, but never the only one — see ltafdb.",
    ),
    "svdb": Dataset(
        key="svdb", slug="svdb", version="1.0.0",
        fs=128, channel=0, role="train", labels="both",
        patient_re=r"^(\d+)$",
        stages=("beat", "quality"),
        note="Supraventricular arrhythmia supplement. Best available PAC coverage. "
             "Also a quality carrier so ectopic beats are not learned as noise.",
    ),
    "ltafdb": Dataset(
        key="ltafdb", slug="ltafdb", version="1.0.0",
        fs=128, channel=0, role="train", labels="both",
        patient_re=r"^(\d+)$",
        stages=("rhythm", "quality", "beat"),
        note="Long-term AF. ~2000 hours annotated. Critical quality carrier: AF is "
             "irregular by definition, and a gate trained only on NSR learns "
             "'irregular = noise' and hides exactly what MUNJI exists to detect. "
             "Beat stage (Nov 2026): its atr beat labels were manually reviewed "
             "(PhysioNet), and it holds 484 ventricular runs of 3+ beats outside "
             "the test split, against 63 in Icentia (data/vt_runs.py). Beat symbols "
             "outside the test split: N, A, V, Q. ODC-By 1.0.",
    ),
    "incartdb": Dataset(
        key="incartdb", slug="incartdb", version="1.0.0",
        fs=257, channel=10, role="train", labels="beat",
        patient_re=r"^P(\d+)$",
        stages=("beat",),
        channel_names=("V5",),
        note="St Petersburg INCART 12-lead: 75 half-hour records from 32 Holter "
             "patients aged 18-80 (mean 58), none paced; coronary disease, MI, "
             "paroxysmal VT, RBBB (patient 8), WPW (29, 30). Wide beats that are not "
             "ventricular, labelled by type, which the beat model must not call PVC. "
             "V5 is the lead nearest CM5. Annotations were placed automatically and "
             "corrected by hand, but their positions were not re-aligned, so a few "
             "may sit off the R peak. Record names do not carry the patient: the "
             "header does, and INCART_PATIENT below holds it. ODC-By 1.0.",
    ),

    # -------------------------------------------------------------- validation
    "mitdb": Dataset(
        key="mitdb", slug="mitdb", version="1.0.0",
        fs=360, channel=0, role="val", labels="both",
        patient_re=r"^(\d+)$",
        stages=("beat", "rhythm"),
        note="Universal benchmark. Split by the de Chazal inter-patient division "
             "(config.MITDB_DS1 / MITDB_DS2): DS1 develops the beat model only and "
             "is used by no other stage; DS2 is the held-out test, which is what "
             "published inter-patient results report on. Records 201 and 202 are "
             "the same subject and the division puts them on opposite sides, so 202 "
             "goes with 201. The four paced records are in neither half (AAMI).",
    ),
    "afdb": Dataset(
        key="afdb", slug="afdb", version="1.0.0",
        fs=250, channel=0, role="val", labels="rhythm",
        patient_re=r"^(\d+)$",
        stages=("rhythm",),
        note="Standard AF benchmark. Beat annotations are machine-generated and "
             "uncorrected — use rhythm labels only.",
    ),

    "challenge2011": Dataset(
        key="challenge2011", slug="challenge-2011", version="1.0.0",
        fs=500, channel=10, role="val", labels="none",
        patient_re=r"^(\d+)$",
        stages=("quality",),
        channel_names=("V5", "v5", "V6", "v6"),
        verified=False,
        note="External quality benchmark — the set every published SQA method "
             "reports on (~1539 records, 10 s, 500 Hz, acceptable/unacceptable). "
             "12-lead and resting, so it sits well outside MUNJI's training "
             "domain; that is the point. Labels live in RECORDS-* files, not WFDB "
             "annotations, so use loader.challenge2011_labels(). V5 is chosen as "
             "the closest lead to CM5.",
    ),

    # ------------------------------------------------------------------- noise
    "nstdb": Dataset(
        key="nstdb", slug="nstdb", version="1.0.0",
        fs=360, channel=0, role="noise", labels="none",
        patient_re=r"^(\w+)$",
        stages=(),
        note="Noise library: bw (baseline wander), em (electrode motion), "
             "ma (muscle artifact). Not a training corpus.",
    ),
}

NOISE_RECORDS = ("bw", "em", "ma")

# INCART patient number -> its records, as each record's header states it
# ("patient N"). Split assignment is per patient, so every record of a patient
# must map to the same id; a record missing here fails patient_re loudly
# instead of becoming a patient of its own. tests/test_beat_data.py re-reads
# the headers when the data is present and checks this table against them.
INCART_PATIENT = {
    1: ('I01', 'I02'), 2: ('I03', 'I04', 'I05'), 3: ('I06', 'I07'),
    4: ('I08',), 5: ('I09', 'I10', 'I11'), 6: ('I12', 'I13', 'I14'),
    7: ('I15',), 8: ('I16', 'I17'), 9: ('I18', 'I19'),
    10: ('I20', 'I21', 'I22'), 11: ('I23', 'I24'), 12: ('I25', 'I26'),
    13: ('I27', 'I28'), 14: ('I29', 'I30', 'I31', 'I32'), 15: ('I33', 'I34'),
    16: ('I35', 'I36', 'I37'), 17: ('I38', 'I39'), 18: ('I40', 'I41'),
    19: ('I42', 'I43'), 20: ('I44', 'I45', 'I46'), 21: ('I47', 'I48'),
    22: ('I49', 'I50'), 23: ('I51', 'I52', 'I53'), 24: ('I54', 'I55', 'I56'),
    25: ('I57', 'I58'), 26: ('I59', 'I60', 'I61'), 27: ('I62', 'I63', 'I64'),
    28: ('I65', 'I66', 'I67'), 29: ('I68', 'I69'), 30: ('I70', 'I71'),
    31: ('I72', 'I73'), 32: ('I74', 'I75'),
}

# Records known to share a subject: (dataset, record) -> the name the
# dataset's patient_re reads the patient from. For MITDB that is the first
# record of the subject; for INCART it is "P" and the patient number.
SAME_SUBJECT = {("mitdb", "202"): "201"}
SAME_SUBJECT.update({("incartdb", rec): f"P{p:02d}"
                     for p, recs in INCART_PATIENT.items() for rec in recs})


def by_role(role: Role) -> list[Dataset]:
    return [d for d in REGISTRY.values() if d.role == role]


def by_stage(stage: str, role: Optional[Role] = None) -> list[Dataset]:
    out = [d for d in REGISTRY.values() if stage in d.stages]
    return [d for d in out if d.role == role] if role else out


def get(key: str) -> Dataset:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]
