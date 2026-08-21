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
    note: str = ""


REGISTRY: dict[str, Dataset] = {
    # ---------------------------------------------------------------- training
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
    "nsrdb": Dataset(
        key="nsrdb", slug="nsrdb", version="1.0.0",
        fs=128, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\d+)$",
        stages=("quality", "rhythm"),
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
        fs=128, channel=0, role="train", labels="rhythm",
        patient_re=r"^(\d+)$",
        stages=("rhythm", "quality"),
        note="Long-term AF. ~2000 hours annotated. Critical quality carrier: AF is "
             "irregular by definition, and a gate trained only on NSR learns "
             "'irregular = noise' and hides exactly what MUNJI exists to detect.",
    ),

    # -------------------------------------------------------------- validation
    "mitdb": Dataset(
        key="mitdb", slug="mitdb", version="1.0.0",
        fs=360, channel=0, role="val", labels="both",
        patient_re=r"^(\d+)$",
        stages=("beat", "rhythm"),
        note="Universal benchmark — never train on it. Records 201 and 202 are the "
             "same subject and must share a split.",
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

# Records known to share a subject. Split assignment uses the first name.
SAME_SUBJECT = {("mitdb", "202"): "201"}


def by_role(role: Role) -> list[Dataset]:
    return [d for d in REGISTRY.values() if d.role == role]


def by_stage(stage: str, role: Optional[Role] = None) -> list[Dataset]:
    out = [d for d in REGISTRY.values() if stage in d.stages]
    return [d for d in out if d.role == role] if role else out


def get(key: str) -> Dataset:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]
