"""SVT and the "fast narrow rhythm" label, by the guideline definitions and EC57.

    python experiments/fast_narrow_template.py --download-shdbaf         # SHDB-AF records with SVT, once
    python experiments/fast_narrow_template.py                           # all sets
    python experiments/fast_narrow_template.py --sets cudb vfdb nsrdb    # a quick look
    python experiments/fast_narrow_template.py --max-records 2           # smoke test

Decision (25 November 2026)
---------------------------
The alert:   row "label: onset 150, slower before" (FINAL_ALERT). It sets
             the exit code: no label touching VT or VF, NSRDB at most 0.29
             false labels per 24 h.
The report:  row "doctor report: abrupt start or end" (FINAL_REPORT), an
             entry in the doctor's report, never an alert. No entry touching
             VT or VF, or the exit code is FAIL.
The other rows stay for comparison and are reported only. The rules, their
basis and every result are in docs/fast_narrow_rhythm.md.

Why
---
The first AI document, which the client approved, promises a "fast narrow
rhythm" output: racing, but each beat still has a normal shape (described,
not named). It is off (config.FAST_NARROW_LABEL): it rested on a QRS width
estimate that failed, and the trained delineator read 5 of 11 sustained VT
episodes narrow in Gate C2. A first rule built on the patient's own beat
(template_wct) passed on CUDB, VFDB, SVDB and NSRDB but failed on LTAFDB: 17
labels touched VT (15 in one patient, inside fast AF with short VT runs) and
fast AF gave 2.5 false labels a day. Few SVT episodes over 150 bpm existed to
measure what it catches. This version follows the definitions and the
reporting standard, and adds SHDB-AF, where SVT is labelled by a cardiologist.

Definitions (2015 ACC/AHA/HRS SVT guideline)
--------------------------------------------
SVT: atrial and/or ventricular rate over 100 bpm at rest, from the bundle of
His or above; it excludes atrial fibrillation. Paroxysmal SVT: regular and
rapid, abrupt onset and termination. Typical atrial flutter is a
macro-reentrant atrial tachycardia, inside the SVT umbrella.

The rule, every check traced to a definition (fixed before the first run)
-------------------------------------------------------------------------
beats, template   template_wct's, unchanged (the rate engine's detector on
                  every 40 s window afresh; the patch-on template)
from above        at least 80% of the window's beats keep the patient's own
                  beat: correlation 0.90 or more (the calibration's own match)
                  and size half to twice
regular           median change between successive RR 40 ms or less, over the
                  beats of that run (below) inside the 30 s: the whole window
                  also holds the calm beats before an onset, and at the start
                  of fast AF they made it look regular
abrupt onset      the newest run over 100 bpm with a beat in the 30 s began
                  with the cycle length 9% or more shorter than the 8
                  intervals before (judged, no override). Over 100 because
                  that is where the definition's tachycardia starts. The first
                  full run judged it at the 150 crossing: a tachycardia
                  hovering round 150 restarted its 150 runs gradually, and 7
                  of the 10 missed episodes failed there. Row 4 (as the first
                  run had it) and "Lorenz, as run before" keep the 150 line
                  and the whole window's regularity, for comparison.
slower before     (row "onset 150, slower before", reported) the onset back
                  at the label's line, 150, as ICDs judge it in their rate
                  zone (Brugada 1998: onset 9% and stability 40 ms programmed
                  in zone 1), with regularity on the run's own beats and the
                  Lorenz score; and before the run's first beat the app's own
                  30 s rate was under 150. Onset is a change from a slower
                  rhythm, and ICDs compare the new average interval with the
                  averages before it; a 30 s rate is not fooled by a few
                  missed beats or a stretch of lost signal, which faked
                  abrupt onsets in the exercise test (STDB 317 and 319). A
                  real jog (GUDB subject 04) leapt past 100 abruptly, so the
                  onset stays at 150. 30 s: the app's rate window, our choice.
doctor report     (row "doctor report: SVT/AFL", reported; an entry in the
                  doctor's report, never an alert) 150 bpm or more, the
                  patient's own beat shape, no VT run, and not AF read off the
                  RR: regular with the Lorenz score under 0, or the intervals
                  on whole multiples of one atrial cycle (atrial tachycardia or
                  flutter with variable conduction: 1 to 4 cycles of 171 to
                  400 ms, within 30 ms, for 80% of them, at least two
                  multiples each 15%); no onset check. Every check but the
                  multiples is the alert's; the multiples are our choices,
                  checked on simulated AF and variable block before the run.
abrupt start/end  (row "doctor report: abrupt start or end", reported; a report
                  entry, never an alert) the doctor's report row, and its fast
                  run over 100 began abruptly (the onset test) or ended
                  abruptly (the same test mirrored at its last fast beat: the
                  cycle length after it 9% or more longer). Paroxysmal SVT is
                  abrupt in onset and termination (2015 ACC/AHA/HRS); a fast
                  stretch inside an AF episode, or sinus speeding up and
                  slowing down, is neither. The end is known only once the run
                  is over: the report is written after the episode, an alert
                  could not wait for it. Either one, not both, so an SVT whose
                  start the patch missed still counts.
150 within 30 s   (one row, reported) the run over 100 got to 150 bpm within
                  30 s of its abrupt start. A re-entrant SVT starts at its
                  rate; a sinus rate that leaps at the start of exercise
                  (vagal release) and then climbs on to 150 passes the onset
                  over 100 but takes longer. 30 s is our choice (the decision
                  window), not a published value.
not AF            RhythmNet, as the app runs it: no SVT while the latest 30 s
                  rhythm window has P(AF) at the decision layer's threshold
                  (0.60) or an AF episode is active (4 of 6, closed after 40
                  quiet windows; decision_layer._Episode)
not VT            no run of 3 or more consecutive beats unlike the template
                  (correlation under 0.6 or size outside half to twice, the
                  WCT rule's unlike): 3 consecutive ventricular beats is VT by
                  definition
over 30 s         every window end, over its last 30 s of beats

Not AF by the RR pattern (a candidate in RhythmNet's place, reported)
---------------------------------------------------------------------
Medtronic's insertable monitors see only R waves, as the patch does. They
call AF from a Lorenz plot of successive RR differences over 2 minutes
(Sarkar, Ritscher and Mehra, IEEE TBME 2008): in sinus rhythm or an
organized atrial tachycardia nearly every point sits in the zero segment
round the origin; in AF the points scatter. Here, fixed before the run, from
the published structure (the device's own thresholds are not public):
  score   occupied 25 ms bins outside the +-75 ms zero segment minus the
          points inside it, per point (-1: all at the origin), over the last
          2 minutes of beats (+-500 ms, 25 ms
          bins, 75 ms mask: US 7,623,911's AF setting; intervals of 220 ms or
          less dropped and at least 30 points: US 10,045,710). No PAC term.
  AF      a score over 0. Unclassified (too few points) never blocks.
The label and the report are scored a second time with it instead of
RhythmNet ("Lorenz, not RhythmNet"), against the same bars.

Two outputs from the same checks:
  label    the app's "fast narrow rhythm" (PRIMARY): rate 150 or more
           (config.FAST_NARROW_BPM, the approved document's line); first
           decision of every episode, re-armed after 30 s
  report   SVT for the doctor's report: rate over 100 (the definition);
           consecutive decisions make one episode, from the first beat of
           its fast run to that run's last fast beat; "PSVT" when it also
           ended abruptly (the cycle length after the run 9% or more longer
           than inside it, the onset test mirrored)

ShockableNet is not applied: the decision layer also holds this label while
ShockableNet sees danger; here the rule stands alone, the worst case.

RhythmNet without leakage
-------------------------
NSRDB and LTAFDB trained RhythmNet. With --rhythm-cv (rhythm_fold*.pt and
rhythm_oof.npz from cross_validate --stage rhythm --save-models) each of their
records is scored by the fold model that held its patient out, as
eval/af_alarm_rate.py does. Without it, only their validation patients are
read, with the final model. Every other set is read with the final model
(--rhythm-ckpt), which never saw it.

Scored
------
label    SVT episodes (SVTA; SHDB-AF's AT, PAT, NOD) of 30 s or more should be
         labelled. A label whose 30 s touches VT or VF is UNSAFE. AFL is
         reported apart. Any other label is false, per 24 h, with the rhythm
         label at it.
report   ANSI/AAMI EC57's episode and duration statistics, as for AF: episode
         sensitivity (reference SVT episodes overlapped by a detected
         episode), episode positive predictivity (detected episodes
         overlapping SVT or AFL), duration sensitivity and positive
         predictivity; gross over all records; for all references and for
         those of 30 s or more. Detected episodes touching VT or VF are
         unsafe; the rest that touch no SVT or AFL are false, per 24 h.
gets     (reported) per SVT episode of 30 s or more, everything the app
         sends or writes during it, whatever name it carries, as the decision
         layer runs it: the label (row "onset 150, slower before"); the rate
         engine's notification, 180 bpm or more held 30 s (MCOT, ARIC;
         config.NOTIFY_TACHY_URGENT_BPM), started inside the episode;
         RhythmNet's AF/AFL alert, 4 of 6 windows at 0.60 and at most one per
         24 h (so only the record's first); the doctor's report entry (row
         "doctor report: SVT/AFL"); the report's fast-rate line, 150 or more
         held over 17 beats (HR_TACHY critical). A name the patient gets need
         not be right: an SVT alerted as AF/AFL still reaches the patient. The
         quality gate is not run here, so the rate rows are an upper bound;
         on the 24 h sets the rate engine runs at their 5 s stride.
Test-split patients are skipped; MITDB is not read.

Pass criteria
-------------
PRIMARY   the label: no unsafe label in any set, and no more than 0.29 false
          labels per 24 h on NSRDB (the bar used for the WCT rule; ours)
REPORTED  the label's SVT and AFL catches; the report's EC57 statistics and
          its unsafe and false episodes; the label without the AF check,
          without the VT check, and as the first run had it (neither).
          For each SVT episode of 30 s or more that reached 150 bpm: each
          check's pass share there and the one that kept the label silent.
          The false labels with and without the AF check and with the
          Lorenz score, by rhythm, each with its Lorenz score.
LORENZ    the Lorenz row (onset over 100) against PRIMARY's bars, plus its SVT150 catches and
          its false labels on LTAFDB, SHDB-AF and SVDB; reported, it does not
          set the exit code.
SHDB-AF holds no VT label, so safety rests on CUDB, VFDB, SVDB and LTAFDB.
By default only its records with an SVT mark are downloaded (--all-shdbaf
for all 98 annotated, about 7 GB), so its false labels per 24 h come from
those records only. The VT check sees only the beats the detector finds: VT
beats it misses cannot block a label.

Reads only.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from munji_ai import config as C  # noqa: E402
from munji_ai.data.windows import _episode_bpm  # noqa: E402
from munji_ai.rules.rate_engine import _trimmed_mean  # noqa: E402


def _experiment(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _experiment("template_wct")

SUSTAIN = T.SUSTAIN
POS_STEP = T.POS_STEP
FAST_BPM = float(C.FAST_NARROW_BPM)          # 150, the label
SVT_BPM = 100.0                              # the definition, the report
ALIKE_C = 0.90                               # template_wct.CAL_MATCH
UNLIKE_C = 0.60                              # the WCT rule's unlike
SHARE = 0.80
VT_RUN = 3                                   # consecutive ventricular beats: VT
MAX_FALSE_NSRDB = 0.29
DANGER = ("VT", "VF")
NAMED = ("SVT", "AFL")
# Lorenz plot of successive RR differences (Sarkar, Ritscher and Mehra 2008;
# Medtronic's insertable monitors, which see only R waves): fixed before the run
LZ_SEC = 120.0                               # the device's 2 minute period
LZ_BIN = 0.025                               # 25 ms bins over +-500 ms, NSR mask 75 ms
LZ_RANGE = 0.500                             # (US 7,623,911's optimized AF setting)
LZ_ZERO = 0.075
LZ_MIN = 30                                  # fewer points: unclassified (US 10,045,710)
LZ_RR_MIN = 0.220                            # intervals of 220 ms or less are not used
LZ_AF = 0.0                                  # AF: more scattered bins than points at the origin
BASE = {"R": FAST_BPM, "on": SVT_BPM, "reg_run": True, "af": True, "vt": True, "lz": False}
PRIMARY = "label: all checks"
NO_AF = "label: no AF check"
LORENZ = "label: Lorenz, not RhythmNet"
LORENZ_150 = "label: Lorenz, as run before"
LORENZ_REACH = "label: Lorenz, 150 within 30 s"
SLOWER = "label: onset 150, slower before"
DOCTOR = "doctor report: SVT/AFL"
DOCTOR_PSVT = "doctor report: abrupt start or end"
FINAL_ALERT = SLOWER          # chosen 25 Nov 2026: docs/fast_narrow_rhythm.md
FINAL_REPORT = DOCTOR_PSVT    # chosen 25 Nov 2026
# RR multiples (atrial tachycardia or flutter with variable conduction): the
# intervals sit on whole multiples of one atrial cycle (LITFL; Medtronic's AT
# detector for organized atrial rhythms). Our choices, fixed before the run
# and checked on simulated AF (never passes) and variable block (passes).
ORG_P = (0.171, 0.400)                       # atrial cycle searched: 150 to 350 bpm
ORG_STEP = 0.002
ORG_TOL = 0.030                              # an interval fits within 30 ms of k cycles
ORG_KMAX = 4                                 # up to 4:1 conduction
ORG_COVER = 0.80                             # 80% of the intervals fit
ORG_SHARE = 0.15                             # at least two multiples, each 15% of them
ORG_MIN_RR = 12
REACH_S = SUSTAIN                            # our choice: the decision window, not a published value
REPORT = "report: over 100 bpm"
REPORT_LZ = "report: Lorenz, not RhythmNet"
# What the patient and the doctor get during an SVT, whatever its name: the
# app's own lines, read from config, none chosen here
NOTIFY_BPM = float(C.NOTIFY_TACHY_URGENT_BPM)  # 180: MCOT, ARIC
NOTIFY_SEC = float(C.NOTIFY_SUSTAIN_SEC)       # held 30 s
REPORT_FAST_BPM = float(C.HR_TACHY["critical"])  # 150: the report's critical tachycardia tier
TACHY_BEATS = int(C.RATE_RUN_BEATS["tachy"])     # held over 17 beats
CONFIGS = [
    (PRIMARY, BASE),
    (NO_AF, {**BASE, "af": False}),
    ("label: no VT check", {**BASE, "vt": False}),
    ("label: first run (neither)", {**BASE, "on": FAST_BPM, "reg_run": False, "af": False, "vt": False}),
    (LORENZ, {**BASE, "af": False, "lz": True}),
    (LORENZ_150, {**BASE, "on": FAST_BPM, "reg_run": False, "af": False, "lz": True}),
    (LORENZ_REACH, {**BASE, "af": False, "lz": True, "reach": True}),
    (SLOWER, {**BASE, "on": FAST_BPM, "af": False, "lz": True, "slower": True}),
    (DOCTOR, {**BASE, "af": False, "lz": True, "organized": True, "no_onset": True}),
    (DOCTOR_PSVT, {**BASE, "af": False, "lz": True, "organized": True, "no_onset": True, "abrupt": True}),
    (REPORT, {**BASE, "R": SVT_BPM}),
    (REPORT_LZ, {**BASE, "R": SVT_BPM, "af": False, "lz": True}),
]
REPORTS = (REPORT, REPORT_LZ)
LONG_SETS = {"nsrdb", "ltafdb", "shdbaf"}    # 24 h: read at the 5 s stride
SPAN_SETS = {"nsrdb", "ltafdb"}              # read over their annotated span
RHYTHM_TRAINED = {"nsrdb", "ltafdb", "icentia11k", "challenge2017"}
SHDB_URL = "https://physionet-open.s3.amazonaws.com/shdb-af/1.0.1/"
SHDB_MAX_ID = 160                            # 128 recordings, numbered with gaps
SHDB_TOKENS = {"(AT": "(SVTA", "(PAT": "(SVTA", "(NOD": "(SVTA", "(AFIB": "(AFIB", "(AFL": "(AFL", "(N": "(N"}
NAN = float("nan")


# ------------------------------------------------------------------ SHDB-AF
def shdb_tokens(aux: list) -> tuple:
    """(indices, tokens) of SHDB-AF's rhythm marks in MUNJI's tokens, one per
    change: the marks are per beat, so repeats are dropped. SHDB-AF's (NOD is
    an intranodal tachycardia, not the nodal rhythm (NOD means elsewhere), and
    (AT and (PAT are atrial and other SVT: all three are SVTA here."""
    idx, tok, last = [], [], None
    for i, a in enumerate(aux):
        a = str(a).split("\x00")[0].strip()
        if not a.startswith("("):
            continue
        t = SHDB_TOKENS.get(a, a)
        if t != last:
            idx.append(i)
            tok.append(t)
            last = t
    return idx, tok


def read_shdb(path: Path):
    """One SHDB-AF record as a loader.Record: channel ECG1 on the model path,
    rhythm marks mapped (shdb_tokens), the whole record as its span."""
    import wfdb
    from munji_ai.data.loader import Record
    from munji_ai.data.preprocess import model_path
    stem = str(Path(path).with_suffix(""))
    rec = wfdb.rdrecord(stem)
    names = [str(s).strip().upper() for s in rec.sig_name]
    ch = names.index("ECG1") if "ECG1" in names else 0
    fs_in = int(round(rec.fs))
    raw = np.nan_to_num(np.asarray(rec.p_signal[:, ch], np.float64))
    x = model_path(raw, fs_in)
    ann = wfdb.rdann(stem, "atr")
    i, tok = shdb_tokens(list(ann.aux_note or []))
    ratio = len(x) / max(len(raw), 1)
    starts = np.round(np.asarray(ann.sample)[i] * ratio).astype(np.int64) if i else np.zeros(0, np.int64)
    name = Path(path).stem
    return Record(dataset="shdbaf", name=name, patient=f"shdbaf:{name}", signal=x, fs=C.TARGET_FS,
                  beat_samples=np.array([0, len(x) - 1]), beat_symbols=np.array(["N", "N"]),
                  rhythm_starts=starts, rhythm_labels=np.array(tok, dtype=object))


def shdb_records(root: Path) -> list:
    """Annotated SHDB-AF records with their signal (.atr and .dat beside the .hea)."""
    return sorted((p for p in Path(root).rglob("*.hea")
                   if p.with_suffix(".atr").exists() and p.with_suffix(".dat").exists()), key=lambda p: p.stem)


def _fetch(url: str, dest: Path, tries: int = 5) -> str:
    """One file, skipped when already whole, retried on passing errors.
    'missing' when the server says it does not exist (404)."""
    import urllib.error
    import urllib.request
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                size = int(r.headers.get("Content-Length", -1))
            if dest.exists() and dest.stat().st_size == size:
                return "kept"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if size >= 0 and tmp.stat().st_size != size:
                raise IOError(f"got {tmp.stat().st_size} of {size} bytes")
            tmp.replace(dest)
            return "fetched"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            if k == tries - 1:
                return f"failed: HTTP {e.code}"
            time.sleep(2 * (k + 1))
        except Exception as e:                       # noqa: BLE001  retried, then reported
            if k == tries - 1:
                return f"failed: {type(e).__name__}: {e}"
            time.sleep(2 * (k + 1))
    return "failed"


def has_svt(atr: Path) -> bool:
    """Does an SHDB-AF annotation file hold an SVT mark (AT, PAT or NOD)?"""
    import wfdb
    try:
        ann = wfdb.rdann(str(atr.with_suffix("")), "atr")
    except Exception:                                # noqa: BLE001  unreadable: not counted
        return False
    return "(SVTA" in shdb_tokens(list(ann.aux_note or []))[1]


def download_shdb(root: Path, workers: int = 8, base: str = SHDB_URL, only_svt: bool = True) -> list:
    """SHDB-AF from PhysioNet's open copy on Amazon S3: every .hea and .atr
    (small), then .dat (about 70 MB each) for the annotated records, only
    those with an SVT mark unless only_svt is False. Returns the files that
    failed."""
    import urllib.error
    import urllib.request
    root.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(base + "RECORDS", timeout=60) as r:
            names = [ln.strip() for ln in r.read().decode().splitlines() if ln.strip()]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # SHDB-AF 1.0.1 has no RECORDS file: its records are 001, 002, ...
        # with gaps (duplicates removed), so every number is tried and the
        # ones that do not exist answer 404
        names = [f"{k:03d}" for k in range(1, SHDB_MAX_ID + 1)]
    failed = []

    def run(files):
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(_fetch, base + f, root / f): f for f in files}
            for k, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res.startswith("failed"):
                    failed.append(f"{futs[fut]} ({res})")
                if k % 25 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} files", flush=True)
    print(f"SHDB-AF: up to {len(names)} records into {root}", flush=True)
    run([f"{n}.{e}" for n in names for e in ("hea", "atr")])
    with_atr = [n for n in names if (root / f"{n}.atr").exists() and (root / f"{n}.hea").exists()]
    want = [n for n in with_atr if has_svt(root / f"{n}.atr")] if only_svt else with_atr
    print(f"{len(with_atr)} annotated, {len(want)} " + ("with SVT marks; " if only_svt else "")
          + f"their signals, about {len(want) * 70 / 1000:.1f} GB", flush=True)
    run([f"{n}.dat" for n in want])
    return failed


# ------------------------------------------------------------------ RhythmNet
def _load_rhythm(path: Path, device):
    import torch
    from munji_ai.models.rhythm_net import build
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build()
    m.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    return m.to(device).eval()


def af_block(p_af: np.ndarray, offset_s: float = 0.0) -> tuple:
    """(window end times in s, blocked) from P(AF) per 30 s window at the
    deployment stride: blocked where that window says AF or an AF episode is
    active, as decision_layer._Episode runs it. offset_s: where the analysed
    span starts in the record."""
    from munji_ai.decision.decision_layer import DecisionConfig, _Episode
    cfg = DecisionConfig()
    ep = _Episode(cfg.af_rule, cfg.af_episode_end)
    pos = np.asarray(p_af) >= cfg.af_threshold
    blocked = np.zeros(len(pos), bool)
    for i, v in enumerate(pos):
        ep.step(bool(v))
        blocked[i] = bool(v) or ep.active
    ends = (np.arange(len(pos)) * C.WINDOW_STRIDE_SEC["rhythm"] + C.WINDOW_SEC["rhythm"]) - offset_s
    return ends, blocked


def blocked_at(t: np.ndarray, af) -> np.ndarray:
    """Per decision time: the latest rhythm window that has ended says AF."""
    if af is None:
        return np.zeros(len(t), bool)
    ends, blocked = af
    k = np.searchsorted(ends, t, side="right") - 1
    return np.where(k >= 0, blocked[np.clip(k, 0, None)] if len(blocked) else False, False)


def af_alert_times(p_af: np.ndarray, offset_s: float = 0.0) -> np.ndarray:
    """When the decision layer sends its AF/AFL alert (DecisionLayer.on_rhythm):
    an AF episode confirmed (4 of 6 windows at 0.60) with no AF alert in the
    24 h before. Seconds from the analysed span's start."""
    from munji_ai.decision.decision_layer import DecisionConfig, _Episode
    cfg = DecisionConfig()
    ep = _Episode(cfg.af_rule, cfg.af_episode_end)
    out, last = [], -np.inf
    for i, v in enumerate(np.asarray(p_af) >= cfg.af_threshold):
        t = i * C.WINDOW_STRIDE_SEC["rhythm"] + C.WINDOW_SEC["rhythm"]
        if ep.step(bool(v)) and t - last >= cfg.af_realert_s:
            last = t
            out.append(t - offset_s)
    return np.asarray(out, float)


def _merged(spans) -> list:
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def rate_engine_near(ends: np.ndarray, wbeats: list, fs: int, step: float, spans) -> tuple:
    """The rate engine and the decision layer's tachycardia family on the 40 s
    windows near the spans (s, e) given: (times a tachycardia notification
    starts, times a window holds the report's fast-rate line). Each span is
    read from far enough before it (the episode's quiet windows plus one
    window) that a notification already open before it is not counted as new.
    Only near the spans: held_rates_for is slow on a 24 h record."""
    from munji_ai.decision.decision_layer import DecisionConfig, RATE_FAMILIES, _Episode
    from munji_ai.rules.rate_engine import held_rates, held_rates_for
    t = np.asarray(ends) / fs
    quiet = DecisionConfig().rate_episode_end
    look = quiet * step + C.RULES_WINDOW_SEC
    starts, fast = [], []
    for s, e in _merged(spans):
        ep = _Episode(tuple(C.CONFIRMATION[RATE_FAMILIES["tachy"]]), quiet)
        for w in np.flatnonzero((t >= s - look) & (t <= e)):
            rr = np.diff(wbeats[w]) / fs
            _, f30 = held_rates_for(rr, NOTIFY_SEC)
            _, f17 = held_rates(rr, TACHY_BEATS)
            if ep.step(bool(np.isfinite(f30) and f30 >= NOTIFY_BPM)):
                starts.append(t[w])
            if np.isfinite(f17) and f17 >= REPORT_FAST_BPM:
                fast.append(t[w])
    return np.unique(starts), np.unique(fast)


def first_in(times: np.ndarray | None, s: float, e: float):
    """Seconds from s to the first time in [s, e]; None if none; NaN when
    the source did not run (times None)."""
    if times is None:
        return NAN
    hit = np.asarray(times)[(np.asarray(times) >= s) & (np.asarray(times) <= e)]
    return float(hit[0] - s) if hit.size else None


# ------------------------------------------------------------------ Lorenz
def af_score(rr: np.ndarray) -> float:
    """The Lorenz plot AF score: occupied 25 ms bins outside the +-75 ms zero
    segment (irregularity evidence) minus the points inside it (origin
    count), per point so that 2 minutes at any rate compare: -1 when every
    point sits at the origin, above 0 when the scatter outweighs it. A point is (dRR_i, dRR_i-1), three intervals in a row; triplets
    with an interval of 220 ms or less are dropped; points past +-500 ms fall
    in the edge bins. NaN with fewer than 30 points. The device also subtracts
    a PAC evidence term; without it, ectopy only raises the score, so it can
    only block an SVT label, never make one."""
    rr = np.asarray(rr, float)
    if rr.size < LZ_MIN + 2:
        return NAN
    d = np.diff(rr)
    x, y = d[1:], d[:-1]
    keep = (rr[:-2] > LZ_RR_MIN) & (rr[1:-1] > LZ_RR_MIN) & (rr[2:] > LZ_RR_MIN)
    x, y = x[keep], y[keep]
    if x.size < LZ_MIN:
        return NAN
    zero = (np.abs(x) <= LZ_ZERO) & (np.abs(y) <= LZ_ZERO)
    lim = LZ_RANGE - 1e-9
    bx = np.floor(np.clip(x[~zero], -lim, lim) / LZ_BIN).astype(np.int64)
    by = np.floor(np.clip(y[~zero], -lim, lim) / LZ_BIN).astype(np.int64)
    occupied = np.unique(bx * 1000 + by).size
    return float(occupied - int(zero.sum())) / x.size


def lorenz_scores(t: np.ndarray, ts: np.ndarray, need: np.ndarray) -> np.ndarray:
    """af_score over the beats of the last 2 minutes at every decision time
    where need is True (NaN elsewhere)."""
    out = np.full(len(t), NAN)
    lo = np.searchsorted(ts, t - LZ_SEC, side="right")
    hi = np.searchsorted(ts, t, side="right")
    for w in np.flatnonzero(need):
        out[w] = af_score(np.diff(ts[lo[w]:hi[w]]))
    return out


# ------------------------------------------------------------------ checks
def alike(corr: dict, c: float) -> np.ndarray:
    """Keeps the template's shape (correlation c or more) and size (half to
    twice). No template (NaN): never alike."""
    with np.errstate(invalid="ignore"):
        return ((corr["shape"] >= c) & (corr["size"] >= 1.0 / T.SIZE_FACTOR)
                & (corr["size"] <= T.SIZE_FACTOR))


def longest_run(flags: np.ndarray) -> int:
    """Longest stretch of consecutive True."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def runs_of(start: np.ndarray) -> dict:
    """Run start index -> last index, from onset_runs' per-beat start."""
    last = {}
    for i, s in enumerate(start):
        if s >= 0:
            last[int(s)] = i
    return last


def termination(ts: np.ndarray, j: int) -> float:
    """The onset test mirrored at a run's last fast beat j: 1 - (the running
    cycle length there) / (median of the next 8 intervals). NaN when the
    recording ends first."""
    rr = np.diff(ts)
    if j < T.RUN_RR or j + T.ONSET_BEFORE > len(rr):
        return NAN
    return float(1.0 - np.median(rr[j - T.RUN_RR:j]) / np.median(rr[j:j + T.ONSET_BEFORE]))


def sudden_onset(t_end: np.ndarray, ts: np.ndarray, start: np.ndarray, drop: dict) -> tuple:
    """Per window end: (sudden, drop, run start) of the newest fast run with a
    beat in the last 30 s. No such run, or a drop that cannot be judged: not
    sudden."""
    n = len(t_end)
    sudden, d_out, s_out = np.zeros(n, bool), np.full(n, np.nan), np.full(n, -1, np.int64)
    if len(ts) == 0:
        return sudden, d_out, s_out
    newest = np.maximum.accumulate(np.where(start >= 0, np.arange(len(ts)), -1))
    i_last = np.searchsorted(ts, t_end, side="right") - 1
    for w, i in enumerate(i_last):
        j = int(newest[i]) if i >= 0 else -1
        if j < 0 or ts[j] <= t_end[w] - SUSTAIN:
            continue
        s0 = int(start[j])
        d = drop.get(s0, np.nan)
        d_out[w], s_out[w] = d, s0
        sudden[w] = bool(np.isfinite(d) and d >= T.ONSET_DROP)
    return sudden, d_out, s_out


def decisions(ends, wbeats, pos, corr, fs, ts) -> dict:
    """At every window end, over its last 30 s of its own beats: enough beats,
    rate, regularity, share alike, the longest run of unlike beats, and per
    rate line the onset of the newest fast run."""
    n = len(ends)
    ok = np.zeros(n, bool)
    bpm, stab, share = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    vt_run = np.zeros(n, np.int64)
    al = alike(corr, ALIKE_C)
    un = T.unlike(corr, ("shape", True, UNLIKE_C))
    S30 = int(SUSTAIN * fs)
    need = SUSTAIN - 60.0 / C.VT_MIN_BPM
    rmin = min(c["R"] for _, c in CONFIGS)
    for w in range(n):
        b = wbeats[w]
        b = b[b >= ends[w] - S30]
        if b.size < 3 or (b[-1] - b[0]) / fs < need:
            continue
        idx = T.index_of(pos, b)
        if idx.size < 3:
            continue
        ok[w] = True
        m = _trimmed_mean(np.sort(np.diff(b) / fs), C.RATE_RUN_TRIM)
        bpm[w] = 60.0 / m if m > 0 else np.nan
        if bpm[w] > rmin:
            stab[w] = T.instability_ms(b, fs)
            share[w] = float(np.mean(al[idx]))
            vt_run[w] = longest_run(un[idx])
    t_end = ends / fs
    with np.errstate(invalid="ignore"):
        lz = lorenz_scores(t_end, ts, ok & (bpm > rmin))
    onset = {}
    for R in {c["R"] for _, c in CONFIGS} | {c["on"] for _, c in CONFIGS}:
        start, drop = fast_runs(ts, R)
        onset[R] = sudden_onset(t_end, ts, start, drop) + (start,)
    before = {R: rate_before(t_end, ok, bpm, ts, onset[R][2]) for R in onset}
    reach = reach_times(ts, onset[SVT_BPM][3], FAST_BPM)
    reach_w = np.array([reach.get(int(s0), NAN) if s0 >= 0 else NAN for s0 in onset[SVT_BPM][2]])
    # regular, over the beats of the newest run over 100 inside the 30 s only:
    # the whole window holds the calm beats before an onset as well
    stab_run = np.full(n, np.nan)
    org = np.zeros(n, bool)
    lo30 = np.searchsorted(ts, t_end - SUSTAIN, side="left")
    hi30 = np.searchsorted(ts, t_end, side="right")
    for w in np.flatnonzero(np.isfinite(stab)):
        s0 = int(onset[SVT_BPM][2][w])
        if s0 >= 0:
            rr = np.diff(ts[max(s0, lo30[w]):hi30[w]])
            if rr.size >= 3:
                stab_run[w] = float(np.median(np.abs(np.diff(rr))) * 1000.0)
            org[w] = organized_rr(rr)[0]
    # the newest run over 100: did it start abruptly, did it end abruptly (its
    # end is in the future of the decision: for the report only)
    last100 = runs_of(onset[SVT_BPM][3])
    start_drop, end_rise = onset[SVT_BPM][1].copy(), np.full(n, NAN)
    ends_of = {}
    for w in np.flatnonzero(onset[SVT_BPM][2] >= 0):
        s0 = int(onset[SVT_BPM][2][w])
        if s0 not in ends_of:
            j1 = last100.get(s0, -1)
            ends_of[s0] = termination(ts, j1) if j1 >= 0 else NAN
        end_rise[w] = ends_of[s0]
    with np.errstate(invalid="ignore"):
        abrupt = (start_drop >= T.ONSET_DROP) | (end_rise >= T.ONSET_DROP)
    return {"t": t_end, "ok": ok, "bpm": bpm, "stab": stab, "share": share, "vt_run": vt_run, "onset": onset,
            "lz": lz, "reach": reach_w, "stab_run": stab_run, "before": before, "organized": org,
            "start_drop": start_drop, "end_rise": end_rise, "abrupt": abrupt}


def rate_before(t_end: np.ndarray, ok: np.ndarray, bpm: np.ndarray, ts: np.ndarray, s_out: np.ndarray) -> np.ndarray:
    """Per decision: the app's own 30 s rate at the last decision that ended
    before the first beat of its newest fast run (s_out); NaN when there is
    no such run or that decision had no usable rate (a gap, noise)."""
    out = np.full(len(t_end), NAN)
    has = s_out >= 0
    if not has.any():
        return out
    w = np.searchsorted(t_end, ts[s_out[has]], side="right") - 1
    vals = np.full(w.size, NAN)
    good = w >= 0
    vals[good] = np.where(ok[w[good]], bpm[w[good]], NAN)
    out[has] = vals
    return out


_ORG_GRID = np.arange(ORG_P[0], ORG_P[1] + 1e-9, ORG_STEP)


def organized_rr(rr) -> tuple:
    """(organized, coverage, cycle): do the intervals sit on whole multiples
    (1 to 4) of one atrial cycle between 171 and 400 ms, within 30 ms, for
    80% of them, with at least two multiples each holding 15%? One multiple
    alone is a regular rhythm, which the regular check covers."""
    rr = np.asarray(rr, float)
    if rr.size < ORG_MIN_RR:
        return False, NAN, NAN
    k = np.rint(rr[None, :] / _ORG_GRID[:, None])
    fit = (k >= 1) & (k <= ORG_KMAX) & (np.abs(rr[None, :] - k * _ORG_GRID[:, None]) <= ORG_TOL)
    cov = fit.mean(axis=1)
    b = int(np.argmax(cov))
    ks, cnt = np.unique(k[b][fit[b]], return_counts=True)
    two = int((cnt / rr.size >= ORG_SHARE).sum()) >= 2
    return bool(cov[b] >= ORG_COVER and two), float(cov[b]), float(_ORG_GRID[b])


def reach_times(ts: np.ndarray, start: np.ndarray, R: float) -> dict:
    """Run start index (onset_runs' runs) -> seconds from its first beat to
    its first beat with the running rate at R or more; NaN: never."""
    out = {}
    if len(ts) <= T.RUN_RR:
        return out
    run = np.full(len(ts), NAN)
    run[T.RUN_RR:] = np.median(np.lib.stride_tricks.sliding_window_view(np.diff(ts), T.RUN_RR), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        at = 60.0 / run >= R - 1e-9
    for s0, j in runs_of(start).items():
        k = np.flatnonzero(at[s0:j + 1])
        out[s0] = float(ts[s0 + k[0]] - ts[s0]) if k.size else NAN
    return out


def fast_runs(ts: np.ndarray, R: float):
    """onset_runs at a rate line: over 100 for the definition, 150 or more
    for the label."""
    return T.onset_runs(ts, R if R == SVT_BPM else R - 1e-9)


CHECKS = ("alike", "regular", "onset", "VT", "not AF", "Lorenz", "reach", "before", "organized", "abrupt")
ROW_CHECKS = {PRIMARY: CHECKS[:5], NO_AF: CHECKS[:4], LORENZ: CHECKS[:4] + ("Lorenz",),
              LORENZ_150: CHECKS[:4] + ("Lorenz",), LORENZ_REACH: CHECKS[:4] + ("Lorenz", "reach"),
              SLOWER: CHECKS[:4] + ("Lorenz", "before"), DOCTOR: ("alike", "VT", "RR ok"),
              DOCTOR_PSVT: ("alike", "VT", "RR ok", "abrupt")}


def passes(dec: dict, af_blocked: np.ndarray, c: dict) -> dict:
    """Per decision, each check on its own (the rate apart), with the row c's
    settings: the onset judged on the fast run over c["on"]; regular over the
    run's own beats (c["reg_run"]) or the whole 30 s. Lorenz: not AF by the Lorenz score;
    unclassified (NaN) never blocks. reach: the run over 100 got to 150 within
    30 s of its start. before: the app's 30 s rate just before the run's first
    beat was under the run's line (no usable rate there fails)."""
    lz = dec.get("lz", np.full(len(dec["t"]), NAN))
    stab = dec["stab_run"] if c.get("reg_run") else dec["stab"]
    with np.errstate(invalid="ignore"):
        return {"alike": dec["share"] >= SHARE, "regular": stab <= T.STABILITY_MS,
                "onset": dec["onset"][c["on"]][0], "VT": dec["vt_run"] < VT_RUN, "not AF": ~af_blocked,
                "Lorenz": ~(lz > LZ_AF),
                "reach": dec.get("reach", np.full(len(dec["t"]), NAN)) <= REACH_S,
                "before": dec["before"][c["on"]] < c["on"] if "before" in dec else np.zeros(len(dec["t"]), bool),
                "organized": dec.get("organized", np.zeros(len(dec["t"]), bool)),
                "abrupt": dec.get("abrupt", np.zeros(len(dec["t"]), bool)),
                "RR ok": ((stab <= T.STABILITY_MS) & ~(lz > LZ_AF)) | dec.get("organized", np.zeros(len(dec["t"]), bool))}


def condition(dec: dict, c: dict, af_blocked: np.ndarray) -> np.ndarray:
    p = passes(dec, af_blocked, c)
    with np.errstate(invalid="ignore"):
        rate = dec["bpm"] > c["R"] if c["R"] == SVT_BPM else dec["bpm"] >= c["R"]
    cond = dec["ok"] & rate & p["alike"]
    if c.get("organized"):
        cond &= p["RR ok"]                       # regular and Lorenz, or RR multiples
    else:
        cond &= p["regular"]
    if not c.get("no_onset"):
        cond &= p["onset"]
    if c["vt"]:
        cond &= p["VT"]
    if c["af"]:
        cond &= p["not AF"]
    if c["lz"] and not c.get("organized"):
        cond &= p["Lorenz"]
    if c.get("reach"):
        cond &= p["reach"]
    if c.get("slower"):
        cond &= p["before"]
    if c.get("abrupt"):
        cond &= p["abrupt"]
    return cond


def why(p: dict, fast: np.ndarray, checks) -> str:
    """Why the label stayed silent over an episode's decisions at 150 bpm or
    more: 'slower' when none reached it, '-' when one passed every check, else
    the check that failed in the most of them ('+' joins a tie)."""
    if fast.size == 0:
        return "slower"
    fails = np.stack([~p[k][fast] for k in checks])
    if (~fails.any(0)).any():
        return "-"
    n = fails.sum(1)
    return "+".join(k for k, v in zip(checks, n) if v == n.max())


def diagnose(dec: dict, ps: dict, s: float, e: float, step: float) -> dict:
    """One reference episode, over the decisions whose 30 s lies at least half
    inside it: the fastest 30 s, the time at 150 bpm or more, each check's pass
    share there (as the new rows judge them; onset150 as before, at the 150
    crossing), the median Lorenz score, and why each row of ROW_CHECKS
    stayed silent. ps: passes() per row of ROW_CHECKS."""
    p = ps[LORENZ]
    ov = np.minimum(dec["t"], e) - np.maximum(dec["t"] - SUSTAIN, s)
    win = np.flatnonzero(dec["ok"] & (ov >= SUSTAIN / 2))
    bpm = dec["bpm"][win]
    with np.errstate(invalid="ignore"):
        fast = win[bpm >= FAST_BPM]
    return {"peak": float(np.nanmax(bpm)) if np.isfinite(bpm).any() else NAN,
            "fast_s": float(fast.size * step),
            "pass": {**{k: float(np.mean(p[k][fast])) if fast.size else NAN for k in CHECKS},
                     "onset150": float(np.mean(ps[LORENZ_150]["onset"][fast])) if fast.size else NAN},
            "lz": float(np.nanmedian(dec["lz"][win])) if np.isfinite(dec["lz"][win]).any() else NAN,
            "why": {n: why(ps[n], fast, ch) for n, ch in ROW_CHECKS.items()}}


def episodes_of(t: np.ndarray, cond: np.ndarray, ok: np.ndarray | None = None) -> list:
    """Consecutive true decisions as (first, last) decision times. A window
    without 30 s of beats (ok False, an edge effect of the beat grid) decides
    nothing, so it neither ends nor starts an episode."""
    ok = np.ones(len(t), bool) if ok is None else ok
    out, cur = [], None
    for tt, c, k in zip(t, cond, ok):
        if not k:
            continue
        if c:
            cur = [cur[0], tt] if cur else [tt, tt]
        elif cur:
            out.append(tuple(cur))
            cur = None
    if cur:
        out.append(tuple(cur))
    return out


def classify(a: float, ivs) -> str:
    """'unsafe:VT', 'unsafe:VF', 'SVT', 'AFL', or 'false:<label at the
    alert>'. Unsafe when the alert's 30 s touches VT or VF at all."""
    return classify_span(a - SUSTAIN, a, ivs, a)


def classify_span(s: float, e: float, ivs, at: float) -> str:
    for kind in DANGER:
        if any(lab == kind and lo < e and s < hi for lo, hi, lab in ivs):
            return "unsafe:" + kind
    for kind in NAMED:
        if any(lab == kind and lo < e and s < hi for lo, hi, lab in ivs):
            return kind
    return "false:" + next((lab for lo, hi, lab in ivs if lo <= at < hi), "unlabelled")


# ------------------------------------------------------------------ record
def process(rec, negative: bool, step: float, af=None, af_fire=None) -> dict:
    """The label and the report for one record. af: af_block's output for the
    analysed span, or None when no RhythmNet was run (the AF check then never
    blocks, and af_checked is False). af_fire: af_alert_times for the same
    span, or None (the AF/AFL alert column is then not known)."""
    b = T.beats_and_template(rec, step, annotated_span=negative)
    x, fs, ivs = b["x"], b["fs"], b["ivs"]
    ends, wbeats, pos, si, ts, first, corr = (b[k] for k in ("ends", "wbeats", "pos", "si", "ts", "first", "corr"))
    dec = decisions(ends, wbeats, pos, corr, fs, ts)
    blk = blocked_at(dec["t"], af)
    cond = {name: condition(dec, c, blk) for name, c in CONFIGS}
    alerts = {name: T.alert_times(dec["t"], cond[name], step) for name, c in CONFIGS if name not in REPORTS}

    start100 = dec["onset"][SVT_BPM][3]
    last = runs_of(start100)
    reports = {}
    for name in REPORTS:
        reports[name] = []
        for f0, f1 in episodes_of(dec["t"], cond[name], dec["ok"]):
            w0 = int(np.argmin(np.abs(dec["t"] - f0)))
            w1 = int(np.argmin(np.abs(dec["t"] - f1)))
            a0, a1 = int(dec["onset"][SVT_BPM][2][w0]), int(dec["onset"][SVT_BPM][2][w1])
            j1 = last.get(a1, -1) if a1 >= 0 else -1
            rise = termination(ts, j1) if j1 >= 0 else NAN
            # the episode runs from its fast run's first beat to that run's
            # last fast beat (the decisions trail by up to 30 s on both sides)
            s_ep = float(ts[a0]) if a0 >= 0 else max(0.0, f0 - SUSTAIN)
            e_ep = float(ts[j1]) if j1 >= 0 else f1
            reports[name].append({"s": s_ep, "e": max(e_ep, s_ep), "end_rise": rise,
                                  "psvt": bool(np.isfinite(rise) and rise >= T.ONSET_DROP)})
    lz_at = {name: {float(a): float(dec["lz"][int(np.argmin(np.abs(dec["t"] - a)))]) for a in al}
             for name, al in alerts.items()}
    detail_rows = {}
    for name in ROW_CHECKS:
        on = dict(CONFIGS)[name]["on"]
        detail_rows[name] = {}
        for a in alerts[name]:
            w = int(np.argmin(np.abs(dec["t"] - a)))
            detail_rows[name][float(a)] = {"bpm": float(dec["bpm"][w]), "drop": float(dec["onset"][on][1][w]),
                                           "before": float(dec["before"][on][w]), "lz": float(dec["lz"][w]),
                                           "stab_run": float(dec["stab_run"][w]),
                                           "start_drop": float(dec["start_drop"][w]),
                                           "end_rise": float(dec["end_rise"][w])}

    detail = {}
    for a in alerts[PRIMARY]:
        w = int(np.argmin(np.abs(dec["t"] - a)))
        detail[float(a)] = {"bpm": float(dec["bpm"][w]), "share": float(dec["share"][w]),
                            "onset": float(dec["onset"][BASE["on"]][1][w]), "stab": float(dec["stab"][w])}
    al = alike(corr, ALIKE_C)
    _, drop_on = fast_runs(ts, BASE["on"])
    ps = {n: passes(dec, blk, dict(CONFIGS)[n]) for n in ROW_CHECKS}   # blk never blocks without RhythmNet
    # the rate engine only where patient_gets reads it: SVT of 30 s or more
    spans = [(s, e + SUSTAIN) for s, e, lab in ivs if lab == "SVT" and e - s >= SUSTAIN]
    r_start, r_fast = rate_engine_near(ends, wbeats, fs, step, spans) if spans else (np.zeros(0), np.zeros(0))
    episodes = []
    for s, e, lab in ivs:
        if lab not in NAMED:
            continue
        i, j = np.searchsorted(ts, [s, e])
        k = si[i:j]
        ref = _episode_bpm(x[int(s * fs):int(e * fs)], fs)
        near = [d for s0, d in sorted(drop_on.items()) if s - 10.0 <= ts[s0] <= s + SUSTAIN]
        wi = np.flatnonzero((dec["t"] > s) & (dec["t"] <= e + SUSTAIN))
        episodes.append({"label": lab, "s": s, "e": e, "dur": e - s,
                         "rate_sig": float(ref) if ref else NAN,
                         "rate_det": float(60.0 / np.median(np.diff(ts[i:j]))) if j - i >= 3 else NAN,
                         "alike": float(np.mean(al[k])) if first is not None and k.size else NAN,
                         "stab": T.instability_ms(pos[k], fs) if k.size else NAN,
                         "onset": near[0] if near else NAN,
                         "af": float(np.mean(blk[wi])) if wi.size and af is not None else NAN,
                         "template": "none" if first is None else ("before" if first[1] <= s else "after"),
                         "rate_alert": first_in(r_start, s, e + SUSTAIN) if (s, e + SUSTAIN) in spans else NAN,
                         "af_alert": first_in(af_fire, s, e + SUSTAIN),
                         "fast_report": (s, e + SUSTAIN) in spans and first_in(r_fast, s, e + SUSTAIN) is not None,
                         **diagnose(dec, ps, s, e, step)})
    return {"dataset": rec.dataset, "name": rec.name, "patient": rec.patient, "hours": len(x) / fs / 3600.0,
            "n_beats": int(len(ts)), "ivs": ivs, "first": first[1] if first else None, "af_checked": af is not None,
            "alerts": alerts, "report": reports[REPORT], "report_lz": reports[REPORT_LZ], "lz_at": lz_at,
            "detail": detail, "detail_rows": detail_rows, "episodes": episodes}


_MODELS: dict = {}


def _work(job) -> dict:
    """One record, in a worker process: read it, run RhythmNet, then the rule."""
    ds, name, raw, step, rhythm, device = job
    t1 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        from munji_ai.data.loader import load_record
        rec = read_shdb(Path(name)) if ds == "shdbaf" else load_record(ds, name, raw)
        lo = int(rec.beat_samples.min()) if ds in SPAN_SETS and len(rec.beat_samples) >= 2 else 0
        af = af_fire = None
        if rhythm is not None:
            if rhythm not in _MODELS:
                _MODELS[rhythm] = _load_rhythm(Path(rhythm), torch.device(device))
            af_eval = _experiment_af()
            p = af_eval.window_scores(_MODELS[rhythm], rec.signal, torch.device(device))
            af = af_block(p, offset_s=lo / rec.fs)
            af_fire = af_alert_times(p, offset_s=lo / rec.fs)
        r = process(rec, ds in SPAN_SETS, step, af, af_fire)
    except Exception as e:                           # noqa: BLE001  one bad record must not end the run
        return {"dataset": ds, "name": Path(name).stem, "error": f"{type(e).__name__}: {e}"}
    r["secs"] = time.time() - t1
    return r


def _experiment_af():
    from munji_ai.eval import af_alarm_rate
    return af_alarm_rate


# ------------------------------------------------------------------ scoring
def score(results: list, name: str) -> dict:
    out = {"long": defaultdict(lambda: [0, 0]), "short": defaultdict(lambda: [0, 0]), "named": defaultdict(int),
           "unsafe": [], "false": defaultdict(int), "false_list": [], "caught": {}, "fast": [0, 0]}
    for r in results:
        al = r["alerts"][name]
        for ep in r["episodes"]:
            hit = al[(al >= ep["s"]) & (al <= ep["e"] + SUSTAIN)]
            key = "long" if ep["dur"] >= SUSTAIN else "short"
            out[key][ep["label"]][0] += int(hit.size > 0)
            out[key][ep["label"]][1] += 1
            if key == "long" and ep["label"] == "SVT" and ep["peak"] >= FAST_BPM:
                out["fast"][0] += int(hit.size > 0)
                out["fast"][1] += 1
            out["caught"][(r["dataset"], r["name"], ep["s"])] = float(hit[0] - ep["s"]) if hit.size else None
        for a in al:
            k = classify(float(a), r["ivs"])
            if k.startswith("unsafe:"):
                out["unsafe"].append((r["dataset"], r["name"], float(a), k[7:]))
            elif k in NAMED:
                out["named"][k] += 1
            else:
                out["false"][r["dataset"]] += 1
                out["false_list"].append((r["dataset"], r["name"], float(a), k[6:]))
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def ec57(results: list, psvt_only: bool = False, key: str = "report") -> dict:
    """ANSI/AAMI EC57 episode and duration statistics for the report, gross
    over all records, against the reference SVT episodes (all, and 30 s or
    more). Detected episodes touching VT or VF are unsafe; those touching no
    SVT or AFL are false."""
    o = defaultdict(float)
    o["false_by"] = defaultdict(int)
    o["unsafe_list"] = []
    for r in results:
        det = [d for d in r[key] if d["psvt"] or not psvt_only]
        refs = [(s, e) for s, e, lab in r["ivs"] if lab == "SVT"]
        named = [(s, e) for s, e, lab in r["ivs"] if lab in NAMED]
        for s, e in refs:
            hit = any(_overlap(s, e, d["s"], d["e"]) > 0 for d in det)
            o["ref"] += 1
            o["ref_hit"] += hit
            if e - s >= SUSTAIN:
                o["ref30"] += 1
                o["ref30_hit"] += hit
            o["ref_dur"] += e - s
            o["ref_dur_hit"] += sum(_overlap(s, e, d["s"], d["e"]) for d in det)
        for d in det:
            k = classify_span(d["s"], d["e"], r["ivs"], d["e"])
            o["det"] += 1
            o["det_dur"] += d["e"] - d["s"]
            o["det_dur_true"] += sum(_overlap(d["s"], d["e"], s, e) for s, e in named)
            if k.startswith("unsafe:"):
                o["unsafe"] += 1
                o["unsafe_list"].append((r["dataset"], r["name"], d["s"], d["e"], k[7:]))
            elif k in NAMED:
                o["det_true"] += 1
            else:
                o["false_by"][r["dataset"]] += 1
    div = lambda a, b: a / b if b else NAN   # noqa: E731
    o["ESe"], o["ESe30"] = div(o["ref_hit"], o["ref"]), div(o["ref30_hit"], o["ref30"])
    o["E+P"] = div(o["det_true"], o["det"])
    o["DSe"], o["D+P"] = div(o["ref_dur_hit"], o["ref_dur"]), div(o["det_dur_true"], o["det_dur"])
    return o


def report(results: list) -> dict:
    hours = defaultdict(float)
    for r in results:
        hours[r["dataset"]] += r["hours"]
    print("\n" + ", ".join(f"{k} {v:.1f} h" for k, v in hours.items()))
    print(f"records without a template (no label): {sum(r['first'] is None for r in results)} of {len(results)}")
    unchecked = sorted({r["dataset"] for r in results if not r["af_checked"]})
    if unchecked:
        print(f"WARNING  RhythmNet did not run on: {', '.join(unchecked)} (the AF check never blocks there)")
    names = [n for n, _ in CONFIGS if n not in REPORTS]
    sc = {n: score(results, n) for n in names}

    long_ = [(r, ep) for r in results for ep in r["episodes"] if ep["dur"] >= SUSTAIN]
    print(f"\nSVT and AFL episodes of 30 s or more ({len(long_)})")
    if long_:
        print(f"{'set':<7}{'record':<8}{'label':<6}{'start':>9}{'dur':>6}{'rate':>6}{'det':>5}{'alike':>7}"
              f"{'onset':>7}{'stab':>6}{'AF':>6}{'lz':>6}{'template':>9}{'1':>6}{'4':>6}")
        for r, ep in long_:
            key = (r["dataset"], r["name"], ep["s"])
            d = [sc[n]["caught"][key] for n in (PRIMARY, "label: first run (neither)")]
            print(f"{r['dataset']:<7}{r['name']:<8}{ep['label']:<6}{T._mmss(ep['s']):>9}{ep['dur']:>6.0f}"
                  f"{T._f(ep['rate_sig']):>6}{T._f(ep['rate_det']):>5}{T._f(ep['alike'], '{:.0%}'):>7}"
                  f"{T._f(ep['onset'], '{:.0%}'):>7}{T._f(ep['stab']):>6}{T._f(ep['af'], '{:.0%}'):>6}"
                  f"{T._f(ep['lz'], '{:.2f}'):>6}{ep['template']:>9}" + "".join(f"{T._f(v):>6}" for v in d))
        print(f"(rate: on the signal; det: from the detector; alike: beats keeping the template (C {ALIKE_C},\n"
              f" size); onset: drop at the first run over 100 near the start; stab: ms; AF: share of its decisions\n"
              f" RhythmNet blocked; lz: median Lorenz AF score, -1 to 1, over {LZ_AF:.0f} is AF; 1, 4: seconds to\n"
              f" the label of rows 1 and 4 below, '-' none)")

    svt30 = [(r, ep) for r, ep in long_ if ep["label"] == "SVT"]
    fast = [(r, ep) for r, ep in svt30 if ep["peak"] >= FAST_BPM]
    print(f"\nSVT episodes of 30 s or more: {len(svt30)}. Reached {FAST_BPM:.0f} bpm in some 30 s: {len(fast)}"
          f" (the rest stay under the label's line: the report's, not the label's)")
    if fast:
        rows = tuple(ROW_CHECKS)
        cols = CHECKS + ("onset150",)
        nums = [str(names.index(n) + 1) for n in rows]
        print(f"{'set':<7}{'record':<8}{'start':>9}{'dur':>6}{'peak':>6}{'at150':>7}"
              + "".join(f"{k:>9}" for k in cols) + "".join(f"{k:>5}" for k in nums)
              + "  why " + " / ".join(nums))
        for r, ep in fast:
            key = (r["dataset"], r["name"], ep["s"])
            d = [sc[n]["caught"][key] for n in rows]
            print(f"{r['dataset']:<7}{r['name']:<8}{T._mmss(ep['s']):>9}{ep['dur']:>6.0f}{T._f(ep['peak']):>6}"
                  f"{ep['fast_s']:>6.0f}s" + "".join(f"{T._f(ep['pass'][k], '{:.0%}'):>9}" for k in cols)
                  + "".join(f"{T._f(v):>5}" for v in d) + "  " + " / ".join(ep["why"][n] for n in rows))
        print(f"(peak: fastest 30 s inside it, from the detector; at150: time at 150 bpm or more; each check: share\n"
              f" of those decisions it passed, judged as row {names.index(LORENZ) + 1} judges them; onset150: the onset as\n"
              f" before, at the 150 crossing; {', '.join(nums)}: seconds to the label of those rows below, '-' none; why: the check that\n"
              f" failed most there, '-' when one decision passed them all)")

    sets = list(hours)
    print("\nLABEL (150 bpm). Episodes labelled / episodes. SVT150: of the SVT episodes of 30 s or more that\n"
          "reached 150 bpm. Unsafe: labels touching VT or VF. False per 24 h.")
    print(f"{'':<4}{'configuration':<30}{'SVT>=30s':>9}{'SVT150':>8}{'SVT<30s':>9}{'AFL>=30s':>9}{'unsafe':>8}"
          + "".join(f"{s:>9}" for s in sets))
    for k, n in enumerate(names, 1):
        s = sc[n]
        f = lambda key, lab: f"{s[key][lab][0]}/{s[key][lab][1]}"   # noqa: E731
        f150 = f"{s['fast'][0]}/{s['fast'][1]}"
        print(f"{k:<4}{n:<30}{f('long', 'SVT'):>9}{f150:>8}{f('short', 'SVT'):>9}"
              f"{f('long', 'AFL'):>9}{len(s['unsafe']):>8}"
              + "".join(f"{s['false'][d] / (hours[d] / 24.0):>9.2f}" for d in sets))

    un = sc[PRIMARY]["unsafe"]
    print(f"\nUnsafe labels, '{PRIMARY}': {len(un)}")
    det = {(r["dataset"], r["name"]): r["detail"] for r in results}
    for d, n, a, k in un:
        v = det[(d, n)].get(a, {})
        print(f"  {d} {n} {T._mmss(a)} in {k}: rate {T._f(v.get('bpm'))}, alike {T._f(v.get('share'), '{:.0%}')}, "
              f"onset {T._f(v.get('onset'), '{:.0%}')}, stab {T._f(v.get('stab'))} ms")
    for row in (FINAL_ALERT, FINAL_REPORT, DOCTOR):
        un = sc[row]["unsafe"]
        what = "labels" if row == FINAL_ALERT else "entries"
        print(f"\nUnsafe {what}, '{row}': {len(un)} (its 30 s touches VT or VF)")
        rows_ = {(r["dataset"], r["name"]): r["detail_rows"].get(row, {}) for r in results}
        for d, n, a, k in un:
            v = rows_[(d, n)].get(a, {})
            print(f"  {d} {n} {T._mmss(a)} in {k}: rate {T._f(v.get('bpm'))}, regular {T._f(v.get('stab_run'))} ms,"
                  f" lz {T._f(v.get('lz'), '{:.2f}')}, start {T._f(v.get('start_drop'), '{:.0%}')},"
                  f" end {T._f(v.get('end_rise'), '{:.0%}')}")
    lz_at = {(r["dataset"], r["name"]): r["lz_at"] for r in results}
    for name in (PRIMARY, NO_AF, LORENZ, LORENZ_REACH, SLOWER, DOCTOR, DOCTOR_PSVT):
        fl = sc[name]["false_list"]
        kinds = defaultdict(int)
        for *_, lab in fl:
            kinds[lab] += 1
        print(f"\nFalse labels, '{name}': {len(fl)}"
              + (" (" + ", ".join(f"{v} in {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
                 + "; most first, time and rhythm label)" if fl else ""))
        by_rec = defaultdict(list)
        for d, n, a, lab in fl:
            by_rec[(d, n)].append((a, lab))
        for (d, n), v in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))[:15]:
            print(f"  {d} {n}: {len(v)}  " + ", ".join(
                f"{T._mmss(a)} {lab} (lz {T._f(lz_at[(d, n)][name].get(a, NAN), '{:.2f}')})" for a, lab in v[:6])
                + (" ..." if len(v) > 6 else ""))

    lorenz_cuts(results)

    print("\nREPORT (over 100 bpm), ANSI/AAMI EC57 episode and duration statistics, gross")
    print(f"{'':<16}{'ESe':>7}{'ESe>=30s':>10}{'E+P':>7}{'DSe':>7}{'D+P':>7}{'episodes':>10}{'unsafe':>8}"
          + "".join(f"{s:>9}" for s in sets))
    e57 = {}
    for label, only, key in (("detected", False, "report"), ("PSVT (ended)", True, "report"),
                             ("Lorenz: detected", False, "report_lz"), ("Lorenz: PSVT", True, "report_lz")):
        o = ec57(results, psvt_only=only, key=key)
        e57[label] = o
        print(f"{label:<16}{T._f(o['ESe'], '{:.2f}'):>7}{T._f(o['ESe30'], '{:.2f}'):>10}{T._f(o['E+P'], '{:.2f}'):>7}"
              f"{T._f(o['DSe'], '{:.2f}'):>7}{T._f(o['D+P'], '{:.2f}'):>7}{int(o['det']):>10}{int(o['unsafe']):>8}"
              + "".join(f"{o['false_by'][d] / (hours[d] / 24.0):>9.2f}" for d in sets))
    o = e57["detected"]
    print(f"(reference SVT episodes: {int(o['ref'])}, {int(o['ref30'])} of 30 s or more; E+P and D+P count\n"
          f" SVT and AFL as true; the last columns are false episodes per 24 h)")
    print(" (the Lorenz rows: the same report with the Lorenz score in RhythmNet's place)")
    for label in ("detected", "Lorenz: detected"):
        for d, n, s, e, k in e57[label]["unsafe_list"][:10]:
            print(f"  unsafe report episode ({label}): {d} {n} {T._mmss(s)} to {T._mmss(e)} touches {k}")
    return {"scores": sc, "hours": dict(hours), "ec57": e57,
            "af_all": not unchecked}


def _got(v) -> bool:
    return v is not None and bool(np.isfinite(v))


GETS = ("alert", "entry", "fast line", "rest")


def patient_gets(results: list, sc: dict) -> dict:
    """What reached the patient and the doctor during each SVT episode of 30 s
    or more, whatever its name ('gets' in the module docstring). Each episode
    counts once, at the first that applies: an alert of any name, else the
    doctor's report entry, else the report's 150 line, else the rest. Counted
    twice, once with each doctor's report row. sc: score() per row name.
    Prints the episodes and the counts; returns, for the row DOCTOR, the counts
    for those that reached 150 bpm ("fast") and for all ("all"), and the same
    under the key DOCTOR_PSVT for that row."""
    entry_rows = (DOCTOR, DOCTOR_PSVT)
    eps = []
    for r in results:
        for ep in r["episodes"]:
            if ep["label"] != "SVT" or ep["dur"] < SUSTAIN:
                continue
            key = (r["dataset"], r["name"], ep["s"])
            v = {"SVT": sc[SLOWER]["caught"].get(key), "rate": ep.get("rate_alert", NAN),
                 "AF/AFL": ep.get("af_alert", NAN), **{row: sc[row]["caught"].get(key) for row in entry_rows}}
            alerts = [v[k] for k in ("SVT", "rate", "AF/AFL") if _got(v[k])]
            v["any"] = min(alerts) if alerts else None
            gets = {row: ("alert" if alerts else "entry" if _got(v[row])
                          else "fast line" if ep.get("fast_report") else "rest") for row in entry_rows}
            eps.append((r, ep, v, gets))

    def counts(row):
        out = {}
        for part in ("fast", "all"):
            sel = [x for x in eps if part == "all" or x[1]["peak"] >= FAST_BPM]
            o = {"n": len(sel), **{g: sum(x[3][row] == g for x in sel) for g in GETS}}
            for k in ("SVT", "rate", "AF/AFL"):
                o[k] = sum(_got(x[2][k]) for x in sel)
            o["AF unknown"] = sum(not np.isfinite(x[2]["AF/AFL"]) for x in sel if x[2]["AF/AFL"] is not None)
            out[part] = o
        return out
    out = counts(DOCTOR)
    out[DOCTOR_PSVT] = counts(DOCTOR_PSVT)

    def cell(x):
        return "-" if x is None else "?" if not np.isfinite(x) else f"{x:.0f}"
    print(f"\nWHAT THE PATIENT AND THE DOCTOR GET during each SVT episode of 30 s or more, whatever its name ({len(eps)})")
    print(f"{'set':<7}{'record':<8}{'start':>9}{'dur':>6}{'peak':>6}{'SVT':>6}{'rate':>6}{'AF/AFL':>8}{'any':>6}"
          f"{'entry':>7}{'abrupt':>8}{'150':>5}  gets (entry / abrupt)")
    for r, ep, v, gets in eps:
        print(f"{r['dataset']:<7}{r['name']:<8}{T._mmss(ep['s']):>9}{ep['dur']:>6.0f}{T._f(ep['peak']):>6}"
              f"{cell(v['SVT']):>6}{cell(v['rate']):>6}{cell(v['AF/AFL']):>8}{cell(v['any']):>6}"
              f"{cell(v[DOCTOR]):>7}{cell(v[DOCTOR_PSVT]):>8}{('yes' if ep.get('fast_report') else 'no'):>5}"
              f"  {gets[DOCTOR]} / {gets[DOCTOR_PSVT]}")
    print(f"(seconds from the episode's start, '-' none, '?' RhythmNet not run. SVT: the label, row '{SLOWER}';\n"
          f" rate: the notification at {NOTIFY_BPM:.0f} bpm or more held {NOTIFY_SEC:.0f} s; AF/AFL: RhythmNet's alert, at"
          f" most one a day;\n any: the first of the three; entry: the doctor's report entry, row '{DOCTOR}';\n abrupt:"
          f" the same with row '{DOCTOR_PSVT}'; 150: the report's fast-rate line, {REPORT_FAST_BPM:.0f} or more\n over"
          f" {TACHY_BEATS} beats. No quality gate here: the rate columns are an upper bound)")
    f, a = out["fast"], out["all"]
    f2, a2 = out[DOCTOR_PSVT]["fast"], out[DOCTOR_PSVT]["all"]
    print(f"\n{'':<44}{'report row: SVT/AFL':>22}{'report row: abrupt':>22}")
    print(f"{'':<44}{'reached 150':>12}{'all':>10}{'reached 150':>12}{'all':>10}")
    rows = (("an alert, any name", "alert"), ("  of them, named SVT (the label)", "SVT"),
            ("  of them, the rate notification", "rate"), ("  of them, AF/AFL", "AF/AFL"),
            ("no alert, an entry in the doctor's report", "entry"), ("no alert, the report's 150 line only", "fast line"),
            ("none of these (the report grades the rate)", "rest"))
    for label, k in rows:
        print(f"{label:<44}{f[k]:>8}/{f['n']:<3}{a[k]:>6}/{a['n']:<3}{f2[k]:>8}/{f2['n']:<3}{a2[k]:>6}/{a2['n']:<3}")
    print("(an episode can get more than one alert, so the three 'of them' rows can add up to more)")
    if a["AF unknown"]:
        print(f"WARNING  RhythmNet did not run on {a['AF unknown']} of these episodes: their AF/AFL column is unknown")
    return out


LZ_CUTS = (-0.8, -0.6, -0.4, -0.2, LZ_AF)


def lorenz_cuts(results: list) -> dict:
    """The Lorenz score at every label of row 2 (no AF check), by what the
    label touched, and what other cut-offs would keep and block. On these
    development patients only: a cut-off chosen here must be confirmed on
    fresh data."""
    groups = defaultdict(list)
    for r in results:
        for a in r["alerts"][NO_AF]:
            k = classify(float(a), r["ivs"])
            g = "SVT/AFL" if k in NAMED else "unsafe" if k.startswith("unsafe:") else \
                "false in AF" if k == "false:AF" else "false, other"
            groups[g].append(r["lz_at"][NO_AF].get(float(a), NAN))
    print(f"\nLorenz score at the labels of row 2 ({NO_AF}); the rule calls AF over {LZ_AF:.0f}")
    print(f"{'labels':<14}{'n':>5}{'unscored':>10}{'min':>7}{'median':>8}{'max':>7}")
    for g in ("SVT/AFL", "false in AF", "false, other", "unsafe"):
        v = np.array(groups.get(g, []), float)
        f = v[np.isfinite(v)]
        print(f"{g:<14}{v.size:>5}{v.size - f.size:>10}" + "".join(
            f"{T._f(q, '{:.2f}'):>{w}}" for q, w in ((f.min() if f.size else NAN, 7), (np.median(f) if f.size else NAN, 8),
                                            (f.max() if f.size else NAN, 7))))
    print(f"{'AF over':<14}" + "".join(f"{c:>8.1f}" for c in LZ_CUTS))
    for g, lab in (("SVT/AFL", "SVT kept"), ("false in AF", "AF blocked"), ("false, other", "other blocked")):
        v = np.array(groups.get(g, []), float)
        cell = (lambda c: f"{int(np.sum(~(v > c)))}/{v.size}") if g == "SVT/AFL" else (lambda c: f"{int(np.sum(v > c))}/{v.size}")
        print(f"{lab:<14}" + "".join(f"{cell(c):>8}" for c in LZ_CUTS))
    print(" (development patients only: a cut-off picked from this table needs fresh data to confirm it)")
    return dict(groups)


def verdict(rep: dict, name: str = FINAL_ALERT) -> tuple:
    """(code, lines): 0 PASS, 2 FAIL, 1 not run. A row without the AF check
    needs no RhythmNet."""
    sc, hours = rep["scores"][name], rep["hours"]
    tag = "ALERT" if name == FINAL_ALERT else "CANDIDATE"
    if "nsrdb" not in hours or not ({"cudb", "vfdb", "ltafdb", "svdb"} & set(hours)):
        return 1, [f"  {tag} not run (needs NSRDB and sets with VT or VF labels)"]
    if dict(CONFIGS)[name]["af"] and not rep.get("af_all", True):
        return 1, [f"  {tag} not run (RhythmNet did not run on every set: see the warning above)"]
    per_day = sc["false"]["nsrdb"] / (hours["nsrdb"] / 24.0)
    safe, fa_ok = not sc["unsafe"], per_day <= MAX_FALSE_NSRDB
    return (0 if safe and fa_ok else 2), [
        f"  SAFETY   {len(sc['unsafe'])} labels touching VT or VF  ->  {'PASS' if safe else 'FAIL'} (none allowed)",
        f"  NSRDB    {per_day:.2f} false labels per 24 h  ->  {'PASS' if fa_ok else 'FAIL'} (at most {MAX_FALSE_NSRDB})"]


# ------------------------------------------------------------------ main
def _rhythm_paths(a) -> tuple:
    """(final checkpoint or None, {(dataset, record): fold checkpoint})."""
    final = next((p for p in [a.rhythm_ckpt] if p and p.exists()), None)
    if final is None:
        final = next((p for p in (Path("checkpoints/rhythm_final/rhythm_net.pt"), Path("checkpoints/rhythm_net.pt"))
                      if p.exists()), None)
    folds = {}
    oof = a.rhythm_cv / "rhythm_oof.npz"
    if oof.exists():
        try:
            fm = _experiment_af().fold_of_records(oof, C.CACHE_DIR)
            folds = {k: str(a.rhythm_cv / f"rhythm_fold{f}.pt") for k, f in fm.items()
                     if (a.rhythm_cv / f"rhythm_fold{f}.pt").exists()}
        except (SystemExit, Exception) as e:          # noqa: BLE001
            print(f"rhythm folds not used: {e}")
    return (str(final) if final else None), folds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["cudb", "vfdb", "svdb", "nsrdb", "ltafdb", "shdbaf"])
    ap.add_argument("--shdbaf", type=Path, default=Path("data/shdbaf"))
    ap.add_argument("--download-shdbaf", action="store_true",
                    help="fetch SHDB-AF from PhysioNet's S3 copy (records with SVT marks), then stop")
    ap.add_argument("--all-shdbaf", action="store_true", help="with --download-shdbaf: every annotated record")
    ap.add_argument("--rhythm-ckpt", type=Path, default=None, help="final RhythmNet (default: checkpoints/rhythm_final/"
                                                                   "rhythm_net.pt or checkpoints/rhythm_net.pt)")
    ap.add_argument("--rhythm-cv", type=Path, default=Path("checkpoints/rhythm_cv"))
    ap.add_argument("--neg-stride", type=float, default=5.0, help="seconds between windows on the 24 h sets")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per set (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    if a.download_shdbaf:
        failed = download_shdb(a.shdbaf, only_svt=not a.all_shdbaf)
        print("done" if not failed else f"{len(failed)} files failed (run again to fetch only those): "
              + "; ".join(failed[:5]))
        return 0 if not failed else 1
    if "mitdb" in a.sets:
        print("MITDB is MUNJI's benchmark, forced whole into the test split: not read here")
        a.sets = [s for s in a.sets if s != "mitdb"]

    final, folds = _rhythm_paths(a)
    print(f"RhythmNet: final {final or 'none'}; out-of-fold models for {len(folds)} records")
    from munji_ai.data.loader import _patient_id
    from munji_ai.data.registry import get
    from munji_ai.data.splits import assign
    from munji_ai.eval.eval_beats_model import eligible
    jobs, skipped = [], defaultdict(int)
    for ds in a.sets:
        step = a.neg_stride if ds in LONG_SETS else POS_STEP
        if ds == "shdbaf":
            recs = shdb_records(a.shdbaf)[: a.max_records or None]
            if not recs:
                print(f"shdbaf: no annotated records under {a.shdbaf} (--download-shdbaf fetches them)")
            jobs += [(ds, str(p), a.raw, step, final, a.device) for p in recs]
            continue
        for name in eligible(ds, a.raw, a.max_records):
            model = final
            if ds in RHYTHM_TRAINED:
                if (ds, name) in folds:
                    model = folds[(ds, name)]
                elif assign(_patient_id(get(ds), name)) != "val" or final is None:
                    skipped[ds] += 1
                    continue
            jobs.append((ds, name, a.raw, step, model, a.device))
    for ds, n in skipped.items():
        print(f"{ds}: {n} records skipped (RhythmNet trained on them and no out-of-fold model)")
    if not jobs:
        print("no records")
        return 1
    print(f"{len(jobs)} records, {a.workers} at a time")
    t0, results, failed = time.time(), [], []

    def done(r):
        if "error" in r:
            failed.append(r)
            print(f"  {r['dataset']} {r['name']}: FAILED {r['error']}", flush=True)
            return
        results.append(r)
        print(f"  {r['dataset']} {r['name']}: {r['hours']:.1f} h, "
              + (f"template at {T._mmss(r['first'])}" if r["first"] is not None else "no template")
              + f", {len(r['alerts'][FINAL_ALERT])} labels, {len(r['report'])} report episodes, {r['secs']:.0f} s",
              flush=True)

    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(_work, j) for j in jobs]):
                done(f.result())
    else:
        for j in jobs:
            done(_work(j))
    if not results:
        print("no record could be read")
        return 1
    results.sort(key=lambda r: (a.sets.index(r["dataset"]), r["name"]))
    rep = report(results)
    if failed:
        print(f"\n{len(failed)} records failed: " + ", ".join(f"{r['dataset']} {r['name']}" for r in failed))
    print(f"\nFAST NARROW RHYTHM, the alert: '{FINAL_ALERT}'")
    code, lines = verdict(rep)
    print("\n".join(lines))
    if code != 1:
        print(f"  ALERT  ->  {'PASS' if code == 0 else 'FAIL'}")
    hrs = rep["hours"]
    for row in (FINAL_REPORT, DOCTOR):
        s3 = rep["scores"][row]
        kind = "the report (a report entry, never an alert)" if row == FINAL_REPORT else "reported, for comparison"
        print(f"\nDOCTOR REPORT row '{row}': {kind}")
        print(f"  SAFETY   {len(s3['unsafe'])} entries touching VT or VF (none allowed)  ->  "
              f"{'PASS' if not s3['unsafe'] else 'FAIL'}")
        print(f"  SVT150   {s3['fast'][0]}/{s3['fast'][1]} SVT episodes that reached 150 bpm entered;  SVT>=30s "
              f"{s3['long']['SVT'][0]}/{s3['long']['SVT'][1]};  AFL>=30s {s3['long']['AFL'][0]}/{s3['long']['AFL'][1]}")
        print("  false entries per 24 h: " + ", ".join(f"{d} {s3['false'][d] / (hrs[d] / 24.0):.2f}" for d in hrs))
        if row == FINAL_REPORT:
            print(f"  REPORT  ->  {'PASS' if not s3['unsafe'] else 'FAIL'}")
            if s3["unsafe"] and code == 0:
                code = 2
    patient_gets(results, rep["scores"])
    for name in (PRIMARY, LORENZ, LORENZ_REACH, LORENZ_150):
        c2, lines2 = verdict(rep, name)
        print(f"\nCANDIDATE '{name}', the same bars (reported, does not set the exit code)")
        print("\n".join(lines2))
        if c2 != 1:
            s2 = rep["scores"][name]
            for d in ("ltafdb", "shdbaf", "svdb"):
                if d in rep["hours"]:
                    print(f"  {d:<8} {s2['false'][d] / (rep['hours'][d] / 24.0):.2f} false labels per 24 h (reported)")
            print(f"  SVT150   {s2['fast'][0]}/{s2['fast'][1]} SVT episodes that reached 150 bpm labelled (reported)")
            print(f"  CANDIDATE  ->  {'PASS' if c2 == 0 else 'FAIL'}")
    print(f"\n{time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
