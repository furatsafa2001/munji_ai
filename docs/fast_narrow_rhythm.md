# Fast narrow rhythm (SVT): the alert and the doctor's report

Chosen on 25 November 2026. Every number below comes from development data
(validation patients and external sets); the test split is untouched and is
kept for the one end-to-end test of all models. The clinical thresholds still
need the client's physician to sign them off.

Code: `experiments/fast_narrow_template.py` (rows `FINAL_ALERT` and
`FINAL_REPORT`), with `experiments/fast_narrow_exercise.py` and
`experiments/fast_narrow_icentia.py` for exercise and daily life.

## What it is

The AI document the client approved promises a "fast narrow rhythm" output:
the heart is racing, but each beat keeps the patient's normal shape. It is
described, not diagnosed. It comes in two forms:

| Output | Who sees it | When |
|---|---|---|
| Alert | patient and caregiver | rare and specific: a sudden fast regular rhythm |
| Doctor's report entry | the physician | more often, still specific: a fast regular rhythm that started or ended abruptly |

Beside them, unchanged, the rate engine sends its own notification for any
rhythm at 180 bpm or more held 30 s, and grades 150 bpm or more as critical
in the report (`docs/rules_engine.md`).

## Definitions (2015 ACC/AHA/HRS SVT guideline)

- **SVT:** a rate over 100 bpm at rest, arising at or above the bundle of His.
  It excludes atrial fibrillation.
- **Paroxysmal SVT:** regular and rapid, with abrupt onset and termination.
- **Typical atrial flutter:** a macro-reentrant atrial tachycardia, inside
  the SVT umbrella.

## Inputs

- **Beats:** the rate engine's detector (Pan-Tompkins++, EC13 floor) on every
  40 s window. Not the beat model, so beat model v3 does not change these
  rules.
- **Template:** the patient's own beat, learned at patch-on from the first
  10 s stretch at 40 to 100 bpm with one main shape: the beat that most of
  the stretch's beats match at correlation 0.90 or more, holding 60% of them;
  the template is their median, so ectopic beats are left out. Where rhythm
  labels exist, the stretch never touches a ventricular, paced, noisy or
  unknown rhythm. It stays fixed for the recording (`template_wct`, "dominant"
  calibration).
- **Decisions:** every 2.5 s (5 s on the 24 h sets in the experiments), over
  the last 30 s.
- **Not used:** RhythmNet, the beat model, QRS width (see "Rejected").

## The alert

All of these, at one decision:

| Check | Value | Basis |
|---|---|---|
| Rate | 30 s trimmed-mean rate of 150 bpm or more | `FAST_NARROW_BPM`; the AHA 150 line and the MCOT tiers |
| Shape | 80% or more of the beats keep the template (correlation 0.90 or more, size half to twice) | the patient's own conducted beat |
| VT lock | no 3 or more consecutive beats unlike the template (correlation under 0.60 or size outside half to twice) | VT: 3 or more consecutive ventricular beats |
| Regular | median change between successive RR intervals of 40 ms or less, over the fast run's own beats | Brugada et al. 1998: stability under 40 ms rejected 31 of 32 AF in the ICD rate zone |
| Not AF | Lorenz plot AF score of 0 or less over the last 2 min (30 points or more) | Sarkar, Ritscher and Mehra 2008 (Medtronic insertable monitors); US 7,623,911 and 10,045,710 |
| Onset | the cycle length drops 9% or more at the start of the run over 150 | Brugada et al. 1998: onset over 9% rejected 65 of 67 sinus tachycardias |
| Slower before | the app's own 30 s rate just before the run's first beat was under 150 | ICD onset compares the new average interval with earlier averages |

Lorenz score: occupied 25 ms bins over plus or minus 500 ms outside the
plus or minus 75 ms zero segment, minus the points inside that segment, per
point; intervals of 220 ms or less are not used.

## The doctor's report entry

All of these, at one decision. An entry is never an alert.

| Check | Value | Basis |
|---|---|---|
| Rate | 30 s trimmed-mean rate of 150 bpm or more | as the alert |
| Shape | as the alert | as the alert |
| VT lock | as the alert | as the alert |
| RR | regular with a Lorenz score of 0 or less, **or** organized on RR multiples | regular: as the alert. Multiples: atrial tachycardia or flutter with variable conduction (2:1, 3:1, 4:1) |
| Abrupt start or end | the run over 100 began with a cycle-length drop of 9% or more, **or** ended with a rise of 9% or more | PSVT: abrupt onset and termination (2015 ACC/AHA/HRS); 9%: Brugada et al. 1998, mirrored at the end |

- **No onset at 150 and no "slower before":** the report catches SVT that
  hovers around 150 or never had a slower rhythm before it.
- **The end is known only after the run:** fine for a report written after
  the episode, impossible for an alert.
- **Either start or end, not both:** an SVT whose start the patch missed
  still counts.
- **RR multiples (our choices):** 80% or more of the intervals within 30 ms
  of 1 to 4 times one atrial cycle of 171 to 400 ms, with at least two
  multiples each holding 15% of the intervals, from 12 intervals or more.
  Checked on simulated data before the first run: AF never passed, variable
  conduction did.

## Results

### Safety and false labels

Per 24 h unless stated. "Unsafe" is a label or entry whose 30 s touches an
annotated VT or VF, even a 3-beat run; none is allowed.

| Data | Alert | Report entry | Earlier report row (no abrupt check), for comparison |
|---|---|---|---|
| Unsafe (CUDB, VFDB, SVDB, LTAFDB, SHDB-AF) | **0** | **0** | 9 |
| NSRDB, healthy, 384 h | 0.00 | 0.00 | 0.12 |
| CUDB | 0.00 | 0.00 | 30.87 |
| VFDB | 0.00 | 0.00 | 0.00 |
| SVDB | 0.00 | 0.71 | 0.71 |
| LTAFDB, AF patients, 1,591 h | 0.17 | 0.14 | 3.39 |
| SHDB-AF | 0.17 | 0.50 | 0.67 |
| Icentia11k, daily life, 199 patients, 476 days | 0.008 (95% upper 0.022) | 0.046 (95% upper 0.070), 9 patients | 0.218, 13 patients |
| Icentia11k, 55 fast sinus episodes (150 or more) | 0 | 3 | 27 |
| STDB stress tests, 28 recordings | 0 | 1 (STDB 317: missed beats faked an abrupt change) | 10 |
| GUDB walking and jogging, 125 recordings | 0 | 1 (s04 jogging: 82 to 108 bpm in about 10 s) | 1 |

Icentia counts every label as false (it has no sustained SVT label).
Alert bar, fixed before the runs: no unsafe label, NSRDB at most 0.29 per
24 h (the wearable-defibrillator study goal, Poole et al. 2022).

### What reaches the patient and the doctor

SVT episodes of 30 s or more, annotated by cardiologists (VFDB, LTAFDB,
SHDB-AF): 76, of which 17 reached 150 bpm. Each counts once, at the first
that applies.

| | Reached 150 (17) | All (76) |
|---|---|---|
| An alert of any name | 5 | 5 |
| of them, the SVT alert | 4 | 4 |
| of them, the rate notification (180 held 30 s) | 2 | 2 |
| No alert, a report entry | 4 | 4 |
| No alert, the report's 150 bpm line only | 8 | 17 |
| None of these (the report grades the rate) | 0 | 50 |

- No episode that reached 150 bpm went unreported.
- Both episodes that reached 180 bpm got the rate notification, as the MCOT
  protocol asks.
- SVT named in the report (entry or alert): 9 of 17.

## Why this design

- **Naming is the weak part, not reaching someone.** A wrong name is worse
  than none: SVT written for what is VT could lead to the wrong treatment.
  Commercial monitors (Zio, MCOT) have technicians who name the rhythm; here
  the physician reading the report plays that part, with the fast-rate line
  in front of them.
- **The report's abrupt check** removed all 9 unsafe entries of the earlier
  report row. All 9 were in LTAFDB 204 and 206: fast, nearly regular AF
  (Lorenz score -0.38 to 0) with a 3-beat VT run inside the 30 s. The VT
  lock missed them because the VT beats looked like the patient's own (6)
  or the detector missed one beat of the run (3)
  (`experiments/doctor_vt_check.py`). The same fast AF caused most false
  entries in AF patients. It cost one SVT episode (SHDB-AF 138 at 22:19).

## Rejected, with the reason

| Option | Result |
|---|---|
| RhythmNet as the AF check | 1 of 17: it calls most fast atrial tachycardia AF |
| Onset judged at 100 bpm for the alert | 5 of 17, but labelled a real jog (GUDB s04) and a missed-beats artifact (STDB 317) |
| QRS width | failed against cardiologists on QTDB and LUDB |
| HRV to tell exercise apart | GUDB jogging at 150 to 165: RMSSD 3.8 to 4.7 ms, SDNN 5.6 to 6.6 ms, as low as SVT |
| Motion gating | the patch has no accelerometer |
| A Lorenz cut-off picked from these results (for example -0.4) | no fresh data to confirm it; the margin was 0.02 |
| VT lock at 2 beats | no basis: VT is 3 or more |
| QRS alternans | rate-related, not specific to SVT (Morady et al. 1987), and breathing moves the amplitude on a chest patch |

## Limits

- **Few episodes:** 17 that reached 150 bpm, from 5 patients, 9 of them from
  SHDB-AF 127. Shares like "4 of 17" are not stable.
- **Development data only.** The thresholds come from the literature, not
  from these results, except: the Lorenz cut-off of 0 (the device value is
  not public; fixed before the first run), the RR-multiple settings, and the
  30 s window for "slower before" (the app's rate window). These are our
  choices.
- **Exercise that starts abruptly** (GUDB s04) enters the report. Without a
  motion sensor this cannot be separated.
- **Missed beats** can fake an abrupt change (STDB 317).
- **Irregular atrial tachycardia** (SHDB-AF 127) often reaches the report
  only as a fast rate.
- **No quality gate** in these runs; the rate notification counts are an
  upper bound.
- **Rate notification timing:** the rate engine's trimmed mean drops the
  longest 3/16 of the intervals, so "held 30 s" is met about 20 s into an
  abrupt run. Noted, not changed.
- **Icentia p00150 at 21:45:** the one alert at 173 bpm with no 30 s over
  150 in the annotations; likely a detector over-count, not yet checked by
  eye.

## Open

1. The client's physician signs off the thresholds: 180 bpm held 30 s for
   the notification (MCOT is a service protocol, not a society guideline;
   ARIC used over 180 for 60 s), 150 bpm for the alert and the report.
2. Port both rules into the app's rate rules. `config.FAST_NARROW_LABEL`
   is still False.
3. The end-to-end test on the test split, once, with all models.
4. Later: a "palpitations" button in the app (SVT usually comes with
   palpitations), a note on abrupt start and end in the report entry, the
   age-predicted maximum rate as a report note.

## Reproduce

```
python experiments/fast_narrow_template.py --download-shdbaf    # once
python experiments/fast_narrow_template.py                      # about 25 min
python experiments/fast_narrow_exercise.py --download           # once, then without --download
python -m munji_ai.data.svt_runs                                # once, for Icentia
python experiments/fast_narrow_icentia.py --workers 4           # about 80 min
python experiments/doctor_vt_check.py --plot                    # the 9 unsafe entries of the earlier row
```

## References

- Page RL et al. 2015 ACC/AHA/HRS guideline for the management of adult
  patients with supraventricular tachycardia. Circulation 2016;133:e506-e574.
- Brugada J et al. J Cardiovasc Electrophysiol 1998: the ICD onset (9%) and
  stability (40 ms) settings, as cited in `experiments/template_wct.py`.
- Sarkar S, Ritscher D, Mehra R. A detector for a chronic implantable atrial
  tachyarrhythmia monitor. IEEE Trans Biomed Eng 2008;55:1219.
- US patents 7,623,911 and 10,045,710 (Medtronic, Lorenz plot AF detection).
- Morady F et al. Determinants of QRS alternans during narrow QRS
  tachycardia. 1987.
- ANSI/AAMI EC57: testing and reporting performance results of cardiac rhythm
  and ST segment measurement algorithms.
- Philips BioTel MCOT notification protocol; ARIC study Zio patch criteria
  (see `docs/rules_engine.md`).
- Poole JE et al. J Cardiovasc Electrophysiol 2022 (wearable
  cardioverter-defibrillator false alarms).
