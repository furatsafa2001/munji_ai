# Rate rules: engine and decision layer

`munji_ai/rules/rate_engine.py` measures; `munji_ai/decision/decision_layer.py`
(`on_rate`) decides. Every clinical number lives in `config.py`, beside its
source and its comparison. Nothing here is a team choice except where it says
so, and those are measurement constants with an acceptance test instead.

## Two layers: graded for the report, sent as a notification

### Graded for the doctor's report (never sent on their own)

| Finding | Condition | Tier | Source |
|---|---|---|---|
| bradycardia | rate held below 60 | info | conventional definition |
| bradycardia | rate held below 50 | warning | 2018 ACC/AHA/HRS: sinus bradycardia < 50 awake |
| extreme bradycardia | rate held below 40 | critical | 2018 ACC/AHA/HRS (symptomatic < 40); NEWS2: <= 40 scores 3 |
| tachycardia | rate held above 100 | info | conventional definition |
| tachycardia | rate held above 110 | warning | NEWS2: 111-130 scores 2 |
| extreme tachycardia | rate held at 150 or more | critical | AHA adult tachyarrhythmia algorithm |

"Held" means over a run of consecutive beats: 5 for brady, 17 for tachy
(PhysioNet/CinC Challenge 2015 alarm definitions). The rate of a run is the
mean RR after dropping the longest and shortest 3/16, the way bedside monitors
average heart rate.

### Sent (ambulatory telemetry practice, MCOT)

| Notification | Condition | Level | Alert severity |
|---|---|---|---|
| sustained bradycardia | 30 bpm or less, held 30 s | emergent | critical |
| sustained tachycardia | 180 bpm or more, held 30 s | urgent | warning |
| pause | no beat for more than 3 s | urgent | warning |
| asystole | no beat for more than 6 s | emergent | critical |
| wide-complex rhythm | 150 bpm for 6 s | ventricular-beat detector, not this engine | |

Sources: physician-notification criteria of a commercial MCOT service (Philips
BioTel protocol, J Cardiol Curr Res): pause >= 3 s, SVT/AF >= 180 for >= 30 s,
bradycardia <= 30 for >= 30 s, wide complex >= 150 for >= 6 s. In a
retrospective series of 26,438 MCOT patients the potentially emergent events
were wide-complex tachycardia, pauses >= 6 s and sustained bradycardia < 30.
The 6 s pause line is also 2018 ESC syncope. The client's abnormality list
gives the same 30 and 180 as its examples.

Why two layers: on one healthy NSRDB day the graded tiers fired about 90
warnings (sleep below 50, walking above 110). Telemetry services do not notify
on those; they report them.

Beat floor: a detected peak counts only above 0.15 mV peak to valley
(ANSI/AAMI EC13; Drew et al. 2014).

## Not reported by the engine

- Wide-complex tachycardia: 14% of real VT windows called wide.
- The fast narrow (SVT) label (`FAST_NARROW_LABEL = False`): the width
  estimate failed on the QT Database (7,246 beats): mean -50 ms, SD 35 ms
  against IEC 60601-2-25 limits of +/-10 ms and 10 ms, correlation 0.25.
  QRS width is still reported as narrow, uncertain or wide, for information.

## Input contract

| Item | Requirement | Why |
|---|---|---|
| Signal | model path (0.5-40 Hz + notch), **millivolts**, never normalised | the amplitude floor is an absolute voltage |
| Window | 40 s every 2.5 s (`RULES_WINDOW_SEC`, `RULES_STRIDE_SEC`) | a 30 s sustained run, and a gap over 6 s, always fit whole |
| Missing samples | `complete=False`, never zero- or hold-filled | a filled gap reads as asystole |
| Lead status | `lead_off=True/False`, `None` if unknown | judges absence findings |

## Decision layer (`on_rate`)

| Rule | Behaviour |
|---|---|
| What is sent | only findings with `notify` (the MCOT table above) |
| Everything else | counted in `report()["rate_report_windows"]` by family and tier |
| Confirmation | per family, from one window (persistence is in the finding) |
| Episodes | one alert per episode; one more if it escalates (urgent to emergent); a new episode after 6 quiet windows |
| Rate findings | need a gate-accepted window |
| Absence findings | need leads on; with lead status unknown, the gate decides (ACLS: check the leads before calling a flat line asystole) |
| Leads off / missing samples | nothing cardiac; counted in `lead_off_windows`, `no_data_windows` |
| Priority | during VF/VT only emergent (critical) rate alerts pass |

## Measurement constants (no clinical number exists for these)

| Constant | Value | What it must pass |
|---|---|---|
| `QRS_AMP_HALF_MS` | 60 | covers a 120 ms complex, the upper limit of normal |
| `QRS_SEARCH_HALF_MS` | 100 | must measure VT complexes; 69% exceed 140 ms (Wellens) |
| 30% threshold, 40 ms lobe bridge | in code | IEC 60601-2-25: mean difference within +/-10 ms, SD 10 ms against experts (`eval_qrs_width_qtdb`) |
| `QRS_UNCERTAIN_MS` | None | not set: the measured error (SD 35 ms) would make the band cover almost everything |

Next candidates for width, judged by the same script and the same IEC limits,
on QT Database records not drawn from MIT-BIH (the test benchmark): the
single-lead wavelet delineator of Martinez et al. 2004 (validated on the QT
Database), and BeatSegmenter mask width.

## Known limits, measured

- Gap length is off by a few tens of ms, either direction (pinned by a test).
- Noise inside the amplitude window adds to the reading: the error is towards
  counting a beat, not towards a false asystole.
- A patient whose CM5 QRS is 0.15 mV or less reads as asystole, by the
  standard's design. Check amplitude on the first patch recordings.

## Open

| Item | Owner |
|---|---|
| A width source that passes IEC 60601-2-25 (Martinez 2004 delineator, BeatSegmenter) | AI track |
| Run `eval_rules_holter` on all NSRDB and on Icentia; `eval_rules_challenge2015` | AI track |
| App: pass `lead_off` and `complete`, never fill gaps, read `STATUS_CHAR` | app |
| Lead-off rule on the final PCB | hardware |
| Chip high-pass at 0.5 Hz vs the documented 0.05 Hz diagnostic path | hardware |

## Run

    python tests/test_rules.py
    python tests/test_decision_layer.py
    python -m munji_ai.rules.rate_engine
    python -m munji_ai.eval.eval_qrs_width_qtdb --data-dir data/qtdb/qt-database-1.0.0
    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb --list-gaps 15
    python -m munji_ai.eval.eval_rules_challenge2015 --data-dir data/challenge2015/training
