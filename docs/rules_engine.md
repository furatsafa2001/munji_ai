# Rate rules: engine and decision layer

`munji_ai/rules/rate_engine.py` measures; `munji_ai/decision/decision_layer.py`
(`on_rate`) decides. Every clinical number lives in `config.py`, beside its
source and its comparison. Nothing here is a team choice except where it says
so, and those are measurement constants with an acceptance test instead.

## Thresholds

| Finding | Condition | Severity | Source |
|---|---|---|---|
| bradycardia | rate held below 60 | info (report only) | conventional definition |
| bradycardia | rate held below 50 | warning | 2018 ACC/AHA/HRS: sinus bradycardia < 50 awake |
| extreme bradycardia | rate held below 40 | critical | 2018 ACC/AHA/HRS (symptomatic < 40); NEWS2: <= 40 scores 3 |
| tachycardia | rate held above 100 | info (report only) | conventional definition |
| tachycardia | rate held above 110 | warning | NEWS2: 111-130 scores 2 |
| extreme tachycardia | rate held at 150 or more | critical | AHA adult tachyarrhythmia algorithm |
| fast narrow | extreme tachycardia, QRS narrow | warning | QRS >= 120 ms is wide (AHA) |
| pause | gap longer than 3 s | warning | 2018 ACC/AHA/HRS; 2018 ESC syncope |
| asystole | gap longer than 6 s | critical | 2018 ESC syncope: asymptomatic pause > 6 s |
| beat | QRS larger than 0.15 mV | (detection) | ANSI/AAMI EC13; Drew et al. 2014 |

"Held" means over a run of consecutive beats: 5 for brady, 17 for tachy
(PhysioNet/CinC Challenge 2015 alarm definitions). The rate of a run is the
mean RR after dropping the longest and shortest 3/16, the way bedside monitors
average heart rate. One missed beat or one pause cannot make or break a run;
AF at 160 still counts; a burst shorter than half the run does not.

The client's abnormality list shows "< 30" and "> 180" in its Example column
for the extreme tiers. They are illustrations there, not thresholds.

## Not reported by the engine

Wide-complex tachycardia. On real VT the engine called 14% of windows wide
(its width estimate reads short); ventricular rhythms are detected in the
decision layer from ventricular beats. The engine still reports QRS width as
`narrow`, `uncertain` or `wide`.

## Input contract

| Item | Requirement | Why |
|---|---|---|
| Signal | model path (0.5-40 Hz + notch), **millivolts**, never normalised | the amplitude floor is an absolute voltage |
| Window | 20 s every 2.5 s (`RULES_WINDOW_SEC`, `RULES_STRIDE_SEC`) | holds a 5-beat run down to 15 bpm; a gap over 6 s always fits whole; latency near EC13's 10 s |
| Missing samples | `complete=False`, never zero- or hold-filled | a filled gap reads as asystole |
| Lead status | `lead_off=True/False`, `None` if unknown | judges absence findings |

## Decision layer (`on_rate`)

| Rule | Behaviour |
|---|---|
| Confirmation | per family (brady, tachy, absence, fast narrow), from one window: persistence is already in the finding (`CONFIRMATION` (1, 1)) |
| Episodes | one alert per episode; one more if it escalates while active; a new episode after 6 quiet windows |
| Info level | counted in `report()["rate_info_windows"]`, never alerted (asymptomatic sinus bradycardia is not associated with adverse outcomes) |
| Rate findings | need a gate-accepted window |
| Absence findings | need leads on; with lead status unknown, the gate decides (ACLS: check the leads before calling a flat line asystole) |
| Leads off / missing samples | nothing cardiac; counted in `lead_off_windows`, `no_data_windows` |
| Priority | during VF/VT only critical rate findings pass; fast narrow is held while ShockableNet saw danger in the last 6 windows |

## Measurement constants (no clinical number exists for these)

| Constant | Value | What it must pass |
|---|---|---|
| `QRS_AMP_HALF_MS` | 60 | covers a 120 ms complex, the upper limit of normal |
| `QRS_SEARCH_HALF_MS` | 100 | must measure VT complexes; 69% exceed 140 ms (Wellens) |
| 30% threshold, 40 ms lobe bridge | in code | IEC 60601-2-25: mean difference within +/-10 ms, SD 10 ms against experts (`eval_qrs_width_qtdb`) |
| `QRS_UNCERTAIN_MS` | None | set to 95% of the measured error once QT Database is run |

On synthetic complexes the width estimate reads about a quarter short, so it
does not meet IEC yet. Until it does, the 120 ms line is not used to name a
rhythm with confidence; extreme tachycardia fires either way.

## Known limits, measured

- Gap length is off by a few tens of ms, either direction (pinned by a test).
- Noise inside the amplitude window adds to the reading: the error is towards
  counting a beat, not towards a false asystole.
- A patient whose CM5 QRS is 0.15 mV or less reads as asystole, by the
  standard's design. Check amplitude on the first patch recordings.

## Open

| Item | Owner |
|---|---|
| Run `eval_qrs_width_qtdb` and set `QRS_UNCERTAIN_MS` | AI track |
| Run `eval_rules_holter` (NSRDB, Icentia) and `eval_rules_challenge2015` | AI track |
| App: pass `lead_off` and `complete`, never fill gaps, read `STATUS_CHAR` | app |
| Lead-off rule on the final PCB | hardware |
| Chip high-pass at 0.5 Hz vs the documented 0.05 Hz diagnostic path | hardware |

## Run

    python tests/test_rules.py
    python tests/test_decision_layer.py
    python -m munji_ai.rules.rate_engine
    python -m munji_ai.eval.eval_qrs_width_qtdb --download --data-dir data/qtdb
    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb
    python -m munji_ai.eval.eval_rules_challenge2015 --data-dir data/challenge2015/training
