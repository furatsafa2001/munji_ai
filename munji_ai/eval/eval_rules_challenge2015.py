"""Run the rate engine on real, expert-labelled alarms.

    python -m munji_ai.eval.eval_rules_challenge2015 --data-dir data/challenge2015/training

Data: PhysioNet/CinC Challenge 2015, "Reducing false arrhythmia alarms in the
ICU" (ODC-BY). 750 training records, 250 Hz, each ending at a bedside-monitor
alarm that experts labelled true or false. Download the ZIP from
physionet.org/content/challenge-2015/1.0.0/ and point --data-dir at the
folder holding the .hea/.mat files.

What it measures
----------------
For the three alarm types this engine covers (asystole, extreme bradycardia,
extreme tachycardia): on the true alarms, how often the engine raises the
same finding; on the false alarms, how often it would have raised it anyway.
The second number is the one that drives caregiver trust.

How
---
The rate engine runs at its deployment window and stride (40 s every 2.5 s)
over the seconds before the alarm; the alarm counts as raised if any window
holds the matching finding. The Challenge places the event's onset within
the 10 s before the alarm.

What it does not measure
------------------------
- Lead status: these recordings carry none, so absence findings are judged
  here at window level only, as if the leads were reported on.
- The quality gate: not applied, so noisy windows count. Expect the false
  alarm column to be worse than deployment.
- Device fit: ICU bedside leads, not the CM5 patch. A precordial lead is
  preferred when present because CM5 is a chest position.
- Definitions differ, on purpose: the Challenge calls extreme tachycardia
  > 140 bpm (this engine >= 150, AHA) and asystole 4 s (this engine > 6 s,
  2018 ESC). True alarms between the two lines are missed by design; the
  held rate and the longest gap are printed so that shows rather than hides.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..data.preprocess import model_path
from .. import config as C
from ..rules.rate_engine import sliding_reports

# alarm keyword in the header -> finding kinds that count as "raised"
ALARMS = {
    "asystole": {"asystole"},
    "bradycardia": {"extreme_bradycardia"},
    "tachycardia": {"extreme_tachycardia"},
}
# finding kinds that are SENT to the caregiver for each alarm type
NOTIFIED = {
    "asystole": {"asystole", "pause"},
    "bradycardia": {"sustained_bradycardia", "profound_bradycardia"},
    "tachycardia": {"sustained_tachycardia"},
}
# lead preference: chest leads first, CM5 being a chest position
LEAD_ORDER = ("V", "V5", "V4", "V6", "MCL", "II", "I", "III", "AVF", "AVL", "AVR")

ALARM_SEC = 300.0      # alarm onset in every record
LOOKBACK_SEC = C.RULES_WINDOW_SEC + 10.0   # whole windows ending at the alarm; onset lies in the last 10 s
AFTER_SEC = 30.0       # long records ("l") carry 30 s past the alarm


def _alarm(comments):
    text = " ".join(comments).lower()
    # "Ventricular_Tachycardia" contains "tachycardia": VT and VF/flutter
    # alarms are ventricular rhythms, outside the rate rules (they belong to the
    # ventricular-beat detector and ShockableNet). The record name's first
    # letter says the same: a, b, t are ours; v, f are not.
    if "ventricular" in text or "flutter" in text or "fib" in text:
        return None, None
    kind = next((k for k in ALARMS if k in text), None)
    if "true alarm" in text:
        label = True
    elif "false alarm" in text:
        label = False
    else:
        label = None
    return kind, label


def _pick_lead(names):
    upper = [n.upper() for n in names]
    for want in LEAD_ORDER:
        if want in upper:
            return upper.index(want)
    return None


ECG_NAMES = {"I", "II", "III", "V", "V1", "V2", "V3", "V4", "V5", "V6",
             "AVR", "AVL", "AVF", "MCL"}


def _run_lead(x, fs, after_sec=0.0):
    """Engine over the seconds before the alarm (and after it, if asked)."""
    a = int((ALARM_SEC - LOOKBACK_SEC) * fs)
    b = int(ALARM_SEC * fs)
    seg = x[a:b]
    if seg.size < (b - a) or np.isnan(seg).any():
        return None
    if after_sec > 0:
        tail = x[b:min(len(x), b + int(after_sec * fs))]
        if tail.size and not np.isnan(tail).any():
            seg = np.concatenate([seg, tail])
    reps = [r for _, r in sliding_reports(model_path(seg, fs), C.TARGET_FS)]
    kinds = set().union(*(r.kinds() for r in reps)) if reps else set()
    gaps = [r.longest_gap_sec for r in reps if np.isfinite(r.longest_gap_sec)]
    slow = [r.slow_run_bpm for r in reps if np.isfinite(r.slow_run_bpm)]
    fast = [r.fast_run_bpm for r in reps if np.isfinite(r.fast_run_bpm)]
    return {"kinds": kinds,
            "flat": any(r.flat for r in reps),
            "gap": max(gaps) if gaps else float("nan"),
            "slow": min(slow) if slow else float("nan"),
            "fast": max(fast) if fast else float("nan")}


def _reason(kind, true, pref, leads):
    """Why the preferred lead missed a true alarm (or raised a false one)."""
    raised_any = any(r and (r["kinds"] & ALARMS[kind]) for r in leads.values())
    if true:
        if pref is None:
            return "no data"
        if kind == "tachycardia" and 140 < pref["fast"] < C.HR_TACHY["critical"]:
            return "140-150 bpm: below our line by design"
        if kind == "asystole" and 4.0 <= pref["gap"] <= C.ASYSTOLE_CRITICAL_SEC:
            return "4-6 s gap: below our line by design"
        if kind == "bradycardia" and C.HR_BRADY["critical"] <= pref["slow"] < 45:
            return "40-45 bpm: at the line, within EC13 rate tolerance"
        if raised_any:
            return "caught on another lead: single-lead limit"
        return "not caught on any lead"
    if pref is not None and pref["flat"]:
        return "flat lead: lead-off on the patch, not a cardiac alert"
    return "raised on a real false alarm"


def evaluate(data_dir: Path, out_csv: Path | None = None) -> dict:
    import wfdb

    rows = []
    for hea in sorted(data_dir.glob("*.hea")):
        rec = wfdb.rdrecord(str(hea.with_suffix("")))
        kind, label = _alarm(rec.comments or [])
        if kind is None or label is None:
            continue
        fs = int(rec.fs)
        leads = {}
        for ch, name in enumerate(rec.sig_name):
            if name.upper() in ECG_NAMES:
                leads[name] = _run_lead(rec.p_signal[:, ch], fs)
        pch = _pick_lead(rec.sig_name)
        pname = rec.sig_name[pch] if pch is not None else None
        pref = leads.get(pname) if pname else None
        raised = bool(pref and (pref["kinds"] & ALARMS[kind]))
        notified = bool(pref and (pref["kinds"] & NOTIFIED[kind]))
        # A notification needs the rate held 30 s, but the Challenge places
        # the event's onset at most 10 s before the alarm. Only long records
        # carry the 30 s that follow; the "sent" figure is judged on them.
        is_long = len(rec.p_signal) >= int((ALARM_SEC + AFTER_SEC) * fs) - 1
        notified_long = None
        if is_long and pch is not None:
            after = _run_lead(rec.p_signal[:, pch], fs, after_sec=AFTER_SEC)
            notified_long = bool(after and (after["kinds"] & NOTIFIED[kind]))
        raised_any = any(r and (r["kinds"] & ALARMS[kind]) for r in leads.values())
        row = {"record": hea.stem, "alarm": kind, "true": label, "lead": pname,
               "status": "ok" if pref is not None else "no_data",
               "raised": raised, "notified": notified, "long": is_long,
               "notified_with_after": notified_long, "raised_any_lead": raised_any,
               "leads": " ".join(leads)}
        if pref is not None:
            row.update({"longest_gap_s": round(pref["gap"], 2),
                        "slowest_held_bpm": round(pref["slow"], 1) if np.isfinite(pref["slow"]) else "",
                        "fastest_held_bpm": round(pref["fast"], 1) if np.isfinite(pref["fast"]) else "",
                        "flat": pref["flat"],
                        "findings": " ".join(sorted(pref["kinds"]))})
        if (label and not raised) or (not label and raised):
            row["reason"] = _reason(kind, label, pref, leads)
        rows.append(row)

    summary = {}
    for kind in ALARMS:
        mine = [r for r in rows if r["alarm"] == kind and r["status"] == "ok"]
        t = [r for r in mine if r["true"]]
        f = [r for r in mine if not r["true"]]
        reasons = {}
        for r in mine:
            if "reason" in r:
                key = ("missed: " if r["true"] else "false: ") + r["reason"]
                reasons[key] = reasons.get(key, 0) + 1
        by_design = sum(1 for r in t if "by design" in r.get("reason", "")
                        or "rate tolerance" in r.get("reason", ""))
        summary[kind] = {
            "by_design": by_design,
            "true_alarms": len(t),
            "raised_on_true": sum(r["raised"] for r in t),
            "raised_on_true_any": sum(r["raised_any_lead"] for r in t),
            "false_alarms": len(f),
            "raised_on_false": sum(r["raised"] for r in f),
            "notified_on_true": sum(r["notified"] for r in t),
            "notified_on_false": sum(r["notified"] for r in f),
            "long_true": sum(1 for r in t if r["long"]),
            "long_false": sum(1 for r in f if r["long"]),
            "long_notified_true": sum(1 for r in t if r["long"] and r["notified_with_after"]),
            "long_notified_false": sum(1 for r in f if r["long"] and r["notified_with_after"]),
            "no_data": sum(1 for r in rows if r["alarm"] == kind and r["status"] != "ok"),
            "reasons": reasons,
        }

    if out_csv:
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return {"rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("rules_challenge2015.csv"))
    args = ap.parse_args()
    res = evaluate(args.data_dir, args.out)

    print(f"\n{'alarm':<12}{'true':>6}{'raised':>8}{'sens':>7}{'at our lines':>14}{'any lead':>10}"
          f"{'false':>8}{'raised':>8}{'rate':>7}{'no data':>9}")
    for kind, s in res["summary"].items():
        sens = s["raised_on_true"] / s["true_alarms"] if s["true_alarms"] else float("nan")
        anyl = s["raised_on_true_any"] / s["true_alarms"] if s["true_alarms"] else float("nan")
        far = s["raised_on_false"] / s["false_alarms"] if s["false_alarms"] else float("nan")
        in_scope = s["true_alarms"] - s["by_design"]
        ours = s["raised_on_true"] / in_scope if in_scope else float("nan")
        print(f"{kind:<12}{s['true_alarms']:>6}{s['raised_on_true']:>8}{sens:>7.2f}{ours:>14.2f}{anyl:>10.2f}"
              f"{s['false_alarms']:>8}{s['raised_on_false']:>8}{far:>7.2f}{s['no_data']:>9}")
    print(f"\n{'sent to the caregiver':<24}{'true':>6}{'sent':>6}{'share':>7}{'false':>8}{'sent':>6}{'share':>7}")
    for kind, s in res["summary"].items():
        a = s["notified_on_true"] / s["true_alarms"] if s["true_alarms"] else float("nan")
        b = s["notified_on_false"] / s["false_alarms"] if s["false_alarms"] else float("nan")
        print(f"  {kind:<22}{s['true_alarms']:>6}{s['notified_on_true']:>6}{a:>7.2f}"
              f"{s['false_alarms']:>8}{s['notified_on_false']:>6}{b:>7.2f}")
    print(f"\n{'sent, long records incl. 30 s after':<36}{'true':>6}{'sent':>6}{'share':>7}{'false':>8}{'sent':>6}{'share':>7}")
    for kind, s in res["summary"].items():
        a = s["long_notified_true"] / s["long_true"] if s["long_true"] else float("nan")
        b = s["long_notified_false"] / s["long_false"] if s["long_false"] else float("nan")
        print(f"  {kind:<34}{s['long_true']:>6}{s['long_notified_true']:>6}{a:>7.2f}"
              f"{s['long_false']:>8}{s['long_notified_false']:>6}{b:>7.2f}")
    print("  (the event starts at most 10 s before the alarm, so a rule that needs 30 s"
          "\n   held can only be judged where the record continues past it)")
    print("  (raised = graded for the report; sent = notified: pause/asystole, a rate"
          "\n   below 40 or at or above 180 held 30 s, or at or below 30 held 30 s)")
    print("\nat our lines: sensitivity once the true alarms that sit below our thresholds by"
          "\ndesign (tachy 140-150, asystole 4-6 s, brady 40-45 within EC13 tolerance) are set"
          "\naside; the Challenge uses ICU limits, the engine uses the clinical sources in config.")
    print("sens: on the preferred lead (a chest lead, closest to CM5). any lead: caught on at"
          "\nleast one ECG lead of the record. The patch has one lead, so 'sens' is the figure;"
          "\nthe gap between the two is the single-lead limit, not the rules.")
    print("\nwhy, per alarm type:")
    for kind, s in res["summary"].items():
        print(f"  {kind}")
        for reason, n in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>4}  {reason}")
    print("\nNo gate and no lead status here; see the module docstring."
          f"\nPer-record detail: {args.out}")


if __name__ == "__main__":
    main()
