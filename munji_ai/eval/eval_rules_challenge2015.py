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
# lead preference: chest leads first, CM5 being a chest position
LEAD_ORDER = ("V", "V5", "V4", "V6", "MCL", "II", "I", "III", "AVF", "AVL", "AVR")

ALARM_SEC = 300.0      # alarm onset in every record
LOOKBACK_SEC = C.RULES_WINDOW_SEC + 10.0   # whole windows ending at the alarm; onset lies in the last 10 s


def _alarm(comments):
    text = " ".join(comments).lower()
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


def evaluate(data_dir: Path, out_csv: Path | None = None) -> dict:
    import wfdb

    rows = []
    for hea in sorted(data_dir.glob("*.hea")):
        rec = wfdb.rdrecord(str(hea.with_suffix("")))
        kind, label = _alarm(rec.comments or [])
        if kind is None or label is None:
            continue
        ch = _pick_lead(rec.sig_name)
        if ch is None:
            continue
        fs = int(rec.fs)
        x = rec.p_signal[:, ch]
        a = int((ALARM_SEC - LOOKBACK_SEC) * fs)
        b = int(ALARM_SEC * fs)
        seg = x[a:b]
        if seg.size < (b - a) or np.isnan(seg).any():
            rows.append({"record": hea.stem, "alarm": kind, "true": label,
                         "lead": rec.sig_name[ch], "status": "no_data"})
            continue
        y = model_path(seg, fs)
        reps = [r for _, r in sliding_reports(y, C.TARGET_FS)]
        kinds = set().union(*(r.kinds() for r in reps)) if reps else set()
        raised = bool(kinds & ALARMS[kind])
        gaps = [r.longest_gap_sec for r in reps if np.isfinite(r.longest_gap_sec)]
        slow = [r.slow_run_bpm for r in reps if np.isfinite(r.slow_run_bpm)]
        fast = [r.fast_run_bpm for r in reps if np.isfinite(r.fast_run_bpm)]
        rows.append({"record": hea.stem, "alarm": kind, "true": label,
                     "lead": rec.sig_name[ch], "status": "ok",
                     "raised": raised,
                     "longest_gap_s": round(max(gaps), 2) if gaps else "",
                     "slowest_held_bpm": round(min(slow), 1) if slow else "",
                     "fastest_held_bpm": round(max(fast), 1) if fast else "",
                     "findings": " ".join(sorted(kinds))})

    summary = {}
    for kind in ALARMS:
        mine = [r for r in rows if r["alarm"] == kind and r["status"] == "ok"]
        t = [r for r in mine if r["true"]]
        f = [r for r in mine if not r["true"]]
        summary[kind] = {
            "true_alarms": len(t),
            "raised_on_true": sum(r["raised"] for r in t),
            "false_alarms": len(f),
            "raised_on_false": sum(r["raised"] for r in f),
            "no_data": sum(1 for r in rows if r["alarm"] == kind
                           and r["status"] != "ok"),
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

    print(f"\n{'alarm':<12}{'true':>6}{'raised':>8}{'sens':>7}"
          f"{'false':>8}{'raised':>8}{'rate':>7}{'no data':>9}")
    for kind, s in res["summary"].items():
        sens = s["raised_on_true"] / s["true_alarms"] if s["true_alarms"] else float("nan")
        far = s["raised_on_false"] / s["false_alarms"] if s["false_alarms"] else float("nan")
        print(f"{kind:<12}{s['true_alarms']:>6}{s['raised_on_true']:>8}{sens:>7.2f}"
              f"{s['false_alarms']:>8}{s['raised_on_false']:>8}{far:>7.2f}{s['no_data']:>9}")
    print("\nsens = engine raised it on a true alarm; rate = engine raised it on a"
          " false one.\nNo gate and no lead status here; see the module docstring."
          f"\nPer-record detail: {args.out}")


if __name__ == "__main__":
    main()
