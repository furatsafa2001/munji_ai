"""The template test on VTaC: real VT from ICU monitors, judged by experts.

    python -m munji_ai.data.vtac --limit 700 --false 700              # once: the alarms
    python -m munji_ai.data.vtac --limit 700 --false 700 --baseline   # once: each patient's own rhythm
    python experiments/template_vtac.py
    python experiments/template_vtac.py --max-events 50               # smoke test

Why
---
template_wct.py had two slow-VT episodes to learn from (VFDB 611 and 607).
VTaC has hundreds of VT alarms from ICU bedside monitors, each judged true or
false by at least two experts, and each patient's ordinary rhythm on the same
monitor four minutes before an alarm (vtac_baseline.npz). That tests the
beat-level core of the rule on far more patients: are VT beats unlike the
patient's own beat, and are the beats that set off a false alarm not?

What runs
---------
template  from the patient's baseline slice, only when it was recorded on the
          same lead as the alarm: template_wct's patch-on calibration (main
          shape of a 10 s stretch at 40 to 100 bpm), its wide template, its
          size. There are no rhythm labels, so no stretch is excluded. The
          baseline is read from the patient's first alarm recording, four
          minutes before that alarm. For the patient's other alarms it comes
          from another recording, maybe hours away, where electrodes and gain
          may have changed; the patch keeps its place. So the scores are
          shown twice: every alarm, and only alarms whose template comes
          from their own recording.
beats     the rate engine's detector on the alarm's 40 s window (the alarm at
          20 s), each beat matched against the template: shape, wide, size.
VT run    VTaC's own definition, five or more consecutive ventricular beats
          above 100 bpm: here five or more consecutive beats unlike the
          template with every RR interval under 0.6 s, between 15 s before
          and 5 s after the alarm.
fast run  the same without the template: five or more consecutive beats with
          every RR under 0.6 s. Found or not, it is the detector's ceiling;
          the rate of the fastest one splits true alarms into 100-150 and 150
          or more, so slow VT can be read on its own.

Scored
------
True alarms (confirmed VT): the share with a VT run found, overall and by
rate. False alarms (artifact, SVT, aberrant beats that fooled the monitor):
the share with one found, lower is better. Alarms whose patient has no
baseline on the same lead, or no template, are counted apart. The VTaC test
split is skipped. This is not the 30 s rule: VTaC's runs are mostly short,
so onset and stability are not scored here.

Reads only.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import template_wct as tw  # noqa: E402
from munji_ai import config as C  # noqa: E402
from munji_ai.rules.rate_engine import detect_beats  # noqa: E402

ALARM_SEC = 20.0                       # the alarm's place in the 40 s window
SEARCH = (-15.0, 5.0)                  # where a VT run is looked for, around the alarm
RUN_BEATS = 5                          # VTaC: five or more consecutive ventricular beats
RUN_MAX_RR = 60.0 / C.VT_MIN_BPM       # above 100 bpm: every RR under 0.6 s
KEYS = [("shape", False, 0.5), ("shape", False, 0.6), ("shape", False, 0.7),
        ("shape", True, 0.6), ("wide", False, 0.6), ("wide", True, 0.6)]


def template_from(x: np.ndarray, fs: int, mode: str = "dominant"):
    """Patch-on calibration on one slice: (shape template, wide template), or None."""
    pre, L, lag = tw._sizes(fs)
    pos, ext = tw.extended(x, np.asarray(detect_beats(x, fs)[0], np.int64), fs)
    if len(pos) < tw.CAL_MIN_BEATS:
        return None
    ts = pos / fs
    cal, _ = tw.calibrations(ts, ext, L, lag, mode)
    if not cal:
        return None
    s, e, tpl = cal[0]
    Lw, wext = tw.wide_windows(x, pos, fs)
    keep = (ts >= s) & (ts < e)
    return tpl, tw.wide_template(ext[keep], wext[keep], tpl, Lw)


def matched_beats(x: np.ndarray, fs: int, tpl, tplw):
    """(beat times in s, match of every beat) on one alarm window."""
    pre, L, lag = tw._sizes(fs)
    pos, ext = tw.extended(x, np.asarray(detect_beats(x, fs)[0], np.int64), fs)
    Lw, wext = tw.wide_windows(x, pos, fs)
    return pos / fs, tw.match(ext, wext, tpl, tplw, L, Lw)


def runs(ts: np.ndarray, ok: np.ndarray) -> list:
    """(beats, median RR s) of every run of consecutive beats where ok holds
    and every RR inside the run is under RUN_MAX_RR."""
    out, i, n = [], 0, len(ts)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and ts[j + 1] - ts[j] < RUN_MAX_RR:
            j += 1
        out.append((j - i + 1, float(np.median(np.diff(ts[i:j + 1]))) if j > i else float("nan")))
        i = j + 1
    return out


def judge(ts: np.ndarray, feat: dict) -> dict:
    """For one alarm: the fastest run's rate, and per KEY whether a VT run is found."""
    sel = (ts >= ALARM_SEC + SEARCH[0]) & (ts <= ALARM_SEC + SEARCH[1])
    t = ts[sel]
    fast = [rr for n, rr in runs(t, np.ones(len(t), bool)) if n >= RUN_BEATS]
    out = {"rate": 60.0 / min(fast) if fast else float("nan")}
    for k in KEYS:
        u = tw.unlike(feat, k)[sel]
        out[k] = any(n >= RUN_BEATS for n, _ in runs(t, u))
    return out


def evaluate(win: dict, base: dict, fs: int, mode: str = "dominant", max_events: int = 0) -> dict:
    """Every alarm judged against its patient's template. win and base: the
    arrays of vtac_windows.npz and vtac_baseline.npz. The baseline of the
    alarm's own recording is used when there is one, else the patient's on
    the same lead; "same" in each row says which."""
    tpl_of, skipped, rows = {}, Counter(), []
    keys = [(str(r), str(e), str(ld)) for r, e, ld in zip(base["record"], base["event"], base["lead"])]
    same = {k: i for i, k in enumerate(keys)}
    patient = {(r, ld): i for i, (r, _, ld) in enumerate(keys)}
    n = len(win["record"]) if not max_events else min(max_events, len(win["record"]))
    for i in range(n):
        if str(win["split"][i]) == "test":
            skipped["test split"] += 1
            continue
        key = (str(win["record"][i]), str(win["event"][i]), str(win["lead"][i]))
        b = same.get(key, patient.get((key[0], key[2])))
        if b is None:
            skipped["no baseline on the alarm's lead"] += 1
            continue
        if b not in tpl_of:
            tpl_of[b] = template_from(np.asarray(base["X"][b], np.float64), fs, mode)
        if tpl_of[b] is None:
            skipped["no template in the baseline"] += 1
            continue
        ts, feat = matched_beats(np.asarray(win["X"][i], np.float64), fs, *tpl_of[b])
        rows.append({"record": key[0], "true": bool(win["decision"][i]), "same": key in same,
                     **judge(ts, feat)})
    return {"rows": rows, "skipped": skipped}


def _share(rows: list, key) -> str:
    if not rows:
        return "-"
    k = sum(bool(r[key]) for r in rows)
    return f"{k}/{len(rows)} ({k / len(rows):.0%})"


def _table(rows: list, title: str) -> None:
    true = [r for r in rows if r["true"]]
    false = [r for r in rows if not r["true"]]
    slow = [r for r in true if 100.0 < r["rate"] < 150.0]
    fast = [r for r in true if r["rate"] >= 150.0]
    no_run = sum(1 for r in true if not np.isfinite(r["rate"]))
    print(f"\n{title}: {len(true)} true VT, {len(false)} false, {len({r['record'] for r in rows})} patients")
    print(f"detector    a fast run of {RUN_BEATS} beats in {len(true) - no_run} of {len(true)} true alarms "
          f"(100-150 bpm: {len(slow)}, 150 or more: {len(fast)}) and "
          f"{sum(1 for r in false if np.isfinite(r['rate']))} of {len(false)} false alarms")
    print(f"{'match':<20}{'true alarms':>16}{'100-150 bpm':>16}{'150 or more':>16}{'false alarms':>16}")
    for k in KEYS:
        m, size, c = k
        name = f"{m} C {c}" + (" + size" if size else "")
        print(f"{name:<20}{_share(true, k):>16}{_share(slow, k):>16}{_share(fast, k):>16}{_share(false, k):>16}")


def report(res: dict) -> None:
    rows = res["rows"]
    print(f"\nalarms      {len(rows)} judged"
          + ("; not judged: " + ", ".join(f"{k} {v}" for k, v in res["skipped"].items())
             if res["skipped"] else ""))
    print(f"VT run found: {RUN_BEATS} beats unlike the template, every RR under {RUN_MAX_RR:.1f} s, "
          f"{-SEARCH[0]:.0f} s before to {SEARCH[1]:.0f} s after the alarm")
    _table(rows, "every alarm")
    _table([r for r in rows if r["same"]], "template from the alarm's own recording")
    print("(true alarms: higher is better; false alarms: lower is better. 100-150 and 150 or more: true"
          "\n alarms by the rate of their fastest run)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--cal", choices=tw.CALS, default="dominant", help="template calibration")
    ap.add_argument("--max-events", type=int, default=0, help="first N alarms only (0 = all)")
    a = ap.parse_args(argv)
    paths = [a.cache / "vtac_windows.npz", a.cache / "vtac_baseline.npz"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing " + ", ".join(str(p) for p in missing) + ". Read them once with:\n"
              "    python -m munji_ai.data.vtac --limit 700 --false 700\n"
              "    python -m munji_ai.data.vtac --limit 700 --false 700 --baseline")
        return 1
    win, base = (dict(np.load(p, allow_pickle=False)) for p in paths)
    print(f"read        {len(win['record'])} alarms, {len(base['record'])} baseline slices")
    report(evaluate(win, base, C.TARGET_FS, a.cal, a.max_events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
