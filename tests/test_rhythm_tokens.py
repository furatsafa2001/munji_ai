"""Checks rhythm-token parsing, and scans the local raw data for unknown tokens.

    python tests/test_rhythm_tokens.py

The first part builds small WFDB records in a temporary folder, so it runs
anywhere. The second part reads every annotation file under data/raw (Icentia
and CinC 2017 excepted: their labels come from elsewhere) and fails on any
rhythm token that is neither mapped, deliberately left as OTHER, nor an ST/T
change marker. That is the check that would have caught "(VF".
"""

import collections
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.config import RAW_DIR
from munji_ai.data.loader import NON_RHYTHM, OTHER_TOKENS, RHYTHM_MAP, load_record
from munji_ai.data.windows import rhythm_intervals

SKIP = {"icentia11k", "challenge2017"}
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def fake_record(root, dataset, name, aux, positions, seconds=60, fs=250):
    import wfdb
    d = Path(root) / dataset
    d.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(0).normal(0, 0.1, seconds * fs)
    wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                fmt=["16"], write_dir=str(d))
    wfdb.wrann(name, "atr", np.array(positions), symbol=["+"] * len(aux),
               aux_note=aux, write_dir=str(d))
    return load_record(dataset, name, Path(root))


def main() -> int:
    print("Token table\n")
    check('"(VF" means ventricular fibrillation', RHYTHM_MAP.get("(VF") == "VF")
    check('"(VFIB" and "(VFL" still map to VF',
          RHYTHM_MAP.get("(VFIB") == "VF" and RHYTHM_MAP.get("(VFL") == "VF")
    st = ["(ST0+", "(ST1-", "(T0+", "(T1--", "(st0-", "(t1+", "(T0+\x00\x13"]
    check("ST/T change markers are recognised", all(NON_RHYTHM.match(s) for s in st))
    check('"(T" (trigeminy) is not mistaken for one',
          not NON_RHYTHM.match("(T") and RHYTHM_MAP.get("(T") == "TRIGEMINY")
    check("no token is both mapped and listed as OTHER",
          not (set(RHYTHM_MAP) & OTHER_TOKENS))

    print("\nIntervals from annotated records\n")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        fs = 250
        rec = fake_record(tmp, "vfdb", "418", ["(N", "(VF", "(N"],
                          [10, 20 * fs, 40 * fs])
        labs = [lab for _, _, lab in rhythm_intervals(rec)]
        check('a "(VF" episode becomes a VF interval', labs == ["NSR", "VF", "NSR"],
              ", ".join(labs))
        rec = fake_record(tmp, "edb", "e0103",
                          ["(N", "(ST0+", "(VT", "(T1-", "(N"],
                          [10, 10 * fs, 20 * fs, 25 * fs, 40 * fs])
        iv = rhythm_intervals(rec)
        labs = [lab for _, _, lab in iv]
        check("ST/T markers no longer cut the rhythm", labs == ["NSR", "VT", "NSR"],
              ", ".join(labs))
        vt = [(s, e) for s, e, lab in iv if lab == "VT"]
        check("the VT run keeps its full length", vt and vt[0][1] - vt[0][0] == 20 * fs)

    print("\nLocal raw data\n")
    import wfdb
    raw = Path(RAW_DIR)
    if not raw.exists():
        print(f"  no {raw} here; scan skipped")
    else:
        for d in sorted(p for p in raw.iterdir() if p.is_dir() and p.name not in SKIP):
            seen = collections.Counter()
            for h in sorted(d.rglob("*.hea")):
                try:
                    ann = wfdb.rdann(str(h.with_suffix("")), "atr")
                except Exception:
                    continue
                for a in ann.aux_note or []:
                    a = str(a).strip("\x00").strip()
                    if a.startswith("(") and not NON_RHYTHM.match(a):
                        seen[a.split("\x00")[0].strip()] += 1
            if not seen:
                continue
            unknown = {k: v for k, v in seen.items()
                       if k not in RHYTHM_MAP and k not in OTHER_TOKENS}
            check(f"{d.name}: every rhythm token is accounted for", not unknown,
                  f"unknown {unknown}" if unknown else f"{sum(seen.values())} markers")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
