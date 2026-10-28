"""Checks data/vtac.py without the network: the PhysioNet calls are replaced.

    python tests/test_vtac.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data import vtac

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    print("Helpers\n")
    check("V lead preferred over II", vtac.pick_lead(["II", "V", "PLETH"]) == 1)
    check("II when there is no V", vtac.pick_lead(["PLETH", "ABP", "II"]) == 2)
    check("no ECG lead gives None", vtac.pick_lead(["PLETH", "ABP"]) is None)
    check("decision parsing", vtac.label({"decision": "True"}) is True
          and vtac.label({"decision": "0"}) is False and vtac.label({"decision": "?"}) is None)
    sm = vtac.split_map([{"record": "r1", "event": "e1", "split": "Train"},
                         {"record": "r2", "event": "e2", "split": "Validation"}])
    check("split column found and normalised", sm["e1"] == "train" and sm["e2"] == "val")
    x = np.ones(1000)
    x[:50] = np.nan
    check("short gaps are filled", np.isfinite(vtac.clean(x)).all())
    x[:500] = np.nan
    check("mostly missing slices are dropped", vtac.clean(x) is None)

    print("\nEnd to end, network replaced\n")
    labels = [{"record": f"p{i // 2}", "event": f"e{i}", "decision": "True" if i % 3 else "False"}
              for i in range(9)]
    splits = [{"record": f"p{i}", "split": "train" if i < 3 else "test"} for i in range(5)]
    calls = []

    def fake_csv(name):
        return labels if name == "event_labels.csv" else splits

    def fake_read(record, event, start_s=None):
        calls.append(event)
        fs = 250
        t = np.arange(int(40 * fs)) / fs
        return np.sin(2 * np.pi * 3 * t), fs, "V"

    real_csv, real_read = vtac.fetch_csv, vtac.read_event
    vtac.fetch_csv, vtac.read_event = fake_csv, fake_read
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cache = Path(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = vtac.main(["--cache", str(cache), "--workers", "2"])
            d = np.load(cache / "vtac_windows.npz")
            n_true = sum(1 for r in labels if r["decision"] == "True")
            check("runs and saves", rc == 0 and d["X"].shape == (n_true, 10000), d["X"].shape)
            check("true alarms only by default", bool(d["decision"].all()))
            check("splits attached by record", set(d["split"]) <= {"train", "test"}
                  and "" not in set(d["split"]))
            check("lead recorded", set(d["lead"]) == {"V"})
            first = len(calls)
            with contextlib.redirect_stdout(io.StringIO()):
                vtac.main(["--cache", str(cache), "--workers", "2"])
            check("a second run reads nothing again (resumable)", len(calls) == first)
            with contextlib.redirect_stdout(io.StringIO()):
                vtac.main(["--cache", str(cache), "--limit", "2", "--false", "2", "--workers", "2"])
            d = np.load(cache / "vtac_windows.npz")
            fr = set(d["record"][~d["decision"]])
            tr = set(d["record"][d["decision"]])
            check("--false adds false alarms, same patients first",
                  (~d["decision"]).sum() == 2 and fr <= tr, f"false from {sorted(fr)}, true {sorted(tr)}")
            starts = []
            def fake_read_b(record, event, start_s=None):
                starts.append(start_s)
                return fake_read(record, event)
            vtac.read_event = fake_read_b
            with contextlib.redirect_stdout(io.StringIO()):
                vtac.main(["--cache", str(cache), "--limit", "6", "--false", "3",
                           "--baseline", "--workers", "2"])
            b = np.load(cache / "vtac_baseline.npz")
            check("--baseline: one slice per patient, from 60 s",
                  len(b["record"]) == len(set(b["record"])) and set(starts) == {60.0},
                  f"{len(b['record'])} slices, starts {set(starts)}")
            vtac.read_event = fake_read
            with contextlib.redirect_stdout(io.StringIO()):
                vtac.main(["--cache", str(cache), "--all", "--workers", "2"])
            d = np.load(cache / "vtac_windows.npz")
            check("--all adds the false alarms", len(d["X"]) == len(labels)
                  and (~d["decision"]).sum() == len(labels) - n_true)
    finally:
        vtac.fetch_csv, vtac.read_event = real_csv, real_read

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
