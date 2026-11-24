"""compare_beat: known answers for its counting, then an end-to-end run.

    python tests/test_compare_beat.py

The end-to-end part writes small WFDB records in the PhysioNet layout to a
temporary folder, with random-weight models; it checks the tool runs through
and refuses records outside val, not what an untrained model scores.
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data.windows import config_hash, paint_ignore, paint_mask
from munji_ai.eval import compare_beat as CB

ID = C.BEAT_CLASS_ID
L = C.window_samples("beat")


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("1. counting, on masks with known answers")
    peaks = np.arange(200, L - 200, 200)                 # 23 beats
    cls = np.array(["N"] * len(peaks), dtype="<U3")
    cls[[3, 10]] = "PAC"
    cls[[6, 14, 15, 16, 17]] = "PVC"                     # one single, one run of 4
    y = paint_ignore(paint_mask(L, peaks, cls), np.array([L - 60]))
    rows = CB.per_beat([y], [y.copy()])
    s = CB.summarise(rows)
    check("a perfect model: every sensitivity 1, no false PVC",
          s["PVC sens, single"] == 1.0 and s["PVC sens, in run"] == 1.0
          and s["false PVC per beat"] == 0.0, str({k: s[k] for k in list(s)[:4]}))
    check("the ignored beat is not counted", s["beats"] == len(peaks), f"{s['beats']} beats")
    check("the run of 4 is found", s["PVCs in runs"] == 4)

    p = y.copy()
    for i in (15, 16):                                   # two run PVCs called PAC
        p[peaks[i] - 17:peaks[i] + 18] = ID["PAC"]
    p[peaks[2] - 12:peaks[2] + 13] = ID["PVC"]           # one N called PVC
    p[peaks[6] - 17:peaks[6] + 18] = 0                   # the single PVC missed
    s = CB.summarise(CB.per_beat([y], [p]))
    check("in-run sensitivity counts the two errors", abs(s["PVC sens, in run"] - 0.5) < 1e-9,
          f"{s['PVC sens, in run']}")
    check("the single PVC counts as missed", s["PVC sens, single"] == 0.0 and s["PVC missed"] == 1)
    check("errors land in the right cells", s["PVC -> PAC"] == 2 and s["N -> PVC"] == 1
          and s["PAC -> PVC"] == 0)
    check("false PVC per beat is over every beat",
          abs(s["false PVC per beat"] - 1 / len(peaks)) < 1e-12, f"{s['false PVC per beat']:.4f}")

    print("\n2. end to end on PhysioNet-layout records")
    import torch
    import wfdb
    from munji_ai.models.beat_segmenter import build

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        torch.manual_seed(0)
        for name in ("a", "b"):
            torch.save({"state_dict": build().state_dict(), "epoch": 1, "val_beat_f1": 0.5},
                       d / f"{name}.pt")
        rng = np.random.default_rng(0)
        ys = [y] * 6
        (d / "cache").mkdir()
        np.savez(d / "cache" / "beat_val.npz",
                 X=np.asarray([rng.normal(0, 1, L) for _ in ys], dtype=C.CACHE_DTYPE),
                 y=np.asarray(ys, np.uint8), dataset=np.array(["icentia11k"] * 3 + ["ltafdb"] * 3),
                 patient=np.array(["p"] * 6), record=np.array(["r"] * 6),
                 offset=np.zeros(6, np.int64), cfg_hash=np.asarray(config_hash("beat")),
                 stage=np.asarray("beat"))

        def write(ds, rec, fs, names, syms):
            out = d / "raw" / ds
            out.mkdir(parents=True, exist_ok=True)
            n = int(fs * 90)
            sig = rng.normal(0, 0.05, (n, len(names)))
            samp = np.arange(int(fs), n - int(fs), int(0.8 * fs))
            sig[samp, :] += 1.0
            wfdb.wrsamp(rec, fs=fs, units=["mV"] * len(names), sig_name=names,
                        p_signal=sig, fmt=["16"] * len(names), write_dir=str(out))
            wfdb.wrann(rec, "atr", samp, symbol=[syms[i % len(syms)] for i in range(len(samp))],
                       write_dir=str(out))
            return len(samp)

        n109 = write("mitdb", "109", 360, ["MLII"], ["L", "L", "L", "V"])
        write("incartdb", "I70", 257, ["I", "II", "III", "AVR", "AVL", "AVF",
                                       "V1", "V2", "V3", "V4", "V5", "V6"], ["N", "A", "V"])
        write("incartdb", "I71", 257, ["V5"], ["N"])
        (d / "splits.json").write_text(json.dumps({
            "mitdb/109": {"dataset": "mitdb", "record": "109", "patient": "mitdb:109",
                          "split": "val", "stages": ["beat"]},
            "incartdb/I70": {"dataset": "incartdb", "record": "I70", "patient": "incartdb:30",
                             "split": "val"},
            # A different patient: one patient in two splits is a leak, and
            # splits.load refuses the whole list (which is right).
            "incartdb/I71": {"dataset": "incartdb", "record": "I71", "patient": "incartdb:99",
                             "split": "test"}}))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = CB.main(["--ckpt", f"a={d / 'a.pt'}", "--ckpt", f"b={d / 'b.pt'}",
                          "--cache", str(d / "cache"), "--raw", str(d / "raw"),
                          "--splits-file", str(d / "splits.json")])
        out = buf.getvalue()
        check("runs through both parts", rc == 0 and "PART 1" in out and "PART 2" in out)
        check("scores every source in the cache", "icentia11k" in out and "ltafdb" in out)
        got = {}
        block = out.split("mitdb 109")[1].split("\n\n")[0]
        for line in block.splitlines():
            parts = line.split()
            if len(parts) > 2 and parts[0] in ("L", "V") and parts[1].replace(",", "").isdigit():
                got[parts[0]] = int(parts[1].replace(",", ""))
        want = {"L": sum(1 for i in range(n109) if i % 4 != 3),
                "V": sum(1 for i in range(n109) if i % 4 == 3)}
        check("reads every annotated beat of a whole record, by symbol", got == want,
              f"read {got}, annotated {want}")
        check("a record outside val is refused", "I71: not in val (test), skipped" in out)
        check("a record missing from the split list is refused", "118: not in val (not listed)" in out)
        from munji_ai.data.loader import _pick_channel
        from munji_ai.data.registry import get
        hdr = wfdb.rdrecord(str(d / "raw" / "incartdb" / "I70"))
        check("the lead read from INCART is V5, by name",
              hdr.sig_name[_pick_channel(get("incartdb"), hdr)] == "V5",
              f"index {_pick_channel(get('incartdb'), hdr)}")
        if not ok:
            print(out)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
