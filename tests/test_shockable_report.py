"""Checks eval/shockable_report.py end to end on a small fake world.

Fake vfdb recordings with annotated VF, fake nsrdb days, a matching cache,
out-of-fold file, untrained fold models and a noise bank, all in a temporary
folder. The numbers are meaningless; the checks are that every section runs,
every table and figure is written, and the EC57 arithmetic is right.

    python tests/test_shockable_report.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import shockable_report as sr
from munji_ai.models.shockable_net import build

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ecg(seconds, fs, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * fs)
        w = max(int(0.01 * fs), 1)
        lo, hi = max(c - 4 * w, 0), min(c + 4 * w, n)
        x[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - c) / w) ** 2)
        t += 0.85
    return x + rng.normal(0, 0.03, n)


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    print("EC57 arithmetic\n")
    a = {"episodes": 4, "caught": 3, "true": 3, "false": 1, "truth_windows": 40,
         "fired_windows": 30, "overlap_windows": 24, "sus_episodes": 2, "sus_caught": 2}
    e = sr.ec57(a)
    check("ESe, E+P, DSe, D+P follow their definitions",
          (e["ESe"], e["E+P"], e["DSe"], e["D+P"]) == (0.75, 0.75, 0.6, 0.8))
    check("sustained detection is reported separately", e["sustained_ESe"] == 1.0)
    f = sr.fired_mask(np.array([0, 1, 1, 0, 0, 1, 0, 1], bool), (2, 3))
    check("2-of-3 confirmation mask", f.tolist() ==
          [False, False, True, True, False, False, False, True])
    check("status helper", sr.status(0.95, 0.90) == "meets"
          and sr.status(4.2, 0.29, higher_is_better=False) == "above")

    print("\nEnd to end on a fake world\n")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, models, out = tmp / "raw", tmp / "cache", tmp / "models", tmp / "out"
        for d in (raw / "vfdb", raw / "nsrdb", cache, models):
            d.mkdir(parents=True)

        fs = 250
        for k, name in enumerate(("418", "419")):
            x = ecg(240, fs, seed=k)
            vf = slice(80 * fs, 160 * fs)                     # 80 s of "VF"
            x[vf] = np.sin(2 * np.pi * 5 * np.arange(vf.stop - vf.start) / fs)
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                        fmt=["16"], write_dir=str(raw / "vfdb"))
            wfdb.wrann(name, "atr", np.array([10, 80 * fs, 160 * fs]),
                       symbol=["+", "+", "+"], aux_note=["(N", "(VF", "(N"],
                       write_dir=str(raw / "vfdb"))
        for k, name in enumerate(("16265", "16272")):
            x = ecg(600, 128, seed=10 + k)
            wfdb.wrsamp(name, fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                        p_signal=np.stack([x, x], 1), fmt=["16", "16"],
                        write_dir=str(raw / "nsrdb"))

        rec = np.array(["418"] * 8 + ["16265"] * 8 + ["419"] * 8 + ["16272"] * 8)
        dset = np.array(["vfdb"] * 8 + ["nsrdb"] * 8 + ["vfdb"] * 8 + ["nsrdb"] * 8)
        y = np.array((["shockable"] * 4 + ["not_shockable"] * 4 + ["not_shockable"] * 8) * 2)
        off = np.tile(np.arange(8) * 625 * 4, 4)
        X = rng.standard_normal((32, 1250)).astype(np.float16)
        for split, sl in (("train", slice(0, 16)), ("val", slice(16, 32))):
            np.savez(cache / f"shockable_{split}.npz", X=X[sl], y=y[sl],
                     dataset=dset[sl], patient=np.char.add(dset[sl], ":" + rec[sl]),
                     record=rec[sl], offset=off[sl], cfg_hash=config_hash())
        np.savez(models / "shockable_oof.npz", y=y, pred=y, score=rng.random(32),
                 patient=rec, record=rec, offset=off,
                 fold=np.array([0] * 16 + [1] * 16))
        torch.manual_seed(0)
        for k in (0, 1):
            torch.save(build().state_dict(), models / f"shockable_fold{k}.pt")
        np.savez(cache / "noise_bank.npz", fs=250,
                 em=rng.standard_normal(30000).astype(np.float32),
                 ma=rng.standard_normal(30000).astype(np.float32))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sr.main(["--models", str(models), "--cache", str(cache), "--raw", str(raw),
                          "--out", str(out), "--day-datasets", "nsrdb",
                          "--arrest-datasets", "vfdb"])
        check("runs end to end", rc == 0, buf.getvalue().strip().splitlines()[-1])
        m = json.loads((out / "metrics.json").read_text())
        check("EC57 section present", "ec57" in m and m["ec57"]["episodes"] == 2,
              f"{m.get('ec57', {}).get('episodes')} episodes")
        check("both annotated VF episodes are sustained (80 s)",
              m["ec57"]["sustained_episodes"] == 2)
        check("daily-life section present", "days" in m and "nsrdb" in m["days"])
        check("noise section present", "noise" in m)
        check("comparison against targets written",
              any(c["measure"].startswith("false alarms per patient-day")
                  for c in m["comparison"]))
        figs = sorted(p.name for p in out.glob("fig*.png"))
        check("figures written", len(figs) >= 6, ", ".join(figs))
        md = (out / "summary.md").read_text(encoding="utf-8")
        check("summary has every section",
              all(h in md for h in ("## Per window", "## EC57", "## False alarms",
                                    "## Per rhythm", "## Against published targets",
                                    "## Limits")))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sr.main(["--models", str(models), "--cache", str(cache), "--raw", str(raw),
                          "--out", str(tmp / "quick"), "--skip-days",
                          "--arrest-datasets", "vfdb"])
        m2 = json.loads((tmp / "quick" / "metrics.json").read_text())
        check("--skip-days leaves the slow section out", rc == 0 and "days" not in m2)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
