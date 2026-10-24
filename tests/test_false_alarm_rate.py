"""Checks the false-alarm-rate evaluation added by fix_false_alarm_rate.py.

Builds a tiny fake world in a temporary folder: two short WFDB recordings,
a matching cache and out-of-fold file, and untrained fold models. Nothing
real is read and nothing in the repo is written.

    python tests/test_false_alarm_rate.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import preprocess as pp
from munji_ai.eval import false_alarm_rate as far
from munji_ai.models.shockable_net import build

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ecg(seconds, fs, hr=70.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * fs)
        w = max(int(0.01 * fs), 1)
        lo, hi = max(c - 4 * w, 0), min(c + 4 * w, n)
        x[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - c) / w) ** 2)
        t += 60.0 / hr
    return x + rng.normal(0, 0.01, n)


def main() -> int:
    rule = C.CONFIRMATION["vf"]

    print("Alarm counting (2 of 3)\n")
    P = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
    T = np.zeros_like(P)
    c = far.count_alarms(P, T, rule)
    check("one confirmed run is one alarm, scattered positives fire again",
          (c["false"], c["true"]) == (2, 0), f"false {c['false']}, true {c['true']}")
    T2 = T.copy()
    T2[1:4] = True
    c2 = far.count_alarms(P, T2, rule)
    check("an alarm overlapping a real episode counts as true",
          (c2["false"], c2["true"]) == (1, 1), f"false {c2['false']}, true {c2['true']}")
    check("the episode is counted and caught",
          (c2["episodes"], c2["caught"]) == (1, 1))
    check("delay counts windows from onset to the first alarm", c2["delays"] == [1],
          f"{c2['delays']}")
    c3 = far.count_alarms(P, T2, (3, 3))
    check("a stricter rule still catches a 3-window episode", c3["caught"] == 1
          and c3["delays"] == [2])
    c4 = far.count_alarms(P, T2, (4, 4))
    check("a rule longer than the episode misses it", c4["caught"] == 0)
    long_T = np.zeros(40, dtype=bool)
    long_T[5:20] = True                      # 15 windows: 40 s at 2.5 s stride
    long_T[30:33] = True                     # 3 windows: 10 s
    c5 = far.count_alarms(long_T.copy(), long_T, rule)
    check("episodes of 30 s or more count as sustained, short runs do not",
          (c5["episodes"], c5["sus_episodes"], c5["sus_caught"]) == (2, 1, 1),
          f"{c5['episodes']} episodes, {c5['sus_episodes']} sustained")
    z = far.count_alarms(np.zeros(20, bool), np.zeros(20, bool), rule)
    check("no positives, no alarms", (z["false"], z["true"]) == (0, 0))

    print("\nWindowing\n")
    sig = ecg(60, C.TARGET_FS)
    W = np.stack([sig[i:i + 1250] for i in (0, 625, 1250)])
    ref = np.stack([pp.normalize(w) for w in W])
    check("row normalisation matches preprocess.normalize",
          np.allclose(far._normalize_rows(W), ref, atol=1e-9))
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    s = far.window_scores(build().eval(), sig, torch.device("cpu"))
    expected = 1 + (len(sig) - win) // stride
    check("one score per deployment-stride window", len(s) == expected,
          f"{len(s)} windows for 60 s")
    check("scores are probabilities", ((s >= 0) & (s <= 1)).all())

    print("\nBeat template veto\n")
    from scipy.signal import butter, filtfilt
    from munji_ai.rules import beat_template as bt

    def beats(seconds, hr, wide=False, seed=0):
        rng = np.random.default_rng(seed)
        fs = C.TARGET_FS
        n = int(seconds * fs)
        x = np.zeros(n)
        t = 0.5
        comps = ((0.12, -0.18, 0.022), (1.0, 0.0, 0.03 if wide else 0.010),
                 (-0.25, 0.08 if wide else 0.04, 0.03 if wide else 0.012),
                 (0.3, 0.25, 0.045))
        while t < seconds - 0.5:
            c = int(t * fs)
            for amp, off, wid in comps:
                k, w = int(off * fs), max(int(wid * fs), 1)
                lo, hi = max(c + k - 4 * w, 0), min(c + k + 4 * w, n)
                if hi > lo:
                    x[lo:hi] += amp * np.exp(-0.5 * ((np.arange(lo, hi) - (c + k)) / w) ** 2)
            t += 60 / hr * rng.uniform(0.98, 1.02)
        return pp.model_path(x + rng.normal(0, 0.01, n), fs)

    fs = C.TARGET_FS
    nsr = beats(180, 72)
    found = bt.build_template(nsr, fs)
    check("a template is built from a clear minute", found is not None
          and found[2] >= bt.MIN_QUALITY, f"quality {found[2]:.3f}" if found else "none")
    tmpl = found[0]
    check("no template when every stretch is excluded",
          bt.build_template(nsr, fs, is_clear=lambda lo, hi: False) is None)
    rng = np.random.default_rng(1)
    b, a = butter(2, [1 / 125, 15 / 125], "band")
    w = nsr[2500:2500 + 1250]
    hits = 0
    for _ in range(20):
        nz = filtfilt(b, a, rng.normal(0, 1, 1250))
        nz *= np.sqrt(np.mean(w ** 2) / np.mean(nz ** 2)) / 10 ** (3 / 20)
        hits += bt.beats_present(w + nz, tmpl, 0.8, 3, fs)
    check("beats are still found under 3 dB artifact", hits >= 18, f"{hits}/20")
    t = np.arange(1250) / fs
    wrong = 0
    for _ in range(50):
        vf = np.sin(2 * np.pi * rng.uniform(4, 7) * t + rng.uniform(0, 6)) \
            * (1 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
        wrong += bt.beats_present(pp.model_path(vf + rng.normal(0, 0.05, 1250), fs),
                                  tmpl, 0.7, 3, fs)
    check("fibrillation-like waves are never mistaken for beats", wrong == 0,
          f"{wrong}/50 vetoed")
    vt = beats(10, 180, wide=True)[:1250]
    check("wide VT does not match a narrow sinus template",
          not bt.beats_present(vt, tmpl, 0.7, 3, fs),
          f"best match {bt.ncc_trace(vt, tmpl).max():.2f}")
    check("too few beats never veto", not bt.beats_present(w, tmpl, 0.8, 50, fs))

    print("\nEnd to end on a fake world\n")
    import wfdb
    from munji_ai.data.windows import config_hash

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, models = tmp / "raw", tmp / "cache", tmp / "models"
        (raw / "nsrdb").mkdir(parents=True)
        cache.mkdir()
        models.mkdir()
        for i, name in enumerate(("16265", "16272")):
            x = ecg(600, 128, seed=i)
            wfdb.wrsamp(name, fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                        p_signal=np.stack([x, x], axis=1), fmt=["16", "16"],
                        write_dir=str(raw / "nsrdb"))

        recs = np.array(["16265"] * 6 + ["16272"] * 6)
        for split, sl in (("train", slice(0, 8)), ("val", slice(8, 12))):
            n = sl.stop - sl.start
            np.savez(cache / f"shockable_{split}.npz",
                     X=np.zeros((n, 1250), np.float32), y=np.array(["not_shockable"] * n),
                     dataset=np.array(["nsrdb"] * n), patient=recs[sl], record=recs[sl],
                     offset=np.arange(n) * 625, cfg_hash=config_hash())
        np.savez(models / "shockable_oof.npz", y=np.array(["not_shockable"] * 12),
                 pred=np.array(["not_shockable"] * 12), score=np.zeros(12),
                 patient=recs, record=recs, offset=np.arange(12) * 625,
                 fold=np.array([0] * 6 + [1] * 6))
        torch.manual_seed(0)
        for f in (0, 1):
            torch.save(build().state_dict(), models / f"shockable_fold{f}.pt")

        folds = far.fold_of_records(models / "shockable_oof.npz", cache)
        check("each record maps to the fold that held its patient out",
              folds == {("nsrdb", "16265"): 0, ("nsrdb", "16272"): 1})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = far.main(["--models", str(models), "--cache", str(cache),
                           "--raw", str(raw), "--datasets", "nsrdb",
                           "--thresholds", "0.5", "0.6", "--rules", "2/3", "4/6"])
        text = buf.getvalue()
        check("runs end to end", rc == 0 and "false alarms per 24 h" in text)
        check("both recordings scored", text.count("  nsrdb  ") == 2
              or text.count("nsrdb       ") >= 2)
        rows = [l for l in text.splitlines()
                if l.strip().startswith(("0.50", "0.60"))]
        check("one row per threshold and rule", len(rows) == 4, f"{len(rows)} rows")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = far.main(["--models", str(models), "--cache", str(cache),
                           "--raw", str(raw), "--datasets", "nsrdb",
                           "--thresholds", "0.5", "--rules", "2/3",
                           "--template", "--ncc", "0.7", "0.8", "--min-beats", "3"])
        text = buf.getvalue()
        rows = [l for l in text.splitlines() if l.strip().startswith("0.50")]
        check("template run adds a row per setting", rc == 0 and len(rows) == 3,
              f"{len(rows)} rows")
        check("template found in both recordings", "nsrdb 2/2 records" in text)
        refused = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                far.main(["--models", str(models), "--cache", str(cache),
                          "--raw", str(raw), "--rules", "5/3"])
            except SystemExit:
                refused = True
        check("an impossible rule is refused", refused)

    print("\ncross_validate --save-models\n")
    from munji_ai.train import cross_validate as cv
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp)
        for split, n in (("train", 48), ("val", 16)):
            np.savez(cache / f"shockable_{split}.npz",
                     X=rng.standard_normal((n, 1250)).astype(np.float32),
                     y=np.array(["shockable", "not_shockable"] * (n // 2)),
                     patient=np.array([f"p{split}{i % 8}" for i in range(n)]),
                     record=np.array([f"r{split}{i % 8}" for i in range(n)]),
                     offset=np.arange(n) * 625, dataset=np.array(["x"] * n),
                     cfg_hash=config_hash())
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cv.main(["--stage", "shockable", "--folds", "2", "--epochs", "1",
                          "--batch", "16", "--cache", str(cache), "--device", "cpu",
                          "--save-models", "--out", str(cache / "out")])
        saved = sorted(p.name for p in (cache / "out").glob("shockable_fold*.pt"))
        check("one model file per fold", rc == 0 and saved ==
              ["shockable_fold0.pt", "shockable_fold1.pt"], ", ".join(saved))
        m = build()
        m.load_state_dict(torch.load(cache / "out" / "shockable_fold0.pt",
                                     map_location="cpu", weights_only=True))
        check("a saved fold model loads back into ShockableNet", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
