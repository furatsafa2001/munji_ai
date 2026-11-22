"""eval_svt without PyTorch: its helpers, and one synthetic record end to end.

The models are stand-ins that answer from the record's own annotation, so
the checks are about the evaluation's plumbing (what runs where, in what
order, how a named window is judged), not about any model.

    python tests/test_eval_svt.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.eval import eval_svt as E

ok = True
FS = C.TARGET_FS


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def spikes(parts):
    """[(seconds, bpm, label)] -> (signal in mV, beat times, rhythm intervals in s)."""
    total = sum(p[0] for p in parts)
    t = np.arange(int(total * FS)) / FS
    x = np.random.default_rng(0).normal(0, 0.02, t.size)
    beats, iv, t0 = [], [], 0.0
    for sec, bpm, lab in parts:
        width = 0.012 if lab != "VT" else 0.04
        for b in np.arange(t0 + 0.2, t0 + sec, 60.0 / bpm):
            x += 1.2 * np.exp(-((t - b) / width) ** 2)
            beats.append((b, lab))
        iv.append((t0, t0 + sec, lab))
        t0 += sec
    return x, beats, iv


class FakeModels:
    """Answers from the annotation: ShockableNet fires in VT, the beat model
    calls VT beats PVC (or N when told to be wrong), RhythmNet never sees AF."""

    def __init__(self, beats, iv, shock_in_vt=True, vt_beats_as="PVC"):
        self.beats, self.iv = beats, iv
        self.shock_in_vt, self.vt_beats_as = shock_in_vt, vt_beats_as
        self.labeller = SimpleNamespace(run=self.run)
        self.calls = {"shock": 0, "rhythm": 0, "beat": 0}

    def _lab(self, t):
        return next((lab for a, b, lab in self.iv if a <= t < b), "NSR")

    def shock(self, ds, name, sig, starts):
        self.calls["shock"] += len(starts)
        win = C.window_samples("shockable")
        return np.array([0.9 if self.shock_in_vt and self._lab((s + win / 2) / FS) == "VT" else 0.05
                         for s in starts])

    def rhythm_probs(self, sig, starts):
        self.calls["rhythm"] += len(starts)
        return np.tile([0.85, 0.05, 0.10], (len(starts), 1))

    def run(self, rec, starts):
        self.calls["beat"] += len(starts)
        win = C.window_samples("beat")
        masks = np.zeros((len(starts), win), np.uint8)
        for i, s in enumerate(starts):
            for b, lab in self.beats:
                k = int(b * FS) - s
                if 10 <= k < win - 10:
                    cls = "N" if lab != "VT" else self.vt_beats_as
                    masks[i, k - 10:k + 10] = C.BEAT_CLASSES.index(cls)
        return masks, np.ones(len(starts), bool)

    def gate_blocks(self, sig):
        return None


def main() -> int:
    print("Helpers\n")
    check("spans merge when they touch or come within the gap",
          E.merge([(0, 10), (12, 20), (40, 50)], gap=2) == [(0, 20), (40, 50)])
    check("regions reach back from each candidate and stop at the record",
          E.regions([30.0, 300.0], 70.0, 310.0) == [(0.0, 30.0), (230.0, 300.0)])
    g = E.grid((3.0, 20.0), win=int(5 * FS), stride=int(2.5 * FS), n=10 ** 6)
    check("windows stay on the record's own grid, inside the span",
          g and g[0] == int(5.0 * FS) and g[-1] + int(5 * FS) <= int(20 * FS), str(g[:3]))
    iv = [(0, 100, "NSR"), (100, 130, "VT"), (130, 400, "SVT")]
    check("a window with any VT in it is judged VT/VF, before SVT",
          E.window_kind(160.0, iv) == "VT/VF")
    check("a window over SVT only is SVT", E.window_kind(200.0, iv) == "SVT")
    check("otherwise the label covering most of it", E.window_kind(60.0, iv) == "NSR")
    check("an episode is named when a named window overlaps it",
          E.episode_named((130, 400), [150.0]) and not E.episode_named((130, 400), [100.0]))
    fast = np.arange(0, 20, 60 / 190)
    check("190 bpm held over a tachy run of annotated beats can be named",
          E.can_be_named((0, 20), fast))
    check("120 bpm cannot, nor 10 beats at 190",
          not E.can_be_named((0, 20), np.arange(0, 20, 0.5))
          and not E.can_be_named((0, 3), np.arange(0, 3, 60 / 190)))
    check("named windows 2.5 s apart are one episode, 30 s apart two",
          E.episodes_of([10, 12.5, 15]) == 1 and E.episodes_of([10, 40]) == 2)

    t = np.arange(40) * 0.4                                  # 150 bpm
    c = np.array(["N"] * 10 + ["PAC"] * 5 + ["N"] * 10 + ["PVC"] * 3 + ["N"] * 12)
    check("runs: 5 PACs and 3 PVCs in a row are found, with their beat counts",
          [r[2] for r in E.class_runs(t, c, "PAC", 3)] == [5]
          and [r[2] for r in E.class_runs(t, c, "PVC", 3)] == [3])
    check("a run at 100 bpm or slower does not count",
          E.class_runs(np.arange(5) * 0.7, np.array(["PAC"] * 5), "PAC", 3) == [])
    check("by beat labels: 3 PVCs in a row in the window is VT, before any SVT run",
          E.beat_kind(16.0, t, c) == "VT (beats)")
    check("... a PAC run without one is SVT, N only is N only, no beats is ''",
          E.beat_kind(9.0, t, c) == "SVT (beats)" and E.beat_kind(3.0, t, c) == "N only (beats)"
          and E.beat_kind(500.0, t, c) == "")

    print("\nOne record end to end (sinus, SVT 190, sinus, VT 190, sinus)\n")
    x, beats, iv = spikes([(60, 75, "NSR"), (90, 190, "SVT"), (60, 75, "NSR"),
                           (60, 190, "VT"), (40, 75, "NSR")])
    rec = SimpleNamespace(signal=x, fs=FS)
    m = FakeModels(beats, iv)
    r = E.run_record("fake", "r1", rec, m)
    kinds = [E.window_kind(t, iv) for t in r["named"]]
    check("the rate engine finds 150 bpm or more in SVT and VT windows",
          r["candidates"] > 0, f"{r['candidates']} of {r['windows']}")
    check("SVT is named", "SVT" in kinds, f"{len(r['named'])} windows")
    check("only over SVT, never over VT", set(kinds) == {"SVT"}, str(sorted(set(kinds))))
    check("the models ran only around candidates",
          0 < m.calls["beat"] < (len(x) / FS) / C.WINDOW_STRIDE_SEC["beat"] + 1, str(m.calls))
    ep = (60.0, 150.0)
    check("the SVT episode counts as named", E.episode_named(ep, r["named"]))

    m = FakeModels(beats, iv, shock_in_vt=False, vt_beats_as="N")
    r = E.run_record("fake", "r1", rec, m)
    kinds = [E.window_kind(t, iv) for t in r["named"]]
    check("a beat model and ShockableNet that both miss the VT: the VT is named "
          "SVT, and the safety count sees it", "VT/VF" in kinds, str(sorted(set(kinds))))

    m = FakeModels(beats, iv, vt_beats_as="N")
    r = E.run_record("fake", "r1", rec, m)
    kinds = [E.window_kind(t, iv) for t in r["named"]]
    check("beat model wrong, ShockableNet right: the hold keeps VT from being named SVT",
          "VT/VF" not in kinds, str(sorted(set(kinds))))

    m = FakeModels(beats, iv)
    r = E.run_record("fake", "r1", rec, m, keep=True)
    kinds_ev = {e[2] for e in r.get("events", [])}
    check("keep=True hands back every event fed, in time order",
          kinds_ev == {"shock", "rhythm", "beats", "rate"}
          and [e[0] for e in r["events"]] == sorted(e[0] for e in r["events"]))

    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location(
        "svt_case_plot", Path(__file__).resolve().parents[1] / "experiments" / "svt_case_plot.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "case.png"
        res = plot.plot_case(rec, "fake/r1", 40.0, 160.0, FakeModels(beats, iv), iv, None, png)
        check("the case plot is drawn, with one summary line per named window",
              png.exists() and png.stat().st_size > 10_000
              and len(res["rows"]) == len(res["named"]) > 0, res["rows"][0] if res["rows"] else "")

    rec2 = SimpleNamespace(signal=spikes([(120, 75, "NSR")])[0], fs=FS)
    m = FakeModels([], [])
    r = E.run_record("fake", "r2", rec2, m)
    check("a record that never reaches 150 runs no model at all",
          r["candidates"] == 0 and sum(m.calls.values()) == 0 and r["named"] == [], str(m.calls))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
