"""False alarms per 24 hours, on whole recordings no fold model has seen.

    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --thresholds 0.55 0.6 --rules 2/3 4/6 6/9 --datasets nsrdb stdb
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --rules 2/3 4/6 6/9 --datasets cudb vfdb        # the cost: delay, misses
    python -m munji_ai.eval.false_alarm_rate --models checkpoints/shock_cv \
        --template --ncc 0.7 0.8 0.9 --min-beats 2 3   # patient-template veto

Why this number
---------------
Window specificity says how often one 5-second window is wrong. A patient does
not experience windows. They experience their phone going off, after the 2 of 3
confirmation rule, a number of times per day. In wearable defibrillator
studies that alarm burden is what decides whether patients keep wearing the
device, so it is the figure the client and a clinician can actually judge.

The cross-validation cache cannot give it: each patient contributes at most a
capped, shuffled sample of windows, not their day. So this script goes back to
the raw recordings and runs the model over every window, end to end, at the
deployment stride.

No leakage
----------
Each recording is scored by the fold model that held its patient out, using
the fold assignment saved in the out-of-fold file. Only patients in the train
and val splits are used; the test split stays untouched.

Longer confirmation
-------------------
--rules scores several N-of-M confirmation rules from one pass over the data.
On arrest recordings (cudb, vfdb) the same table shows what a longer rule
costs: episodes caught and the median delay from onset to alarm. Artifact is
usually brief and fibrillation is not, so a longer rule should remove false
alarms faster than it delays true ones; this measures whether it does.

Beat-template veto
------------------
--template checks every VF candidate against the patient's own normal beat,
the way wearable defibrillators do (rules/beat_template.py). The template is
built from the first clear minute of each recording, standing in for the
baseline recorded at setup. A candidate is vetoed when enough beats matching
it are still found at a plausible rate: the heart is beating under the
artifact. --ncc and --min-beats sweep the match threshold and beat count, and
the veto is scored on arrest recordings too, so its cost on real VF shows.

What it does not do
-------------------
The quality gate is not applied, so every window reaches the model. In the
field the gate removes the worst artifact first, so these figures are an upper
bound on the alarm rate, not an estimate of it.

Reads only. Writes nothing and trains nothing.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR, TARGET_FS
from ..data.loader import load_record
from ..data.windows import (SHOCKABLE, dominant_label, load_cache,
                            rhythm_intervals, split_vt_by_rate)
from ..models.shockable_net import build

DEFAULT_DATASETS = ("nsrdb", "stdb", "edb", "fantasia")

# An episode counts as sustained when annotated shockable for at least this
# long: the conventional 30 s line between sustained and non-sustained VT.
# A non-sustained run ends on its own; a sustained one is the emergency.
SUSTAINED_S = 30.0
WINDOW_S = C.WINDOW_SEC["shockable"]
STRIDE_S = C.WINDOW_STRIDE_SEC["shockable"]


def fold_of_records(oof_path: Path, cache_dir: Path) -> dict:
    """(dataset, record) -> fold, from the saved out-of-fold assignment."""
    with np.load(oof_path, allow_pickle=False) as d:
        records = d["record"].astype(str)
        folds = d["fold"].astype(int)
    datasets = np.concatenate([load_cache("shockable", s, cache_dir)["dataset"].astype(str)
                               for s in ("train", "val")])
    if len(datasets) != len(records):
        raise SystemExit("the cache no longer matches the out-of-fold file; "
                         "re-run cross_validate")
    out = {}
    for ds, rec, f in zip(datasets, records, folds):
        out.setdefault((ds, rec), int(f))
    return out


def load_fold_models(models_dir: Path, device) -> dict:
    models = {}
    for p in sorted(models_dir.glob("shockable_fold*.pt")):
        f = int(p.stem.replace("shockable_fold", ""))
        m = build()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models[f] = m.to(device).eval()
    if not models:
        raise SystemExit(f"no shockable_fold*.pt in {models_dir}; run cross_validate "
                         f"with --save-models")
    return models


def _normalize_rows(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """preprocess.normalize, applied to every row at once."""
    q25, med, q75 = np.percentile(W, [25, 50, 75], axis=1)
    scale = (q75 - q25) / 1.349
    return (W - med[:, None]) / (scale[:, None] + eps)


@torch.no_grad()
def window_scores(model, signal: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    if len(signal) < win:
        return np.zeros(0, dtype=np.float32)
    starts = np.arange(0, len(signal) - win + 1, stride)
    out = np.empty(len(starts), dtype=np.float32)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        W = signal[s[:, None] + np.arange(win)[None, :]]
        x = torch.from_numpy(_normalize_rows(W).astype(np.float32)).to(device)
        out[i:i + len(s)] = model.predict_proba(x).cpu().numpy()
    return out


def truth_flags(rec, n: int) -> np.ndarray:
    """True where the annotated rhythm makes the window genuinely shockable."""
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    flags = np.zeros(n, dtype=bool)
    if not iv:
        return flags
    for i in range(n):
        lo = i * stride
        lab = dominant_label(iv, lo, lo + win, C.SHOCKABLE_PURITY,
                             excluded=C.SHOCKABLE_EXCLUDED)
        flags[i] = lab in SHOCKABLE
    return flags


def count_alarms(pred: np.ndarray, truth: np.ndarray, rule) -> dict:
    """Alarms under N-of-M; a run of confirmed windows is one alarm.

    false   alarms whose confirming windows contain no annotated shockable one
    true    alarms that do
    episodes / caught / delays
            contiguous runs of shockable windows, whether an alarm fired while
            each was running, and how many windows after onset it fired
    """
    n, m = rule
    csum = np.concatenate([[0], np.cumsum(pred)])
    idx = np.arange(len(pred))
    fired = (csum[idx + 1] - csum[np.maximum(0, idx - m + 1)]) >= n
    out = {"false": 0, "true": 0, "episodes": 0, "caught": 0, "delays": [],
           "sus_episodes": 0, "sus_caught": 0}

    for i in np.flatnonzero(fired):
        if i == 0 or not fired[i - 1]:
            lo = max(0, i - m + 1)
            out["true" if truth[lo:i + 1].any() else "false"] += 1

    t = np.concatenate([[False], truth, [False]])
    starts = np.flatnonzero(~t[:-1] & t[1:])
    ends = np.flatnonzero(t[:-1] & ~t[1:])          # exclusive
    sustained_windows = int(np.ceil((SUSTAINED_S - WINDOW_S) / STRIDE_S)) + 1
    for a, b in zip(starts, ends):
        out["episodes"] += 1
        hit = np.flatnonzero(fired[a:b])
        sustained = (b - a) >= sustained_windows
        out["sus_episodes"] += int(sustained)
        if hit.size:
            out["caught"] += 1
            out["sus_caught"] += int(sustained)
            out["delays"].append(int(hit[0]))
    return out


def template_vetoes(signal, scores, truth, thr_min, cfgs, bt) -> dict:
    """Per-window veto flags for each (match, beats) setting, for candidates only.

    The template comes from the first clear minute: no window there is
    annotated shockable. Returns {"missing": True} plus all-False flags when no
    usable baseline exists, so that record is scored without a veto.
    """
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")

    def is_clear(lo, hi):
        i0 = max(0, (lo - win) // stride + 1)
        i1 = min(len(truth), hi // stride + 1)
        return not truth[i0:i1].any()

    out = {cfg: np.zeros(len(scores), dtype=bool) for cfg in cfgs}
    found = bt.build_template(signal, TARGET_FS, is_clear)
    if found is None:
        out["missing"] = True
        return out
    tmpl = found[0]
    for i in np.flatnonzero(scores >= thr_min):
        x = signal[i * stride:i * stride + win]
        c = bt.ncc_trace(x, tmpl)
        for cfg in cfgs:
            out[cfg][i] = bt.beats_present(x, tmpl, cfg[0], cfg[1], TARGET_FS, ncc=c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True,
                    help="folder with shockable_fold*.pt and shockable_oof.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55, 0.6])
    ap.add_argument("--rules", nargs="+", default=None, metavar="N/M",
                    help="confirmation rules to compare, e.g. 2/3 4/6 6/9 "
                         "(default: config.CONFIRMATION['vf'])")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-records", type=int, default=None,
                    help="per dataset, for a quick trial run")
    ap.add_argument("--template", action="store_true",
                    help="also score the patient beat-template veto")
    ap.add_argument("--ncc", type=float, nargs="+", default=[0.7, 0.8, 0.9],
                    help="template match thresholds to compare")
    ap.add_argument("--min-beats", type=int, nargs="+", default=[3],
                    help="matched beats needed in a window to veto")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    if a.rules:
        rules = [tuple(int(v) for v in r.split("/")) for r in a.rules]
    else:
        rules = [tuple(C.CONFIRMATION["vf"])]
    for n_, m_ in rules:
        if not 1 <= n_ <= m_:
            raise SystemExit(f"bad rule {n_}/{m_}: need 1 <= N <= M")
    step = C.WINDOW_STRIDE_SEC["shockable"]
    folds = fold_of_records(a.models / "shockable_oof.npz", a.cache)
    models = load_fold_models(a.models, device)
    print(f"models      {len(models)} folds from {a.models}")
    print(f"rules       " + "  ".join(f"{n_}/{m_} (~{(m_ - 1) * step + 5:g} s)"
                                      for n_, m_ in rules))
    configs = [None]
    if a.template:
        from ..rules import beat_template as bt
        configs += [(c, b) for c in a.ncc for b in a.min_beats]
        print(f"template    veto at match " + ", ".join(f"{c:g}" for c in a.ncc)
              + f" with {', '.join(map(str, a.min_beats))} beats")
    print(f"gate        not applied (upper bound on alarms)\n")
    no_template = defaultdict(int)

    hours = defaultdict(float)
    n_rec = defaultdict(int)
    acc = defaultdict(lambda: {"false": 0, "true": 0, "episodes": 0,
                               "caught": 0, "delays": [],
                               "sus_episodes": 0, "sus_caught": 0})
    per_record = defaultdict(lambda: defaultdict(int))
    t0 = time.time()

    for ds in a.datasets:
        keys = sorted(k for k in folds if k[0] == ds)[: a.max_records]
        if not keys:
            print(f"  {ds:<9} no records in the out-of-fold file, skipped")
            continue
        for j, (_, name) in enumerate(keys, 1):
            f = folds[(ds, name)]
            if f not in models:
                continue
            rec = load_record(ds, name, a.raw)
            s = window_scores(models[f], rec.signal, device)
            truth = truth_flags(rec, len(s))
            hours[ds] += len(rec.signal) / TARGET_FS / 3600
            n_rec[ds] += 1

            veto = {None: np.zeros(len(s), dtype=bool)}
            if a.template:
                veto.update(template_vetoes(rec.signal, s, truth, min(a.thresholds),
                                            configs[1:], bt))
                if veto.get("missing"):
                    no_template[ds] += 1
                veto.pop("missing", None)

            for t in a.thresholds:
                for cfg in configs:
                    pred = (s >= t) & ~veto[cfg]
                    for r in rules:
                        c = count_alarms(pred, truth, r)
                        k = acc[(ds, t, r, cfg)]
                        for key in ("false", "true", "episodes", "caught",
                                    "sus_episodes", "sus_caught"):
                            k[key] += c[key]
                        k["delays"] += c["delays"]
                        if cfg is None:
                            per_record[(t, r)][f"{ds}:{name}"] += c["false"]
            print(f"  {ds:<9} {j:>3}/{len(keys)}  {name:<10} "
                  f"{len(rec.signal) / TARGET_FS / 3600:5.1f} h   "
                  f"{time.time() - t0:6.0f} s elapsed", flush=True)

    used = [ds for ds in a.datasets if n_rec[ds]]
    if not used:
        print("nothing scored")
        return 1

    if a.template:
        print(f"\n  template found in "
              + ", ".join(f"{d} {n_rec[d] - no_template[d]}/{n_rec[d]}" for d in used)
              + " records (no template: veto off for that record)")
    print(f"\n  false alarms per 24 h ({', '.join(f'{d} {hours[d]:.0f} h' for d in used)})")
    head = f"  {'thr':>5}{'rule':>6}"
    if a.template:
        head += f"{'veto':>10}"
    head += "".join(f"{d:>10}" for d in used) + f"{'all':>10}"
    has_ep = any(acc[(d, t, r, cfg)]["episodes"] for d in used for t in a.thresholds
                 for r in rules for cfg in configs)
    if has_ep:
        head += f"{'caught':>10}{'sustained':>11}{'median delay':>14}"
    print(head)
    for t in a.thresholds:
        for r in rules:
            for cfg in configs:
                row = f"  {t:>5.2f}{f'{r[0]}/{r[1]}':>6}"
                if a.template:
                    row += f"{'off' if cfg is None else f'{cfg[0]:g}/{cfg[1]}':>10}"
                tot = 0
                for d in used:
                    fa = acc[(d, t, r, cfg)]["false"]
                    tot += fa
                    row += f"{fa / hours[d] * 24:>10.1f}"
                row += f"{tot / sum(hours[d] for d in used) * 24:>10.1f}"
                if has_ep:
                    ep = sum(acc[(d, t, r, cfg)]["episodes"] for d in used)
                    ca = sum(acc[(d, t, r, cfg)]["caught"] for d in used)
                    dl = [x for d in used for x in acc[(d, t, r, cfg)]["delays"]]
                    med = f"{np.median(dl) * step + 5:.0f} s" if dl else "-"
                    se = sum(acc[(d, t, r, cfg)]["sus_episodes"] for d in used)
                    sc = sum(acc[(d, t, r, cfg)]["sus_caught"] for d in used)
                    row += f"{f'{ca}/{ep}':>10}{f'{sc}/{se}':>11}{med:>14}"
                print(row)

    t, r = a.thresholds[0], rules[0]
    worst = sorted(per_record[(t, r)].items(), key=lambda kv: -kv[1])[:5]
    if worst and worst[0][1]:
        total = sum(per_record[(t, r)].values())
        print(f"\n  records with the most false alarms (threshold {t:.2f}, rule {r[0]}/{r[1]})")
        for name, fa in worst:
            print(f"    {name:<16}{fa:>6}   {fa / max(total, 1):>5.0%} of all")

    print("\n" + "=" * 66)
    print("  Per 24 h is the rate a patient would feel with the gate off.")
    print("  Alarms overlapping annotated VF or fast VT count as true, not false.")
    if has_ep:
        print("  Delay is from episode onset to alarm, including the 5 s window.")
        print(f"  Sustained: annotated shockable for {SUSTAINED_S:g} s or more.")
    if a.template:
        print("  veto m/b: vetoed when b beats match the patient template at m or above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
