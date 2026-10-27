"""Can MUNJI tell real VT from a false VT alarm? Tested inside VTaC.

    python -m munji_ai.eval.vt_detect_feasibility --vtac data/cache/vtac_windows.npz

Why this question
-----------------
VF against VT could not be learnt across sources: trained with VTaC's ICU VT,
a classifier learnt "ICU monitor" rather than "VT". This question avoids that
trap. True and false alarms in VTaC come from the same monitors, the same
hospitals and, where possible, the same patients (munji_ai.data.vtac --false
takes false alarms from the true-alarm patients first), so the only systematic
difference between the two sides is the rhythm.

It is also the question the client's outputs need answered. ShockableNet
covers VT at 150 bpm and above; VT from 100 to 149 bpm and wide-complex
tachycardia are not covered, and the rate engine found only 14% of real VT.
VTaC's definition of VT, five or more ventricular beats above 100 bpm, is the
client's.

What it reports
---------------
Every event gives three 5 s windows around its alarm. Candidates are scored
by patient on VTaC's train and validation splits (out of fold), then once on
its held-out test split:

    shockable_net   today's ShockableNet score, untouched: what we have now
    features        published measures (complexity, entropy, spectrum, ...)
    embeddings      ShockableNet's 384 pooled features
    both            the two together

Per event, the three windows' probabilities are averaged, as an alarm would.
Events are also split by the rate measured from the signal: VT under 150 bpm
(the gap) and 150 and above.

Verdict: an event-level AUC of 0.85 or more on the test split makes a VT
detector worth building. A pass is evidence on ICU leads; the patch's CM5
lead is close to VTaC's V lead, but transfer to the patch is not yet shown.

Reads only. Writes nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..data.preprocess import normalize
from ..data.windows import _episode_bpm
from ..models.shockable_net import SHOCKABLE_IDX, build
from .vf_vt_feasibility import FS, auc, embeddings, features, make_clf

STARTS_S = (15.0, 17.5, 20.0)


def event_windows(path: Path):
    d = np.load(path)
    win = C.window_samples("shockable")
    raw, ev, y, grp, split, bpm = [], [], [], [], [], []
    for k, (x, rec, dec, sp) in enumerate(zip(d["X"], d["record"], d["decision"], d["split"])):
        seg = np.asarray(x[int(15 * FS):int(25 * FS)], dtype=np.float64)
        b = _episode_bpm(seg, FS)
        for s0 in STARTS_S:
            raw.append(np.asarray(x[int(s0 * FS):int(s0 * FS) + win], dtype=np.float64))
            ev.append(k)
            y.append(int(bool(dec)))
            grp.append(str(rec))
            split.append(str(sp))
            bpm.append(float(b) if b else np.nan)
    return (raw, np.array(ev), np.array(y), np.array(grp), np.array(split), np.array(bpm))


@torch.no_grad()
def shockable_scores(model, Xn: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(Xn), 256):
        logits = model(torch.from_numpy(Xn[i:i + 256].astype(np.float32)))
        out.append(torch.softmax(logits, -1)[:, SHOCKABLE_IDX].numpy())
    return np.concatenate(out)


def per_event(ev: np.ndarray, values: np.ndarray) -> tuple:
    ids = np.unique(ev)
    return ids, np.array([np.nanmean(values[ev == i]) for i in ids])


def oof(F, y, groups, kind):
    from sklearn.model_selection import GroupKFold

    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(5, len(np.unique(groups)))).split(F, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf(kind)
        clf.fit(F[tr], y[tr])
        p[te] = clf.predict_proba(F[te])[:, 1]
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtac", type=Path, default=Path("data/cache/vtac_windows.npz"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/shockable_final/shockable_net.pt"))
    a = ap.parse_args(argv)

    raw, ev, y, grp, split, bpm = event_windows(a.vtac)
    dev = np.isin(split, ["train", "val"])
    test = split == "test"
    n_ev = lambda m, v: len(np.unique(ev[m & (y == v)]))
    print(f"events      true VT {n_ev(dev, 1):,} + {n_ev(test, 1):,} test, "
          f"false alarms {n_ev(dev, 0):,} + {n_ev(test, 0):,} test")
    print(f"patients    {len(np.unique(grp[dev])):,} development, {len(np.unique(grp[test])):,} test")
    if n_ev(dev, 0) < 10 or n_ev(dev, 1) < 10:
        print("too few false or true alarms; read more with munji_ai.data.vtac --false N")
        return 1

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print("computing features ...", flush=True)
    Xn = np.stack([normalize(x) for x in raw])
    Ff = np.stack([features(x) for x in raw])
    Fe = embeddings(model, Xn)
    ps = shockable_scores(model, Xn)
    sets = {"features": Ff, "embeddings": Fe, "both": np.hstack([Ff, Fe])}
    plan = [("features", "forest"), ("features", "logistic"),
            ("embeddings", "logistic"), ("both", "logistic")]

    def ev_auc(mask, p):
        ids, pe = per_event(ev[mask], p[mask])
        _, ye = per_event(ev[mask], y[mask].astype(float))
        return auc(ye.astype(int), pe)

    def rate_aucs(mask, p):
        ids, pe = per_event(ev[mask], p[mask])
        _, ye = per_event(ev[mask], y[mask].astype(float))
        _, be = per_event(ev[mask], bpm[mask])
        ye = ye.astype(int)
        neg = ye == 0
        out = []
        for lo, hi in ((100, 150), (150, 400)):
            pos = (ye == 1) & (be >= lo) & (be < hi)
            sel = neg | pos
            out.append((auc(ye[sel], pe[sel]) if pos.any() else float("nan"), int(pos.sum())))
        return out

    rows = [("shockable_net (as is)", ps, ps)]
    for fs_name, kind in plan:
        p_dev = oof(sets[fs_name][dev], y[dev], grp[dev], kind)
        p_all = np.full(len(y), np.nan)
        p_all[dev] = p_dev
        if test.any():
            clf = make_clf(kind)
            clf.fit(sets[fs_name][dev], y[dev])
            p_all[test] = clf.predict_proba(sets[fs_name][test])[:, 1]
        rows.append((f"{fs_name}, {kind}", p_all, p_all))

    print(f"\nevent-level AUC, true VT against false VT alarms")
    print(f"  {'candidate':<26}{'dev (by patient)':>18}{'test split':>12}"
          f"{'VT <150 test':>14}{'VT 150+ test':>14}")
    best = None
    for name, p, _ in rows:
        a_dev = ev_auc(dev, p)
        a_test = ev_auc(test, p) if test.any() else float("nan")
        (s_auc, s_n), (f_auc, f_n) = rate_aucs(test, p) if test.any() else ((np.nan, 0),) * 2
        print(f"  {name:<26}{a_dev:>18.3f}{a_test:>12.3f}{s_auc:>14.3f}{f_auc:>14.3f}")
        if name != "shockable_net (as is)" and (best is None or a_test > best[1]):
            best = (name, a_test)
    if test.any():
        (_, n_slow), (_, n_fast) = rate_aucs(test, ps)
        print(f"  test VT events by measured rate: under 150 bpm {n_slow}, 150 and above {n_fast}")

    print(f"\nbest on the test split: {best[0]}, event AUC {best[1]:.3f}")
    print("worth building a VT detector" if best[1] >= 0.85 else
          "not enough to build on yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
