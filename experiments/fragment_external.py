"""Train VT-vs-VF on the ECG Fragment Database, test on uncurated windows.

    python experiments/fragment_external.py                 # test on CUDB
    python experiments/fragment_external.py --vfdb-logo     # held-out VFDB patients

--vfdb-logo: CUDB has VT in one patient only, so this tests on VFDB itself,
one patient at a time: train on the curated fragments of every other patient,
test on every pure 2 s window of the held-out patient from the original
recording. Patients never overlap; the only difference between training and
testing is curated against uncurated.

On the Fragment Database, with cardiologist-revised labels, VT against VF
reached AUC 0.94-0.96 leaving one patient out. Its fragments were hand-picked
and ambiguous ones excluded, so that number may be optimistic. This trains on
every fragment (VFDB patients) and tests on CUDB, a different hospital and
different patients that nobody curated: every 2 s window lying wholly inside
an annotated VT or VF episode (flutter excluded). MUNJI's test split stays out.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location("fc", Path(__file__).with_name("fragment_check.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

WIN, STRIDE = 500, 250
VF_TOKENS, VT_TOKENS = {"(VF", "(VFIB"}, {"(VT"}


def fragments(root: Path, with_patient: bool = False):
    held_out = fc.test_split_records()
    X, y, g = [], [], []
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in (".txt", ".hea")):
        m = fc.NAME.match(f.stem)
        if not m or m.group(1) in held_out:
            continue
        lab = m.group(2).upper()
        if lab not in ("VF", "VTHR", "VTTDP"):
            continue
        x = fc.load_fragment(f)
        if len(x) >= 400:
            X.append(fc.spectrum(x))
            y.append(int(lab != "VF"))
            g.append(m.group(1))
    if with_patient:
        return np.array(X), np.array(y), np.array(g)
    return np.array(X), np.array(y)


def pure_windows(rec):
    starts = np.asarray(rec.rhythm_starts, dtype=int)
    toks = [str(t).split("\x00")[0].strip() for t in rec.rhythm_labels]
    ends = np.append(starts[1:], len(rec.signal))
    out = []
    for s, e, t in zip(starts, ends, toks):
        lab = 0 if t in VF_TOKENS else 1 if t in VT_TOKENS else None
        if lab is None:
            continue
        for lo in range(s, e - WIN + 1, STRIDE):
            out.append((lab, np.asarray(rec.signal[lo:lo + WIN], dtype=np.float64)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ecg_fragments"))
    ap.add_argument("--datasets", nargs="+", default=["cudb"])
    ap.add_argument("--vfdb-logo", action="store_true",
                    help="test on held-out VFDB patients' uncurated windows")
    a = ap.parse_args(argv)
    if a.vfdb_logo:
        return vfdb_logo(a)

    from munji_ai.data.loader import list_records, load_record
    from munji_ai.data import splits
    try:
        manifest = splits.load()
        test = {(e["dataset"], e["record"]) for e in manifest.values() if e["split"] == "test"}
    except Exception:
        test = set()

    Xtr, ytr = fragments(a.root)
    print(f"train   Fragment Database: VT {ytr.sum()}, VF {(ytr == 0).sum()} fragments")
    Xte, yte, gte = [], [], []
    for ds in a.datasets:
        for name in list_records(ds):
            if (ds, name) in test:
                continue
            for lab, x in pure_windows(load_record(ds, name)):
                Xte.append(fc.spectrum(x))
                yte.append(lab)
                gte.append(f"{ds}:{name}")
    Xte, yte, gte = np.array(Xte), np.array(yte), np.array(gte)
    print(f"test    {', '.join(a.datasets)}: VT {yte.sum()} windows from {len(set(gte[yte == 1]))} patients, "
          f"VF {(yte == 0).sum()} from {len(set(gte[yte == 0]))} patients")
    if len(set(yte)) < 2:
        print("the test set lacks VT or VF")
        return 1
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "random forest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    print("\nVT against VF on data no one hand-picked")
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        lo, hi = fc.bootstrap(yte, p, gte)
        print(f"  {name:<15} AUC {fc.auc(yte, p):.3f}   95% interval {lo:.2f} to {hi:.2f}")
        for thr in (0.5, 0.7):
            print(f"     at {thr}: VT called VT {np.mean(p[yte == 1] >= thr):.0%}, "
                  f"VF called VT {np.mean(p[yte == 0] >= thr):.0%}")
    return 0


def vfdb_logo(a) -> int:
    from munji_ai.data.loader import load_record

    Xf, yf, gf = fragments(a.root, with_patient=True)
    patients = sorted(set(gf))
    print(f"fragments   VT {yf.sum()}, VF {(yf == 0).sum()} from {len(patients)} VFDB patients")
    p_all, y_all, g_all = [], [], []
    clfs = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    raw = {}
    for pat in patients:
        w = pure_windows(load_record("vfdb", pat))
        if w:
            raw[pat] = (np.array([fc.spectrum(x) for _, x in w]), np.array([l for l, _ in w]))
    n_vt = sum(1 for v in raw.values() if (v[1] == 1).any())
    n_vf = sum(1 for v in raw.values() if (v[1] == 0).any())
    print(f"uncurated   pure 2 s windows from the original annotations: "
          f"{sum(int(v[1].sum()) for v in raw.values())} VT ({n_vt} patients), "
          f"{sum(int((v[1] == 0).sum()) for v in raw.values())} VF ({n_vf} patients)")
    print("\ntrain on other patients' curated fragments, test on this patient's uncurated windows")
    for name, make in clfs.items():
        P, Y, G = [], [], []
        for pat, (Xr, yr) in raw.items():
            tr = gf != pat
            if len(set(yf[tr])) < 2:
                continue
            clf = make()
            clf.fit(Xf[tr], yf[tr])
            P.append(clf.predict_proba(Xr)[:, 1]); Y.append(yr); G.append(np.full(len(yr), pat))
        P, Y, G = np.concatenate(P), np.concatenate(Y), np.concatenate(G)
        lo, hi = fc.bootstrap(Y, P, G)
        print(f"  {name:<15} AUC {fc.auc(Y, P):.3f}   95% interval {lo:.2f} to {hi:.2f}   "
              f"(at 0.5: VT called VT {np.mean(P[Y == 1] >= 0.5):.0%}, VF called VT {np.mean(P[Y == 0] >= 0.5):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
