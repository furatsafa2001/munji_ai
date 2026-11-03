"""VT against VF on the ECG Fragment Database, with cardiologist-revised labels.

    python experiments/fragment_check.py --root data/raw/ecg_fragments

The ECG Fragment Database (PhysioNet, 2022) re-labels VFDB in 2 s fragments,
reviewed by a cardiologist: 412 of 1,016 annotations were revised, torsade de
pointes (VTTdP) is split from other fast VT (VTHR), and flutter (VFL) from VF.
Same 22 patients as VFDB, but cleaner labels. Its authors report 94.8% for six
classes, with fragments chosen for variety, not split by patient.

Here every comparison is scored leaving one patient out, with a 95% patient
bootstrap interval. Patients in MUNJI's test split are left out.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 250
NAME = re.compile(r"^(\d+)_C_([A-Za-z]+)_(\d+)s_frag$", re.IGNORECASE)


def load_fragment(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".hea":
        import wfdb
        rec = wfdb.rdrecord(str(path.with_suffix("")))
        x = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
        x = x[np.isfinite(x)]
        if rec.fs and abs(rec.fs - FS) > 1:
            from math import gcd
            g = gcd(int(rec.fs), FS)
            x = resample_poly(x, FS // g, int(rec.fs) // g)
        return x
    vals = []
    for line in path.read_text(errors="ignore").split():
        try:
            vals.append(float(line.replace(",", ".")))
        except ValueError:
            continue
    x = np.asarray(vals, dtype=np.float64)
    if len(x) >= 700:                                  # 2 s at 360 Hz
        x = resample_poly(x, 25, 36)
    return x


def spectrum(x):
    x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-8)
    f, p = welch(x, FS, nperseg=min(256, len(x)), nfft=512)
    m = (f >= 0.5) & (f <= 25)
    p = p[m] / (p[m].sum() + 1e-12)
    return np.log(p + 1e-8)


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap(y, p, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in pats}
    out = []
    for _ in range(n):
        sel = np.concatenate([idx[k] for k in rng.choice(pats, len(pats), replace=True)])
        a = auc(y[sel], p[sel])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def test_split_records() -> set:
    try:
        from munji_ai.data import splits
        m = splits.load()
        return {e["record"] for e in m.values() if e["dataset"] == "vfdb" and e["split"] == "test"}
    except Exception:
        return set()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ecg_fragments"))
    a = ap.parse_args(argv)

    held_out = test_split_records()
    X, lab, grp = [], [], []
    files = sorted(p for p in a.root.rglob("*") if p.suffix.lower() in (".txt", ".hea"))
    for f in files:
        m = NAME.match(f.stem)
        if not m or m.group(1) in held_out:
            continue
        x = load_fragment(f)
        if len(x) < 400:
            continue
        X.append(spectrum(x))
        lab.append(m.group(2).upper())
        grp.append(m.group(1))
    X, lab, grp = np.array(X), np.array(lab), np.array(grp)
    if not len(lab):
        sample = sorted(p for p in a.root.rglob("*") if p.is_file())[:12]
        print(f"no fragments matched under {a.root}; first files found:")
        for p in sample:
            print(f"  {p.relative_to(a.root)}")
        return 1
    print(f"fragments  {len(lab):,} from {len(set(grp))} patients "
          f"({len(held_out)} test-split patients left out)")
    for k in ("VF", "VFL", "VTTDP", "VTHR", "VTLR"):
        print(f"  {k:<6} {int((lab == k).sum()):>4} fragments from {len(set(grp[lab == k])):>2} patients")

    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    tests = [
        ("fast VT (VTHR) against VF", {"VTHR"}, {"VF"}),
        ("fast VT against VF and flutter", {"VTHR"}, {"VF", "VFL"}),
        ("all VT (VTHR, TdP) against VF", {"VTHR", "VTTDP"}, {"VF"}),
    ]
    for title, pos, neg in tests:
        m = np.isin(lab, list(pos | neg))
        y = np.isin(lab[m], list(pos)).astype(int)
        g = grp[m]
        if len(set(g[y == 1])) < 2 or len(set(g[y == 0])) < 2:
            print(f"\n{title}: too few patients")
            continue
        print(f"\n{title}  ({len(set(g[y == 1]))} patients with VT, {len(set(g[y == 0]))} with VF)")
        for name, make in models.items():
            p = np.full(len(y), np.nan)
            for tr, te in LeaveOneGroupOut().split(X[m], y, g):
                if len(set(y[tr])) < 2:
                    continue
                clf = make()
                clf.fit(X[m][tr], y[tr])
                p[te] = clf.predict_proba(X[m][te])[:, 1]
            ok = np.isfinite(p)
            lo, hi = bootstrap(y[ok], p[ok], g[ok])
            print(f"  {name:<15} AUC {auc(y[ok], p[ok]):.3f}   95% interval {lo:.2f} to {hi:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
