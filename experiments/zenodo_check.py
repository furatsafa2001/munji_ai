"""External test: VT against VF on new patients from the Zenodo curated set.

    python experiments/zenodo_check.py

The Curated Six-Class ECG Dataset (Zenodo 2026, Bischoff) holds hand-reviewed
VT ("vtachy") and VF/flutter ("fibv") examples from two sources: VFDB, and the
PhysioNet/CinC Challenge 2015 ICU alarms. Challenge patients are new to MUNJI.

This trains on the ECG Fragment Database (VFDB patients, cardiologist labels)
and tests on the Zenodo examples, reported separately by source:

    challenge   new patients, new hospital: the external test
    vfdb        the same patients as training: shown only for comparison

2 s windows every 1 s, full spectra, AUC with a bootstrap over source records.
The archive (54 MB) is downloaded on first run.
"""

import argparse
import bz2
import csv
import importlib.util
import io
import sys
import urllib.request
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location("fc", Path(__file__).with_name("fragment_check.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

URL = "https://zenodo.org/records/22818527/files/curated-six-class-ecg-dataset-v1.0.1.zip?download=1"
FS, WIN, STRIDE, MAX_WIN = 250, 500, 250, 20


def fetch(root: Path) -> None:
    if any(root.rglob("MANIFEST.csv")):
        return
    print("downloading the Zenodo archive (54 MB)...", flush=True)
    data = urllib.request.urlopen(URL, timeout=600).read()
    root.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(root)
    # the package may itself contain zips
    for z in list(root.rglob("*.zip")):
        zipfile.ZipFile(z).extractall(z.parent)


def column(cols, *keys, values=None, rows=None):
    """A column by its values first (any cell containing one of `values`), else by name."""
    if values and rows:
        for c in cols:
            cells = [str(r.get(c, "")).lower() for r in rows]
            if any(v in cell for cell in cells for v in values):
                return c
    for c in cols:
        if keys and all(k in c.lower() for k in keys):
            return c
    return None


def class_of(cell: str):
    cell = str(cell).lower()
    return "vtachy" if "vtachy" in cell else "fibv" if "fibv" in cell else None


def read_header(hea: Path):
    lines = [l for l in hea.read_text(errors="ignore").splitlines() if l.strip() and not l.startswith("#")]
    first = lines[0].split()
    fs = float(first[2].split("/")[0]) if len(first) > 2 else FS
    names = [" ".join(l.split()[8:]) if len(l.split()) > 8 else "" for l in lines[1:]]
    return fs, names


def read_signal(base: Path, channel: str):
    hea = base.with_suffix(".hea") if base.suffix != ".hea" else base
    csvbz = Path(str(hea)[:-4] + ".csv.bz2")
    fs, names = read_header(hea)
    text = bz2.decompress(csvbz.read_bytes()).decode(errors="ignore").splitlines()
    rows = [r.split(",") for r in text if r.strip()]
    head = rows[0]
    try:
        float(head[0])
        header = None
    except ValueError:
        header = [h.strip() for h in head]
        rows = rows[1:]
    idx = 0
    if channel:
        if header and channel in header:
            idx = header.index(channel)
        elif channel in names:
            idx = names.index(channel)
        elif channel.isdigit() and int(channel) < len(rows[0]):
            idx = int(channel)
    x = np.array([float(r[idx]) if idx < len(r) and r[idx].strip() not in ("", "NaN", "nan") else np.nan
                  for r in rows])
    x = x[np.isfinite(x)]
    if abs(fs - FS) > 1:
        f = int(round(fs))
        g = gcd(f, FS)
        x = resample_poly(x, FS // g, f // g)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", type=Path, default=Path("data/raw/zenodo_six_class"))
    ap.add_argument("--fragments", type=Path, default=Path("data/raw/ecg_fragments"))
    a = ap.parse_args(argv)

    fetch(a.zenodo)
    man = next(a.zenodo.rglob("MANIFEST.csv"))
    rows = list(csv.DictReader(man.open(newline="", encoding="utf-8", errors="ignore")))
    cols = list(rows[0].keys())
    c_cls = column(cols, "class", values=["vtachy", "fibv"], rows=rows)
    c_src = column(cols, "source", "dataset", values=["vfdb", "challenge"], rows=rows)
    c_grp = column(cols, "source_component_id") or column(cols, "source", "record")
    c_ch = column(cols, "selected", "channel")
    c_path = next((c for c in cols if any(str(r[c]).endswith((".hea", ".csv.bz2")) for r in rows[:50])), None)
    src_of = lambda r: "vfdb" if any("vfdb" in str(v).lower() for k, v in r.items() if k != c_cls) else "challenge"
    if not all((c_cls, c_grp, c_path)):
        print(f"manifest columns not recognised: {cols}\nfirst row: {rows[0]}")
        return 1

    keep = [r for r in rows if class_of(r[c_cls])]
    print(f"manifest   {len(rows)} examples; VT and VF/flutter: {len(keep)}   "
          f"(class column '{c_cls}', source '{c_src}', group '{c_grp}')")
    if not keep:
        print(f"columns: {cols}")
        for r in rows[:3]:
            print(f"  {r}")
        return 1
    for cls in ("vtachy", "fibv"):
        for src in ("challenge", "vfdb"):
            sel = [r for r in keep if class_of(r[c_cls]) == cls and src_of(r) == src]
            if sel:
                print(f"  {cls:<7} {src:<28} {len(sel):>4} examples from {len({r[c_grp] for r in sel}):>3} records")

    Xtr, ytr = fc_train(a.fragments)
    print(f"\ntrain      Fragment Database: VT {ytr.sum()}, VF {(ytr == 0).sum()} fragments")
    X, y, g, s = [], [], [], []
    for r in keep:
        try:
            x = read_signal(man.parent / r[c_path], r.get(c_ch, "") or "")
        except Exception as e:
            print(f"  skipped {r[c_path]}: {e}")
            continue
        n = max(0, (len(x) - WIN) // STRIDE + 1)
        for i in range(min(n, MAX_WIN)):
            X.append(fc.spectrum(x[i * STRIDE:i * STRIDE + WIN]))
            y.append(int(class_of(r[c_cls]) == "vtachy"))
            g.append(str(r[c_grp]))
            s.append(src_of(r))
    X, y, g, s = np.array(X), np.array(y), np.array(g), np.array(s)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "random forest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
    for src, title in (("challenge", "NEW patients (Challenge 2015): the external test"),
                       ("vfdb", "VFDB (same patients as training, for comparison only)")):
        m = s == src
        if not m.any() or len(set(y[m])) < 2:
            print(f"\n{title}: needs both VT and VF")
            continue
        print(f"\n{title}\n  windows VT {int(y[m].sum())} ({len(set(g[m & (y == 1)]))} records), "
              f"VF {int((y[m] == 0).sum())} ({len(set(g[m & (y == 0)]))} records)")
        for name, clf in models.items():
            p = clf.predict_proba(X[m])[:, 1]
            lo, hi = fc.bootstrap(y[m], p, g[m])
            print(f"  {name:<15} AUC {fc.auc(y[m], p):.3f}   95% interval {lo:.2f} to {hi:.2f}")
    return 0


def fc_train(root: Path):
    held_out = fc.test_split_records()          # MUNJI's test split stays untouched
    X, y = [], []
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
    return np.array(X), np.array(y)


if __name__ == "__main__":
    raise SystemExit(main())
