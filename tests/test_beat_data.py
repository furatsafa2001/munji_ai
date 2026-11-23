"""Beat-stage data: MITDB DS1/DS2, INCART patients, LTAFDB, runs, accounting.

    python tests/test_beat_data.py

Synthetic records and empty header files only; no PhysioNet needed. When
data/raw/incartdb is present, the INCART patient table is also checked against
the record headers themselves.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from munji_ai import config as C
from munji_ai.data import splits as S
from munji_ai.data import windows as W
from munji_ai.data.loader import Record, _patient_id
from munji_ai.data.registry import INCART_PATIENT, get
from test_pipeline import synth_ecg


def load_all(path):
    """Read every array and close the file: Windows will not delete an open npz."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def record(name, patient, seconds=600, pvc_every=0, runs_at=(), seed=0):
    """Synthetic beat record: isolated PVCs every `pvc_every` beats, and a run
    of three consecutive PVCs starting at each beat index in `runs_at`."""
    x, fs, peaks = synth_ecg(seconds=seconds, hr=72, fs=250, seed=seed)
    sym = ["N"] * len(peaks)
    if pvc_every:
        for i in range(pvc_every, len(peaks), pvc_every):
            sym[i] = "V"
    for i in runs_at:
        for j in range(i, min(i + 3, len(peaks))):
            sym[j] = "V"
    return Record(dataset="incartdb", name=name, patient=patient, signal=x, fs=fs,
                  beat_samples=np.asarray(peaks, np.int64),
                  beat_symbols=np.asarray(sym, "<U2"),
                  rhythm_starts=np.array([0], np.int64),
                  rhythm_labels=np.array(["(N"], object))


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("1. MITDB: de Chazal DS1/DS2, 202 with 201, paced nowhere")
    every = set(C.MITDB_DS1) | set(C.MITDB_DS2) | set(C.MITDB_PACED)
    check("DS1, DS2 and paced cover the 48 records once",
          len(every) == 48 and not set(C.MITDB_DS1) & set(C.MITDB_DS2),
          f"{len(C.MITDB_DS1)} + {len(C.MITDB_DS2)} + {len(C.MITDB_PACED)}")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "mitdb").mkdir()
        for r in every:
            (Path(d) / "mitdb" / f"{r}.hea").write_text("")
        m = S.build(["mitdb"], root=Path(d))
        by = lambda sp: {e["record"] for e in m.values() if e["split"] == sp}
        ds1 = (set(C.MITDB_DS1) | set(C.MITDB_MOVED_TO_DS1))
        check("DS1 validation records", by("val") == set(C.MITDB_DS1_VAL), str(sorted(by("val"))))
        check("DS1 training records, with 202", by("train") == ds1 - set(C.MITDB_DS1_VAL),
              f"{len(by('train'))} records")
        check("test is DS2 without 202", by("test") == set(C.MITDB_DS2) - {"202"},
              f"{len(by('test'))} records")
        check("paced records are excluded", by("excluded") == set(C.MITDB_PACED))
        stg = {e["record"]: e.get("stages") for e in m.values()}
        check("DS1 (and 202) are beat-only, nothing else is limited",
              all(stg[r] == ["beat"] for r in ds1) and
              all(stg[r] is None for r in every - ds1))
        check("201 and 202 are one patient in one split",
              m["mitdb/201"]["patient"] == m["mitdb/202"]["patient"]
              and m["mitdb/201"]["split"] == m["mitdb/202"]["split"] == "train")
        check("records_in: the beat stage sees DS1",
              set(S.records_in(m, "train", "mitdb", stage="beat")) == by("train"))
        check("records_in: no other caller does",
              S.records_in(m, "train", "mitdb") == []
              and S.records_in(m, "val", "mitdb", stage="rhythm") == [])
        check("records_in: test unchanged for every caller",
              set(S.records_in(m, "test", "mitdb")) == by("test")
              and set(S.records_in(m, "test", "mitdb", stage="rhythm")) == by("test"))
        (Path(d) / "mitdb" / "999.hea").write_text("")
        try:
            S.build(["mitdb"], root=Path(d))
            check("an unknown MITDB record fails loudly", False)
        except ValueError:
            check("an unknown MITDB record fails loudly", True)
    check("in_stage: unlimited entries reach every stage",
          W.in_stage({"split": "train"}, "rhythm") and W.in_stage({"split": "train"}, None))

    print("\n2. INCART: patients from the headers, V5, beat stage")
    recs = [r for v in INCART_PATIENT.values() for r in v]
    check("75 records, 32 patients, each record once",
          len(recs) == 75 and len(set(recs)) == 75 and len(INCART_PATIENT) == 32)
    ds = get("incartdb")
    ids = {r: _patient_id(ds, r) for r in recs}
    check("records of one patient share an id",
          ids["I29"] == ids["I30"] == ids["I31"] == ids["I32"] == "incartdb:14")
    check("32 distinct patient ids", len(set(ids.values())) == 32)
    try:
        _patient_id(ds, "I76")
        check("a record missing from the table fails loudly", False)
    except ValueError:
        check("a record missing from the table fails loudly", True)
    check("V5 is the lead read", ds.channel_names == ("V5",))
    check("INCART feeds the beat stage only", ds.stages == ("beat",))
    check("LTAFDB now feeds the beat stage too", "beat" in get("ltafdb").stages)
    raw = Path(C.RAW_DIR) / "incartdb"
    if raw.exists() and any(raw.rglob("*.hea")):
        import re
        import wfdb
        bad = []
        for h in raw.rglob("*.hea"):
            com = " ; ".join(wfdb.rdheader(str(h.with_suffix(""))).comments)
            mm = re.search(r"patient (\d+)", com)
            if not mm or ids.get(h.stem) != f"incartdb:{int(mm.group(1)):02d}":
                bad.append(h.stem)
        check("the table matches every INCART header", not bad, f"mismatches: {bad[:5]}")
    else:
        print("  ----  INCART not downloaded here: header check skipped")

    print("\n3. ventricular runs and the ectopic ceiling")
    check("longest run by name and by id",
          W.longest_pvc_run(["N", "PVC", "PVC", "N", "PVC", "PVC", "PVC"]) == 3
          and W.longest_pvc_run([1, 3, 3, 1]) == 2)
    mk = np.zeros(400, np.uint8)
    for s, c in ((20, 1), (80, 3), (140, 3), (200, C.BEAT_IGNORE_ID), (260, 3), (320, 2)):
        mk[s:s + 30] = c
    check("beat classes read off a mask, ignore skipped",
          W.mask_beat_classes(mk).tolist() == [1, 3, 3, 3, 2])
    rec = record("I01", "incartdb:01", seconds=600, pvc_every=5, runs_at=(100, 300, 500))
    rng = np.random.default_rng(0)
    X0, Y0, _ = W.extract_beat(rec, rng, cap=150, ectopic_room=0)
    n_runs = sum(W.longest_pvc_run(W.mask_beat_classes(y)) >= 3 for y in Y0)
    ect0 = sum(bool(np.isin(y, W._ECTOPIC_IDS).any()) for y in Y0)
    check("with no ectopic room left, run windows are still all kept",
          n_runs > 0 and ect0 == n_runs, f"{n_runs} run windows, {ect0} ectopic")
    X5, Y5, _ = W.extract_beat(rec, np.random.default_rng(0), cap=150, ectopic_room=5)
    ect5 = sum(bool(np.isin(y, W._ECTOPIC_IDS).any()) for y in Y5)
    check("the ceiling bounds the other ectopic windows", ect5 == n_runs + 5,
          f"{ect5} = {n_runs} runs + 5")

    print("\n4. build_stage: per-patient accounting, determinism")
    recs3 = [record(f"I{i:02d}", "incartdb:14", seconds=900, pvc_every=3,
                    runs_at=(400,) if i == 3 else (), seed=i) for i in (1, 2, 3)]
    manifest = {f"incartdb/{r.name}": {"dataset": "incartdb", "record": r.name,
                                       "patient": r.patient, "split": "train"}
                for r in recs3}
    real_iter = W.iter_records
    W.iter_records = lambda key, root=None, names=None: (r for r in recs3 if r.name in names)
    try:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            W.build_stage("beat", manifest, out_dir=Path(d1), raw_dir=Path(d1), verbose=False)
            W.build_stage("beat", manifest, out_dir=Path(d2), raw_dir=Path(d2), verbose=False)
            a = load_all(Path(d1) / "beat_train.npz")
            b = load_all(Path(d2) / "beat_train.npz")
            used = set(a["record"].tolist())
            check("every record of an ectopy-heavy patient is read",
                  used == {"I01", "I02", "I03"}, f"records used: {sorted(used)}")
            runs3 = sum(W.longest_pvc_run(W.mask_beat_classes(y)) >= 3 for y in a["y"])
            check("the run in the third record reaches the cache", runs3 > 0, f"{runs3} windows")
            check("two builds give identical caches",
                  np.array_equal(a["X"], b["X"]) and np.array_equal(a["y"], b["y"])
                  and np.array_equal(a["offset"], b["offset"]))
            comp = W.beat_composition(a["y"], a["dataset"], a["patient"])
            check("composition is reported per dataset",
                  comp["incartdb"]["windows"] == len(a["y"]) and comp["incartdb"]["runs3"] == runs3)
    finally:
        W.iter_records = real_iter
    check("the patient seed is the same in every run", W.patient_seed(0, "x") == W.patient_seed(0, "x")
          and W.patient_seed(0, "x") != W.patient_seed(0, "y"))
    ws = [np.random.default_rng(i).normal(0, 3, 5000) for i in range(20)]
    at_once = np.asarray(ws, dtype=C.CACHE_DTYPE)
    each = np.asarray([np.asarray(w, dtype=C.CACHE_DTYPE) for w in ws], dtype=C.CACHE_DTYPE)
    check("casting each window as it comes equals casting all at the end",
          at_once.dtype == each.dtype and np.array_equal(at_once.view(np.uint16), each.view(np.uint16)),
          "bit for bit")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
