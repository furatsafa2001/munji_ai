"""Compare beat-model checkpoints on the same windows and the same records.

    python -m munji_ai.eval.compare_beat
    python -m munji_ai.eval.compare_beat --ckpt v2=checkpoints/v2/beat_segmenter.pt --ckpt v3=checkpoints/v3/beat_segmenter.pt
    python -m munji_ai.eval.compare_beat --part val --limit 2000      # a quick look

Every model is scored on identical inputs, so a difference between columns is
a difference between models, not between samples.

Part 1, val: the validation cache
---------------------------------
Per-beat scores (the training script's matching, 150 ms) overall and per source
dataset, then per expert beat, the model's label read off its cleaned mask over
the beat's region:

    false PVC per beat   true N and PAC beats called PVC, over every expert beat:
                         the rate the VT rule's simulation assumed under 0.5%, and
                         the false burden a patient with no PVCs at all would see
    PVC sens, single     PVCs alone or in pairs
    PVC sens, in run     PVCs inside 3 or more consecutive PVCs: where v2 failed
    PVC -> PAC, PAC -> PVC, PVC -> N, PVC missed   where the errors go

Part 2, wide: whole validation records with wide beats that are not ventricular
-------------------------------------------------------------------------------
MITDB 109 (left bundle branch block) and 118 (right), and INCART patient 30
(I70, I71: WPW, wide supraventricular beats, bundle branch blocks, AF). Every
annotated beat is read, not a sample of windows, and reported by its original
symbol: what share of L, R, N, A and V beats each model calls N, PAC, PVC, or
misses. A wide beat called PVC is a false PVC burden, and in a long run of them
a false sustained-VT alert.

Each beat is read from the window (20 s, every 10 s) whose centre is nearest,
so no beat is judged at a window edge. A record the split list does not put in
val is skipped with a message: this part never reads train or test.

Not here: sustained VT and false alerts per day, which eval_beats_model already
measures (--part vt, --part nsrdb, with --beat-ckpt). Its --part runs is not a
fair comparison for a model trained on LTAFDB, because vt_runs.npz holds the
runs of LTAFDB's training patients; in-run sensitivity is measured here instead,
on validation patients only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .. import config as C
from ..data import preprocess as pp
from ..data import splits as S
from ..data.loader import load_record
from ..data.windows import load_cache
from ..train.train_beat import beat_scores

ID = C.BEAT_CLASS_ID
NAME = {0: "missed", ID["N"]: "N", ID["PAC"]: "PAC", ID["PVC"]: "PVC"}
WIN = C.window_samples("beat")
STRIDE = WIN // 2
RUN = 3
WIDE_RECORDS = (("mitdb", "109", "left bundle branch block"),
                ("mitdb", "118", "right bundle branch block"),
                ("incartdb", "I70", "WPW, wide supraventricular beats"),
                ("incartdb", "I71", "WPW, AF, bundle branch blocks"))
DEFAULT_CKPTS = ("v2=checkpoints/v2/beat_segmenter.pt", "v3=checkpoints/v3/beat_segmenter.pt")


class Model:
    def __init__(self, name: str, path: Path):
        import torch

        from ..models.beat_segmenter import build, clean_mask

        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.net = build()
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.name, self.path, self.torch, self.clean = name, path, torch, clean_mask
        self.epoch, self.val_f1 = blob.get("epoch"), blob.get("val_beat_f1")

    def masks(self, X, batch: int = 32) -> np.ndarray:
        """Cleaned masks, as deployment reads them."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(X), batch):
                xb = self.torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32))
                out.append(self.clean(self.net(xb).argmax(dim=1).numpy()))
        return np.concatenate(out) if out else np.zeros((0, WIN), np.uint8)


def regions(mask) -> list:
    """Expert beat regions [(start, end, class id)]; ignore-marked samples skipped."""
    m = np.asarray(mask).astype(int)
    out, i, n = [], 0, len(m)
    while i < n:
        c = m[i]
        if c == 0 or c == C.BEAT_IGNORE_ID:
            i += 1
            continue
        j = i
        while j < n and m[j] == c:
            j += 1
        out.append((i, j, c))
        i = j
    return out


def stretch(classes) -> list:
    """For each beat, the length of the consecutive-PVC stretch it is in (0 if not a PVC)."""
    out, i, n = [0] * len(classes), 0, len(classes)
    while i < n:
        if classes[i] != ID["PVC"]:
            i += 1
            continue
        j = i
        while j < n and classes[j] == ID["PVC"]:
            j += 1
        out[i:j] = [j - i] * (j - i)
        i = j
    return out


def label_at(pred_mask, lo: int, hi: int) -> int:
    """The model's label over [lo, hi): majority, background counting as missed."""
    seg = np.asarray(pred_mask[max(lo, 0):max(hi, 0)], dtype=np.int64)
    return int(np.bincount(seg, minlength=4)[:4].argmax()) if seg.size else 0


def per_beat(Y, P) -> list:
    """(true class, predicted class, run length) for every expert beat."""
    rows = []
    for y, p in zip(Y, P):
        regs = regions(y)
        for (s, e, c), k in zip(regs, stretch([c for _, _, c in regs])):
            rows.append((c, label_at(p, s, e), k))
    return rows


def summarise(rows) -> dict:
    a = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    t, p, k = a[:, 0], a[:, 1], a[:, 2]
    pvc = t == ID["PVC"]

    def share(sel, val):
        return float((p[sel] == val).mean()) if sel.any() else float("nan")

    return {
        "beats": int(len(t)),
        "false PVC per beat": float(((t != ID["PVC"]) & (p == ID["PVC"])).sum() / max(len(t), 1)),
        "PVC sens, single": share(pvc & (k < RUN), ID["PVC"]),
        "PVC sens, in run": share(pvc & (k >= RUN), ID["PVC"]),
        "PVCs in runs": int((pvc & (k >= RUN)).sum()),
        "PVC -> PAC": int((pvc & (p == ID["PAC"])).sum()),
        "PVC -> N": int((pvc & (p == ID["N"])).sum()),
        "PVC missed": int((pvc & (p == 0)).sum()),
        "PAC -> PVC": int(((t == ID["PAC"]) & (p == ID["PVC"])).sum()),
        "N -> PVC": int(((t == ID["N"]) & (p == ID["PVC"])).sum()),
    }


def fmt(v) -> str:
    if isinstance(v, float):
        return "-" if np.isnan(v) else (f"{v:.3%}" if v < 0.02 else f"{v:.3f}")
    return f"{v:,}"


def part_val(models, cache: Path, limit: int) -> None:
    d = load_cache("beat", "val", cache)
    X, Y, ds = d["X"], np.asarray(d["y"]), np.asarray(d["dataset"]).astype(str)
    if limit:
        X, Y, ds = X[:limit], Y[:limit], ds[:limit]
    print(f"\nPART 1  validation cache: {len(X):,} windows, "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(Counter(ds.tolist()).items())))
    preds = {}
    for m in models:
        print(f"  predicting with {m.name} ...", flush=True)
        preds[m.name] = m.masks(X)

    names = [m.name for m in models]
    head = f"{'':<26}" + "".join(f"{n:>14}" for n in names)

    print("\nper-beat scores (150 ms matching), all sources")
    print(head)
    scores = {n: beat_scores(Y, preds[n]) for n in names}
    for key in ("N", "PAC", "PVC", "QRS (any beat)"):
        for stat in ("sensitivity", "precision"):
            print(f"{key + ' ' + stat[:4]:<26}"
                  + "".join(f"{scores[n]['per_class'][key][stat]:>14.3f}" for n in names))
    print(f"{'macro F1 (beats)':<26}" + "".join(f"{scores[n]['macro_f1']:>14.3f}" for n in names))

    print("\nmacro F1 (beats) by source")
    print(head)
    for src in sorted(set(ds.tolist())):
        sel = ds == src
        print(f"{src:<26}" + "".join(
            f"{beat_scores(Y[sel], preds[n][sel])['macro_f1']:>14.3f}" for n in names))

    print("\nper expert beat (label read off the cleaned mask over the beat)")
    print(head)
    summ = {n: summarise(per_beat(Y, preds[n])) for n in names}
    for key in summ[names[0]]:
        print(f"{key:<26}" + "".join(f"{fmt(summ[n][key]):>14}" for n in names))
    print("'false PVC per beat' is a floor: false beats on background are not counted.")


def wide_record(models, ds_key, name, raw: Path) -> dict | None:
    rec = load_record(ds_key, name, raw)
    n = len(rec.signal)
    if n < WIN:
        return None
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    starts = np.asarray(starts)
    X = np.stack([pp.normalize(rec.signal[s:s + WIN]) for s in starts]).astype(np.float32)
    centres = starts + WIN / 2.0
    out = {}
    for m in models:
        P = m.masks(X)
        tally: dict = {}
        for b, sym in zip(rec.beat_samples, rec.beat_symbols):
            inside = np.flatnonzero((starts <= b) & (b < starts + WIN))
            if not inside.size:
                continue
            w = inside[np.argmin(np.abs(centres[inside] - b))]
            cls = {"PAC": "PAC", "PVC": "PVC"}.get(_mapped(sym), "N")
            h = int(round(C.QRS_HALFWIDTH_MS[cls] * rec.fs / 1000.0))
            pos = int(b - starts[w])
            tally.setdefault(str(sym), Counter())[NAME[label_at(P[w], pos - h, pos + h + 1)]] += 1
        out[m.name] = tally
    return out


def _mapped(sym: str) -> str:
    from ..data.loader import BEAT_MAP
    return BEAT_MAP.get(str(sym), "?")


def part_wide(models, raw: Path, splits_file: Path | None) -> None:
    print("\nPART 2  wide beats that are not ventricular, whole validation records")
    manifest = S.load(splits_file) if splits_file else S.load()
    for ds_key, name, what in WIDE_RECORDS:
        e = manifest.get(f"{ds_key}/{name}")
        if e is None or e["split"] != "val":
            print(f"\n  {ds_key} {name}: not in val ({e['split'] if e else 'not listed'}), skipped")
            continue
        try:
            res = wide_record(models, ds_key, name, raw)
        except FileNotFoundError as err:
            print(f"\n  {ds_key} {name}: {err}")
            continue
        if not res:
            print(f"\n  {ds_key} {name}: shorter than one window, skipped")
            continue
        print(f"\n  {ds_key} {name} ({what})")
        print(f"  {'symbol':<8}{'beats':>8}  " + "  ".join(
            f"{m.name + ' ' + lab:>12}" for m in models for lab in ("PVC", "PAC", "missed")))
        syms = sorted({s for t in res.values() for s in t}, key=lambda s: -res[models[0].name][s].total())
        for s in syms:
            n = res[models[0].name][s].total()
            cells = []
            for m in models:
                c = res[m.name][s]
                cells += [f"{c[lab] / max(n, 1):>12.1%}" for lab in ("PVC", "PAC", "missed")]
            print(f"  {s:<8}{n:>8,}  " + "  ".join(cells))
        for m in models:
            t = res[m.name]
            non_v = sum(c.total() for s, c in t.items() if _mapped(s) != "PVC")
            false_v = sum(c["PVC"] for s, c in t.items() if _mapped(s) != "PVC")
            print(f"  {m.name}: non-ventricular beats called PVC {false_v:,} of {non_v:,} "
                  f"= {false_v / max(non_v, 1):.2%}")
    print("\nA patient's daily PVC burden of 0.7% already carried a 90% specificity for heart"
          "\nfailure within 15 years in the Cardiovascular Health Study (adults 65+), so a"
          "\nfalse PVC share near that level on these beats is clinically visible.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=None,
                    help="name=path, repeatable (default: v2 and v3)")
    ap.add_argument("--part", choices=("val", "wide", "all"), default="all")
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--splits-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="val windows to use (0 = all)")
    a = ap.parse_args(argv)

    models = []
    for item in a.ckpt or DEFAULT_CKPTS:
        name, _, path = item.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"{name}: {p} not found, left out")
            continue
        models.append(Model(name, p))
        print(f"{name:<4} {p}  epoch {models[-1].epoch}  val beat F1 at training "
              f"{models[-1].val_f1 if models[-1].val_f1 is None else round(models[-1].val_f1, 3)}")
    if not models:
        print("no checkpoints to compare")
        return 1
    print("\nThe F1 stored in each checkpoint was measured on the cache it was trained "
          "with;\nonly the columns below share their inputs.")

    if a.part in ("val", "all"):
        part_val(models, a.cache, a.limit)
    if a.part in ("wide", "all"):
        part_wide(models, a.raw, a.splits_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
