"""Plot evaluation figures for a trained checkpoint.

    python -m munji_ai.eval.plot_eval --stage shockable
    python -m munji_ai.eval.plot_eval --stage rhythm

Reads a saved checkpoint, runs it over a split, and writes four figures:

    {stage}_confusion.png     where the model confuses what
    {stage}_roc_pr.png        ROC and precision-recall, operating point marked
    {stage}_probability.png   how confident the model is, split by true class
    {stage}_per_patient.png   per-patient F1, which a pooled score hides

Defaults to the validation split. The test split needs --split test and an
explicit confirmation, because once you have looked at it to tune anything it
is no longer a held-out estimate.

Figure labels are in English deliberately: matplotlib has no reliable Arabic
shaping without extra packages, and scientific figures are read in English
anyway. The surrounding document can be in Arabic.

Trains nothing and changes nothing. Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import CACHE_DIR
from ..data.windows import load_cache

# One ink colour, one accent, one muted. Anything more and the eye starts
# reading colour as meaning where there is none.
INK = "#1a1a1a"
ACCENT = "#1D9E75"
WARN = "#D85A30"
MUTED = "#B4B2A9"

STAGES = {
    "shockable": {"positive": "shockable", "floor": 0.90},
    "rhythm": {"positive": "AF", "floor": None},
}


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(stage: str, ckpt_path: Path, device):
    if stage == "shockable":
        from ..models.shockable_net import build
    elif stage == "rhythm":
        from ..models.rhythm_net import build
    else:
        raise ValueError(f"unknown stage {stage!r}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 128) -> np.ndarray:
    """Softmax probabilities for every window."""
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
        out.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ curves

def roc_points(y_true_bin: np.ndarray, score: np.ndarray):
    """ROC without sklearn — sort by score and sweep every threshold.

    Written out rather than imported so the figures do not depend on a
    package that may or may not be pinned in requirements.
    """
    order = np.argsort(-score)
    y = y_true_bin[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos, n_neg = max(y.sum(), 1), max((1 - y).sum(), 1)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def pr_points(y_true_bin: np.ndarray, score: np.ndarray):
    order = np.argsort(-score)
    y = y_true_bin[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(y.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


def binary_at(y_true_bin: np.ndarray, score: np.ndarray, thr: float):
    pred = score >= thr
    tp = int((y_true_bin & pred).sum())
    fn = int((y_true_bin & ~pred).sum())
    fp = int((~y_true_bin & pred).sum())
    tn = int((~y_true_bin & ~pred).sum())
    return (tp / max(tp + fn, 1), tn / max(tn + fp, 1))


def macro_f1_of(y_true, y_pred, classes) -> float:
    """Macro F1 over the classes actually present in y_true.

    Averaging over an absent class scores it 0 and drags a flawless record
    to 0.5 — which is how 167 healthy-only patients once showed up as
    'failing'. Present classes only.
    """
    present = [c for c in classes if (y_true == c).any()]
    if not present:
        return float("nan")
    scores = []
    for c in present:
        t, p = y_true == c, y_pred == c
        tp = int((t & p).sum())
        prec = tp / max(int(p.sum()), 1)
        rec = tp / max(int(t.sum()), 1)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


def error_rate(y_true, y_pred) -> float:
    """Share of windows got wrong. Defined for every label mix, so it is the
    right axis for a bar chart that puts healthy-only and arrest-only
    patients side by side."""
    return float((y_true != y_pred).mean())


# ------------------------------------------------------------------- plots

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_confusion(y_true, y_pred, classes, out: Path, title: str):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.4, 1.5 * n + 2.0))
    ax.imshow(row, cmap="Greens", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}\n{row[i, j]:.1%}",
                    ha="center", va="center", fontsize=10,
                    color="white" if row[i, j] > 0.55 else INK)

    ax.set_xticks(range(n), classes, fontsize=9)
    ax.set_yticks(range(n), classes, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("predicted", fontsize=10, color=INK)
    ax.set_ylabel("true", fontsize=10, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=12)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def plot_curves(y_bin, score, out: Path, positive: str, floor, thr=0.50):
    fpr, tpr, auc = roc_points(y_bin, score)
    rec, prec, ap = pr_points(y_bin, score)
    sens, spec = binary_at(y_bin, score, thr)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    a1.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    a1.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"AUC = {auc:.3f}")
    a1.plot(1 - spec, sens, "o", color=INK, markersize=7, zorder=5,
            label=f"operating point (thr {thr:.2f})")
    if floor is not None:
        a1.axhline(floor, color=WARN, linewidth=1, linestyle=":",
                   label=f"required sensitivity {floor:.2f}")
    a1.set_xlabel("false positive rate  (1 - specificity)", fontsize=10)
    a1.set_ylabel("sensitivity", fontsize=10)
    a1.set_title("ROC", fontsize=11, pad=10)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.02)
    style(a1)

    base = y_bin.mean()
    a2.axhline(base, color=MUTED, linestyle="--", linewidth=1,
               label=f"prevalence {base:.1%}")
    a2.plot(rec, prec, color=ACCENT, linewidth=2, label=f"AP = {ap:.3f}")
    a2.set_xlabel("recall", fontsize=10)
    a2.set_ylabel("precision", fontsize=10)
    a2.set_title(f"precision-recall  ({positive})", fontsize=11, pad=10)
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    a2.set_xlim(-0.02, 1.02); a2.set_ylim(-0.02, 1.02)
    style(a2)

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    return auc, ap


def plot_probability(y_true, score, positive: str, out: Path):
    """Where the scores sit, split by true class.

    A model that is right for the right reason pushes the two groups to
    opposite ends. One that is merely lucky leaves them piled near 0.5, and
    that shows up here long before it shows up in an F1.
    """
    pos = score[y_true == positive]
    neg = score[y_true != positive]
    bins = np.linspace(0, 1, 41)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.hist(neg, bins=bins, color=MUTED, alpha=0.85,
            label=f"not {positive}  (n={len(neg):,})")
    ax.hist(pos, bins=bins, color=ACCENT, alpha=0.75,
            label=f"{positive}  (n={len(pos):,})")
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=1,
               label="decision threshold")
    ax.set_xlabel(f"P({positive})", fontsize=10)
    ax.set_ylabel("windows", fontsize=10)
    ax.set_title("score distribution by true class", fontsize=11, pad=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def plot_per_patient(y_true, y_pred, patients, classes, out: Path):
    """Per-patient F1. A pooled score lets one easy patient carry a hard one.

    In a monitoring product the unit that matters is the person, not the
    window: a model that works on thirty patients and fails on the thirty
    first has failed for that person completely.
    """
    by = defaultdict(lambda: ([], []))
    for t, p, q in zip(y_true, y_pred, patients):
        by[q][0].append(t)
        by[q][1].append(p)

    # A patient with one window scores either 0.0 or 1.0, nothing between, so
    # a per-patient ranking over them is noise with a ranking printed on it.
    # CinC 2017 contributes one 30-second recording per subject — 1,104 of the
    # 6,617 "patients" here had a single window, and every one that was
    # misclassified appeared at the top of this list at error 1.000.
    #
    # Patients below MIN_WINDOWS are counted in the mean but kept out of the
    # chart and the printed list; their contribution is already in the
    # window-level figures above, which is where single-window records belong.
    MIN_WINDOWS = 5
    all_rows = [(q, error_rate(np.array(t), np.array(p)), len(t))
                for q, (t, p) in by.items()]
    rows = [r for r in all_rows if r[2] >= MIN_WINDOWS]
    thin = len(all_rows) - len(rows)
    if not rows:                      # every patient is a single recording
        rows = all_rows
        thin = 0
    rows.sort(key=lambda r: -r[1])
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    mean = float(vals.mean())

    # With 199 patients a bar per patient is unreadable at any sane figure
    # width. Show the worst 40 — which is where every patient above the
    # threshold lives — and summarise the rest in the title.
    shown = rows[:40]
    hidden = len(rows) - len(shown)
    names = [r[0] for r in shown]
    vals_all = vals
    vals = np.array([r[1] for r in shown])

    fig, ax = plt.subplots(figsize=(max(7.0, 0.30 * len(shown) + 3), 4.6))
    ax.bar(range(len(shown)), vals,
           color=[WARN if v > 0.20 else ACCENT for v in vals], width=0.68)
    ax.axhline(mean, color=INK, linestyle="--", linewidth=1,
               label=f"mean over all {len(rows)} patients: {mean:.3f}")
    ax.axhline(0.20, color=WARN, linestyle=":", linewidth=1,
               label="0.20 — above this, inspect the patient")
    ax.set_xticks(range(len(shown)), names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("window error rate", fontsize=10)
    ax.set_ylim(0, max(0.5, float(vals.max()) * 1.1))
    extra = (f"  —  {hidden} further below "
             f"{float(vals_all[len(shown):].max()):.3f}" if hidden else "")
    if thin:
        extra += f"  ·  {thin:,} single-recording patients not shown"
    ax.set_title(f"per-patient error rate — worst {len(shown)} of "
                 f"{len(rows)} patients{extra}", fontsize=11, pad=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    return rows, thin, MIN_WINDOWS


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES), default="shockable")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--yes-touch-test", action="store_true",
                    help="required to run on the test split")
    ap.add_argument("--oof", type=Path, default=None,
                    help="plot from cross-validation predictions instead of "
                         "a checkpoint — every patient scored by a model that "
                         "never saw them; this is the figure to publish")
    a = ap.parse_args(argv)

    if a.split == "test" and not a.yes_touch_test:
        print("Refusing to read the test split without --yes-touch-test.\n"
              "Once you have looked at it, it is no longer held out. Use val\n"
              "for anything you intend to act on.")
        return 1

    cfg = STAGES[a.stage]
    positive = cfg["positive"]
    a.out.mkdir(parents=True, exist_ok=True)

    if a.oof is not None:
        # Cross-validation predictions: one score per window, each produced
        # by a fold model that never trained on that patient. No checkpoint
        # is loaded and no inference runs — the file already holds the
        # honest numbers.
        if not a.oof.exists():
            print(f"no out-of-fold file at {a.oof} — run cross_validate first")
            return 1
        d = np.load(a.oof, allow_pickle=False)
        y_true = np.asarray(d["y"], dtype=str)
        y_pred = np.asarray(d["pred"], dtype=str)
        score = np.asarray(d["score"], dtype=float)
        patients = np.asarray(d["patient"], dtype=str)
        classes = sorted(set(y_true.tolist()))
        tag = f"{a.stage}_oof"
        title_src = f"out-of-fold, {len(set(patients)):,} patients"
        print(f"stage       {a.stage}")
        print(f"source      {a.oof}  ({len(y_true):,} windows, "
              f"{len(set(patients)):,} patients, cross-validated)\n")
    else:
        ckpt_path = a.ckpt or Path("checkpoints") / f"{a.stage}_net.pt"
        if not ckpt_path.exists():
            print(f"no checkpoint at {ckpt_path} — train the model first")
            return 1
        device = pick_device(a.device)
        model, ckpt = load_model(a.stage, ckpt_path, device)
        classes = list(ckpt["classes"])
        d = load_cache(a.stage, a.split, a.cache)
        y_true = np.asarray(d["y"], dtype=str)
        patients = np.asarray(d["patient"], dtype=str)
        P = predict(model, d["X"], device)
        y_pred = np.array([classes[i] for i in P.argmax(1)])
        score = P[:, classes.index(positive)]
        tag = f"{a.stage}_{a.split}"
        title_src = f"{a.split} (epoch {ckpt.get('epoch', '?')})"
        print(f"stage       {a.stage}")
        print(f"split       {a.split}  ({len(y_true):,} windows, "
              f"{len(set(patients)):,} patients)")
        print(f"checkpoint  {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
        print(f"device      {device}\n")

    y_bin = y_true == positive

    plot_confusion(y_true, y_pred, classes, a.out / f"{tag}_confusion.png",
                   f"{a.stage} — {title_src}")
    auc, apr = plot_curves(y_bin, score, a.out / f"{tag}_roc_pr.png",
                           positive, cfg["floor"], a.threshold)
    plot_probability(y_true, score, positive, a.out / f"{tag}_probability.png")
    rows, thin, min_windows = plot_per_patient(
        y_true, y_pred, patients, classes,
        a.out / f"{tag}_per_patient.png")

    sens, spec = binary_at(y_bin, score, a.threshold)
    print(f"  macro F1      {macro_f1_of(y_true, y_pred, classes):.3f}")
    print(f"  {positive} sensitivity  {sens:.3f}")
    print(f"  {positive} specificity  {spec:.3f}")
    print(f"  ROC AUC       {auc:.3f}")
    print(f"  average prec. {apr:.3f}")

    if thin:
        print(f"\n  {thin:,} patients with fewer than {min_windows} windows are")
        print("  excluded from the per-patient view: a one-window record scores")
        print("  0 or 1 and ranks as noise. They are covered by the window-level")
        print("  figures above.")

    weak = [r for r in rows if r[1] > 0.20]
    if weak:
        print(f"\n  {len(weak)} patient(s) with error rate above 0.20 "
              f"(of {len(rows):,} with >= {min_windows} windows):")
        for name, err, n in weak[:25]:
            print(f"    {name:<24} error {err:.3f}   ({n} windows)")
        if len(weak) > 25:
            print(f"    ... and {len(weak) - 25} more")
        print("  A pooled score hides these. Inspect them before quoting any "
              "figure.")

    print(f"\nwrote 4 figures to {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
