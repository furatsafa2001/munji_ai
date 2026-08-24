"""Evaluation metrics.

Written before the models on purpose: the metric defines what "done" means,
and choosing it after seeing results invites picking whichever number looks
best.

Accuracy is the wrong headline number for every stage here. Class balance is
skewed everywhere — most beats are normal, most windows are not VF — so a
model that predicts the majority class scores well and detects nothing.

Convention throughout: the *positive* class is the thing being detected.
For the quality gate the positive class is "reject", so

    sensitivity = of the windows that are truly unusable, how many were caught
    specificity = of the windows that are truly usable, how many were passed

Specificity is what matters for the gate. Rejecting good signal blinds every
downstream model, which is worse than passing a slightly noisy window.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import QUALITY_CLASSES

# Target from the detection matrix.
GATE_TARGET = {"specificity": 0.95, "sensitivity": 0.90}


# ----------------------------------------------------------------- primitives


def confusion(y_true, y_pred, labels: list[str]) -> np.ndarray:
    """Rows are true classes, columns predicted."""
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        ti, pi = idx.get(str(t)), idx.get(str(p))
        if ti is not None and pi is not None:
            m[ti, pi] += 1
    return m


def binary_scores(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    def safe(a, b):
        return float(a) / float(b) if b else float("nan")

    sens = safe(tp, tp + fn)
    prec = safe(tp, tp + fp)
    return {
        "sensitivity": sens,
        "specificity": safe(tn, tn + fp),
        "precision": prec,
        "npv": safe(tn, tn + fn),
        "f1": safe(2 * prec * sens, prec + sens) if (prec + sens) else float("nan"),
        "support": int(tp + fn),
    }


def per_class(y_true, y_pred, labels: list[str]) -> dict[str, dict]:
    """One-vs-rest scores for each class."""
    m = confusion(y_true, y_pred, labels)
    total = m.sum()
    out = {}
    for i, lab in enumerate(labels):
        tp = int(m[i, i])
        fn = int(m[i].sum() - tp)
        fp = int(m[:, i].sum() - tp)
        out[lab] = binary_scores(tp, fp, int(total - tp - fn - fp), fn)
    return out


def macro_f1(y_true, y_pred, labels: list[str]) -> float:
    f1s = [v["f1"] for v in per_class(y_true, y_pred, labels).values()]
    f1s = [f for f in f1s if not np.isnan(f)]
    return float(np.mean(f1s)) if f1s else float("nan")


# ---------------------------------------------------------------- gate views


def _binary(y_true, y_pred, positive: set[str]) -> dict[str, float]:
    t = np.isin(np.asarray(y_true, dtype=str), list(positive))
    p = np.isin(np.asarray(y_pred, dtype=str), list(positive))
    return binary_scores(int((t & p).sum()), int((~t & p).sum()),
                         int((~t & ~p).sum()), int((t & ~p).sum()))


def gate_views(y_true, y_pred) -> dict[str, dict]:
    """The gate's headline view: is 'unusable' correctly identified.

    With two classes there is only one hard decision. The finer distinction
    downstream stages need comes from thresholding P(usable) at different
    points (see config.GATE_MIN_P), not from a third label — see
    gate_views_scored for that.
    """
    return {"reject": _binary(y_true, y_pred, {"unusable"})}


def gate_views_scored(y_true, p_usable, thresholds=None) -> dict[str, dict]:
    """Scores at each downstream operating point.

    One model, several thresholds. The beat classifier wants clean morphology
    and so sets a high bar; the rule engine only needs R-peak timing and can
    accept more. Reporting both makes the trade explicit instead of burying it
    in a class definition.
    """
    from ..config import GATE_MIN_P

    p = np.asarray(p_usable, dtype=float)
    out = {}
    for name, thr in (thresholds or GATE_MIN_P).items():
        pred = np.where(p >= thr, "usable", "unusable")
        s = _binary(y_true, pred, {"unusable"})
        s["threshold"] = thr
        s["pass_rate"] = float((pred == "usable").mean())
        out[name] = s
    return out


def passes_target(y_true, y_pred, target: dict | None = None) -> tuple[bool, dict]:
    """Check the gate against the detection-matrix target.

    Measured on the 'reject' view: specificity is the share of genuinely usable
    windows that were not thrown away.
    """
    t = target or GATE_TARGET
    got = gate_views(y_true, y_pred)["reject"]
    ok = all(got[k] >= v for k, v in t.items())
    return ok, {k: {"target": v, "got": got[k], "pass": got[k] >= v}
                for k, v in t.items()}


# --------------------------------------------------------- patient breakdown


def by_patient(y_true, y_pred, patients, labels: list[str],
               min_windows: int = 20) -> dict:
    """Per-patient scores, then the spread across patients.

    A pooled number hides the failure mode that matters most: a model that
    works well on 90% of patients and fails completely on the other 10% looks
    fine pooled, and is not fine. The worst-patient figure and the 10th
    percentile are the honest summary.
    """
    groups = defaultdict(lambda: ([], []))
    for t, p, pat in zip(np.asarray(y_true, dtype=str),
                         np.asarray(y_pred, dtype=str),
                         np.asarray(patients, dtype=str)):
        groups[pat][0].append(t)
        groups[pat][1].append(p)

    scores = {}
    for pat, (t, p) in groups.items():
        if len(t) < min_windows:
            continue
        scores[pat] = macro_f1(t, p, labels)
    if not scores:
        return {"n_patients": 0}

    vals = np.array(list(scores.values()))
    worst = min(scores, key=scores.get)
    return {
        "n_patients": len(scores),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p10": float(np.percentile(vals, 10)),
        "median": float(np.median(vals)),
        "worst_patient": worst,
        "worst_score": float(scores[worst]),
        "below_0.7": int((vals < 0.7).sum()),
    }


# -------------------------------------------------------------- rhythm bias


def rhythm_rejection_bias(y_pred, source_dataset, normal_keys=("nsrdb",),
                          reject_label: str = "unusable") -> dict:
    """Does the gate reject arrhythmic signal more often than normal signal?

    The failure this catches is subtle and severe. A gate trained only on
    normal sinus rhythm learns "regular = clean, irregular = noise". Atrial
    fibrillation is irregular by definition and ventricular fibrillation has no
    organised QRS at all, so both look like noise to such a model. The gate
    then discards exactly the events MUNJI exists to detect — and its own
    accuracy score stays excellent, because it is measured on the sinus-rhythm
    data it was trained on.

    Published work measures the same thing: the best single-lead quality
    algorithms still misclassify under 20% of high-quality AF episodes as
    noise, and that is the state of the art.

    Compares rejection rate on windows from arrhythmic corpora against
    rejection rate on windows from normal-rhythm corpora. A ratio near 1.0 is
    the goal; above about 1.5 the gate is biased and its headline numbers are
    not trustworthy.
    """
    pred = np.asarray(y_pred, dtype=str)
    src = np.asarray(source_dataset, dtype=str)
    is_normal = np.isin(src, list(normal_keys))

    def rate(mask):
        return float((pred[mask] == reject_label).mean()) if mask.any() else float("nan")

    normal_rate = rate(is_normal)
    arr_rate = rate(~is_normal)

    per_ds = {}
    for ds in sorted(set(src.tolist())):
        m = src == ds
        per_ds[ds] = {"n": int(m.sum()), "reject_rate": rate(m)}

    # A zero normal-rejection rate is not a missing value — if the gate rejects
    # arrhythmic windows while passing every normal one, that is total bias and
    # must be reported as such rather than silently becoming nan.
    if normal_rate > 0:
        ratio = arr_rate / normal_rate
    elif arr_rate > 0:
        ratio = float("inf")
    else:
        ratio = 1.0
    return {
        "normal_reject_rate": normal_rate,
        "arrhythmic_reject_rate": arr_rate,
        "ratio": ratio,
        "biased": bool(ratio > 1.5),
        "per_dataset": per_ds,
    }


def format_bias(bias: dict) -> str:
    lines = ["rhythm rejection bias",
             f"  normal rhythm      {bias['normal_reject_rate']:.3f}",
             f"  arrhythmic         {bias['arrhythmic_reject_rate']:.3f}",
             f"  ratio              {bias['ratio']:>5.2f}   "
             f"{'BIASED — gate hides arrhythmias' if bias['biased'] else 'OK'}",
             "  per dataset"]
    for ds, v in bias["per_dataset"].items():
        lines.append(f"    {ds:<14} n={v['n']:>7,}  reject {v['reject_rate']:.3f}")
    return "\n".join(lines)


# ------------------------------------------------------------------- reports


def report(y_true, y_pred, labels: list[str] | None = None,
           patients=None, title: str = "") -> str:
    labels = list(labels or QUALITY_CLASSES)
    lines = []
    if title:
        lines += [title, "=" * len(title)]

    m = confusion(y_true, y_pred, labels)
    w = max(11, max(len(l) for l in labels) + 2)
    lines += ["", "confusion (rows = true, cols = predicted)",
              " " * w + "".join(f"{l:>{w}}" for l in labels)]
    for i, l in enumerate(labels):
        lines.append(f"{l:<{w}}" + "".join(f"{v:>{w},}" for v in m[i]))

    lines += ["", f"{'class':<{w}}{'sens':>9}{'spec':>9}{'prec':>9}"
                  f"{'f1':>9}{'support':>10}"]
    for lab, s in per_class(y_true, y_pred, labels).items():
        lines.append(f"{lab:<{w}}{s['sensitivity']:>9.3f}{s['specificity']:>9.3f}"
                     f"{s['precision']:>9.3f}{s['f1']:>9.3f}{s['support']:>10,}")
    lines.append(f"{'macro F1':<{w}}{'':>27}{macro_f1(y_true, y_pred, labels):>9.3f}")

    if set(labels) == set(QUALITY_CLASSES):
        gv = gate_views(y_true, y_pred)
        gw = max(w, max(len(k) for k in gv) + 2)
        lines += ["", f"{'gate view':<{gw}}{'sens':>9}{'spec':>9}{'prec':>9}{'f1':>9}"]
        for name, s in gv.items():
            lines.append(f"{name:<{gw}}{s['sensitivity']:>9.3f}{s['specificity']:>9.3f}"
                         f"{s['precision']:>9.3f}{s['f1']:>9.3f}")
        ok, detail = passes_target(y_true, y_pred)
        lines += ["", "target (reject view)"]
        for k, v in detail.items():
            mark = "PASS" if v["pass"] else "FAIL"
            lines.append(f"  {k:<14} target {v['target']:.2f}  got {v['got']:.3f}  {mark}")
        lines.append(f"  overall: {'PASS' if ok else 'FAIL'}")

    if patients is not None:
        p = by_patient(y_true, y_pred, patients, labels)
        if p.get("n_patients"):
            lines += ["", f"per-patient macro F1 across {p['n_patients']} patients",
                      f"  mean {p['mean']:.3f}   median {p['median']:.3f}   "
                      f"p10 {p['p10']:.3f}   std {p['std']:.3f}",
                      f"  worst: {p['worst_patient']} at {p['worst_score']:.3f}",
                      f"  patients below 0.70: {p['below_0.7']}"]
    return "\n".join(lines)
