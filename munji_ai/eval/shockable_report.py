"""One validation report for the shockable detector: numbers, figures, comparison.

    python -m munji_ai.eval.shockable_report --models checkpoints/shock_cv_hn
    python -m munji_ai.eval.shockable_report --models checkpoints/shock_cv_hn --skip-days

Everything here comes from cross-validation: every window, recording and
episode is scored by the fold model that never saw that patient. The test
split is not read. The report gathers what was measured piecemeal so far into
one place, at the chosen operating point, for the model card and the client
document.

Sections
--------
    windows      sensitivity, specificity, ROC AUC, confusion, spread over folds
    per rhythm   one figure per rhythm class, all windows and AHA protocol
    per source   specificity per database (arrest recordings vs the rest)
    noise        held-out NSTDB noise at a fixed SNR, clean vs noisy
    EC57         episode sensitivity and positive predictivity, duration
                 sensitivity and positive predictivity, on full arrest
                 recordings; sustained episodes (30 s or more) separately;
                 delay from onset to alarm
    daily life   false alarms per 24 h on full ambulatory and resting days
    comparison   each figure next to the published target it answers to

Primary standard
----------------
ANSI/AAMI EC57 is the standard for testing rhythm algorithms in continuous
monitors and evaluates VF by episode and by duration. AHA AED performance
goals are per analysed segment, written for a device that decides from one
segment; they are reported too, as a comparison, never hidden. The episode
matching here is EC57-style (overlap of a confirmed alarm with an annotated
episode), not a certified EC57 run.

Outputs, in --out
-----------------
    summary.md       every table, with sources, ready for the document
    metrics.json     the same numbers, machine-readable
    fig1_roc.png ... fig8_folds.png

Reads only; trains nothing. The daily-life section re-reads about 700 hours of
raw recordings and takes most of the runtime (about 45 minutes on a laptop
CPU); --skip-days leaves it out.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR, TARGET_FS
from ..data.loader import load_record
from ..data.windows import load_cache
from .aha_protocol import quality, rhythm_and_purity
from .false_alarm_rate import (SUSTAINED_S, count_alarms, fold_of_records,
                               load_fold_models, truth_flags, window_scores)

POS = "shockable"
STEP = C.WINDOW_STRIDE_SEC["shockable"]
WIN_S = C.WINDOW_SEC["shockable"]
SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70)

# Published targets each figure answers to. Values only from the sources named.
AHA = "AHA AED performance goals (Kerber et al., Circulation 1997)"
EC57_WEARABLE = ("Wearable VF detection evaluated with ANSI/AAMI EC57 "
                 "(Sensors 2017;17:2445): episode Se and +P above 0.80")
WCD_GOAL = ("Wearable defibrillator study performance goal "
            "(Poole et al., J Cardiovasc Electrophysiol 2022)")
WCD_BEST = "Kestra ASSURE observed rate (Poole et al., 2022)"


# --------------------------------------------------------------------- helpers

def rates(t: np.ndarray, p: np.ndarray) -> dict:
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def roc(t: np.ndarray, s: np.ndarray):
    order = np.argsort(-s, kind="stable")
    ts = t[order]
    tpr = np.concatenate([[0], np.cumsum(ts) / max(ts.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(~ts) / max((~ts).sum(), 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid")
                           else np.trapz(tpr, fpr))


def fired_mask(pred: np.ndarray, rule) -> np.ndarray:
    n, m = rule
    csum = np.concatenate([[0], np.cumsum(pred)])
    i = np.arange(len(pred))
    return (csum[i + 1] - csum[np.maximum(0, i - m + 1)]) >= n


def status(value, target, higher_is_better=True):
    if value is None:
        return "not measured"
    ok = value >= target if higher_is_better else value <= target
    return "meets" if ok else "below" if higher_is_better else "above"


# -------------------------------------------------------------------- sections

def window_section(y, s, folds, thr):
    t = y == POS
    out = {"threshold": thr}
    for name, cut in (("default_0.50", 0.5), ("operating", thr)):
        out[name] = rates(t, s >= cut)
    fpr, tpr, auc = roc(t, s)
    out["auc"] = auc
    out["per_fold"] = {int(f): rates(t[folds == f], s[folds == f] >= thr)
                       for f in np.unique(folds)}
    out["sweep"] = {f"{c:.2f}": rates(t, s >= c) for c in np.round(np.arange(0.05, 1.0, 0.05), 2)}
    return out, (fpr, tpr)


def rhythm_section(y, s, records, offsets, datasets, X, thr, raw):
    n = C.window_samples("shockable")
    iv: dict = {}
    lp = [rhythm_and_purity(r, ds, o, n, iv, raw)
          for r, ds, o in zip(records, datasets, offsets)]
    labels = np.array([x[0] for x in lp])
    purity = np.array([x[1] for x in lp])
    q = quality(X)
    q_thr = float(np.quantile(q, 0.20))
    aha_mask = (purity >= 0.95) & (q >= q_thr)
    t, p = y == POS, s >= thr

    def table(mask):
        rows = []
        groups = defaultdict(list)
        for i in np.flatnonzero(mask):
            groups[(labels[i], bool(t[i]))].append(i)
        for (lab, is_pos), idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(idx) < 10:
                continue
            idx = np.array(idx)
            val = float(p[idx].mean() if is_pos else (~p[idx]).mean())
            name = lab
            if lab in ("VT", "VF", "VFL"):
                name = f"{lab} ({'shockable' if is_pos else 'slow, excluded'})"
            target = None
            if is_pos and lab in ("VF", "VFL"):
                target = 0.90
            elif is_pos and lab == "VT":
                target = 0.75
            elif not is_pos and lab in ("N", "NSR"):
                target = 0.99
            elif not is_pos and lab not in ("VT", "UNLABELLED"):
                target = 0.95
            rows.append({"rhythm": name, "windows": int(len(idx)),
                         "metric": "sensitivity" if is_pos else "specificity",
                         "value": val, "target": target})
        return rows

    return {"all_windows": table(np.ones(len(y), bool)),
            "aha_protocol": table(aha_mask),
            "aha_kept_share": float(aha_mask.mean()),
            "quality_threshold": q_thr}


def source_section(y, s, datasets, thr):
    neg = y != POS
    out = {}
    for ds in sorted(set(datasets[neg])):
        m = neg & (datasets == ds)
        out[ds] = {"negatives": int(m.sum()), "specificity": float((s[m] < thr).mean())}
    arrest = np.isin(datasets, ["cudb", "vfdb"])
    out["_not_arrest"] = {"negatives": int((neg & ~arrest).sum()),
                          "specificity": float((s[neg & ~arrest] < thr).mean())}
    out["_arrest"] = {"negatives": int((neg & arrest).sum()),
                      "specificity": float((s[neg & arrest] < thr).mean())}
    return out


@torch.no_grad()
def noise_section(y, s, folds, X, models, cache, thr, snr, device):
    from ..data.train_augment import (BANK_NAME, add_noise_fixed, load_noise_bank,
                                      split_bank)
    _, held = split_bank(load_noise_bank(cache / BANK_NAME))
    noisy = np.zeros(len(y), dtype=np.float32)
    for f, model in models.items():
        idx = np.flatnonzero(folds == f)
        if idx.size == 0:
            continue
        Xn = add_noise_fixed(X[idx].astype(np.float32), held, snr, seed=1000 + f)
        out = []
        for i in range(0, len(Xn), 512):
            out.append(model.predict_proba(torch.from_numpy(Xn[i:i + 512]).to(device)).cpu().numpy())
        noisy[idx] = np.concatenate(out)
    t = y == POS
    return {"snr_db": snr, "clean": rates(t, s >= thr), "noisy": rates(t, noisy >= thr)}


def continuous_section(models, folds_map, datasets, thresholds, rule, raw, device):
    acc = {t: defaultdict(lambda: defaultdict(float)) for t in thresholds}
    delays = {t: [] for t in thresholds}
    t0 = time.time()
    for ds in datasets:
        keys = sorted(k for k in folds_map if k[0] == ds)
        for j, (_, name) in enumerate(keys, 1):
            f = folds_map[(ds, name)]
            if f not in models:
                continue
            rec = load_record(ds, name, raw)
            sc = window_scores(models[f], rec.signal, device)
            truth = truth_flags(rec, len(sc))
            hours = len(rec.signal) / TARGET_FS / 3600
            for t in thresholds:
                pred = sc >= t
                c = count_alarms(pred, truth, rule)
                fired = fired_mask(pred, rule)
                a = acc[t][ds]
                a["hours"] += hours
                for k in ("false", "true", "episodes", "caught", "sus_episodes", "sus_caught"):
                    a[k] += c[k]
                a["truth_windows"] += truth.sum()
                a["fired_windows"] += fired.sum()
                a["overlap_windows"] += (fired & truth).sum()
                delays[t] += [d for d in c["delays"]]
            if j % 10 == 0 or j == len(keys):
                print(f"  {ds:<9} {j:>3}/{len(keys)}   {time.time() - t0:6.0f} s", flush=True)
    return acc, delays


def ec57(a: dict) -> dict:
    return {"episodes": int(a["episodes"]),
            "ESe": a["caught"] / max(a["episodes"], 1),
            "E+P": a["true"] / max(a["true"] + a["false"], 1),
            "DSe": a["overlap_windows"] / max(a["truth_windows"], 1),
            "D+P": a["overlap_windows"] / max(a["fired_windows"], 1),
            "sustained_episodes": int(a["sus_episodes"]),
            "sustained_ESe": a["sus_caught"] / max(a["sus_episodes"], 1)}


# --------------------------------------------------------------------- figures

def figures(out, w, rocxy, rh, noise, arrest, days, delays, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        files.append(name)

    fpr, tpr = rocxy
    op = w["operating"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC {w['auc']:.3f})")
    ax.plot(1 - op["specificity"], op["sensitivity"], "o", ms=8,
            label=f"operating point {thr:.2f}")
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    ax.set_xlabel("1 - specificity (per window)")
    ax.set_ylabel("sensitivity (per window)")
    ax.set_title("ShockableNet, cross-validated")
    ax.legend(loc="lower right")
    save(fig, "fig1_roc.png")

    cuts = sorted(float(k) for k in w["sweep"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cuts, [w["sweep"][f"{c:.2f}"]["sensitivity"] for c in cuts], lw=2, label="sensitivity")
    ax.plot(cuts, [w["sweep"][f"{c:.2f}"]["specificity"] for c in cuts], lw=2, label="specificity")
    ax.axvline(thr, ls="--", c="k", lw=1, label=f"operating {thr:.2f}")
    ax.axhline(0.90, ls=":", c="tab:red", lw=1, label="AHA VF 0.90 (per window)")
    ax.set_xlabel("threshold on P(shockable)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-window rates against the threshold")
    ax.legend(fontsize=8)
    save(fig, "fig2_threshold.png")

    if days and arrest:
        ts = sorted(days)
        fa = [sum(days[t][d]["false"] for d in days[t]) /
              max(sum(days[t][d]["hours"] for d in days[t]), 1e-9) * 24 for t in ts]
        se = [ec57(_sum(arrest[t]))["sustained_ESe"] for t in ts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(ts, fa, "o-", lw=2, label="false alarms per 24 h (daily life)")
        ax.axhline(0.29, ls=":", c="tab:red", lw=1, label="WCD study goal 0.29/day")
        ax.set_xlabel("threshold on P(shockable)")
        ax.set_ylabel("false alarms per 24 h")
        ax2 = ax.twinx()
        ax2.plot(ts, se, "s--", c="tab:green", lw=2, label="sustained VF/VT detected")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("share of sustained episodes detected")
        ax.axvline(thr, ls="--", c="k", lw=1)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
        ax.set_title("Operating curve: alarms against safety")
        save(fig, "fig3_operating_curve.png")

    rows = rh["all_windows"]
    rows_aha = {r["rhythm"]: r for r in rh["aha_protocol"]}
    names = [r["rhythm"] for r in rows if r["rhythm"] != "UNLABELLED"][:9]
    if names:
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, 4))
        v1 = [next(r["value"] for r in rows if r["rhythm"] == n) for n in names]
        v2 = [rows_aha[n]["value"] if n in rows_aha else np.nan for n in names]
        ax.bar(x - 0.2, v1, 0.4, label="all windows")
        ax.bar(x + 0.2, v2, 0.4, label="AHA protocol")
        for i, n in enumerate(names):
            tgt = next(r["target"] for r in rows if r["rhythm"] == n)
            if tgt:
                ax.plot([i - 0.45, i + 0.45], [tgt, tgt], c="tab:red", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("sensitivity (shockable) or specificity")
        ax.set_title(f"Per rhythm at threshold {thr:.2f}; red lines are AHA targets")
        ax.legend(fontsize=8)
        save(fig, "fig4_per_rhythm.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not shockable", "shockable"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not shockable", "shockable"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion at {thr:.2f}")
    save(fig, "fig5_confusion.png")

    if noise:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(2)
        ax.bar(x - 0.2, [noise["clean"]["sensitivity"], noise["clean"]["specificity"]], 0.4, label="clean")
        ax.bar(x + 0.2, [noise["noisy"]["sensitivity"], noise["noisy"]["specificity"]], 0.4,
               label=f"NSTDB noise {noise['snr_db']:g} dB")
        ax.set_xticks(x); ax.set_xticklabels(["sensitivity", "specificity"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Robustness to motion noise at {thr:.2f}")
        ax.legend(fontsize=8)
        save(fig, "fig6_noise.png")

    if delays and delays.get(thr):
        d = np.array(delays[thr]) * STEP + WIN_S
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(d, bins=np.arange(0, d.max() + STEP * 2, STEP), edgecolor="white")
        ax.axvline(np.median(d), ls="--", c="k", label=f"median {np.median(d):.0f} s")
        ax.set_xlabel("seconds from episode onset to alarm (incl. 5 s window)")
        ax.set_ylabel("episodes")
        ax.set_title(f"Detection delay at {thr:.2f}")
        ax.legend()
        save(fig, "fig7_delay.png")

    pf = w["per_fold"]
    fig, ax = plt.subplots(figsize=(6, 4))
    fs = sorted(pf)
    ax.plot(fs, [pf[f]["sensitivity"] for f in fs], "o-", label="sensitivity")
    ax.plot(fs, [pf[f]["specificity"] for f in fs], "s-", label="specificity")
    ax.set_xticks(fs); ax.set_xlabel("fold"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Spread over folds at {thr:.2f} (per window)")
    ax.legend(fontsize=8)
    save(fig, "fig8_folds.png")
    return files


def _sum(per_ds: dict) -> dict:
    out = defaultdict(float)
    for a in per_ds.values():
        for k, v in a.items():
            out[k] += v
    return out


# --------------------------------------------------------------------- summary

def summary_md(m: dict, figs: list[str]) -> str:
    thr = m["threshold"]
    w = m["windows"]
    L = [f"# ShockableNet validation report", "",
         f"Cross-validated (5 folds, grouped by patient). Every window, recording and "
         f"episode is scored by a model that never saw that patient. Test split not used.",
         f"Operating point: threshold **{thr:.2f}** on P(shockable), confirmation "
         f"**{m['rule'][0]} of {m['rule'][1]}** windows ({STEP:g} s stride).", "",
         "## Per window", "",
         "| | at 0.50 | at operating point |", "|---|---|---|",
         f"| sensitivity | {w['default_0.50']['sensitivity']:.3f} | {w['operating']['sensitivity']:.3f} |",
         f"| specificity | {w['default_0.50']['specificity']:.3f} | {w['operating']['specificity']:.3f} |",
         f"| ROC AUC | {w['auc']:.3f} | |", ""]
    if "ec57" in m:
        e = m["ec57"]
        L += ["## EC57-style evaluation on full arrest recordings (cudb, vfdb)", "",
              "| metric | value |", "|---|---|",
              f"| episode sensitivity (ESe) | {e['ESe']:.3f} ({e['episodes']} episodes) |",
              f"| episode positive predictivity (E+P) | {e['E+P']:.3f} |",
              f"| duration sensitivity (DSe) | {e['DSe']:.3f} |",
              f"| duration positive predictivity (D+P) | {e['D+P']:.3f} |",
              f"| sustained episodes (30 s or more) detected | {e['sustained_ESe']:.3f} "
              f"({e['sustained_episodes']} episodes) |",
              f"| median delay, onset to alarm | {m.get('median_delay_s', float('nan')):.0f} s |", "",
              "E+P on these recordings counts alarms during the quiet stretches of "
              "arrests as false, so it understates precision in a home setting.", ""]
    if "days" in m:
        L += ["## False alarms on full days (gate off, upper bound)", "",
              "| source | hours | false alarms | per 24 h |", "|---|---|---|---|"]
        for ds, a in m["days"].items():
            L.append(f"| {ds} | {a['hours']:.0f} | {int(a['false'])} | {a['per_24h']:.2f} |")
        L.append("")
    L += ["## Per rhythm at the operating point", "",
          "| rhythm | windows | metric | all windows | AHA protocol | AHA target |",
          "|---|---|---|---|---|---|"]
    aha_rows = {r["rhythm"]: r for r in m["per_rhythm"]["aha_protocol"]}
    for r in m["per_rhythm"]["all_windows"]:
        a = aha_rows.get(r["rhythm"])
        L.append(f"| {r['rhythm']} | {r['windows']:,} | {r['metric']} | {r['value']:.3f} | "
                 f"{a['value']:.3f} |" if a else
                 f"| {r['rhythm']} | {r['windows']:,} | {r['metric']} | {r['value']:.3f} | — |")
        L[-1] += f" {r['target'] if r['target'] else '—'} |"
    L += ["", f"AHA protocol keeps single-rhythm, artifact-free windows "
              f"({m['per_rhythm']['aha_kept_share']:.0%} of all).", "",
          "## Specificity per source", "", "| source | negatives | specificity |", "|---|---|---|"]
    for ds, v in m["per_source"].items():
        L.append(f"| {ds.strip('_').replace('_', ' ')} | {v['negatives']:,} | {v['specificity']:.4f} |")
    if "noise" in m:
        n = m["noise"]
        L += ["", f"## Motion noise (NSTDB, held-out segments, {n['snr_db']:g} dB)", "",
              "| | clean | noisy |", "|---|---|---|",
              f"| sensitivity | {n['clean']['sensitivity']:.3f} | {n['noisy']['sensitivity']:.3f} |",
              f"| specificity | {n['clean']['specificity']:.3f} | {n['noisy']['specificity']:.3f} |"]
    L += ["", "## Against published targets", "",
          "| measure | ours | target | status | source |", "|---|---|---|---|---|"]
    for c in m["comparison"]:
        ours = "—" if c["ours"] is None else f"{c['ours']:.4g}"
        L.append(f"| {c['measure']} | {ours} | {c['target']} | {c['status']} | {c['source']} |")
    L += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    L += ["", "## Limits", "",
          "- Public recordings only; out-of-hospital VF is known to be harder than these databases.",
          "- The quality gate was not applied to daily-life windows, so alarm rates are upper bounds.",
          "- The operating point was chosen on these same cross-validation results; the "
          "test split will confirm it once.",
          "- Thresholds and confirmation are engineering proposals pending cardiologist review."]
    return "\n".join(L) + "\n"


def comparison(m: dict) -> list[dict]:
    rows = []
    def pick(names, metric):
        hits = [r for r in m["per_rhythm"]["all_windows"]
                if r["rhythm"] in names and r["metric"] == metric]
        return max(hits, key=lambda r: r["windows"])["value"] if hits else None
    vf = pick(("VF (shockable)", "VFL (shockable)"), "sensitivity")
    vt = pick(("VT (shockable)",), "sensitivity")
    nsr = pick(("NSR", "N"), "specificity")
    rows.append({"measure": "VF sensitivity, per window", "ours": vf, "target": ">= 0.90",
                 "status": status(vf, 0.90), "source": AHA})
    rows.append({"measure": "rapid VT sensitivity, per window", "ours": vt, "target": ">= 0.75",
                 "status": status(vt, 0.75), "source": AHA})
    rows.append({"measure": "normal sinus specificity, per window", "ours": nsr, "target": ">= 0.99",
                 "status": status(nsr, 0.99), "source": AHA})
    ns = m["per_source"]["_not_arrest"]["specificity"]
    rows.append({"measure": "specificity outside arrest recordings, per window", "ours": ns,
                 "target": ">= 0.99", "status": status(ns, 0.99), "source": AHA + " (NSR goal)"})
    if "ec57" in m:
        e = m["ec57"]
        rows.append({"measure": "VF/VT episode sensitivity (ESe)", "ours": e["ESe"], "target": "> 0.80",
                     "status": status(e["ESe"], 0.80), "source": EC57_WEARABLE})
        rows.append({"measure": "VF/VT episode positive predictivity (E+P)", "ours": e["E+P"],
                     "target": "> 0.80", "status": status(e["E+P"], 0.80), "source": EC57_WEARABLE})
        rows.append({"measure": "sustained VF/VT detected", "ours": e["sustained_ESe"],
                     "target": "1.00 (no sustained episode missed)",
                     "status": status(e["sustained_ESe"], 1.0), "source": "safety requirement"})
    if "days_all_per_24h" in m:
        fa = m["days_all_per_24h"]
        rows.append({"measure": "false alarms per patient-day", "ours": fa, "target": "<= 0.29",
                     "status": status(fa, 0.29, higher_is_better=False), "source": WCD_GOAL})
        rows.append({"measure": "false alarms per patient-day, best device", "ours": fa,
                     "target": "0.00075", "status": status(fa, 0.00075, higher_is_better=False),
                     "source": WCD_BEST})
    return rows


# ------------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--out", type=Path, default=Path("reports") / "shockable")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--snr", type=float, default=6.0)
    ap.add_argument("--skip-days", action="store_true",
                    help="leave out the full-day false-alarm section (the slow part)")
    ap.add_argument("--day-datasets", nargs="+", default=["nsrdb", "stdb", "edb", "fantasia"])
    ap.add_argument("--arrest-datasets", nargs="+", default=["cudb", "vfdb"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    thr, rule = a.threshold, tuple(C.CONFIRMATION["vf"])
    a.out.mkdir(parents=True, exist_ok=True)

    with np.load(a.models / "shockable_oof.npz", allow_pickle=False) as d:
        y, s = d["y"].astype(str), d["score"].astype(np.float64)
        records, offsets = d["record"].astype(str), d["offset"].astype(np.int64)
        folds = d["fold"].astype(int)
    parts = [load_cache("shockable", k, a.cache) for k in ("train", "val")]
    X = np.concatenate([p["X"] for p in parts])
    datasets = np.concatenate([np.asarray(p["dataset"], dtype=str) for p in parts])
    if len(X) != len(y):
        raise SystemExit("the cache no longer matches the out-of-fold file")
    models = load_fold_models(a.models, device)
    print(f"source      {a.models}  ({len(y):,} windows, {len(models)} fold models)")
    print(f"operating   threshold {thr:.2f}, rule {rule[0]}/{rule[1]}\n")

    m: dict = {"threshold": thr, "rule": list(rule)}
    print("windows ...", flush=True)
    m["windows"], rocxy = window_section(y, s, folds, thr)
    print("per rhythm (reads annotations, a minute or two) ...", flush=True)
    m["per_rhythm"] = rhythm_section(y, s, records, offsets, datasets, X, thr, a.raw)
    m["per_source"] = source_section(y, s, datasets, thr)
    print("noise ...", flush=True)
    try:
        m["noise"] = noise_section(y, s, folds, X, models, a.cache, thr, a.snr, device)
    except FileNotFoundError as e:
        print(f"  skipped: {e}")

    fmap = fold_of_records(a.models / "shockable_oof.npz", a.cache)
    sweep = sorted(set(SWEEP) | {thr})
    print("arrest recordings, EC57-style ...", flush=True)
    arrest, delays = continuous_section(models, fmap, a.arrest_datasets, sweep, rule, a.raw, device)
    if any(arrest[thr].values()):
        tot = _sum(arrest[thr])
        m["ec57"] = ec57(tot)
        m["ec57_per_threshold"] = {f"{t:.2f}": ec57(_sum(arrest[t])) for t in sweep}
        if delays[thr]:
            m["median_delay_s"] = float(np.median(delays[thr]) * STEP + WIN_S)

    days = None
    if not a.skip_days:
        print("daily life, full recordings (the slow part) ...", flush=True)
        days, _ = continuous_section(models, fmap, a.day_datasets, sweep, rule, a.raw, device)
        m["days"] = {ds: {"hours": v["hours"], "false": v["false"],
                          "per_24h": v["false"] / max(v["hours"], 1e-9) * 24}
                     for ds, v in days[thr].items()}
        tot = _sum(days[thr])
        m["days_all_per_24h"] = tot["false"] / max(tot["hours"], 1e-9) * 24
        m["days_per_threshold"] = {f"{t:.2f}": _sum(days[t])["false"] /
                                   max(_sum(days[t])["hours"], 1e-9) * 24 for t in sweep}

    m["comparison"] = comparison(m)
    figs = figures(a.out, m["windows"], rocxy, m["per_rhythm"], m.get("noise"),
                   arrest if "ec57" in m else None, days, delays, thr)
    (a.out / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    (a.out / "summary.md").write_text(summary_md(m, figs), encoding="utf-8")

    print("\n" + "=" * 66)
    for c in m["comparison"]:
        ours = "-" if c["ours"] is None else f"{c['ours']:.4g}"
        print(f"  {c['measure']:<48}{ours:>9}  {c['target']:<10} {c['status']}")
    print(f"\nwrote {a.out / 'summary.md'}, metrics.json and {len(figs)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
