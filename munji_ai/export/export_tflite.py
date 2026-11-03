"""Export MUNJI's models to TFLite for the Flutter app, and prove they still agree.

    python -m munji_ai.export.export_tflite --model shockable \
        --checkpoint checkpoints/shockable_final/shockable_net.pt
    python -m munji_ai.export.export_tflite --model rhythm \
        --checkpoint checkpoints/rhythm_final/rhythm_net.pt

Run where litert-torch installs (Linux or macOS; Google Colab works). On
Windows the converter has no wheels.

What it writes, in --out (default exports/)
-------------------------------------------
    <name>.tflite        float32 model with softmax folded in
                           shockable  [1, 1250] -> [1]  P(shockable)
                           rhythm     [1, 7500] -> [1, 3]  P(NSR), P(AF), P(OTHER)
    <name>.json          what the app needs to call it correctly: input spec,
                         preprocessing, classes, threshold, confirmation rule,
                         training provenance, verification results, SHA-256
    <name>_golden.json   validation windows with the expected outputs, for a
                         Dart unit test of the app's inference

Why verify
----------
Every reported number belongs to the PyTorch model. A converted model can
differ: operators are lowered, fused, sometimes approximated. So the TFLite
file is scored on the same validation windows and must agree before it ships:
the largest probability difference, and whether any window crosses the alert
threshold differently. The test split is not used; it is kept for the one
end-to-end test after every model is converted.

The decision settings come from config.CONFIRMATION, the single source of
truth, not from the checkpoint: a checkpoint records the rule in force when it
was trained, which can be older than the adopted one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from ..config import CACHE_DIR
from ..data.windows import load_cache

SPECS = {
    "shockable": {"name": "shockable_net", "stage": "shockable", "rule_key": "vf",
                  "classes": ["not_shockable", "shockable"], "positive": "shockable",
                  "threshold": 0.60},
    "rhythm": {"name": "rhythm_net", "stage": "rhythm", "rule_key": "af",
               "classes": list(C.RHYTHM_CLASSES), "positive": "AF", "threshold": 0.60},
}


class Probabilities(torch.nn.Module):
    """The network with softmax folded in, so the app reads probabilities."""

    def __init__(self, net, index: int | None):
        super().__init__()
        self.net, self.index = net, index

    def forward(self, x):
        p = torch.softmax(self.net(x), dim=-1)
        return p[:, self.index] if self.index is not None else p


def load_model(kind: str, checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if kind == "shockable":
        from ..models.shockable_net import SHOCKABLE_IDX, build
        net, index = build(), SHOCKABLE_IDX
    else:
        from ..models.rhythm_net import build
        net, index = build(), None
    net.load_state_dict(ck["state_dict"])
    return Probabilities(net.eval(), index).eval(), ck


def convert(model, n_samples: int, out_path: Path) -> None:
    try:
        import litert_torch as conv
    except ImportError:
        try:
            import ai_edge_torch as conv          # the package's former name
        except ImportError:
            raise SystemExit("litert-torch is not installed. In Colab or Linux:\n"
                             "  pip install litert-torch\n"
                             "It has no Windows wheels; run the export there.")
    conv.convert(model, (torch.zeros(1, n_samples),)).export(str(out_path))


def score_tflite(path: Path, X: np.ndarray) -> np.ndarray:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    rows = []
    for x in X:
        it.set_tensor(inp["index"], x[None].astype(np.float32))
        it.invoke()
        rows.append(it.get_tensor(out["index"]).reshape(-1).copy())
    return np.stack(rows).astype(np.float32)


@torch.no_grad()
def score_torch(model, X: np.ndarray) -> np.ndarray:
    out = [model(torch.from_numpy(X[i:i + 256].astype(np.float32))).numpy()
           for i in range(0, len(X), 256)]
    return np.concatenate(out).reshape(len(X), -1).astype(np.float32)


def rates(y: np.ndarray, pos: np.ndarray, positive: str) -> dict:
    t = y == positive
    return {"sensitivity": float((t & pos).sum() / max(t.sum(), 1)),
            "specificity": float((~t & ~pos).sum() / max((~t).sum(), 1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="alert threshold on the positive class (default: the adopted one)")
    ap.add_argument("--out", type=Path, default=Path("exports"))
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--golden", type=int, default=10, help="windows per class in the golden file")
    ap.add_argument("--max-diff", type=float, default=1e-4)
    a = ap.parse_args(argv)

    spec = SPECS[a.model]
    stage, name = spec["stage"], spec["name"]
    model, ck = load_model(a.model, a.checkpoint)
    thr = float(a.threshold if a.threshold is not None
                else ck.get("threshold", spec["threshold"]))
    rule = list(C.CONFIRMATION[spec["rule_key"]])
    ck_rule = ck.get("confirmation")
    n = C.window_samples(stage)
    a.out.mkdir(parents=True, exist_ok=True)
    tfl = a.out / f"{name}.tflite"

    print(f"model       {a.model}: {a.checkpoint} (epoch {ck.get('epoch')})")
    print(f"decision    threshold {thr} on P({spec['positive']}), confirmation {rule[0]} of {rule[1]}"
          + (f"  (checkpoint recorded {tuple(ck_rule)}; config wins)"
             if ck_rule and tuple(ck_rule) != tuple(rule) else ""))
    t0 = time.time()
    convert(model, n, tfl)
    print(f"converted   {tfl}  ({tfl.stat().st_size / 1e3:.0f} kB, {time.time() - t0:.0f} s)")

    val = load_cache(stage, "val", a.cache)
    X, y = val["X"].astype(np.float32), np.asarray(val["y"], dtype=str)
    p_t, p_l = score_torch(model, X), score_tflite(tfl, X)
    k = spec["classes"].index(spec["positive"]) if p_t.shape[1] > 1 else 0
    diff = np.abs(p_t - p_l)
    flips = int(((p_t[:, k] >= thr) != (p_l[:, k] >= thr)).sum())
    r_t = rates(y, p_t[:, k] >= thr, spec["positive"])
    r_l = rates(y, p_l[:, k] >= thr, spec["positive"])
    ok = float(diff.max()) <= a.max_diff and flips == 0
    print(f"\nverified on the validation split ({len(X):,} windows)")
    print(f"  largest probability difference   {diff.max():.2e}  (mean {diff.mean():.2e})")
    print(f"  windows crossing {thr:.2f} differently  {flips}")
    print(f"  sensitivity  PyTorch {r_t['sensitivity']:.4f}   TFLite {r_l['sensitivity']:.4f}")
    print(f"  specificity  PyTorch {r_t['specificity']:.4f}   TFLite {r_l['specificity']:.4f}")

    rng = np.random.default_rng(0)
    pos_i, neg_i = np.flatnonzero(y == spec["positive"]), np.flatnonzero(y != spec["positive"])
    pick = np.concatenate([rng.choice(pos_i, min(a.golden, len(pos_i)), replace=False),
                           rng.choice(neg_i, min(a.golden, len(neg_i)), replace=False)])
    out_names = (spec["classes"] if p_l.shape[1] > 1 else [f"p_{spec['positive']}"])
    golden = {"model": tfl.name, "output_order": out_names, "threshold": thr,
              "note": "inputs are already preprocessed and normalised windows",
              "cases": [{"label": str(y[i]), "input": [round(float(v), 6) for v in X[i]],
                         "expected": [round(float(v), 6) for v in p_l[i]]} for i in pick]}
    (a.out / f"{name}_golden.json").write_text(json.dumps(golden))

    meta = {
        "model": name, "file": tfl.name, "exported": date.today().isoformat(),
        "sha256": hashlib.sha256(tfl.read_bytes()).hexdigest(),
        "input": {"shape": [1, n], "dtype": "float32", "sample_rate_hz": C.TARGET_FS,
                  "window_s": C.WINDOW_SEC[stage], "stride_s": C.WINDOW_STRIDE_SEC[stage],
                  "leads": 1},
        "preprocessing": {
            "resample_hz": C.TARGET_FS,
            "bandpass_hz": list(C.MODEL_BAND), "bandpass": "Butterworth, order 4, zero-phase",
            "notch_hz": C.MAINS_HZ, "notch_q": C.NOTCH_Q, "notch": "IIR notch, zero-phase",
            "normalise": "(x - median(x)) / ((q75 - q25) / 1.349 + 1e-8), per window",
            "note": "Training filtered whole recordings with zero-phase filters. The app "
                    "must reproduce this on a rolling buffer; check it with raw-signal "
                    "golden vectors before release."},
        "output": {"order": out_names, "shape": list(p_l.shape[1:]) or [1], "range": [0, 1]},
        "decision": {"positive_class": spec["positive"], "threshold": thr,
                     "confirmation_n_of_m": rule,
                     "meaning": "alert when n of the last m windows reach the threshold"},
        "training": {k2: ck.get(k2) for k2 in ("epoch", "augment", "gamma", "extra_negatives")
                     if k2 in ck},
        "verification": {"split": "val", "windows": int(len(X)),
                         "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                         "threshold_flips": flips, "pytorch": r_t, "tflite": r_l,
                         "passed": bool(ok)},
    }
    (a.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {tfl.name}, {name}.json, {name}_golden.json in {a.out}")
    print("PASSED: the TFLite model agrees with PyTorch" if ok
          else "FAILED: the TFLite model disagrees; do not ship it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
