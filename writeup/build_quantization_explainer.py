"""Refresh the data inside post 4's interactive explainer, writeup/quantization_explainer.html.

    uv run python writeup/build_quantization_explainer.py

The page walks through how post 4 quantized the fine-tuned models (published as a claude.ai
artifact): a weight-rounding lab on simulated weights, what got shrunk, the checks, and the results.
This script rewrites only the page's JSON block (the `quant-data` script tag); the rest of the page
is edited by hand.

Reads results/quantization.csv (analysis/quantization.py), the 18 post 4 records' `trained` test
predictions, and the bf16 reproduction checks in results/checks/quantization/. Per model and
precision: 10 equal-width calibration bins, the headline numbers, and 40 test customers at evenly
spaced quantiles of the model's own bf16 predictions.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent / "runners"))

from quantize import PRECISIONS, QUANT_MODELS  # noqa: E402
from records import ROOT, read_records  # noqa: E402

PAGE = HERE / "quantization_explainer.html"
SUMMARY = ROOT / "results" / "quantization.csv"
REPRO_DIR = ROOT / "results" / "checks" / "quantization"
BLOCK = re.compile(r'(<script type="application/json" id="quant-data">)(.*?)(</script>)', re.S)
GIB_TO_GB = 1024**3 / 1e9  # the CSV's memory columns are GiB; the page shows decimal GB
N_CUSTOMERS = 40
FIELDS = {  # page key: (CSV column, scale)
    "auc": ("auc", 1), "ece": ("ece", 1), "ece_cal": ("ece_calibrated", 1),
    "noise": ("ece_noise_p95", 1), "mean": ("mean_predicted", 1),
    "chg": ("mean_abs_change_vs_bf16", 1), "moved5": ("moved_5pts_share", 1),
    "rho": ("spearman_vs_bf16", 1), "lat": ("latency_ms_median", 1),
    "mem": ("gpu_weights_gb", GIB_TO_GB),
}  # fmt: skip


def test_predictions(contestant: str) -> dict[str, tuple[float, int]]:
    out = {r["id"]: (r["p_default"], r["label"]) for r in read_records(contestant)
           if r["split"] == "test" and r["condition"] == "trained"}  # fmt: skip
    if len(out) != 2000:
        raise SystemExit(f"{contestant}: {len(out)} trained test records, expected 2000")
    return out


def calibration_bins(p: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """[mean prediction, observed rate, count] for each non-empty 10-point band."""
    out = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (p >= lo) & ((p < hi) if b < 9 else (p <= hi))
        if m.any():
            out.append([round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum())])
    return out


def build() -> dict:
    rows = {r["contestant"]: r for r in csv.DictReader(SUMMARY.open(encoding="utf-8"))}
    data: dict = {"models": {}, "repro": {}}
    for model in QUANT_MODELS:
        preds = {p: test_predictions(f"{model}-{p}") for p in PRECISIONS}
        ids = sorted(preds["bf16"])
        y = np.array([preds["bf16"][i][1] for i in ids], float)
        per = {}
        for p in PRECISIONS:
            x = np.array([preds[p][i][0] for i in ids])
            row = rows[f"{model}-{p}"]
            per[p] = {"rel": calibration_bins(x, y)}
            for key, (col, scale) in FIELDS.items():
                per[p][key] = None if row[col] == "" else round(float(row[col]) * scale, 5)
        base = np.array([preds["bf16"][i][0] for i in ids])
        order = np.argsort(base, kind="stable")
        picks = [order[round(q * (len(ids) - 1))] for q in np.linspace(0.01, 0.99, N_CUSTOMERS)]
        customers = [
            {"id": ids[k], "y": int(y[k]), "p": [round(preds[p][ids[k]][0], 4) for p in PRECISIONS]}
            for k in picks
        ]
        observed = round(float(y.mean()), 4)
        data["models"][model] = {"per": per, "cust": customers, "observed": observed}
        check = json.loads((REPRO_DIR / f"repro_{model}-bf16.json").read_text("utf-8"))
        data["repro"][model] = {
            "mean": round(check["mean_abs_diff"], 5), "max": round(check["max_abs_diff"], 4),
            "rows": [[round(r["reference"], 4), round(r["new_path"], 4)] for r in check["rows"]],
        }  # fmt: skip
    return data


def main() -> None:
    blob = json.dumps(build(), separators=(",", ":"))
    if "</" in blob:
        raise SystemExit("the data would close the page's script tag")
    page = PAGE.read_text("utf-8")
    new, n = BLOCK.subn(lambda m: m.group(1) + blob + m.group(3), page)
    if n != 1:
        raise SystemExit(f"expected one quant-data block in {PAGE.name}, found {n}")
    PAGE.write_text(new, "utf-8")
    print(f"{PAGE.relative_to(ROOT)}: data refreshed ({len(blob):,} bytes)")


if __name__ == "__main__":
    main()
