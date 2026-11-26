"""Post 4 speed addendum (PLAN.md §12): the tuned setup's speed, cost and honesty.

    uv run python analysis/speed_report.py   # -> results/speed.csv, speed_throughput.csv, .json

- Throughput (results/checks/speed/*.json, runners/throughput.py): customers per second and cost per
  1,000 decisions at 1, 8 and 32 at a time, on the same 500 tune customers for every version.
- Honesty, for the tags whose arithmetic changes (hqq-4-aoint4, hqq-4-gemlite, fp8-dynamic):
  scored on test against the model's own post 4 bf16 run with analysis/quantization.py's
  measures, Holm over this addendum's own rows (post 4's 15-row family is unchanged).
- Reproduction checks for bf16-tuned and bnb-nf4-tuned against post 4's bf16 / bnb-nf4 runs
  (results/checks/quantization/repro_*.json).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runners"))

import quantization as qa  # noqa: E402
from quantize import QUANT_MODELS, contestant_name  # noqa: E402
from records import PREDICTIONS_DIR, ROOT  # noqa: E402
from robustness import holm  # noqa: E402
from speed import FULL_RUN, REPRODUCE, SPEED_TAGS, speed_name  # noqa: E402

SPEED_DIR = ROOT / "results" / "checks" / "speed"
OUT_CSV = ROOT / "results" / "speed.csv"
OUT_THROUGHPUT = ROOT / "results" / "speed_throughput.csv"
OUT_JSON = ROOT / "results" / "speed.json"


def throughput_rows(directory: Path = SPEED_DIR) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        d = json.loads(path.read_text("utf-8"))
        for batch, b in d["by_batch"].items():
            rows.append({"contestant": d["contestant"], "batch": int(batch),
                         "customers": b["customers"],
                         "customers_per_second": b["customers_per_second"],
                         "cost_per_1000_usd": b["cost_per_1000_usd"],
                         "latency_ms_median": b.get("latency_ms_median"),
                         "max_abs_diff_vs_batch_1": b["max_abs_diff_vs_batch_1"]})  # fmt: skip
    return rows


def honesty_rows(directory: Path = PREDICTIONS_DIR) -> tuple[list[dict], dict]:
    rows, not_scored = [], {}
    for model in QUANT_MODELS:
        for tag in FULL_RUN:
            name = speed_name(model, tag)
            out = qa.score_pair(name, contestant_name(model, "bf16"), model, tag, directory)
            if isinstance(out, str):
                not_scored[name] = out
            else:
                rows.append(out)
    for key in ("auc_vs_bf16_p", "ece_vs_bf16_p"):
        adjusted = holm({r["contestant"]: r[key] for r in rows if key in r})
        for r in rows:
            if r["contestant"] in adjusted:
                r[f"{key}_holm"] = adjusted[r["contestant"]]
    return rows, not_scored


def reproduction(directory: Path = qa.REPRO_DIR) -> dict:
    out = {}
    for model in QUANT_MODELS:
        for tag in REPRODUCE:
            path = directory / f"repro_{speed_name(model, tag)}.json"
            if path.exists():
                d = json.loads(path.read_text("utf-8"))
                out[d["contestant"]] = {k: d[k] for k in ("reference", "n", "mean_abs_diff",
                                                          "max_abs_diff", "passed")}  # fmt: skip
    return out


def build(predictions: Path = PREDICTIONS_DIR, speed_dir: Path = SPEED_DIR,
          repro_dir: Path = qa.REPRO_DIR) -> dict:  # fmt: skip
    rows, not_scored = honesty_rows(predictions)
    return {"honesty": rows, "not_scored": not_scored, "throughput": throughput_rows(speed_dir),
            "reproduction_checks": reproduction(repro_dir), "tags": list(SPEED_TAGS),
            "holm_family": "the addendum's full-run rows, AUC and ECE vs own bf16"}  # fmt: skip


def _write_csv(path: Path, rows: list[dict]) -> None:
    first = ("contestant", "model", "precision", "batch")
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in first, k))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    out = build()
    if not out["honesty"] and not out["throughput"]:
        print(f"nothing to score yet ({len(out['not_scored'])} full-run records pending)")
        return
    if out["honesty"]:
        _write_csv(OUT_CSV, out["honesty"])
    if out["throughput"]:
        _write_csv(OUT_THROUGHPUT, out["throughput"])
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    for r in out["throughput"]:
        print(f"{r['contestant']:28} {r['batch']:>2} at a time: {r['customers_per_second']:7.1f}/s "
              f"${r['cost_per_1000_usd']:.4f} per 1,000")  # fmt: skip
    for r in out["honesty"]:
        print(f"{r['contestant']:28} AUC {r['auc']:.3f} ({r['auc_vs_bf16']:+.4f}) ECE "
              f"{100 * r['ece']:.1f} ({100 * r['ece_vs_bf16']:+.2f}) |chg| "
              f"{100 * r['mean_abs_change_vs_bf16']:.1f}")  # fmt: skip
    for name, why in out["not_scored"].items():
        print(f"not scored: {name}: {why}")
    for name, r in out["reproduction_checks"].items():
        print(f"reproduction {name} vs {r['reference']}: mean |diff| {r['mean_abs_diff']:.4f} "
              f"-> {'PASS' if r['passed'] else 'FAIL'}")  # fmt: skip
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
