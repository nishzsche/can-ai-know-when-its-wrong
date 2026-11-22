"""Score every contestant x condition on `test` and write results/summary.csv (+ detail JSON).

Reads only results/predictions/ (CLAUDE.md). Contestants whose files are missing or incomplete
are skipped with a note, so this can run before Sol finishes and again after.

- results/summary.csv: one row per contestant x condition, all headline metrics with bootstrap
  95% intervals for AUC, ECE and calibration-in-the-large, plus cost per 1,000 decisions and
  median latency. Recalibrated rows carry the cost and latency of the model call they need.
- results/summary_detail.json: reliability by decile, risk bands and approval thresholds, for
  the figures and the write-up.
- results/summary_excluding_memorized.csv: the same table with the test customers GPT-6.2 Sol
  reproduced from the data file left out, for every contestant (written once
  data/memdepth.py has run). The recalibrated rows keep their calibrators, fitted on all of tune.

    uv run python analysis/summarize.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as m  # noqa: E402
from records import CONDITIONS, PREDICTIONS_DIR, ROOT, read_records  # noqa: E402

CONTESTANTS = (
    "scorecard", "gbm",
    "jev", "kev-4b", "kev-0.8b", "laya", "qwen-logprobs", "frontier-verbal",
)  # fmt: skip
BASELINES = ("scorecard", "gbm")
N_TEST = 2000
SUMMARY_CSV = ROOT / "results" / "summary.csv"
DETAIL_JSON = ROOT / "results" / "summary_detail.json"
# The with/without comparison: everyone re-scored on test minus the customers GPT-6.2 Sol
# reproduced verbatim in the memorization tests (data/contamination.md).
MEMORIZED_JSON = ROOT / "data" / "memcheck" / "results" / "openai_gpt-6.2-sol__depth.json"
EXCLUDED_CSV = ROOT / "results" / "summary_excluding_memorized.csv"

COLUMNS = [
    "contestant", "kind", "condition", "n", "observed_rate", "mean_predicted",
    "auc", "auc_lo", "auc_hi", "gini", "ks", "brier",
    "ece", "ece_lo", "ece_hi", "citl", "citl_lo", "citl_hi", "cal_slope", "cal_intercept",
    "cost_per_1000_usd", "latency_ms_median", "model_version",
]  # fmt: skip


def score_group(recs: list[dict], n_boot: int) -> tuple[dict, dict]:
    recs = sorted(recs, key=lambda r: r["id"])
    y = np.array([r["label"] for r in recs], dtype=float)
    p = np.array([r["p_default"] for r in recs], dtype=float)
    row = m.summarize(y, p, n_boot=n_boot)
    row["cost_per_1000_usd"] = 1000 * statistics.mean(r["cost_usd"] for r in recs)
    row["latency_ms_median"] = statistics.median(r["latency_ms"] for r in recs)
    row["model_version"] = "; ".join(sorted({r["model_version"] for r in recs}))
    detail = {
        "deciles": m.reliability_by_decile(y, p),
        "risk_bands": m.risk_bands(y, p),
        "approval": [m.approval(y, p, t) for t in m.APPROVAL_THRESHOLDS],
    }
    return row, detail


def memorized_ids(path: Path = MEMORIZED_JSON) -> frozenset[str]:
    """Customers the frontier contestant reproduced from the data file (data/memdepth.py)."""
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text("utf-8"))["memorized_ids"])


def build(
    directory: Path = PREDICTIONS_DIR, n_boot: int = 1000, exclude: frozenset[str] = frozenset()
) -> tuple[list[dict], dict]:
    """Score each complete contestant x condition on test, leaving out the `exclude` ids.

    Completeness is checked on the full test set first, so every contestant is scored on the
    same customers either way.
    """
    rows, details = [], {}
    for contestant in CONTESTANTS:
        if not (directory / f"{contestant}.jsonl").exists():
            print(f"{contestant}: no predictions yet, skipped")
            continue
        test = [r for r in read_records(contestant, directory) if r["split"] == "test"]
        for condition in CONDITIONS:
            group = [r for r in test if r["condition"] == condition]
            if not group:
                continue
            if len(group) != N_TEST or len({r["id"] for r in group}) != N_TEST:
                print(f"{contestant}/{condition}: {len(group)} test records, skipped")
                continue
            group = [r for r in group if r["id"] not in exclude]
            row, detail = score_group(group, n_boot)
            kind = "baseline (trained on 20,000)" if contestant in BASELINES else "AI (zero-shot)"
            rows.append({"contestant": contestant, "kind": kind, "condition": condition, **row})
            details.setdefault(contestant, {})[condition] = detail
    return rows, details


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items() if k in COLUMNS})  # fmt: skip


def compare_without_memorized(rows: list[dict], exclude: frozenset[str]) -> None:
    """Re-score everyone without the memorized customers; write the CSV and print the change."""
    without, _ = build(exclude=exclude)
    write_csv(without, EXCLUDED_CSV)
    full = {(r["contestant"], r["condition"]): r for r in rows}
    n_out = N_TEST - (without[0]["n"] if without else N_TEST)
    print(f"\nWithout the {n_out} memorized test customer(s) (same customers for everyone):")
    print(f"{'contestant':<16}{'condition':<20}{'n':>6}{'dAUC':>9}{'dECE':>9}{'dCITL':>9}")
    for r in without:
        f = full[(r["contestant"], r["condition"])]
        print(f"{r['contestant']:<16}{r['condition']:<20}{r['n']:>6}"
              f"{r['auc'] - f['auc']:>+9.4f}{r['ece'] - f['ece']:>+9.4f}"
              f"{r['citl'] - f['citl']:>+9.4f}")  # fmt: skip
    print(f"Saved -> {EXCLUDED_CSV}")


def main() -> None:
    rows, details = build()
    write_csv(rows, SUMMARY_CSV)
    DETAIL_JSON.write_text(json.dumps(details, indent=1) + "\n", encoding="utf-8")
    print(f"{'contestant':<16}{'condition':<20}{'AUC':>6}{'ECE':>7}{'mean p':>8}{'actual':>8}"
          f"{'slope':>7}{'$/1k':>8}")  # fmt: skip
    for r in rows:
        print(f"{r['contestant']:<16}{r['condition']:<20}{r['auc']:>6.3f}{r['ece']:>7.3f}"
              f"{r['mean_predicted']:>8.3f}{r['observed_rate']:>8.3f}{r['cal_slope']:>7.2f}"
              f"{r['cost_per_1000_usd']:>8.3f}")  # fmt: skip
    print(f"Saved -> {SUMMARY_CSV}, {DETAIL_JSON}")
    exclude = memorized_ids()
    if exclude:
        compare_without_memorized(rows, exclude)
    else:
        print(f"No {MEMORIZED_JSON.name} yet: skipped the without-memorized comparison")


if __name__ == "__main__":
    main()
