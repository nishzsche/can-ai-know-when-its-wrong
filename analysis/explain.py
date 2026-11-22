"""Why the results look the way they do: three descriptive breakdowns of the saved test predictions.

    uv run python analysis/explain.py   # results/explain.json, plus a printed summary

1. By repayment status: actual default rate vs each contestant's mean prediction, grouped by the
   customer's most recent repayment status (PAY_0). Shows where over- or under-statement comes from.
2. Spread: how much each contestant's predictions vary (SD, 10th-90th percentile). A model can be
   calibrated on average just by saying roughly the same number for everyone.
3. Base-rate sensitivity: calibration if the same customers came from portfolios with a different
   default rate (5% to 50%). Defaulters and non-defaulters are reweighted to the target rate;
   every prediction stays as it was, since out-of-the-box answers don't depend on the portfolio.
   AUC doesn't change under this reweighting (it compares defaulters with non-defaulters), so
   only calibration is shown.

Descriptive only: nothing is fitted on test. Predictions come from results/predictions/; the
grouping variable (PAY_0) comes from data/splits/test.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import ROOT, read_records  # noqa: E402

OUT = ROOT / "results" / "explain.json"
AI = ("jev", "kev-4b", "kev-0.8b", "laya", "qwen-logprobs", "frontier-verbal")
REFERENCE = ("scorecard", "trained")
BASE_RATES = (0.05, 0.10, 0.221, 0.35, 0.50)
STATUS_GROUPS = {  # PAY_0 code -> group label (documented meaning only)
    -2: "not late (code -2)", -1: "paid on time (code -1)", 0: "not late (code 0)",
    1: "1 month late", 2: "2 months late",
}  # fmt: skip


def predictions(contestant: str, condition: str) -> pd.DataFrame:
    recs = [r for r in read_records(contestant)
            if r["split"] == "test" and r["condition"] == condition]  # fmt: skip
    return pd.DataFrame({"id": [r["id"] for r in recs], "p": [r["p_default"] for r in recs],
                         "y": [r["label"] for r in recs]})  # fmt: skip


def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: int = 10) -> float:
    """ECE with sample weights: sum over equal-width bins of W_b/W * |weighted mean p - y|."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            wb = w[m]
            total += wb.sum() * abs(np.average(p[m], weights=wb) - np.average(y[m], weights=wb))
    return float(total / w.sum())


def by_status(groups: dict[str, pd.DataFrame], status: pd.Series) -> list[dict]:
    rows = []
    group = status.map(lambda c: STATUS_GROUPS.get(c, "3+ months late"))
    order = [*STATUS_GROUPS.values(), "3+ months late"]
    any_df = next(iter(groups.values()))
    for label in order:
        ids = set(group.index[group == label])
        sub = any_df[any_df["id"].isin(ids)]
        row = {"status": label, "n": len(sub), "actual": float(sub["y"].mean())}
        for name, df in groups.items():
            row[name] = float(df[df["id"].isin(ids)]["p"].mean())
        rows.append(row)
    return rows


def spread(groups: dict[str, pd.DataFrame]) -> dict[str, dict]:
    return {name: {"mean": float(df["p"].mean()), "sd": float(df["p"].std()),
                   "p10": float(df["p"].quantile(0.1)), "p90": float(df["p"].quantile(0.9))}
            for name, df in groups.items()}  # fmt: skip


def base_rate_sensitivity(groups: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for target in BASE_RATES:
        row = {"default_rate": target}
        for name, df in groups.items():
            y, p = df["y"].to_numpy(float), df["p"].to_numpy(float)
            observed = y.mean()
            w = np.where(y == 1, target / observed, (1 - target) / (1 - observed))
            row[name] = {
                "ece": weighted_ece(y, p, w),
                "mean_predicted": float(np.average(p, weights=w)),
            }
        rows.append(row)
    return rows


def main() -> None:
    groups = {c: predictions(c, "out_of_box") for c in AI}
    groups["scorecard"] = predictions(*REFERENCE)
    status = pd.read_parquet(ROOT / "data" / "splits" / "test.parquet").set_index("id")["PAY_0"]
    result = {
        "by_status": by_status(groups, status),
        "spread": spread(groups),
        "base_rate_sensitivity": base_rate_sensitivity(groups),
        "kev_0.8b_base_rate": {
            "mean_out_of_box": float(groups["kev-0.8b"]["p"].mean()),
            "mean_told_base_rate": float(predictions("kev-0.8b", "base_rate")["p"].mean()),
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    names = list(groups)
    print("1. By most recent repayment status: actual default rate vs mean prediction")
    print(f"{'status':<24}{'n':>5}{'actual':>8}" + "".join(f"{n[:9]:>10}" for n in names))
    for r in result["by_status"]:
        print(f"{r['status']:<24}{r['n']:>5}{r['actual']:>8.2f}"
              + "".join(f"{r[n]:>10.2f}" for n in names))  # fmt: skip
    print("\n2. Spread of predictions")
    for n, s in result["spread"].items():
        print(
            f"{n:<16} mean {s['mean']:.2f}  sd {s['sd']:.2f}  "
            f"10th-90th pct {s['p10']:.2f}-{s['p90']:.2f}"
        )
    print("\n3. ECE if the portfolio's default rate were different (same customers, reweighted)")
    print(f"{'rate':<8}" + "".join(f"{n[:9]:>10}" for n in names))
    for r in result["base_rate_sensitivity"]:
        print(f"{r['default_rate']:<8.3f}" + "".join(f"{r[n]['ece']:>10.3f}" for n in names))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
