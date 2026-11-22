"""The classic credit scorecard: weight-of-evidence (WoE) binning + logistic regression.

    uv run python baselines/scorecard.py

Fits on `train` only (20,000 labelled customers), predicts `tune` and `test`, and writes
results/predictions/scorecard.jsonl (condition `trained`) plus results/baselines/scorecard.json
(the bin table and coefficients, for the write-up).

How it works: each of the 19 financial fields is cut into bins (repayment codes by code value,
amounts by train quantiles), small bins are merged into the neighbour with the closest default
rate, and each bin is replaced by its WoE = ln(share of defaulters / share of non-defaulters).
A logistic regression on the 19 WoE columns gives the probability of default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from common import (
    BASELINE_DIR,
    build_records,
    load_split,
    quick_report,
    timed_predict,
    train_fingerprint,
)
from load import FEATURES, PAY_STATUS
from records import write_records
from sklearn.linear_model import LogisticRegression

CONTESTANT = "scorecard"
# Repayment codes are categories: one bin per code -2, -1, 0, 1, 2 and one for 3 or more.
STATUS_EDGES = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])


@dataclass
class WoEBinner:
    """Fit bins and WoE values per column on training data; transform any data with them."""

    max_bins: int = 10
    min_bin_share: float = 0.05  # every bin holds at least 5% of training customers
    smoothing: float = 0.5  # added to bin counts so an all-good or all-bad bin stays finite
    categorical: tuple[str, ...] = PAY_STATUS
    edges_: dict[str, np.ndarray] = field(default_factory=dict)  # inner edges per column
    woe_: dict[str, np.ndarray] = field(default_factory=dict)
    stats_: dict[str, list[dict]] = field(default_factory=dict)

    def _initial_edges(self, col: str, x: np.ndarray) -> np.ndarray:
        if col in self.categorical:
            return STATUS_EDGES.copy()
        qs = np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1])
        return np.unique(qs)

    @staticmethod
    def _bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        # bin 0: x < edges[0]; bin i: edges[i-1] <= x < edges[i]; last bin: x >= edges[-1]
        return np.searchsorted(edges, x, side="right")

    def _merge_small_bins(self, x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
        edges = list(edges)
        while edges:
            idx = self._bin(x, np.array(edges))
            n_bins = len(edges) + 1
            counts = np.bincount(idx, minlength=n_bins)
            bads = np.bincount(idx, weights=y, minlength=n_bins)
            shares = counts / len(x)
            small = int(np.argmin(shares))
            if shares[small] >= self.min_bin_share:
                break
            rate = np.divide(bads, counts, out=np.full(n_bins, np.nan), where=counts > 0)
            left, right = small - 1, small + 1
            if left < 0:
                merge_with = right
            elif right >= n_bins:
                merge_with = left
            elif np.isnan(rate[small]):
                merge_with = left if counts[left] <= counts[right] else right
            else:
                gap_l, gap_r = abs(rate[small] - rate[left]), abs(rate[small] - rate[right])
                merge_with = left if gap_l <= gap_r else right
            # Merging bin i with bin i+1 removes edge i; with bin i-1 removes edge i-1.
            edges.pop(min(small, merge_with))
        return np.array(edges)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> WoEBinner:
        y = np.asarray(y, dtype=float)
        total_bad, total_good = y.sum(), len(y) - y.sum()
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            edges = self._merge_small_bins(x, y, self._initial_edges(col, x))
            idx = self._bin(x, edges)
            n_bins = len(edges) + 1
            counts = np.bincount(idx, minlength=n_bins)
            bads = np.bincount(idx, weights=y, minlength=n_bins)
            goods = counts - bads
            s, k = self.smoothing, n_bins
            woe = np.log(((bads + s) / (total_bad + s * k)) / ((goods + s) / (total_good + s * k)))
            self.edges_[col], self.woe_[col] = edges, woe
            bounds = [-np.inf, *edges, np.inf]
            self.stats_[col] = [
                {
                    "from": None if np.isinf(bounds[i]) else float(bounds[i]),
                    "to": None if np.isinf(bounds[i + 1]) else float(bounds[i + 1]),
                    "customers": int(counts[i]),
                    "default_rate": float(bads[i] / counts[i]) if counts[i] else None,
                    "woe": float(woe[i]),
                }
                for i in range(n_bins)
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            idx = self._bin(X[col].to_numpy(dtype=float), self.edges_[col])
            out[col] = self.woe_[col][idx]
        return pd.DataFrame(out, index=X.index)


def fit_scorecard(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> tuple[WoEBinner, LogisticRegression]:
    binner = WoEBinner().fit(X_train, y_train)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(binner.transform(X_train), y_train)
    return binner, model


def predict(binner: WoEBinner, model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(binner.transform(X))[:, 1]


def main() -> None:
    _, X_train, y_train = load_split("train")
    binner, model = fit_scorecard(X_train, y_train)
    model_version = f"woe-logreg (sklearn {sklearn.__version__}; train {train_fingerprint()})"

    print(
        quick_report(
            "scorecard (train, in-sample)", "train", predict(binner, model, X_train), y_train
        )
    )
    records = []
    for split in ("tune", "test"):
        df, X, y = load_split(split)
        p, ms = timed_predict(lambda X_: predict(binner, model, X_), X)
        print(quick_report("scorecard", split, p, y))
        records += build_records(
            df, p, split=split, contestant=CONTESTANT, condition="trained",
            model_version=model_version, latency_ms=ms,
        )  # fmt: skip
    path = write_records(CONTESTANT, records)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_version": model_version,
        "intercept": float(model.intercept_[0]),
        "coefficients": dict(zip(FEATURES, map(float, model.coef_[0]), strict=True)),
        "bins": binner.stats_,
    }
    (BASELINE_DIR / "scorecard.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"Wrote {len(records):,} records -> {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
