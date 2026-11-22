"""Shared helpers for the trained baselines (scorecard, gbm)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "analysis"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from load import DEMOGRAPHIC, FEATURES, TARGET  # noqa: E402
from records import make_record, utc_now  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from splits import MANIFEST, SPLIT_DIR  # noqa: E402

BASELINE_DIR = ROOT / "results" / "baselines"
SEED = 20261122


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(full frame, feature matrix, labels) for one split. Demographics never appear."""
    df = pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
    assert not set(DEMOGRAPHIC) & set(df.columns), "demographic columns in a split file"
    return df, df[list(FEATURES)], df[TARGET].to_numpy()


def train_fingerprint() -> str:
    """Short hash of the train ids, so model_version pins the exact training data."""
    return json.loads(MANIFEST.read_text("utf-8"))["splits"]["train"]["ids_sha256"][:12]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins (quick check; metrics.py is canonical)."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return total / len(p)


def quick_report(name: str, split: str, p: np.ndarray, y: np.ndarray) -> str:
    return (
        f"{name:<26} {split:<5} AUC {roc_auc_score(y, p):.4f}  Brier {brier_score_loss(y, p):.4f}  "
        f"ECE {ece(p, y):.4f}  mean p {p.mean():.4f} vs actual {y.mean():.4f}"
    )


def timed_predict(fn, X) -> tuple[np.ndarray, float]:
    """Predict and return (probabilities, milliseconds per customer)."""
    start = time.perf_counter()
    p = np.asarray(fn(X), dtype=float)
    per_row_ms = (time.perf_counter() - start) * 1000 / len(X)
    return p, per_row_ms


def build_records(
    df: pd.DataFrame,
    p: np.ndarray,
    *,
    split: str,
    contestant: str,
    condition: str,
    model_version: str,
    latency_ms: float,
) -> list[dict]:
    ts = utc_now()
    return [
        make_record(
            id=row_id,
            split=split,
            contestant=contestant,
            condition=condition,
            model_version=model_version,
            p_default=float(prob),
            label=int(label),
            latency_ms=latency_ms,
            cost_usd=0.0,  # local CPU
            ts=ts,
        )
        for row_id, prob, label in zip(df["id"], p, df[TARGET], strict=True)
    ]
