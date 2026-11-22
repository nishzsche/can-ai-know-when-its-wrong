"""Tests for the trained baselines: WoE binning, saved predictions, and reproducibility."""

import json

import numpy as np
import pandas as pd
import pytest
import records as rc
from common import BASELINE_DIR, load_split, train_fingerprint
from scorecard import WoEBinner, fit_scorecard, predict
from splits import MANIFEST

# ---- WoE binning on synthetic data -----------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 5_000
    amount = np.concatenate([np.zeros(n // 2), rng.lognormal(8, 1.5, n - n // 2)])
    status = rng.choice(
        [-2, -1, 0, 1, 2, 3, 5, 8], size=n, p=[0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.03, 0.02]
    )
    logit = -1.5 + 0.8 * np.clip(status, -1, 3) - 0.00002 * amount
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"PAY_0": status, "BILL_AMT1": amount})
    return X, y


def test_every_bin_holds_at_least_the_minimum_share(synthetic):
    X, y = synthetic
    binner = WoEBinner().fit(X, y)
    for col in X.columns:
        shares = [b["customers"] / len(X) for b in binner.stats_[col]]
        assert min(shares) >= binner.min_bin_share
        assert sum(b["customers"] for b in binner.stats_[col]) == len(X)


def test_woe_rises_with_default_rate(synthetic):
    X, y = synthetic
    binner = WoEBinner().fit(X, y)
    for col in X.columns:
        bins = binner.stats_[col]
        order_rate = np.argsort([b["default_rate"] for b in bins], kind="stable")
        order_woe = np.argsort([b["woe"] for b in bins], kind="stable")
        assert list(order_rate) == list(order_woe)


def test_repayment_codes_three_and_up_share_a_bin(synthetic):
    X, y = synthetic
    binner = WoEBinner().fit(X, y)
    t = binner.transform(pd.DataFrame({"PAY_0": [3, 5, 8], "BILL_AMT1": [0, 0, 0]}))
    assert t["PAY_0"].nunique() == 1


def test_transform_handles_unseen_values_without_nan(synthetic):
    X, y = synthetic
    binner = WoEBinner().fit(X, y)
    t = binner.transform(pd.DataFrame({"PAY_0": [-9, 99], "BILL_AMT1": [-1e9, 1e12]}))
    assert np.isfinite(t.to_numpy()).all()


def test_constant_column_gets_one_finite_bin():
    X = pd.DataFrame({"BILL_AMT1": np.zeros(100)})
    y = np.array([0, 1] * 50)
    binner = WoEBinner().fit(X, y)
    assert len(binner.stats_["BILL_AMT1"]) == 1
    assert np.isfinite(binner.woe_["BILL_AMT1"]).all()


# ---- saved predictions -----------------------------------------------------------------------

EXPECTED = {
    "scorecard": {("trained", "tune"), ("trained", "test")},
    "gbm": {("trained", "tune"), ("trained", "test"), ("trained_calibrated", "test")},
}


def _records(contestant):
    path = rc.PREDICTIONS_DIR / f"{contestant}.jsonl"
    if not path.exists():
        pytest.skip(f"run `uv run python baselines/{contestant}.py` first")
    return rc.read_records(contestant)


@pytest.mark.parametrize("contestant", sorted(EXPECTED))
def test_saved_predictions_cover_tune_and_test_exactly(contestant):
    recs = _records(contestant)
    groups = {(r["condition"], r["split"]) for r in recs}
    assert groups == EXPECTED[contestant]
    truth = {s: load_split(s)[0] for s in ("tune", "test")}
    for condition, split in groups:
        part = [r for r in recs if (r["condition"], r["split"]) == (condition, split)]
        df = truth[split]
        assert [r["id"] for r in part] == df["id"].tolist()
        assert [r["label"] for r in part] == df["default_next_month"].tolist()
        assert all(train_fingerprint() in r["model_version"] for r in part)


def test_scorecard_predictions_are_reproducible():
    """Refitting on train reproduces the saved test probabilities exactly."""
    recs = [r for r in _records("scorecard") if r["split"] == "test"]
    _, X_train, y_train = load_split("train")
    _, X_test, _ = load_split("test")
    binner, model = fit_scorecard(X_train, y_train)
    np.testing.assert_allclose(predict(binner, model, X_test), [r["p_default"] for r in recs])


def test_gbm_predictions_are_reproducible():
    import gbm

    recs = [r for r in _records("gbm") if r["condition"] == "trained"]
    rounds = json.loads((BASELINE_DIR / "gbm.json").read_text("utf-8"))["rounds"]
    _, X_train, y_train = load_split("train")
    model = gbm.fit_gbm(X_train, y_train, rounds)
    for split in ("tune", "test"):
        _, X, _ = load_split(split)
        saved = [r["p_default"] for r in recs if r["split"] == split]
        np.testing.assert_allclose(model.predict(X), saved, rtol=1e-9)


def test_baselines_never_saw_demographics():
    cols = set(json.loads(MANIFEST.read_text("utf-8"))["columns"])
    assert not {"SEX", "EDUCATION", "MARRIAGE", "AGE"} & cols
