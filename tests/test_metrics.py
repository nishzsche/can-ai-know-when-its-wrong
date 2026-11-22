"""Tests for analysis/metrics.py: hand-computed cases plus cross-checks against scikit-learn."""

import math

import metrics as m
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

RNG = np.random.default_rng(0)


def _synthetic(n=4000, slope=1.0, shift=0.0, seed=1):
    """y drawn from true risks q; p = sigmoid(slope * logit(q) + shift)."""
    rng = np.random.default_rng(seed)
    q = 1 / (1 + np.exp(-rng.normal(-1.3, 1.0, n)))
    y = (rng.random(n) < q).astype(int)
    p = 1 / (1 + np.exp(-(slope * m.logit(q) + shift)))
    return y, p


# ---- input checks -------------------------------------------------------------------------------


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        m.auc([0, 1], [0.5])
    with pytest.raises(ValueError):
        m.brier([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.ece([0, 1], [0.1, 1.2])
    with pytest.raises(ValueError):
        m.auc([1, 1], [0.1, 0.2])


# ---- ranking ------------------------------------------------------------------------------------


def test_auc_hand_cases():
    assert m.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert m.auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == 0.0
    assert m.auc([0, 1], [0.5, 0.5]) == 0.5  # a tie counts half
    assert m.gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0


def test_auc_matches_sklearn_with_ties():
    y, p = _synthetic()
    p = np.round(p, 2)  # force many ties
    assert m.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_ks_hand_case_and_bounds():
    # perfect separation -> 1; identical scores -> 0
    assert m.ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert m.ks([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.0
    # 0: {0.1, 0.4}, 1: {0.3, 0.9}: at t=0.1 gap 0.5; at t=0.3 F0=0.5, F1=0.5; at 0.4 F0=1, F1=.5
    assert m.ks([0, 0, 1, 1], [0.1, 0.4, 0.3, 0.9]) == 0.5


def test_ks_matches_roc_definition():
    from sklearn.metrics import roc_curve

    y, p = _synthetic(seed=3)
    fpr, tpr, _ = roc_curve(y, p)
    assert m.ks(y, p) == pytest.approx(np.max(tpr - fpr), abs=1e-12)


# ---- honesty ------------------------------------------------------------------------------------


def test_brier_matches_sklearn():
    y, p = _synthetic()
    assert m.brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_ece_hand_case():
    # bin 0: p=0.05 (y=0), bin 9: p=0.95, 0.95 (y=1, 0)
    y = [0, 1, 0]
    p = [0.05, 0.95, 0.95]
    expected = (1 * abs(0.05 - 0) + 2 * abs(0.95 - 0.5)) / 3
    assert m.ece(y, p) == pytest.approx(expected)
    assert m.ece([1], [1.0]) == 0.0  # p = 1.0 lands in the last bin


def test_citl_and_means():
    assert m.calibration_in_the_large([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)
    assert m.mean_predicted([0, 1], [0.2, 0.4]) == pytest.approx(0.3)
    assert m.observed_rate([0, 1, 1, 1]) == 0.75


def test_calibration_slope_recovers_known_miscalibration():
    for slope, shift in [(1.0, 0.0), (0.5, 0.0), (2.0, 0.0), (1.0, 0.8)]:
        y, p = _synthetic(n=40000, slope=slope, shift=shift, seed=7)
        b, a = m.calibration_slope_intercept(y, p)
        # p = sigmoid(slope*z + shift), z the true logit  =>  z = (logit p - shift) / slope
        assert b == pytest.approx(1 / slope, rel=0.06)
        if slope == 1.0:
            assert a == pytest.approx(-shift, abs=0.05)


def test_calibration_slope_matches_sklearn_unpenalised():
    y, p = _synthetic(seed=11, slope=0.7)
    lr = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(m.logit(p)[:, None], y)
    b, _ = m.calibration_slope_intercept(y, p)
    assert b == pytest.approx(lr.coef_[0][0], rel=1e-5)


def test_calibration_intercept_is_offset_fit():
    # Constant p = 0.5 (offset 0) and 20% defaults: the intercept is logit(0.2).
    y = np.array([0] * 80 + [1] * 20)
    p = np.full(100, 0.5)
    _, a = m.calibration_slope_intercept(np.r_[y, 0, 1], np.r_[p, 0.4, 0.6])  # slope needs spread
    a_const = m._fit_logistic(np.ones((100, 1)), y.astype(float), offset=m.logit(p))[0]
    assert a_const == pytest.approx(math.log(0.2 / 0.8), abs=1e-8)
    assert a == pytest.approx(math.log(21 / 81), abs=0.02)


def test_logit_metrics_survive_exact_zero_and_one():
    y = [0, 1, 0, 1, 1, 0]
    p = [0.0, 1.0, 0.2, 0.7, 0.9, 0.1]
    b, a = m.calibration_slope_intercept(y, p)
    assert math.isfinite(b) and math.isfinite(a)


def test_reliability_by_decile():
    y = np.array([0] * 9 + [1] + [0] * 5 + [1] * 5)
    p = np.r_[np.full(10, 0.1), np.full(10, 0.5)]
    rows = m.reliability_by_decile(y, p, n_groups=2)
    assert [r["n"] for r in rows] == [10, 10]
    assert rows[0]["promised"] == pytest.approx(0.1) and rows[0]["actual"] == pytest.approx(0.1)
    assert rows[1]["promised"] == pytest.approx(0.5) and rows[1]["actual"] == pytest.approx(0.5)
    rows10 = m.reliability_by_decile(*_synthetic())
    assert len(rows10) == 10 and sum(r["n"] for r in rows10) == 4000
    assert all(a["p_max"] <= b["p_min"] for a, b in zip(rows10[:-1], rows10[1:], strict=True))


# ---- decision view ------------------------------------------------------------------------------


def test_risk_bands_edges():
    y = [0, 0, 1, 1, 1]
    p = [0.05, 0.10, 0.19, 0.40, 1.0]
    rows = m.risk_bands(y, p)
    assert [r["n"] for r in rows] == [1, 2, 0, 2]  # 0.10 -> 10-20%; 0.40 and 1.0 -> >=40%
    assert rows[2]["promised"] is None and rows[2]["share"] == 0
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
    assert rows[1]["actual"] == pytest.approx(0.5)


def test_approval():
    y = [0, 1, 0, 1]
    p = [0.05, 0.12, 0.18, 0.5]
    r = m.approval(y, p, 0.15)
    assert r["approval_rate"] == 0.5 and r["default_rate_approved"] == 0.5
    assert m.approval(y, p, 0.01)["default_rate_approved"] is None


# ---- uncertainty --------------------------------------------------------------------------------


def test_bootstrap_ci_covers_point_estimate_and_is_reproducible():
    y, p = _synthetic(n=2000)
    lo, hi = m.bootstrap_ci(m.auc, y, p, n_boot=300)
    assert lo < m.auc(y, p) < hi
    assert (lo, hi) == m.bootstrap_ci(m.auc, y, p, n_boot=300)


def test_summarize_keys():
    y, p = _synthetic(n=1000)
    s = m.summarize(y, p, n_boot=50)
    for key in ("auc", "gini", "ks", "brier", "ece", "citl", "cal_slope", "cal_intercept",
                "auc_lo", "auc_hi", "ece_lo", "ece_hi", "citl_lo", "citl_hi"):  # fmt: skip
        assert math.isfinite(s[key])


# ---- recalibration ------------------------------------------------------------------------------

import recalibrate as rc  # noqa: E402


def test_platt_fixes_overconfidence_and_isotonic_is_monotone():
    y, p = _synthetic(n=6000, slope=2.0, shift=0.5, seed=5)
    platt = rc.Platt().fit(p, y)
    assert platt.b == pytest.approx(0.5, rel=0.1)
    q = platt.predict(p)
    assert abs(m.calibration_in_the_large(y, q)) < 0.01
    iso = rc.Isotonic().fit(p, y)
    order = np.argsort(p)
    assert np.all(np.diff(iso.predict(p[order])) >= 0)


def test_choice_uses_cv_on_tune_only():
    y, p = _synthetic(n=2000, slope=0.6, seed=9)
    model, info = rc.choose_and_fit(p, y.astype(float))
    assert info["chosen"] in ("platt", "isotonic")
    assert info["cv_log_loss"][info["chosen"]] == min(info["cv_log_loss"].values())
    # a logit-linear miscalibration is exactly Platt's family, so Platt should win
    assert info["chosen"] == "platt"


# ---- analysis/explain.py ---------------------------------------------------------------------

import explain  # noqa: E402


def test_weighted_ece_matches_ece_with_equal_weights():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p).astype(float)
    assert explain.weighted_ece(y, p, np.ones(500)) == pytest.approx(m.ece(y, p))


def test_reweighting_hits_the_target_rate_and_keeps_predictions():
    import pandas as pd

    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.99, 400)
    y = (rng.uniform(size=400) < 0.25).astype(int)
    df = pd.DataFrame({"id": range(400), "p": p, "y": y})
    rows = explain.base_rate_sensitivity({"m": df})
    observed = next(r for r in rows if r["default_rate"] == 0.221)
    assert observed["m"]["mean_predicted"] == pytest.approx(p.mean(), abs=0.02)
    for r in rows:
        target = r["default_rate"]
        w = np.where(y == 1, target / y.mean(), (1 - target) / (1 - y.mean()))
        assert np.average(y, weights=w) == pytest.approx(target)
