"""Metrics for "can AI know when it's wrong?" (PLAN.md section 4). Pure numpy, unit-tested.

Ranking: AUC, Gini, KS.
Honesty: reliability by risk decile, calibration-in-the-large, calibration slope and intercept,
ECE (10 equal-width bins), Brier score.
Decision view: risk bands (promised vs actual), "approve if predicted risk < X".
Uncertainty: percentile bootstrap intervals.

Conventions:
- `y` is 0/1 (1 = defaulted), `p` is the predicted probability of default.
- Logit-based metrics clip p to [EPS, 1 - EPS] first: isotonic calibration can output exact
  0 or 1, and some models give probabilities like 4e-5.
- Calibration slope comes from the logistic fit  logit P(y=1) = a + b * logit(p).
  Calibration intercept is the "calibration-in-the-large" intercept: the same fit with the
  slope fixed at 1 (logit(p) as an offset), as in Van Calster et al. (2019). A perfectly
  calibrated model has slope 1 and intercept 0. Slope < 1: predictions too extreme
  (overconfident); slope > 1: too timid. Intercept < 0: predictions too high on average.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

EPS = 1e-6
RISK_BANDS = (0.0, 0.10, 0.20, 0.40, 1.0)  # <10%, 10-20%, 20-40%, >=40%
APPROVAL_THRESHOLDS = (0.10, 0.15, 0.20)


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y and p must be 1-D and the same length, got {y.shape}, {p.shape}")
    if len(y) == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("p must be finite and in [0, 1]")
    return y, p


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---- ranking ------------------------------------------------------------------------------------


def auc(y, p) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks, so ties count half)."""
    y, p = _arrays(y, p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    # average ranks over tied groups
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between the score distributions of defaulters and payers."""
    y, p = _arrays(y, p)
    thresholds = np.unique(p)
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("KS needs both classes")
    cdf_pos = np.searchsorted(pos, thresholds, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, thresholds, side="right") / len(neg)
    return float(np.max(np.abs(cdf_neg - cdf_pos)))


# ---- honesty ------------------------------------------------------------------------------------


def brier(y, p) -> float:
    y, p = _arrays(y, p)
    return float(np.mean((p - y) ** 2))


def mean_predicted(y, p) -> float:
    _, p = _arrays(y, p)
    return float(p.mean())


def observed_rate(y, p=None) -> float:
    y = np.asarray(y, dtype=float)
    return float(y.mean())


def calibration_in_the_large(y, p) -> float:
    """Mean predicted minus observed default rate (positive = promises more defaults than occur)."""
    y, p = _arrays(y, p)
    return float(p.mean() - y.mean())


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins: sum of n_b/N * |mean p_b - mean y_b|."""
    y, p = _arrays(y, p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)  # p = 1.0 goes in the last bin
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(y))


def _fit_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """Unpenalised logistic regression by Newton-Raphson. x: (n, k) design incl. intercept."""
    beta = np.zeros(x.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(100):
        eta = np.clip(x @ beta + off, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        grad = x.T @ (y - mu)
        hess = (x * w[:, None]).T @ x
        step = np.linalg.solve(hess + 1e-10 * np.eye(len(beta)), grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """(slope, intercept): slope from logit P = a + b*logit(p); intercept with slope fixed at 1."""
    y, p = _arrays(y, p)
    lp = logit(p)
    ones = np.ones((len(y), 1))
    slope = float(_fit_logistic(np.column_stack([ones, lp]), y)[1])
    intercept = float(_fit_logistic(ones, y, offset=lp)[0])
    return slope, intercept


def reliability_by_decile(y, p, n_groups: int = 10) -> list[dict]:
    """Customers sorted by predicted risk and cut into equal-size groups (risk deciles).

    For each group: n, mean predicted rate ("promised") and actual default rate. Ties in p are
    broken by input order (stable sort), so the result is deterministic.
    """
    y, p = _arrays(y, p)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for k, chunk in enumerate(np.array_split(order, n_groups), start=1):
        rows.append({
            "decile": k,
            "n": int(len(chunk)),
            "promised": float(p[chunk].mean()),
            "actual": float(y[chunk].mean()),
            "p_min": float(p[chunk].min()),
            "p_max": float(p[chunk].max()),
        })  # fmt: skip
    return rows


# ---- decision view ------------------------------------------------------------------------------


def risk_bands(y, p, edges: Sequence[float] = RISK_BANDS) -> list[dict]:
    """Share of customers per predicted-risk band, with promised vs actual default rate.

    Bands are [lo, hi), except the last, which includes 1.0. Empty bands are reported with
    n = 0 and None rates.
    """
    y, p = _arrays(y, p)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = i == len(edges) - 2
        mask = (p >= lo) & ((p <= hi) if last else (p < hi))
        n = int(mask.sum())
        rows.append({
            "band": f"{lo:.0%}–{hi:.0%}" if not last else f"≥{lo:.0%}",
            "lo": lo, "hi": hi, "n": n, "share": n / len(y),
            "promised": float(p[mask].mean()) if n else None,
            "actual": float(y[mask].mean()) if n else None,
        })  # fmt: skip
    return rows


def approval(y, p, threshold: float) -> dict:
    """Approve customers with predicted risk < threshold."""
    y, p = _arrays(y, p)
    approved = p < threshold
    n = int(approved.sum())
    return {
        "threshold": threshold,
        "approval_rate": n / len(y),
        "default_rate_approved": float(y[approved].mean()) if n else None,
        "promised_approved": float(p[approved].mean()) if n else None,
        "n_approved": n,
    }


# ---- uncertainty --------------------------------------------------------------------------------


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y,
    p,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 20261122,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling customers with replacement."""
    y, p = _arrays(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():  # a one-class resample can't score AUC/KS; vanishingly rare
            continue
        stats.append(metric(yb, p[idx]))
    alpha = (1 - level) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return float(lo), float(hi)


def summarize(y, p, n_boot: int = 1000) -> dict:
    """Every headline metric for one contestant x condition, with bootstrap intervals."""
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "n": int(len(y)),
        "observed_rate": observed_rate(y),
        "mean_predicted": mean_predicted(y, p),
        "auc": auc(y, p),
        "gini": gini(y, p),
        "ks": ks(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "citl": calibration_in_the_large(y, p),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
    for name, fn in (("auc", auc), ("ece", ece), ("citl", calibration_in_the_large)):
        lo, hi = bootstrap_ci(fn, y, p, n_boot=n_boot)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out
