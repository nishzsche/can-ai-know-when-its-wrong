"""Robustness checks the review panel asked for (2026-11-23). Descriptive; nothing is tuned on test.

    uv run python analysis/robustness.py   # results/robustness.json, plus a printed summary

1. Paired AUC: scorecard minus each AI contestant x condition on the same test customers, with
   a paired bootstrap (2,000 resamples) and a Holm correction over the AI rows.
2. Calibration measured several ways: ECE with 10 equal-width bins (the headline) and with 10
   equal-mass bins (deciles), Brier score, and a Hosmer-Lemeshow test by decile. Plus the ECE a
   perfectly calibrated model would score here ("noise floor"): simulate outcomes from each
   model's own probabilities and take the median and 95th percentile.
3. Approval view: approve the 80% of customers each model rates safest; default rate among them.
4. Label-free shift: move each AI model's predictions (on the logit scale) so their average
   equals the portfolio's known default rate (22.1%, from train). This needs no outcome per
   customer, only the overall rate, and leaves ranking unchanged.
5. A scorecard trained on the 2,000 tune customers only (the labels the AI models' recalibration
   used), scored on test.

Predictions come from results/predictions/; the small-data scorecard reads data/splits/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))

import metrics as m  # noqa: E402
from records import ROOT, read_records  # noqa: E402

OUT = ROOT / "results" / "robustness.json"
AI = ("jev", "kev-4b", "kev-0.8b", "laya", "qwen-logprobs", "frontier-verbal")
BASE_RATE = 0.221  # train default rate (22.12%); also what the base-rate sentence states
SEED = 20261123


def load(contestant: str, condition: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    recs = sorted((r for r in read_records(contestant)
                   if r["split"] == "test" and r["condition"] == condition),
                  key=lambda r: r["id"])  # fmt: skip
    return (np.array([r["label"] for r in recs], float),
            np.array([r["p_default"] for r in recs], float), [r["id"] for r in recs])  # fmt: skip


def ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        total += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total / len(y))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    from scipy.stats import chi2

    order = np.argsort(p, kind="stable")
    stat = 0.0
    for chunk in np.array_split(order, n_bins):
        n, obs, exp = len(chunk), y[chunk].sum(), p[chunk].sum()
        pbar = min(max(exp / n, 1e-9), 1 - 1e-9)
        stat += (obs - exp) ** 2 / (n * pbar * (1 - pbar))
    return {"statistic": float(stat), "p_value": float(chi2.sf(stat, n_bins - 2))}


def ece_noise_floor(p: np.ndarray, sims: int = 500, seed: int = SEED) -> dict:
    """ECE a perfectly calibrated model with these probabilities would score on n customers."""
    rng = np.random.default_rng(seed)
    vals = [m.ece((rng.uniform(size=len(p)) < p).astype(float), p) for _ in range(sims)]
    return {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95))}


def approval(y: np.ndarray, p: np.ndarray, rate: float = 0.8) -> float:
    keep = np.argsort(p, kind="stable")[: int(round(rate * len(p)))]
    return float(y[keep].mean())


def shift_to_rate(p: np.ndarray, target: float = BASE_RATE) -> np.ndarray:
    """Add one constant on the logit scale so the average prediction equals `target`."""
    from scipy.optimize import brentq

    z = m.logit(p)
    c = brentq(lambda c: (1 / (1 + np.exp(-(z + c)))).mean() - target, -20, 20)
    return 1 / (1 + np.exp(-(z + c)))


def paired_auc(y, p_ref, p_other, n_boot: int = 2000, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(m.auc(y[idx], p_ref[idx]) - m.auc(y[idx], p_other[idx]))
    diffs = np.array(diffs)
    est = m.auc(y, p_ref) - m.auc(y, p_other)
    # two-sided bootstrap p-value for "no difference"
    pval = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"difference": float(est), "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)), "p_value": pval}  # fmt: skip


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def small_scorecard() -> dict:
    """WoE scorecard trained on the 2,000 tune customers only, scored on test."""
    from common import load_split
    from scorecard import fit_scorecard, predict

    _, X_tune, y_tune = load_split("tune")
    _, X_test, y_test = load_split("test")
    binner, model = fit_scorecard(X_tune, y_tune)
    p = predict(binner, model, X_test)
    y = np.asarray(y_test, float)
    return {"trained_on": "tune (2,000)", "auc": m.auc(y, p), "ece": m.ece(y, p),
            "ece_equal_mass": ece_equal_mass(y, p), "brier": m.brier(y, p)}  # fmt: skip


def main() -> None:
    y, p_card, ids = load("scorecard", "trained")
    rows = {"scorecard/trained": p_card}
    _, p_gbm, _ = load("gbm", "trained")
    rows["gbm/trained"] = p_gbm
    for c in AI:
        for cond in ("out_of_box", "base_rate", "recalibrated"):
            yy, pp, ii = load(c, cond)
            assert ii == ids and np.array_equal(yy, y), f"{c}/{cond} customers differ"
            rows[f"{c}/{cond}"] = pp

    calibration = {}
    for name, p in rows.items():
        calibration[name] = {
            "ece_equal_width": m.ece(y, p), "ece_equal_mass": ece_equal_mass(y, p),
            "brier": m.brier(y, p), "auc": m.auc(y, p),
            "hosmer_lemeshow": hosmer_lemeshow(y, p), "approval_80_default_rate": approval(y, p),
            "ece_noise_floor": ece_noise_floor(p),
        }  # fmt: skip

    auc_tests = {name: paired_auc(y, p_card, rows[name])
                 for name in rows if name.split("/")[1] in ("out_of_box", "base_rate")}  # fmt: skip
    adjusted = holm({k: v["p_value"] for k, v in auc_tests.items()})
    for k in auc_tests:
        auc_tests[k]["p_holm"] = adjusted[k]

    shifted = {}
    for c in AI:
        p = rows[f"{c}/out_of_box"]
        q = shift_to_rate(p)
        shifted[c] = {"ece_before": m.ece(y, p), "ece_after": m.ece(y, q),
                      "ece_equal_mass_after": ece_equal_mass(y, q),
                      "ece_told_base_rate": m.ece(y, rows[f"{c}/base_rate"]),
                      "ece_recalibrated": m.ece(y, rows[f"{c}/recalibrated"])}  # fmt: skip

    result = {"n": len(y), "observed_rate": float(y.mean()), "calibration": calibration,
              "paired_auc_vs_scorecard": auc_tests, "label_free_shift": shifted,
              "scorecard_on_2000": small_scorecard()}  # fmt: skip
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("Paired AUC, scorecard minus model (out of box / base rate):")
    for k, v in sorted(auc_tests.items(), key=lambda kv: kv[1]["difference"]):
        print(f"  {k:<28}{v['difference']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  "
              f"p {v['p_value']:.3f}  Holm {v['p_holm']:.3f}")  # fmt: skip
    print("\nCalibration (equal-width ECE | equal-mass ECE | floor p95 | HL p | approve-80% DR):")
    for k, v in calibration.items():
        print(f"  {k:<28}{v['ece_equal_width']:.3f} | {v['ece_equal_mass']:.3f} | "
              f"{v['ece_noise_floor']['p95']:.3f} | {v['hosmer_lemeshow']['p_value']:.3f} | "
              f"{v['approval_80_default_rate']:.3f}")  # fmt: skip
    print("\nLabel-free shift to 22.1% (ECE before -> after; told base rate; recalibrated):")
    for c, v in shifted.items():
        print(f"  {c:<16}{v['ece_before']:.3f} -> {v['ece_after']:.3f}   "
              f"told {v['ece_told_base_rate']:.3f}  recal {v['ece_recalibrated']:.3f}")  # fmt: skip
    s = result["scorecard_on_2000"]
    print(f"\nScorecard trained on 2,000 tune customers: AUC {s['auc']:.3f}, ECE {s['ece']:.3f}, "
          f"equal-mass ECE {s['ece_equal_mass']:.3f}")  # fmt: skip
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
