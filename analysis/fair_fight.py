"""M6: the fair-fight scoreboard (PLAN.md §11).

Every model that learned from the 20,000 `train` customers, next to its own zero-shot result and
the two baselines, on the same 2,000 `test` customers. Descriptive only; nothing is tuned on test.

    uv run python analysis/fair_fight.py      # results/fair_fight.csv and results/fair_fight.json

Rows (whatever is present; a contestant without a complete test set is listed and skipped):
- each AI model: zero-shot (`out_of_box`), then `trained`, then `trained_calibrated`
  (Jev and Sol "trained" = shown their 20 nearest `train` customers);
- tabular foundation models and the `neighbours-20` reference: `trained`, `trained_calibrated`;
- the scorecard (`trained`) and LightGBM (`trained`, `trained_calibrated`).

Metrics: AUC with a bootstrap interval, ECE (10 equal-width bins, plus equal-mass), the ECE a
perfectly calibrated model would score here (noise floor, 95th percentile), calibration-in-the-
large, calibration slope, Brier, and the default rate among the 80% each model rates safest.
Paired bootstrap AUC against the scorecard and against LightGBM, Holm-corrected across rows.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as m  # noqa: E402
from records import ROOT  # noqa: E402
from robustness import (  # noqa: E402
    approval,
    ece_equal_mass,
    ece_noise_floor,
    holm,
    load,
    paired_auc,
)

OUT_CSV = ROOT / "results" / "fair_fight.csv"
OUT_JSON = ROOT / "results" / "fair_fight.json"
N_TEST = 2000
# trained contestant -> its zero-shot self
PAIRS = {
    "kev-4b-ft": "kev-4b",
    "kev-0.8b-ft": "kev-0.8b",
    "qwen-ft": "qwen-logprobs",
    "laya-ft": "laya",
    "jev-examples": "jev",
    "sol-examples": "frontier-verbal",
}
TABULAR = ("kumo-tabular", "tabicl-v2")
REFERENCES = ("neighbours-20",)
BASELINES = {"scorecard": ("trained",), "gbm": ("trained", "trained_calibrated")}
TRAINED = ("trained", "trained_calibrated")


def plan() -> list[tuple[str, str, str]]:
    """(contestant, condition, kind) in display order."""
    rows = [
        (c, cond, "baseline (trained on 20,000)")
        for c, conds in BASELINES.items()
        for cond in conds
    ]
    rows += [
        (c, cond, "reference: 20 nearest train customers") for c in REFERENCES for cond in TRAINED
    ]
    for trained, zero_shot in PAIRS.items():
        rows.append((zero_shot, "out_of_box", "AI, zero-shot"))
        kind = (
            "AI, 20 examples in the prompt"
            if trained.endswith("-examples")
            else "AI, fine-tuned on train"
        )
        rows += [(trained, cond, kind) for cond in TRAINED]
    rows += [
        (c, cond, "tabular foundation model, train in context") for c in TABULAR for cond in TRAINED
    ]
    return rows


def score(y: np.ndarray, p: np.ndarray) -> dict:
    s = m.summarize(y, p, n_boot=1000)
    floor = ece_noise_floor(p)
    return {
        "auc": s["auc"], "auc_lo": s["auc_lo"], "auc_hi": s["auc_hi"],
        "ece": s["ece"], "ece_lo": s["ece_lo"], "ece_hi": s["ece_hi"],
        "ece_equal_mass": ece_equal_mass(y, p),
        "ece_noise_p95": floor.get("p95"),
        "mean_predicted": s["mean_predicted"], "observed_rate": s["observed_rate"],
        "citl": s["citl"], "cal_slope": s["cal_slope"], "brier": s["brier"],
        "default_rate_at_80pct_approval": approval(y, p, 0.8),
    }  # fmt: skip


# The test family (fixed 2026-11-24 after the review panel; the first version corrected over every
# row, zero-shot and duplicate calibrated rows included). Ranking tests use raw `trained` output
# only: the calibration maps are monotone, so `trained_calibrated` has the same AUC.
FAMILY = ("kumo-tabular", "tabicl-v2", "kev-4b-ft", "kev-0.8b-ft", "qwen-ft", "laya-ft",
          "jev-examples", "sol-examples")  # fmt: skip


def _gap(row: dict, name: str, d: dict, sign: float = -1.0) -> None:
    """Store a paired difference as contestant minus reference (sign -1 flips paired_auc's
    reference-minus-contestant), with its interval and p-value."""
    lo, hi = sorted((sign * d["lo"], sign * d["hi"]))
    row.update({name: sign * d["difference"], f"{name}_lo": lo, f"{name}_hi": hi,
                f"{name}_p": d["p_value"]})  # fmt: skip


def build() -> tuple[list[dict], list[str]]:
    y_ref, p_card, ids_ref = load("scorecard", "trained")
    _, p_gbm, ids_gbm = load("gbm", "trained")
    _, p_nb, _ = load("neighbours-20", "trained")
    assert ids_ref == ids_gbm and len(ids_ref) == N_TEST, "baseline test sets differ"
    rows, skipped = [], []
    for contestant, condition, kind in plan():
        try:
            y, p, ids = load(contestant, condition)
        except FileNotFoundError:
            skipped.append(f"{contestant}/{condition} (no file)")
            continue
        if len(ids) != N_TEST:
            skipped.append(f"{contestant}/{condition} ({len(ids)} of {N_TEST} test customers)")
            continue
        assert ids == ids_ref and np.array_equal(y, y_ref), f"{contestant}: test customers differ"
        row = {"contestant": contestant, "condition": condition, "kind": kind, **score(y, p)}
        if condition in ("trained", "out_of_box"):
            for ref_name, p_ref in (("scorecard", p_card), ("gbm", p_gbm)):
                if contestant != ref_name:
                    _gap(row, f"auc_vs_{ref_name}", paired_auc(y, p_ref, p))
        if contestant in FAMILY and condition == "trained":
            row["in_test_family"] = True
            _gap(row, "auc_vs_neighbours", paired_auc(y, p_nb, p))
            if contestant in PAIRS:
                _, p_zs, _ = load(PAIRS[contestant], "out_of_box")
                _gap(row, "auc_gain_vs_zero_shot", paired_auc(y, p_zs, p))
        rows.append(row)
    family = [r for r in rows if r.get("in_test_family")]
    for key in ("auc_vs_scorecard", "auc_vs_gbm", "auc_vs_neighbours", "auc_gain_vs_zero_shot"):
        members = [r for r in family if f"{key}_p" in r]
        adjusted = holm({r["contestant"]: r[f"{key}_p"] for r in members})
        for r in members:
            r[f"{key}_p_holm"] = adjusted[r["contestant"]]
    return rows, skipped


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _split(contestant: str, condition: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    from records import read_records

    recs = sorted((r for r in read_records(contestant)
                   if r["split"] == split and r["condition"] == condition),
                  key=lambda r: r["id"])  # fmt: skip
    return (np.array([r["label"] for r in recs], float),
            np.array([r["p_default"] for r in recs], float))  # fmt: skip


def incremental(contestant: str, n_boot: int = 2000, seed: int = 20261124) -> dict:
    """Does the model add anything beyond its 20 neighbours' outcome rate? Two logistic models
    fitted on tune, (a) the neighbours' rate alone and (b) the rate plus the model's own answer,
    then compared on test by log loss, with a paired bootstrap of the difference."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    def feats(split: str):
        y, nb = _split("neighbours-20", "trained", split)
        y2, p = _split(contestant, "trained", split)
        assert np.array_equal(y, y2)
        nb_s = (nb * 20 + 0.5) / 21  # the rate of 20 outcomes, kept off exactly 0 and 1
        return y, _logit(nb_s)[:, None], np.column_stack([_logit(nb_s), _logit(p)])

    y_tu, a_tu, b_tu = feats("tune")
    y_te, a_te, b_te = feats("test")
    fa = LogisticRegression(C=1e6, max_iter=1000).fit(a_tu, y_tu)
    fb = LogisticRegression(C=1e6, max_iter=1000).fit(b_tu, y_tu)
    pa, pb = fa.predict_proba(a_te)[:, 1], fb.predict_proba(b_te)[:, 1]

    def ll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    diff = ll(y_te, pa) - ll(y_te, pb)  # positive: the model's answer helps
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {"log_loss_neighbours_only": float(ll(y_te, pa).mean()),
            "log_loss_with_model": float(ll(y_te, pb).mean()),
            "improvement": float(diff.mean()),
            "improvement_lo": float(np.percentile(boots, 2.5)),
            "improvement_hi": float(np.percentile(boots, 97.5)),
            "auc_neighbours_only": m.auc(y_te, pa), "auc_with_model": m.auc(y_te, pb),
            "spearman_with_neighbours": float(spearmanr(b_te[:, 0], b_te[:, 1])[0])}  # fmt: skip


def temperature_only(contestant: str) -> dict:
    """One temperature fitted on tune (the step Kev's maker recipe ends with), applied to test."""
    from scipy.optimize import minimize_scalar

    y_tu, p_tu = _split(contestant, "trained", "tune")
    y_te, p_te = _split(contestant, "trained", "test")
    z_tu, z_te = _logit(p_tu), _logit(p_te)

    def nll(t: float) -> float:
        q = np.clip(1 / (1 + np.exp(-z_tu / t)), 1e-9, 1 - 1e-9)
        return float(-(y_tu * np.log(q) + (1 - y_tu) * np.log(1 - q)).mean())

    t = float(minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded").x)
    q = 1 / (1 + np.exp(-z_te / t))
    return {"temperature": t, "test_mean_before": float(p_te.mean()),
            "test_mean_after": float(q.mean()), "test_ece_before": m.ece(y_te, p_te),
            "test_ece_after": m.ece(y_te, q)}  # fmt: skip


def main() -> None:
    rows, skipped = build()
    extras = {
        "test_family": list(FAMILY),
        "test_family_note": "Holm over the eight trained AI models' raw output, separately per "
        "comparison; the first version (2026-11-24 morning) corrected over every row, zero-shot "
        "and duplicate calibrated rows included, which made every gap harder to detect.",
        "incremental_over_neighbours": {
            c: incremental(c) for c in ("sol-examples", "jev-examples")
        },
        "temperature_only_on_tune": {c: temperature_only(c) for c in ("kev-4b-ft", "kev-0.8b-ft")},
    }
    cols = sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("contestant", "condition", "kind"), k),
    )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    payload = {"rows": rows, "skipped": skipped, **extras}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    head = ("contestant", "condition", "AUC", "ECE", "mean p", "vs card", "vs gbm")
    print("{:<16} {:<19} {:>6} {:>6} {:>7} {:>8} {:>7}".format(*head))
    for r in rows:
        vs_c = "" if "auc_vs_scorecard" not in r else f"{r['auc_vs_scorecard']:+.3f}"
        vs_g = "" if "auc_vs_gbm" not in r else f"{r['auc_vs_gbm']:+.3f}"
        print(f"{r['contestant']:<16} {r['condition']:<19} {r['auc']:>6.3f} {r['ece']:>6.3f} "
              f"{r['mean_predicted']:>7.3f} {vs_c:>8} {vs_g:>7}")  # fmt: skip
    print("\nTest family (raw trained output; Holm within the eight):")
    for r in rows:
        if r.get("in_test_family"):
            parts = []
            for key, lab in (
                ("auc_vs_scorecard", "card"),
                ("auc_vs_gbm", "gbm"),
                ("auc_vs_neighbours", "nbrs"),
                ("auc_gain_vs_zero_shot", "vs cold"),
            ):
                if key in r:
                    lo, hi = r[key + "_lo"], r[key + "_hi"]
                    p_raw, p_holm = r[key + "_p"], r[key + "_p_holm"]
                    parts.append(f"{lab} {r[key]:+.3f} [{lo:+.3f},{hi:+.3f}] "
                                 f"p {p_raw:.3f} holm {p_holm:.3f}")  # fmt: skip
            print(f"  {r['contestant']:<13} " + " | ".join(parts))
    for c, d in extras["incremental_over_neighbours"].items():
        print(f"{c}: log loss {d['log_loss_neighbours_only']:.4f} -> "
              f"{d['log_loss_with_model']:.4f} (improvement {d['improvement']:+.4f} "
              f"[{d['improvement_lo']:+.4f},{d['improvement_hi']:+.4f}]); "
              f"spearman with neighbours {d['spearman_with_neighbours']:.2f}")  # fmt: skip
    for c, d in extras["temperature_only_on_tune"].items():
        print(f"{c}: temperature {d['temperature']:.2f}; test mean "
              f"{d['test_mean_before']:.3f} -> {d['test_mean_after']:.3f}; "
              f"ECE {d['test_ece_before']:.3f} -> {d['test_ece_after']:.3f}")  # fmt: skip
    if skipped:
        print("not yet scored: " + ", ".join(skipped))
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} "
          "(AUC gaps: the contestant minus the reference)")  # fmt: skip


if __name__ == "__main__":
    main()
