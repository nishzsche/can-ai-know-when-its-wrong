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
TABULAR = ("kumo-tabular", "tabicl-v2", "tabpfn-3.5")
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


def build() -> tuple[list[dict], list[str]]:
    y_ref, p_card, ids_ref = load("scorecard", "trained")
    _, p_gbm, ids_gbm = load("gbm", "trained")
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
        for ref_name, p_ref in (("scorecard", p_card), ("gbm", p_gbm)):
            if contestant == ref_name:
                continue
            d = paired_auc(y, p_ref, p)
            k = f"auc_gap_vs_{ref_name}"
            row.update({k: d["difference"], f"{k}_lo": d["lo"], f"{k}_hi": d["hi"],
                        f"{k}_p": d["p_value"]})  # fmt: skip
        rows.append(row)
    for ref_name in ("scorecard", "gbm"):
        key = f"auc_gap_vs_{ref_name}_p"
        adjusted = holm({f"{r['contestant']}/{r['condition']}": r[key] for r in rows if key in r})
        for r in rows:
            if key in r:
                r[f"{key}_holm"] = adjusted[f"{r['contestant']}/{r['condition']}"]
    return rows, skipped


def main() -> None:
    rows, skipped = build()
    cols = sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("contestant", "condition", "kind"), k),
    )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps({"rows": rows, "skipped": skipped}, indent=2) + "\n", "utf-8")
    head = ("contestant", "condition", "AUC", "ECE", "mean p", "vs card", "vs gbm")
    print("{:<16} {:<19} {:>6} {:>6} {:>7} {:>8} {:>7}".format(*head))
    for r in rows:
        gap_c = r.get("auc_gap_vs_scorecard")
        gap_g = r.get("auc_gap_vs_gbm")
        vs_c = "" if gap_c is None else f"{-gap_c:+.3f}"
        vs_g = "" if gap_g is None else f"{-gap_g:+.3f}"
        print(f"{r['contestant']:<16} {r['condition']:<19} {r['auc']:>6.3f} {r['ece']:>6.3f} "
              f"{r['mean_predicted']:>7.3f} {vs_c:>8} {vs_g:>7}")  # fmt: skip
    if skipped:
        print("not yet scored: " + ", ".join(skipped))
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} "
          "(AUC columns: the contestant minus the baseline)")  # fmt: skip


if __name__ == "__main__":
    main()
