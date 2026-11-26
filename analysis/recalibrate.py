"""The `recalibrated` condition: each AI contestant's out-of-the-box probabilities, recalibrated.

PLAN.md: "condition 1's probabilities, recalibrated on `tune` (Platt scaling or isotonic
regression)". No new model calls: the calibrator is fitted on the saved `tune` out_of_box
probabilities and applied to the saved `test` out_of_box probabilities.

Method choice, made on tune only: for each contestant, 5-fold stratified cross-validation on tune
compares Platt scaling (logistic regression on logit(p), slope and intercept; temperature scaling
is its no-intercept special case) with isotonic regression, by log loss. The lower one is refitted
on all of tune and applied to test. The choice and CV scores go in results/recalibration.json and
in each record's model_version. Test labels are never used.

Recalibrated records copy latency and cost from the out_of_box test record they come from
(the calibrator itself costs nothing measurable). Re-running replaces earlier recalibrated
records, so it's safe to run again when a contestant (e.g. Sol) completes.

    uv run python analysis/recalibrate.py

M6 (PLAN.md §11): the contestants that learned from `train` get `trained_calibrated` the same way,
from their `trained` probabilities, exactly as the baselines' calibrated condition is a fit on
tune. Contestants whose runs aren't complete are skipped and picked up on a later run.

Post 4 (PLAN.md §12): each fine-tuned model at each precision (runners/quantize.py, e.g.
`kev-0.8b-ft-bnb-nf4`) gets `trained_calibrated` the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from metrics import EPS, logit  # noqa: E402
from quantize import all_contestants  # noqa: E402
from records import ROOT, make_record, read_records, utc_now, write_records  # noqa: E402
from speed import FULL_RUN, all_speed_contestants  # noqa: E402

AI_CONTESTANTS = ("jev", "kev-0.8b", "kev-4b", "laya", "qwen-logprobs", "frontier-verbal")
M6_CONTESTANTS = ("kev-4b-ft", "kev-0.8b-ft", "qwen-ft", "laya-ft", "jev-examples",
                  "sol-examples", "neighbours-20", "kumo-tabular", "tabicl-v2")  # fmt: skip
QUANT_CONTESTANTS = all_contestants() + all_speed_contestants(FULL_RUN)  # post 4 and its addendum
N_CUSTOMERS = 2000
CV_FOLDS = 5
SEED = 20261122
OUT_JSON = ROOT / "results" / "recalibration.json"


class Platt:
    name = "platt"

    def fit(self, p, y):
        from metrics import _fit_logistic

        x = np.column_stack([np.ones(len(p)), logit(p)])
        self.a, self.b = (float(v) for v in _fit_logistic(x, np.asarray(y, float)))
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-(self.a + self.b * logit(p))))

    def params(self):
        return {"intercept": self.a, "slope": self.b}


class Isotonic:
    name = "isotonic"

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression

        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)

    def params(self):
        return {"n_thresholds": int(len(self.iso.X_thresholds_))}


def log_loss(y, q) -> float:
    q = np.clip(q, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def cv_log_loss(method_cls, p, y, folds: int = CV_FOLDS, seed: int = SEED) -> float:
    from sklearn.model_selection import StratifiedKFold

    losses = []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(p, y):
        model = method_cls().fit(p[tr], y[tr])
        losses.append(log_loss(y[va], model.predict(p[va])))
    return float(np.mean(losses))


def choose_and_fit(p_tune, y_tune) -> tuple[object, dict]:
    scores = {cls.name: cv_log_loss(cls, p_tune, y_tune) for cls in (Platt, Isotonic)}
    best = min(scores, key=scores.get)
    model = {"platt": Platt, "isotonic": Isotonic}[best]().fit(p_tune, y_tune)
    return model, {"cv_log_loss": scores, "chosen": best, "params": model.params()}


def recalibrate(
    contestant: str, source: str = "out_of_box", target: str = "recalibrated"
) -> dict | None:
    records = read_records(contestant)
    base = [r for r in records if r["condition"] != target]
    tune = sorted((r for r in base if r["condition"] == source and r["split"] == "tune"),
                  key=lambda r: r["id"])  # fmt: skip
    test = sorted((r for r in base if r["condition"] == source and r["split"] == "test"),
                  key=lambda r: r["id"])  # fmt: skip
    if len(tune) != N_CUSTOMERS or len(test) != N_CUSTOMERS:
        print(f"{contestant}: skipped ({source} tune {len(tune)}, test {len(test)})")
        return None
    p_tune = np.array([r["p_default"] for r in tune])
    y_tune = np.array([r["label"] for r in tune], dtype=float)
    model, info = choose_and_fit(p_tune, y_tune)
    q_test = np.clip(model.predict(np.array([r["p_default"] for r in test])), 0.0, 1.0)

    versions = sorted({r["model_version"] for r in test})
    suffix = f"{target}: {info['chosen']} fit on tune {source} ({CV_FOLDS}-fold CV choice)"
    ts = utc_now()
    new = [
        make_record(
            id=r["id"], split="test", contestant=contestant, condition=target,
            model_version=f"{r['model_version']}; {suffix}", p_default=float(q),
            label=r["label"], latency_ms=r["latency_ms"], cost_usd=r["cost_usd"], ts=ts,
        )
        for r, q in zip(test, q_test, strict=True)
    ]  # fmt: skip
    write_records(contestant, base + new)
    print(
        f"{contestant}: {info['chosen']} (CV log loss platt {info['cv_log_loss']['platt']:.4f}, "
        f"isotonic {info['cv_log_loss']['isotonic']:.4f}); wrote {len(new)} test records"
    )
    return {**info, "source_model_versions": versions, "fitted_at": ts}


def main() -> None:
    summary = json.loads(OUT_JSON.read_text("utf-8")) if OUT_JSON.exists() else {}
    jobs = [(c, "out_of_box", "recalibrated") for c in AI_CONTESTANTS]
    jobs += [(c, "trained", "trained_calibrated") for c in M6_CONTESTANTS + QUANT_CONTESTANTS]
    for contestant, source, target in jobs:
        if not (ROOT / "results" / "predictions" / f"{contestant}.jsonl").exists():
            print(f"{contestant}: no predictions file yet")
            continue
        info = recalibrate(contestant, source, target)
        if info:
            summary[contestant] = info
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
