"""Gradient boosting (LightGBM): the "modern ML" reference, raw and isotonic-calibrated.

    uv run python baselines/gbm.py

- The number of boosting rounds is chosen by 5-fold cross-validation inside `train`, so model
  selection never touches `tune` or `test`.
- `trained`: the raw LightGBM probabilities, for `tune` and `test`.
- `trained_calibrated`: isotonic regression fitted on the raw `tune` probabilities, applied to
  `test`. It's only written for `test`, since its tune predictions would be in-sample.

Writes results/predictions/gbm.jsonl and results/baselines/gbm.json (settings, rounds, feature
importances).
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from common import (
    BASELINE_DIR,
    SEED,
    build_records,
    load_split,
    quick_report,
    timed_predict,
    train_fingerprint,
)
from load import FEATURES
from records import write_records
from sklearn.isotonic import IsotonicRegression

CONTESTANT = "gbm"
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


def choose_rounds(X_train, y_train) -> int:
    cv = lgb.cv(
        PARAMS,
        lgb.Dataset(X_train, y_train),
        num_boost_round=MAX_ROUNDS,
        nfold=5,
        stratified=True,
        seed=SEED,
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return len(cv["valid binary_logloss-mean"])


def fit_gbm(X_train, y_train, rounds: int) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(X_train, y_train), num_boost_round=rounds)


def fit_isotonic(p_tune: np.ndarray, y_tune: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_tune, y_tune)


def main() -> None:
    _, X_train, y_train = load_split("train")
    rounds = choose_rounds(X_train, y_train)
    model = fit_gbm(X_train, y_train, rounds)
    base_version = f"lightgbm {lgb.__version__}; {rounds} rounds; train {train_fingerprint()}"

    df_tune, X_tune, y_tune = load_split("tune")
    df_test, X_test, y_test = load_split("test")
    p_tune, ms_tune = timed_predict(model.predict, X_tune)
    p_test, ms_test = timed_predict(model.predict, X_test)

    iso = fit_isotonic(p_tune, y_tune)
    p_test_cal, ms_cal = timed_predict(iso.predict, p_test)

    print(f"Boosting rounds chosen by 5-fold CV on train: {rounds}")
    print(quick_report("gbm raw (train, in-sample)", "train", model.predict(X_train), y_train))
    print(quick_report("gbm raw", "tune", p_tune, y_tune))
    print(quick_report("gbm raw", "test", p_test, y_test))
    print(quick_report("gbm isotonic (fit on tune)", "test", p_test_cal, y_test))
    extremes = int(((p_test_cal == 0) | (p_test_cal == 1)).sum())
    if extremes:
        print(
            f"Note: {extremes} calibrated test probabilities are exactly 0 or 1 (isotonic steps)."
        )

    common = {"contestant": CONTESTANT}
    records = (
        build_records(df_tune, p_tune, split="tune", condition="trained",
                      model_version=base_version, latency_ms=ms_tune, **common)
        + build_records(df_test, p_test, split="test", condition="trained",
                        model_version=base_version, latency_ms=ms_test, **common)
        + build_records(df_test, p_test_cal, split="test", condition="trained_calibrated",
                        model_version=f"{base_version}; isotonic fit on tune",
                        latency_ms=ms_test + ms_cal, **common)
    )  # fmt: skip
    path = write_records(CONTESTANT, records)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    gain = model.feature_importance(importance_type="gain")
    summary = {
        "model_version": base_version,
        "params": PARAMS,
        "rounds": rounds,
        "feature_importance_gain_share": dict(
            zip(FEATURES, map(float, gain / gain.sum()), strict=True)
        ),
        "isotonic_steps": int(len(iso.X_thresholds_)),
        "calibrated_test_probs_exactly_0_or_1": extremes,
    }
    (BASELINE_DIR / "gbm.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"Wrote {len(records):,} records -> {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
