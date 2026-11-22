"""False-base-rate check: does GPT-6.2 Sol listen to the base-rate sentence?

Sol recognizes the dataset from any summary and knows its default rate (22.12%), and the true
base-rate sentence barely moves its average (0.260 out of the box, 0.253 told 22.1%). Two
readings fit that: it already uses the rate it knows, or it ignores the sentence. This check
tells it a different rate. On a fixed random sample of test customers it sends the `base_rate`
request exactly, except that the sentence says "about 10.0%" instead of "about 22.1%".

- If Sol's average moves toward 10%, it listens to the sentence, so its out-of-the-box closeness
  to the true rate comes from what it already knows.
- If its average stays put, it ignores stated rates.

Pass-through = (mean told 10% - mean told 22.1%) / (0.10 - 0.221): 1 = follows the sentence
fully, 0 = ignores it. The false rate is a diagnostic only, never a contestant condition: output
goes to results/checks/, and the headline results are unchanged.

    # 500 test customers (fixed seed), resumable
    uv run --group runners python runners/checks/sol_false_rate.py --workers 4
    # compare with the same customers' real base-rate and out-of-the-box predictions
    uv run --group runners python runners/checks/sol_false_rate.py --report
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "analysis"))

from frontier_verbal import DEFAULT_RPM, QUESTION_ID, FrontierVerbalPredictor  # noqa: E402
from harness import ROOT, build_parser, load_requests, resolve_args, run  # noqa: E402

TRUE_RATE, FALSE_RATE = 0.221, 0.100
TRUE_SENTENCE = "In this portfolio, about 22.1% of customers miss next month's payment."
FALSE_SENTENCE = TRUE_SENTENCE.replace("22.1%", "10.0%")
N_CUSTOMERS = 500
SAMPLE_SEED = 20261123
CHECK_DIR = ROOT / "results" / "checks" / "false_rate"
SUMMARY_JSON = ROOT / "results" / "checks" / "sol_false_rate_summary.json"


def with_false_rate(request: dict) -> dict:
    """The base-rate request with the sentence's rate changed; nothing else differs."""
    out = copy.deepcopy(request)
    q = out["questions"][QUESTION_ID]
    if TRUE_SENTENCE not in q["instructions"]:
        raise ValueError("not a base-rate request: the 22.1% sentence is missing")
    q["instructions"] = q["instructions"].replace(TRUE_SENTENCE, FALSE_SENTENCE)
    return out


def sample(rows: list[dict], n: int = N_CUSTOMERS) -> list[dict]:
    return sorted(random.Random(SAMPLE_SEED).sample(rows, n), key=lambda r: r["id"])


class FalseRateSol(FrontierVerbalPredictor):
    contestant = "frontier-verbal-false-rate-10"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_version = f"{self.model_version}; base-rate sentence says 10.0%"
        self.meta = {**self.meta, "false_sentence": FALSE_SENTENCE, "true_sentence": TRUE_SENTENCE,
                     "sample_seed": SAMPLE_SEED, "n_customers": N_CUSTOMERS}  # fmt: skip

    def predict(self, request: dict):
        return super().predict(with_false_rate(request))


def report(n_boot: int = 2000, seed: int = SAMPLE_SEED) -> dict:
    import metrics as m
    from records import read_records

    sol = [r for r in read_records("frontier-verbal") if r["split"] == "test"]
    true_rate = {r["id"]: r for r in sol if r["condition"] == "base_rate"}
    oob = {r["id"]: r for r in sol if r["condition"] == "out_of_box"}
    false_rate = {r["id"]: r for r in read_records(FalseRateSol.contestant, CHECK_DIR)}
    ids = sorted(false_rate)
    y = np.array([true_rate[i]["label"] for i in ids], dtype=float)
    p = {
        "out_of_box": np.array([oob[i]["p_default"] for i in ids]),
        "told_22.1%": np.array([true_rate[i]["p_default"] for i in ids]),
        "told_10.0%": np.array([false_rate[i]["p_default"] for i in ids]),
    }

    def pass_through(a, b):  # a: told 10%, b: told 22.1%
        return (a.mean() - b.mean()) / (FALSE_RATE - TRUE_RATE)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        boot.append(pass_through(p["told_10.0%"][idx], p["told_22.1%"][idx]))
    shift = p["told_10.0%"] - p["told_22.1%"]
    result = {
        "n": len(ids),
        "observed_rate": float(y.mean()),
        "mean_predicted": {k: float(v.mean()) for k, v in p.items()},
        "median_predicted": {k: float(np.median(v)) for k, v in p.items()},
        "pass_through": {
            "estimate": float(pass_through(p["told_10.0%"], p["told_22.1%"])),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
        },  # fmt: skip
        "share_lower_when_told_10": float(np.mean(shift < 0)),
        "share_unchanged": float(np.mean(shift == 0)),
        "share_higher_when_told_10": float(np.mean(shift > 0)),
        "metrics": {
            k: {"auc": m.auc(y, v), "ece": m.ece(y, v), "citl": m.calibration_in_the_large(y, v)}
            for k, v in p.items()
        },  # fmt: skip
        "cost_usd": float(sum(false_rate[i]["cost_usd"] for i in ids)),
        "false_sentence": FALSE_SENTENCE,
    }
    SUMMARY_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"n={result['n']}  actual default rate {result['observed_rate']:.3f}")
    for k in p:
        mt = result["metrics"][k]
        print(f"{k:<12} mean p {result['mean_predicted'][k]:.3f}  median "
              f"{result['median_predicted'][k]:.3f}  AUC {mt['auc']:.3f}  "
              f"ECE {mt['ece']:.3f}")  # fmt: skip
    pt = result["pass_through"]
    print(f"pass-through {pt['estimate']:.2f} [{pt['lo']:.2f}, {pt['hi']:.2f}]  "
          f"(1 = follows the sentence, 0 = ignores it)")  # fmt: skip
    print(f"told 10% vs 22.1%: lower {result['share_lower_when_told_10']:.0%}, unchanged "
          f"{result['share_unchanged']:.0%}, higher "
          f"{result['share_higher_when_told_10']:.0%}")  # fmt: skip
    print(f"cost ${result['cost_usd']:.2f}; saved -> {SUMMARY_JSON}")
    return result


def main() -> None:
    ap = build_parser(__doc__)
    ap.set_defaults(split="test", condition="base_rate")
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM, help="requests-per-minute cap")
    ap.add_argument("--report", action="store_true", help="compare with the real base-rate run")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.condition != "base_rate" or args.split != "test":
        ap.error("the false-base-rate check runs on test, base_rate requests only")
    if args.out_dir is None and not args.smoke:
        args.out_dir = CHECK_DIR
    args = resolve_args(args)
    from env import load_env

    load_env()
    rows = load_requests("test")
    run(FalseRateSol(rpm=args.rpm), args, requests=sample(rows)[: args.limit or None])


if __name__ == "__main__":
    main()
