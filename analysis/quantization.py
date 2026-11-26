"""Post 4: does shrinking a model make it less honest? (PLAN.md §12, post 4 spec)

Each fine-tuned model (Kev-0.8B, Kev-4B, Qwen) at each precision, against its own bf16 run served
through the same code path (`<model>-bf16`), on the same 2,000 `test` customers. Descriptive only;
the lending cut-offs are chosen on `tune`, and nothing is tuned on test.

    uv run python analysis/quantization.py    # -> results/quantization.csv and .json
    uv run python analysis/quantization.py --record-failure kev-4b-ft-hqq-2 --stage load \\
        --error "AssertionError: ..."         # a precision that couldn't run (never substituted)

Per row:
- drift without labels: mean prediction, mean shift and mean |change| per customer against bf16,
  the largest change, and rank correlation with bf16;
- ranking: AUC with a bootstrap interval, and the paired bootstrap difference from the model's own
  bf16 run (Holm over the 15 quantized rows);
- honesty: ECE (10 equal-width bins, and equal-mass), the noise floor a perfectly calibrated model
  would score (95th percentile), calibration-in-the-large and slope, raw and after the same
  correction fitted on tune (`trained_calibrated`, analysis/recalibrate.py); and the change in ECE
  from bf16 with a paired bootstrap, Holm over the 15 quantized rows (added after the review panel);
- lending (post 2, loss ratio 4:1): profit per 1,000 customers approving at the model's own
  percentages (p < 20%), and at the cut-off that maximizes profit on tune;
- decisions (added while writing, after the runs): the share of approve/decline decisions that
  differ from bf16's, each at its own tune cut-off; the same model swapped in at bf16's tune
  cut-off (approval, profit, decisions changed); and the share of customers moved 5+ points;
- cost: median latency, cost per 1,000 decisions at the L4 rate, the GPU memory the model holds
  after loading (weights), and the peak while scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runners"))

import metrics as m  # noqa: E402
from accuracy import profit, tune_cutoff  # noqa: E402
from quantize import PRECISIONS, QUANT_MODELS, QUANTIZED, contestant_name  # noqa: E402
from records import PREDICTIONS_DIR, ROOT, read_records, utc_now  # noqa: E402
from robustness import ece_equal_mass, ece_noise_floor, holm, paired_auc  # noqa: E402

OUT_CSV = ROOT / "results" / "quantization.csv"
OUT_JSON = ROOT / "results" / "quantization.json"
FAILURES = ROOT / "results" / "checks" / "quant_failures.json"
REPRO_DIR = ROOT / "results" / "checks" / "quantization"
N_TEST = 2000
LOSS = 4.0  # post 2's headline loss ratio: one default costs four good customers' profit
GB = 1024**3


def split_arrays(
    contestant: str, condition: str, split: str, directory: Path = PREDICTIONS_DIR
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(ids, labels, probabilities) for one condition and split, sorted by id."""
    path = directory / f"{contestant}.jsonl"
    if not path.exists():
        return [], np.array([]), np.array([])
    recs = sorted((r for r in read_records(contestant, directory)
                   if r["split"] == split and r["condition"] == condition),
                  key=lambda r: r["id"])  # fmt: skip
    return ([r["id"] for r in recs], np.array([r["label"] for r in recs], float),
            np.array([r["p_default"] for r in recs], float))  # fmt: skip


def drift(p_ref: np.ndarray, p: np.ndarray) -> dict:
    """How far one precision's predictions moved from bf16's, without looking at outcomes."""
    from scipy.stats import spearmanr

    d = p - p_ref
    return {"mean_shift_vs_bf16": float(d.mean()),
            "mean_abs_change_vs_bf16": float(np.abs(d).mean()),
            "max_abs_change_vs_bf16": float(np.abs(d).max()),
            "spearman_vs_bf16": float(spearmanr(p_ref, p)[0])}  # fmt: skip


def lending(y: np.ndarray, p: np.ndarray, y_tune: np.ndarray, p_tune: np.ndarray,
            loss: float = LOSS) -> dict:  # fmt: skip
    own = p < 1 / (1 + loss)
    cut = tune_cutoff(y_tune, p_tune, loss)
    tuned = p <= cut
    return {"approve_own": float(own.mean()), "profit_own": profit(y, own, loss),
            "tune_cutoff": cut, "approve_tune_cutoff": float(tuned.mean()),
            "profit_tune_cutoff": profit(y, tuned, loss)}  # fmt: skip


def paired_ece(y: np.ndarray, p_ref: np.ndarray, p: np.ndarray, n_boot: int = 2000,
               seed: int = 20261126) -> dict:  # fmt: skip
    """Change in ECE (shrunk minus bf16) with a paired bootstrap over customers: the same
    resamples for both, and the same two-sided p-value as robustness.paired_auc. The noise floor
    (ece_noise_p95) is for the level of ECE; this tests the change. Added after the review panel."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(m.ece(y[idx], p[idx]) - m.ece(y[idx], p_ref[idx]))
    diffs = np.array(diffs)
    pval = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"ece_vs_bf16": m.ece(y, p) - m.ece(y, p_ref),
            "ece_vs_bf16_lo": float(np.percentile(diffs, 2.5)),
            "ece_vs_bf16_hi": float(np.percentile(diffs, 97.5)), "ece_vs_bf16_p": pval}  # fmt: skip


def decisions(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray, cut: float, cut_ref: float,
              loss: float = LOSS) -> dict:  # fmt: skip
    """What changes for a lender (added while writing post 4, after the runs). Each model approves
    at its own tune cut-off; "swapped in" keeps the bf16 run's tune cut-off, as a lender would who
    replaced the model without re-checking it. Nothing here is chosen on test."""
    ref, own, swapped = p_ref <= cut_ref, p <= cut, p <= cut_ref
    return {"decisions_changed_vs_bf16": float((own != ref).mean()),
            "approve_bf16_cutoff": float(swapped.mean()),
            "profit_bf16_cutoff": profit(y, swapped, loss),
            "decisions_changed_bf16_cutoff": float((swapped != ref).mean()),
            "moved_5pts_share": float((np.abs(p - p_ref) >= 0.05 - 1e-9).mean())}  # fmt: skip


def cost(contestant: str, directory: Path = PREDICTIONS_DIR) -> dict:
    """Latency and cost from the test records; memory from the run metadata and raw records."""
    test = [r for r in read_records(contestant, directory)
            if r["split"] == "test" and r["condition"] == "trained"]  # fmt: skip
    out = {"latency_ms_median": statistics.median(r["latency_ms"] for r in test),
           "cost_per_1000_usd": 1000 * statistics.fmean(r["cost_usd"] for r in test)}  # fmt: skip
    meta_path = directory / "raw" / f"{contestant}.meta.json"
    runs = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else []
    loads = [r["quantization"]["memory_after_load"]["allocated"] for r in runs
             if r.get("quantization", {}).get("memory_after_load")]  # fmt: skip
    out["gpu_weights_gb"] = max(loads) / GB if loads else None
    raw_path = directory / "raw" / f"{contestant}.jsonl"
    peaks = []
    if raw_path.exists():
        for line in raw_path.read_text("utf-8").splitlines():
            if line:
                r = json.loads(line)
                if r.get("split") == "test" and "raw" in r:
                    peak = r["raw"].get("gpu_max_allocated")
                    if peak is not None:
                        peaks.append(peak)
    out["gpu_peak_gb"] = max(peaks) / GB if peaks else None
    return out


def score_row(model: str, precision: str, directory: Path = PREDICTIONS_DIR) -> dict | str:
    """One model at one precision, or the reason it can't be scored yet."""
    name = contestant_name(model, precision)
    ref = contestant_name(model, "bf16")
    ids, y, p = split_arrays(name, "trained", "test", directory)
    if len(ids) != N_TEST:
        return f"{name}: {len(ids)} of {N_TEST} test customers"
    ids_ref, y_ref, p_ref = split_arrays(ref, "trained", "test", directory)
    if ids_ref != ids or not np.array_equal(y, y_ref):
        return f"{name}: its bf16 reference ({ref}) isn't complete on the same customers"
    _, y_tu, p_tu = split_arrays(name, "trained", "tune", directory)
    _, _, q = split_arrays(name, "trained_calibrated", "test", directory)
    s = m.summarize(y, p, n_boot=1000)
    row = {"model": model, "precision": precision, "contestant": name,
           "model_version": read_records(name, directory)[0]["model_version"],
           "auc": s["auc"], "auc_lo": s["auc_lo"], "auc_hi": s["auc_hi"],
           "ece": s["ece"], "ece_equal_mass": ece_equal_mass(y, p),
           "ece_noise_p95": ece_noise_floor(p).get("p95"),
           "citl": s["citl"], "cal_slope": s["cal_slope"], "brier": s["brier"],
           "mean_predicted": s["mean_predicted"], "observed_rate": s["observed_rate"]}  # fmt: skip
    if len(q) == N_TEST:
        row.update({"ece_calibrated": m.ece(y, q), "mean_predicted_calibrated": float(q.mean())})
    if precision != "bf16":
        row.update(drift(p_ref, p))
        d = paired_auc(y, p_ref, p)  # reference minus contestant; stored the other way round
        lo, hi = sorted((-d["lo"], -d["hi"]))
        row.update({"auc_vs_bf16": -d["difference"], "auc_vs_bf16_lo": lo,
                    "auc_vs_bf16_hi": hi, "auc_vs_bf16_p": d["p_value"]})  # fmt: skip
        row.update(paired_ece(y, p_ref, p))
    if len(y_tu) == N_TEST:
        row.update(lending(y, p, y_tu, p_tu))
        _, y_tu_ref, p_tu_ref = split_arrays(ref, "trained", "tune", directory)
        if precision != "bf16" and len(p_tu_ref) == N_TEST:
            row.update(decisions(y, p, p_ref, row["tune_cutoff"],
                                 tune_cutoff(y_tu_ref, p_tu_ref, LOSS)))  # fmt: skip
    row.update(cost(name, directory))
    return row


def build(directory: Path = PREDICTIONS_DIR, failures: dict | None = None) -> dict:
    failures = failures if failures is not None else load_failures()
    rows, not_scored = [], {}
    for model in QUANT_MODELS:
        for precision in PRECISIONS:
            name = contestant_name(model, precision)
            if name in failures:
                not_scored[name] = {"recorded_failure": failures[name]}
                continue
            out = score_row(model, precision, directory)
            if isinstance(out, str):
                not_scored[name] = {"incomplete": out}
            else:
                rows.append(out)
    family = [r for r in rows if r["precision"] in QUANTIZED]
    adjusted = holm({r["contestant"]: r["auc_vs_bf16_p"] for r in family})
    adjusted_ece = holm({r["contestant"]: r["ece_vs_bf16_p"] for r in family})
    for r in family:
        r["auc_vs_bf16_p_holm"] = adjusted[r["contestant"]]
        r["ece_vs_bf16_p_holm"] = adjusted_ece[r["contestant"]]
    repro = {}
    for path in sorted(REPRO_DIR.glob("repro_*.json")) if REPRO_DIR.exists() else []:
        d = json.loads(path.read_text("utf-8"))
        repro[d["contestant"]] = {k: d[k] for k in ("reference", "n", "mean_abs_diff",
                                                    "max_abs_diff", "passed")}  # fmt: skip
    return {"rows": rows, "not_scored": not_scored, "reproduction_checks": repro,
            "holm_family": "the 15 quantized rows (3 models x 5 precisions), AUC vs own bf16",
            "loss_ratio": LOSS}  # fmt: skip


def load_failures(path: Path = FAILURES) -> dict:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def record_failure(name: str, stage: str, error: str, path: Path = FAILURES) -> dict:
    """Record that one model x precision couldn't run. It is reported, never replaced."""
    from quantize import split_name

    model, precision = split_name(name)
    failures = load_failures(path)
    failures[name] = {"model": model, "precision": precision, "stage": stage, "error": error,
                      "recorded_at": utc_now()}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2) + "\n", "utf-8")
    return failures[name]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record-failure", metavar="NAME", help="e.g. kev-4b-ft-hqq-2")
    ap.add_argument("--stage", choices=("load", "smoke", "run"), default="load")
    ap.add_argument("--error", default="", help="the error message, as printed")
    args = ap.parse_args()
    if args.record_failure:
        print(record_failure(args.record_failure, args.stage, args.error))
        return
    out = build()
    rows = out["rows"]
    if not rows:
        print(f"nothing to score yet ({len(out['not_scored'])} model x precision pairs pending)")
        return
    cols = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("model", "precision", "contestant"), k))  # fmt: skip
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    head = ("model", "precision", "AUC", "vs bf16", "holm p", "ECE", "cal'd", "mean p", "|chg|",
            "ms", "GB")
    print("{:<12} {:<9} {:>6} {:>8} {:>7} {:>6} {:>6} {:>7} {:>6} {:>6} {:>5}".format(*head))
    for r in rows:
        gap = f"{r['auc_vs_bf16']:+.3f}" if "auc_vs_bf16" in r else ""
        hp = f"{r['auc_vs_bf16_p_holm']:.3f}" if "auc_vs_bf16_p_holm" in r else ""
        chg = f"{r['mean_abs_change_vs_bf16']:.3f}" if "mean_abs_change_vs_bf16" in r else ""
        cal = f"{r['ece_calibrated']:.3f}" if "ece_calibrated" in r else ""
        gb = f"{r['gpu_weights_gb']:.1f}" if r.get("gpu_weights_gb") else ""
        print(f"{r['model']:<12} {r['precision']:<9} {r['auc']:>6.3f} {gap:>8} {hp:>7} "
              f"{r['ece']:>6.3f} {cal:>6} {r['mean_predicted']:>7.3f} {chg:>6} "
              f"{r['latency_ms_median']:>6.0f} {gb:>5}")  # fmt: skip
    for name, why in out["not_scored"].items():
        print(f"not scored: {name}: {why}")
    for name, d in out["reproduction_checks"].items():
        print(f"reproduction check {name} vs {d['reference']}: mean |diff| "
              f"{d['mean_abs_diff']:.4f} ({'pass' if d['passed'] else 'FAIL'})")  # fmt: skip
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
