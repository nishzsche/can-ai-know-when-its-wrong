"""Post 5: the same customers in other layouts (PLAN.md §12, post 5 spec).

Each zero-shot AI contestant's `out_of_box` answers on the 2,000 test customers in the four formats
(results/checks/formats/) and in the layout check's realigned table (results/checks/disguise/,
`aligned`), each against its own template v2 run (results/predictions/). A diagnostic, not a
contestant condition: nothing is tuned on test, and the headline results stay on v2.

    uv run python analysis/layouts.py    # -> results/layouts.csv and .json

Per model and layout. The spec's measures are the mean shift, AUC, ECE and the change per customer;
the paired tests, the 5-point share and the decisions were added while writing, after the runs:
- drift without labels: mean prediction, mean shift and mean |change| per customer against v2, the
  share of customers moved 5+ points, and rank correlation with v2;
- run-to-run noise: the same per-customer measures between v2 and the disguise check's control
  (the original summaries asked again for 200 test customers);
- ranking: AUC, and the paired bootstrap difference from v2;
- honesty: ECE, the noise floor a perfectly calibrated model would score (95th percentile), and the
  paired bootstrap change from v2;
- Holm over all 30 rows (six models, five layouts) for both paired tests;
- decisions (post 2's lender, loss ratio 4:1): the cut-off chosen on v2's tune answers, applied to
  v2 and to the layout on test; the share of approve/decline decisions that differ, approval and
  profit. The same between v2 and the control gives the noise.
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
from accuracy import profit, tune_cutoff  # noqa: E402
from quantization import LOSS, paired_ece  # noqa: E402
from records import PREDICTIONS_DIR, ROOT, read_records, utc_now  # noqa: E402
from robustness import ece_noise_floor, holm, paired_auc  # noqa: E402

OUT_CSV = ROOT / "results" / "layouts.csv"
OUT_JSON = ROOT / "results" / "layouts.json"
FORMAT_DIR = ROOT / "results" / "checks" / "formats"
CHECK_DIR = ROOT / "results" / "checks" / "disguise"
N_TEST = 2000

MODELS = ("jev", "kev-4b", "kev-0.8b", "laya", "qwen-logprobs", "frontier-verbal")
LAYOUTS = {  # tag -> (where its records are, file suffix, label)
    "markdown": (FORMAT_DIR, "markdown", "Markdown table"),
    "csv": (FORMAT_DIR, "csv", "CSV"),
    "json": (FORMAT_DIR, "json", "JSON"),
    "prose": (FORMAT_DIR, "prose", "Sentences"),
    "aligned": (CHECK_DIR, "aligned", "Table, bills realigned"),
}


def answers(name: str, directory: Path, split: str = "test") -> dict[str, tuple[int, float]]:
    """{id: (label, p_default)} for one record file's `out_of_box` answers on one split."""
    if not (directory / f"{name}.jsonl").exists():
        return {}
    return {r["id"]: (r["label"], r["p_default"]) for r in read_records(name, directory)
            if r["split"] == split and r["condition"] == "out_of_box"}  # fmt: skip


def arrays(ref: dict, other: dict, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([ref[i][0] for i in ids], float)
    assert np.array_equal(y, [other[i][0] for i in ids]), "labels differ between runs"
    return y, np.array([ref[i][1] for i in ids]), np.array([other[i][1] for i in ids])


def moved(p_ref: np.ndarray, p: np.ndarray) -> dict:
    """How far answers moved between two runs, without outcomes."""
    from scipy.stats import spearmanr

    d = p - p_ref
    return {"mean_shift": float(d.mean()), "mean_abs_change": float(np.abs(d).mean()),
            "moved_5pts_share": float((np.abs(d) >= 0.05 - 1e-9).mean()),
            "spearman": float(spearmanr(p_ref, p)[0])}  # fmt: skip


def cutoff(model: str) -> float:
    """v2's approval cut-off for post 2's lender, chosen on the model's tune answers."""
    tune = answers(model, PREDICTIONS_DIR, "tune")
    assert len(tune) == N_TEST, f"{model}: {len(tune)} tune answers"
    y, p = (np.array(v, float) for v in zip(*tune.values(), strict=True))
    return tune_cutoff(y, p, LOSS)


def noise(model: str, cut: float) -> dict:
    """v2 against the same summaries asked again (the disguise check's 200-customer control)."""
    ref, ctrl = answers(model, PREDICTIONS_DIR), answers(f"{model}-control", CHECK_DIR)
    ids = sorted(set(ref) & set(ctrl))
    _, p0, p1 = arrays(ref, ctrl, ids)
    out = {f"noise_{k}": v for k, v in moved(p0, p1).items() if k != "spearman"}
    out.update({"noise_n": len(ids),
                "noise_decisions_changed": float(((p0 <= cut) != (p1 <= cut)).mean())})  # fmt: skip
    return out


def score(model: str, layout: str, cut: float) -> dict:
    directory, suffix, label = LAYOUTS[layout]
    ref, other = answers(model, PREDICTIONS_DIR), answers(f"{model}-{suffix}", directory)
    ids = sorted(ref)
    assert len(ids) == N_TEST and sorted(other) == ids, f"{model}-{suffix}: incomplete"
    y, p0, p1 = arrays(ref, other, ids)
    row = {"model": model, "layout": layout, "label": label,
           "model_version": read_records(f"{model}-{suffix}", directory)[0]["model_version"],
           "observed_rate": float(y.mean()),
           "mean_v2": float(p0.mean()), "mean": float(p1.mean())}  # fmt: skip
    row.update(moved(p0, p1))
    d = paired_auc(y, p0, p1)  # v2 minus layout; stored the other way round
    lo, hi = sorted((-d["lo"], -d["hi"]))
    row.update({"auc_v2": m.auc(y, p0), "auc": m.auc(y, p1), "auc_vs_v2": -d["difference"],
                "auc_vs_v2_lo": lo, "auc_vs_v2_hi": hi, "auc_vs_v2_p": d["p_value"]})  # fmt: skip
    e = paired_ece(y, p0, p1)
    row.update({"ece_v2": m.ece(y, p0), "ece": m.ece(y, p1),
                "ece_noise_p95": ece_noise_floor(p1)["p95"]})  # fmt: skip
    row.update({k.replace("bf16", "v2"): v for k, v in e.items()})
    a0, a1 = p0 <= cut, p1 <= cut
    row.update({"tune_cutoff_v2": cut, "approve_v2": float(a0.mean()), "approve": float(a1.mean()),
                "profit_v2": profit(y, a0, LOSS), "profit": profit(y, a1, LOSS),
                "decisions_changed": float((a0 != a1).mean())})  # fmt: skip
    return row


def build() -> list[dict]:
    rows = []
    for model in MODELS:
        cut = cutoff(model)
        base = noise(model, cut)
        rows += [score(model, layout, cut) | base for layout in LAYOUTS]
    for test in ("auc", "ece"):
        adj = holm({f"{r['model']}/{r['layout']}": r[f"{test}_vs_v2_p"] for r in rows})
        for r in rows:
            r[f"{test}_vs_v2_p_holm"] = adj[f"{r['model']}/{r['layout']}"]
    return rows


def main() -> None:
    rows = build()
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps({"generated": utc_now(), "loss_ratio": LOSS,
                                    "holm_family": len(rows), "rows": rows}, indent=2) + "\n",
                        encoding="utf-8")  # fmt: skip
    print(f"{'model':16}{'layout':10}{'mean':>14}{'|chg|':>7}{'5pt+':>6}{'rho':>6}{'AUC':>14}"
          f"{'dAUC [95%]':>22}{'pH':>6}{'ECE':>14}{'pH':>6}{'flip':>6}{'noise':>12}")  # fmt: skip
    for r in rows:
        print(f"{r['model']:16}{r['layout']:10}{r['mean_v2']:6.3f}>{r['mean']:6.3f}"
              f"{100 * r['mean_abs_change']:7.1f}{100 * r['moved_5pts_share']:6.0f}"
              f"{r['spearman']:6.2f}{r['auc_v2']:6.3f}>{r['auc']:6.3f}"
              f"{r['auc_vs_v2']:+8.3f} [{r['auc_vs_v2_lo']:+.3f},{r['auc_vs_v2_hi']:+.3f}]"
              f"{r['auc_vs_v2_p_holm']:6.3f}{r['ece_v2']:6.3f}>{r['ece']:6.3f}"
              f"{r['ece_vs_v2_p_holm']:6.3f}{100 * r['decisions_changed']:6.1f}"
              f"{100 * r['noise_mean_abs_change']:6.1f}/"
              f"{100 * r['noise_decisions_changed']:4.1f}")  # fmt: skip
    print(f"Saved -> {OUT_CSV}, {OUT_JSON}")


if __name__ == "__main__":
    main()
