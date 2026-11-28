"""Post 5: the same customers in other layouts (PLAN.md §12, post 5 spec and addendum).

Each zero-shot AI contestant's `out_of_box` answers on the 2,000 test customers in the four formats
(results/checks/formats/) and in the layout check's realigned table (results/checks/disguise/,
`aligned`), each against its own template v2 run (results/predictions/). A diagnostic, not a
contestant condition: nothing is tuned on test, and the headline results stay on v2.

    uv run python analysis/layouts.py            # -> results/layouts.csv and .json
    uv run python analysis/layouts.py --trained  # addendum part B -> results/layouts_trained.*

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

`--trained` (the addendum's part B, measures fixed before its runs): post 3's four trained models
on the same five layouts (results/checks/layouts_trained/, runners/layout.py), each against its
own reference on the original layout: post 4's full-size `<model>-bf16` run for Kev-4B, Kev-0.8B
and Qwen, M6's `laya-ft` for Laya. The cut-off comes from the reference's tune answers. The noise
is the original layout asked again in the same session, on all 2,000 test customers. Holm over 20.
Each row also carries the same model's cold (post 5) movement in that layout, for the comparison
the addendum asks about; the cold models had no training, so that's a comparison of conditions.

Added after the addendum's results, for post 5b (PLAN.md §12), in both modes and at the end of each
row so the earlier columns are unchanged: how close the reference's test answers sit to its tune
cut-off (the median distance, and the share within 1, 2 and 5 points), the mean shift of the
customers within 5 points of it, and the changed decisions split by direction. Label-free.
"""

from __future__ import annotations

import argparse
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

# The addendum's part B: reference run -> the same model answering cold in post 5.
TRAINED_DIR = ROOT / "results" / "checks" / "layouts_trained"  # = runners/layout.py LAYOUT_DIR
TRAINED_REFS = {"kev-4b-ft-bf16": "kev-4b", "kev-0.8b-ft-bf16": "kev-0.8b",
                "qwen-ft-bf16": "qwen-logprobs", "laya-ft": "laya"}  # fmt: skip
OUT_TRAINED_CSV = ROOT / "results" / "layouts_trained.csv"
OUT_TRAINED_JSON = ROOT / "results" / "layouts_trained.json"
NEAR_CUTOFF = (0.01, 0.02, 0.05)  # post 5b's windows around the cut-off: 1, 2 and 5 points


def answers(name: str, directory: Path, split: str = "test",
            condition: str = "out_of_box") -> dict[str, tuple[int, float]]:  # fmt: skip
    """{id: (label, p_default)} for one record file's answers in one condition and split."""
    if not (directory / f"{name}.jsonl").exists():
        return {}
    return {r["id"]: (r["label"], r["p_default"]) for r in read_records(name, directory)
            if r["split"] == split and r["condition"] == condition}  # fmt: skip


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


def cutoff(name: str, directory: Path = PREDICTIONS_DIR, condition: str = "out_of_box") -> float:
    """The reference's approval cut-off for post 2's lender, chosen on its own tune answers."""
    tune = answers(name, directory, "tune", condition)
    assert len(tune) == N_TEST, f"{name}: {len(tune)} tune answers"
    y, p = (np.array(v, float) for v in zip(*tune.values(), strict=True))
    return tune_cutoff(y, p, LOSS)


def noise_between(ref: dict, again: dict, cut: float) -> dict:
    """Run-to-run noise: the reference against the same summaries asked again."""
    ids = sorted(set(ref) & set(again))
    _, p0, p1 = arrays(ref, again, ids)
    out = {f"noise_{k}": v for k, v in moved(p0, p1).items() if k != "spearman"}
    out.update({"noise_n": len(ids),
                "noise_decisions_changed": float(((p0 <= cut) != (p1 <= cut)).mean())})  # fmt: skip
    return out


def noise(model: str, cut: float) -> dict:
    """v2 against the same summaries asked again (the disguise check's 200-customer control)."""
    return noise_between(answers(model, PREDICTIONS_DIR), answers(f"{model}-control", CHECK_DIR),
                         cut)  # fmt: skip


def compare(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, cut: float, ref: str = "v2") -> dict:
    """Every measure of a run (p1) against its reference (p0) on the same customers. Keys for the
    reference's own values end in `_<ref>`."""
    row = {"observed_rate": float(y.mean()), f"mean_{ref}": float(p0.mean()),
           "mean": float(p1.mean())}  # fmt: skip
    row.update(moved(p0, p1))
    d = paired_auc(y, p0, p1)  # reference minus run; stored the other way round
    lo, hi = sorted((-d["lo"], -d["hi"]))
    row.update({f"auc_{ref}": m.auc(y, p0), "auc": m.auc(y, p1), f"auc_vs_{ref}": -d["difference"],
                f"auc_vs_{ref}_lo": lo, f"auc_vs_{ref}_hi": hi,
                f"auc_vs_{ref}_p": d["p_value"]})  # fmt: skip
    e = paired_ece(y, p0, p1)
    row.update({f"ece_{ref}": m.ece(y, p0), "ece": m.ece(y, p1),
                "ece_noise_p95": ece_noise_floor(p1)["p95"]})  # fmt: skip
    row.update({k.replace("bf16", ref): v for k, v in e.items()})
    a0, a1 = p0 <= cut, p1 <= cut
    row.update({f"tune_cutoff_{ref}": cut, f"approve_{ref}": float(a0.mean()),
                "approve": float(a1.mean()), f"profit_{ref}": profit(y, a0, LOSS),
                "profit": profit(y, a1, LOSS),
                "decisions_changed": float((a0 != a1).mean())})  # fmt: skip
    return row


def near_cutoff(p0: np.ndarray, p1: np.ndarray, cut: float) -> dict:
    """Post 5b: how close the reference's answers (p0) sit to its cut-off, how far the run (p1)
    moved the customers within 5 points of it on average, and which way the decisions that changed
    went (approve when p <= cut, as in compare). No outcomes needed."""
    dist = np.abs(p0 - cut)
    a0, a1 = p0 <= cut, p1 <= cut
    near = dist <= 0.05 + 1e-9
    out = {"cutoff_distance_median": float(np.median(dist))}
    out.update({f"near_cutoff_share_{round(100 * w)}": float((dist <= w + 1e-9).mean())
                for w in NEAR_CUTOFF})  # fmt: skip
    out["near_cutoff_shift_5"] = float((p1 - p0)[near].mean()) if near.any() else float("nan")
    out.update({"flips_to_decline": float((a0 & ~a1).mean()),
                "flips_to_approve": float((~a0 & a1).mean())})  # fmt: skip
    return out


def score(model: str, layout: str, cut: float) -> dict:
    directory, suffix, label = LAYOUTS[layout]
    ref, other = answers(model, PREDICTIONS_DIR), answers(f"{model}-{suffix}", directory)
    ids = sorted(ref)
    assert len(ids) == N_TEST and sorted(other) == ids, f"{model}-{suffix}: incomplete"
    y, p0, p1 = arrays(ref, other, ids)
    version = read_records(f"{model}-{suffix}", directory)[0]["model_version"]
    row = {"model": model, "layout": layout, "label": label, "model_version": version}
    row.update(compare(y, p0, p1, cut))
    row["_late"] = near_cutoff(p0, p1, cut)
    return row


def add_holm(rows: list[dict], key: str, tests: tuple[str, ...] = ("auc", "ece"),
             ref: str = "v2") -> None:  # fmt: skip
    """Holm-adjusted p for each paired test, over all `rows` as one family."""
    for test in tests:
        adj = holm({r[key]: r[f"{test}_vs_{ref}_p"] for r in rows})
        for r in rows:
            r[f"{test}_vs_{ref}_p_holm"] = adj[r[key]]


def finish(rows: list[dict]) -> list[dict]:
    """Holm over all rows, then post 5b's measures (`_late`) last, so earlier columns keep their
    places."""
    for r in rows:
        r["_key"] = f"{r['model']}/{r['layout']}"
    add_holm(rows, "_key")
    for r in rows:
        del r["_key"]
        r.update(r.pop("_late"))
    return rows


def build() -> list[dict]:
    rows = []
    for model in MODELS:
        cut = cutoff(model)
        base = noise(model, cut)
        rows += [score(model, layout, cut) | base for layout in LAYOUTS]
    return finish(rows)


def cold() -> dict[tuple[str, str], dict]:
    """Post 5's published rows, for the trained models' cold comparison."""
    with OUT_CSV.open(encoding="utf-8") as f:
        return {(r["model"], r["layout"]): r for r in csv.DictReader(f)}


def build_trained() -> list[dict]:
    """The addendum's part B: post 3's trained models in post 5's layouts."""
    rows, published = [], cold()
    for ref_name, cold_name in TRAINED_REFS.items():
        cut = cutoff(ref_name, PREDICTIONS_DIR, "trained")
        ref = answers(ref_name, PREDICTIONS_DIR, "test", "trained")
        ids = sorted(ref)
        assert len(ids) == N_TEST, f"{ref_name}: {len(ids)} test answers"
        again = answers(f"{ref_name}-again", TRAINED_DIR, "test", "trained")
        assert sorted(again) == ids, f"{ref_name}-again: incomplete"
        base = noise_between(ref, again, cut)
        for layout in LAYOUTS:
            name = f"{ref_name}-{layout}"
            other = answers(name, TRAINED_DIR, "test", "trained")
            assert sorted(other) == ids, f"{name}: incomplete"
            y, p0, p1 = arrays(ref, other, ids)
            c = published[cold_name, layout]
            row = {"model": ref_name, "layout": layout, "label": LAYOUTS[layout][2],
                   "model_version": read_records(name, TRAINED_DIR)[0]["model_version"],
                   **compare(y, p0, p1, cut), **base,
                   "cold_model": cold_name,
                   "cold_mean_abs_change": float(c["mean_abs_change"]),
                   "cold_decisions_changed": float(c["decisions_changed"]),
                   "cold_auc_vs_v2": float(c["auc_vs_v2"]),
                   "_late": near_cutoff(p0, p1, cut)}  # fmt: skip
            rows.append(row)
    return finish(rows)


def write(rows: list[dict], out_csv: Path, out_json: Path, holm_family: int | None = None) -> None:
    """`holm_family`: the number of tests each Holm correction covered (default: every row)."""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    family = len(rows) if holm_family is None else holm_family
    out_json.write_text(json.dumps({"generated": utc_now(), "loss_ratio": LOSS,
                                    "holm_family": family, "rows": rows}, indent=2) + "\n",
                        encoding="utf-8")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--trained", action="store_true", help="the addendum's part B")
    trained = ap.parse_args().trained
    rows = build_trained() if trained else build()
    out_csv, out_json = (OUT_TRAINED_CSV, OUT_TRAINED_JSON) if trained else (OUT_CSV, OUT_JSON)
    write(rows, out_csv, out_json)
    print(f"{'model':18}{'layout':10}{'mean':>14}{'|chg|':>7}{'5pt+':>6}{'rho':>6}{'AUC':>14}"
          f"{'dAUC [95%]':>22}{'pH':>6}{'ECE':>14}{'pH':>6}{'flip':>6}{'noise':>12}"
          + ("  cold |chg|" if trained else ""))  # fmt: skip
    for r in rows:
        print(f"{r['model']:18}{r['layout']:10}{r['mean_v2']:6.3f}>{r['mean']:6.3f}"
              f"{100 * r['mean_abs_change']:7.1f}{100 * r['moved_5pts_share']:6.0f}"
              f"{r['spearman']:6.2f}{r['auc_v2']:6.3f}>{r['auc']:6.3f}"
              f"{r['auc_vs_v2']:+8.3f} [{r['auc_vs_v2_lo']:+.3f},{r['auc_vs_v2_hi']:+.3f}]"
              f"{r['auc_vs_v2_p_holm']:6.3f}{r['ece_v2']:6.3f}>{r['ece']:6.3f}"
              f"{r['ece_vs_v2_p_holm']:6.3f}{100 * r['decisions_changed']:6.1f}"
              f"{100 * r['noise_mean_abs_change']:6.1f}/"
              f"{100 * r['noise_decisions_changed']:4.1f}"
              + (f"{100 * r['cold_mean_abs_change']:9.1f}" if trained else ""))  # fmt: skip
    print(f"Saved -> {out_csv}, {out_json}")


if __name__ == "__main__":
    main()
