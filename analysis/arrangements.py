"""Post 5 addendum, part A (PLAN.md §12): the table models with their table laid out another way.

Kumo-Tabular and TabICL-v2, with all 20,000 `train` customers in context, answering the 2,000
`test` customers with the same 19 fields rearranged (data/arrange.py; records from
baselines/tabular_fm.py --arrangement in results/checks/arrangements/). Measures fixed before the
runs, as for post 5's layouts:

    uv run python analysis/arrangements.py   # -> results/arrangements.csv and .json

- **Reference:** the same session's `original` run, per model and setting. Its tune answers set
  post 2's lender's cut-off (4:1). Its distance from M6's run (`results/predictions/<model>`) is
  reported as the reproduction (tune and test).
- **Per arrangement against the reference:** mean shift, mean |change| per customer, the share
  moved 5+ points, rank correlation; paired bootstrap of AUC and ECE; decisions changed at the
  reference's cut-off (layouts.compare).
- **Seed noise:** `original` with four other seeds against the reference, the same measures. An
  arrangement counts as moving the answers beyond seed noise when its mean |change| per customer
  is larger than the largest of the four seeds' (`beyond_seed_noise`), the author's question 1.
- **Settings:** `defaults` (the makers' defaults, as in post 3: the headline) and `e1` (one
  estimator with the makers' column shuffle off: a diagnostic of the bare model). Each is its own
  Holm family: 2 models x 14 arrangements = 28.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))

import metrics as m  # noqa: E402
from arrange import ARRANGE_DIR, ARRANGEMENTS, run_name  # noqa: E402
from layouts import N_TEST, add_holm, answers, arrays, compare, cutoff, moved, write  # noqa: E402
from records import PREDICTIONS_DIR, ROOT, read_records  # noqa: E402

OUT_CSV = ROOT / "results" / "arrangements.csv"
OUT_JSON = ROOT / "results" / "arrangements.json"
MODELS = ("kumo-tabular", "tabicl-v2")
OTHERS = tuple(a for a in ARRANGEMENTS if a != "original")  # 14
SEED_OFFSETS = (1, 2, 3, 4)
SETTINGS = {"defaults": None, "e1": 1}  # name -> --estimators
CONDITION = "trained"


def seed_noise(ref: dict, model: str, estimators: int | None, cut: float) -> dict:
    """The reference against `original` at four other seeds: the largest and average move."""
    ids, runs = sorted(ref), []
    for k in SEED_OFFSETS:
        other = answers(run_name(model, "original", k, estimators), ARRANGE_DIR, "test", CONDITION)
        assert sorted(other) == ids, f"{model} seed {k}: incomplete"
        y, p0, p1 = arrays(ref, other, ids)
        runs.append({**moved(p0, p1), "auc": m.auc(y, p1),
                     "decisions_changed": float(((p0 <= cut) != (p1 <= cut)).mean())})  # fmt: skip
    pick = lambda key, f: float(f([r[key] for r in runs]))  # noqa: E731
    return {"seed_mean_abs_change_max": pick("mean_abs_change", max),
            "seed_mean_abs_change_mean": pick("mean_abs_change", np.mean),
            "seed_moved_5pts_max": pick("moved_5pts_share", max),
            "seed_decisions_changed_max": pick("decisions_changed", max),
            "seed_auc_min": pick("auc", min), "seed_auc_max": pick("auc", max)}  # fmt: skip


def reproduction(model: str, ref_name: str) -> dict:
    """M6's run against this session's original (defaults), on tune and test."""
    out = {}
    for split in ("tune", "test"):
        m6 = answers(model, PREDICTIONS_DIR, split, CONDITION)
        new = answers(ref_name, ARRANGE_DIR, split, CONDITION)
        ids = sorted(new)
        assert sorted(m6) == ids, f"{model}: M6 and the session's {split} customers differ"
        _, p0, p1 = arrays(m6, new, ids)
        d = np.abs(p1 - p0)
        out[f"repro_{split}_mean_abs"] = float(d.mean())
        out[f"repro_{split}_max_abs"] = float(d.max())
    return out


def build() -> list[dict]:
    rows = []
    for setting, est in SETTINGS.items():
        family = []
        for model in MODELS:
            ref_name = run_name(model, "original", 0, est)
            cut = cutoff(ref_name, ARRANGE_DIR, CONDITION)
            ref = answers(ref_name, ARRANGE_DIR, "test", CONDITION)
            ids = sorted(ref)
            assert len(ids) == N_TEST, f"{ref_name}: {len(ids)} test answers"
            base = seed_noise(ref, model, est, cut)
            if setting == "defaults":
                base |= reproduction(model, ref_name)
            for arrangement in OTHERS:
                name = run_name(model, arrangement, 0, est)
                other = answers(name, ARRANGE_DIR, "test", CONDITION)
                assert sorted(other) == ids, f"{name}: incomplete"
                y, p0, p1 = arrays(ref, other, ids)
                row = {"setting": setting, "model": model, "arrangement": arrangement,
                       "model_version": read_records(name, ARRANGE_DIR)[0]["model_version"],
                       **compare(y, p0, p1, cut, ref="original"), **base}  # fmt: skip
                row["beyond_seed_noise"] = row["mean_abs_change"] > base["seed_mean_abs_change_max"]
                row["_key"] = f"{model}/{arrangement}"
                family.append(row)
        add_holm(family, "_key", ref="original")
        for r in family:
            del r["_key"]
        rows += family
    return rows


def main() -> None:
    rows = build()
    write(rows, OUT_CSV, OUT_JSON, holm_family=len(rows) // len(SETTINGS))  # one family per setting
    print(f"{'setting':9}{'model':14}{'arrangement':17}{'|chg|':>7}{'seeds':>7}{'5pt+':>6}"
          f"{'AUC':>14}{'pH':>7}{'ECE':>14}{'pH':>7}{'flip':>6}  beyond seeds")  # fmt: skip
    for r in rows:
        print(f"{r['setting']:9}{r['model']:14}{r['arrangement']:17}"
              f"{100 * r['mean_abs_change']:7.2f}{100 * r['seed_mean_abs_change_max']:7.2f}"
              f"{100 * r['moved_5pts_share']:6.1f}{r['auc_original']:7.3f}>{r['auc']:6.3f}"
              f"{r['auc_vs_original_p_holm']:7.3f}{r['ece_original']:7.3f}>{r['ece']:6.3f}"
              f"{r['ece_vs_original_p_holm']:7.3f}{100 * r['decisions_changed']:6.1f}"
              f"  {'yes' if r['beyond_seed_noise'] else 'no'}")  # fmt: skip
    print(f"Saved -> {OUT_CSV}, {OUT_JSON}")


if __name__ == "__main__":
    main()
