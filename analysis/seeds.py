"""M6 seed check: how much does re-running a fine-tune move the results? (PLAN.md §11, choice 6)

    uv run python analysis/seeds.py       # results/seeds.json, plus a printed table

The main runs are seed 0 (results/predictions/). Seeds 1 and 2 of Kev-0.8B and Laya are scored into
results/checks/m6_seeds/seed{1,2}/ by L4_RUNBOOK.md §6.8, with the same data, settings and code;
only the training seed differs (for Laya, its 400-row calibration slice is fixed by its own seed and
stays the same). For each run: test AUC, raw ECE, ECE after the same correction fitted on that
run's own tune predictions (Platt vs isotonic by CV, as analysis/recalibrate.py), and mean
prediction. The spread across seeds is the training noise the bootstrap intervals don't capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as m  # noqa: E402
from recalibrate import choose_and_fit  # noqa: E402
from records import PREDICTIONS_DIR, ROOT, read_records  # noqa: E402

SEED_DIR = ROOT / "results" / "checks" / "m6_seeds"
OUT = ROOT / "results" / "seeds.json"
MODELS = ("kev-0.8b-ft", "laya-ft")


def split(contestant: str, directory: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not (directory / f"{contestant}.jsonl").exists():
        return None
    recs = sorted((r for r in read_records(contestant, directory=directory)
                   if r["split"] == name and r["condition"] == "trained"),
                  key=lambda r: r["id"])  # fmt: skip
    if len(recs) != 2000:
        return None
    return (np.array([r["label"] for r in recs], float),
            np.array([r["p_default"] for r in recs], float))  # fmt: skip


def score(contestant: str, directory: Path) -> dict | None:
    tune, test = split(contestant, directory, "tune"), split(contestant, directory, "test")
    if tune is None or test is None:
        return None
    model, info = choose_and_fit(tune[1], tune[0])
    q = np.clip(model.predict(test[1]), 0, 1)
    return {"auc": m.auc(*test), "ece_raw": m.ece(*test), "ece_corrected": m.ece(test[0], q),
            "mean_predicted": float(test[1].mean()), "correction": info["chosen"]}  # fmt: skip


def main() -> None:
    dirs = {0: PREDICTIONS_DIR, 1: SEED_DIR / "seed1", 2: SEED_DIR / "seed2"}
    out = {}
    for c in MODELS:
        runs = {s: score(c, d) for s, d in dirs.items()}
        runs = {s: r for s, r in runs.items() if r}
        if not runs:
            continue
        spread = {k: [min(r[k] for r in runs.values()), max(r[k] for r in runs.values())]
                  for k in ("auc", "ece_raw", "ece_corrected", "mean_predicted")}  # fmt: skip
        out[c] = {"seeds": runs, "range": spread}
        print(f"{c}: seeds {sorted(runs)}")
        for s, r in sorted(runs.items()):
            print(f"  seed {s}: AUC {r['auc']:.3f}  ECE raw {r['ece_raw']:.3f}  corrected "
                  f"{r['ece_corrected']:.3f}  mean p {r['mean_predicted']:.3f}")  # fmt: skip
        print("  range: " + ", ".join(f"{k} {lo:.3f}-{hi:.3f}" for k, (lo, hi) in spread.items()))
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
