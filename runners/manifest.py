"""Check M3 coverage and write results/run_manifest.json.

    uv run python runners/manifest.py                     # exit 1 if anything is missing
    uv run python runners/manifest.py --allow-incomplete  # write it anyway (e.g. on the L4)
    uv run python runners/manifest.py --m6                # M6 contestants -> run_manifest_m6.json
    uv run python runners/manifest.py --quant             # post 4 -> run_manifest_quant.json
    uv run python runners/manifest.py --addendum          # post 5 addendum -> ..._addendum.json

For every contestant: are exactly the expected customers there for each (condition, split)?
Does each file come from a single model version? Also recorded: total cost, median latency, run
dates, each run's software environment (from <predictions>/raw/<contestant>.meta.json), the git
commit and the data versions. PLAN.md section 6: pin model revisions and library versions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))  # arrange

from arrange import ARRANGE_DIR, ARRANGEMENTS, run_name  # noqa: E402
from harness import RUN_CONDITIONS, load_requests  # noqa: E402
from layout import AGAIN, LAYOUT_DIR, LAYOUTS  # noqa: E402
from quantize import all_contestants  # noqa: E402
from records import PREDICTIONS_DIR, TEST_ONLY_CONDITIONS, read_records, utc_now  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "run_manifest.json"
MANIFEST_M6 = ROOT / "results" / "run_manifest_m6.json"
MANIFEST_QUANT = ROOT / "results" / "run_manifest_quant.json"
QUANT_FAILURES = ROOT / "results" / "checks" / "quant_failures.json"
SPLITS = ("tune", "test")

AI_CONTESTANTS = ("jev", "kev-0.8b", "kev-4b", "laya", "qwen-logprobs", "frontier-verbal")
GROUPS = {
    **{c: [(cond, s) for cond in RUN_CONDITIONS for s in SPLITS] for c in AI_CONTESTANTS},
    "scorecard": [("trained", "tune"), ("trained", "test")],
    "gbm": [("trained", "tune"), ("trained", "test"), ("trained_calibrated", "test")],
}
# M6 (PLAN.md §11): every contestant that learned from `train`, plus the neighbours reference.
M6_GROUPS = {
    c: [("trained", "tune"), ("trained", "test"), ("trained_calibrated", "test")]
    for c in ("kev-4b-ft", "kev-0.8b-ft", "qwen-ft", "laya-ft", "jev-examples", "sol-examples",
              "neighbours-20", "kumo-tabular", "tabicl-v2")
}  # fmt: skip
# Post 4 (PLAN.md §12): every fine-tuned model at every precision, bf16 reference included.
QUANT_GROUPS = {
    c: [("trained", "tune"), ("trained", "test"), ("trained_calibrated", "test")]
    for c in all_contestants()
}
# Post 5 addendum (PLAN.md §12). Part A: every table-model run (original answers tune too).
MANIFEST_ADDENDUM = ROOT / "results" / "run_manifest_addendum.json"
ADDENDUM_FAILURES = ROOT / "results" / "checks" / "addendum_failures.json"
ARRANGE_GROUPS = {
    run_name(model, arrangement, k, est): [("trained", "tune"), ("trained", "test")]
    if arrangement == "original" else [("trained", "test")]
    for model in ("kumo-tabular", "tabicl-v2") for est in (None, 1)
    for arrangement in ARRANGEMENTS
    for k in ((0, 1, 2, 3, 4) if arrangement == "original" else (0,))
}  # fmt: skip
# Part B: the trained models in each layout and the original asked again, test only.
LAYOUT_GROUPS = {
    f"{ref}-{layout}": [("trained", "test")]
    for ref in ("kev-4b-ft-bf16", "kev-0.8b-ft-bf16", "qwen-ft-bf16", "laya-ft")
    for layout in (*LAYOUTS, AGAIN)
}  # fmt: skip


def expected_ids() -> dict[str, list[str]]:
    return {s: [row["id"] for row in load_requests(s)] for s in SPLITS}


def summarize(
    contestant: str, groups: list, expected: dict[str, list[str]], directory: Path
) -> dict:
    path = directory / f"{contestant}.jsonl"
    if not path.exists():
        return {"present": False, "complete": False, "groups": {}}
    recs = read_records(contestant, directory=directory)  # validates every record
    out_groups, complete = {}, True
    for condition, split in groups:
        got = [r["id"] for r in recs if (r["condition"], r["split"]) == (condition, split)]
        want = set(expected[split])
        missing, extra = want - set(got), set(got) - want
        ok = not missing and not extra and len(got) == len(want)
        complete &= ok
        out_groups[f"{condition}/{split}"] = {"records": len(got), "missing": len(missing),
                                              "unexpected": len(extra), "complete": ok}  # fmt: skip
    # One model version per condition (gbm's calibrated condition legitimately has its own).
    by_condition: dict[str, set[str]] = {}
    for r in recs:
        by_condition.setdefault(r["condition"], set()).add(r["model_version"])
    complete &= all(len(v) == 1 for v in by_condition.values())
    versions = sorted(set().union(*by_condition.values())) if by_condition else []
    meta_path = directory / "raw" / f"{contestant}.meta.json"
    runs = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else []
    # Spend and latency count model calls only: derived conditions (recalibrated,
    # trained_calibrated) copy their source record's cost and latency but make no new call.
    calls = [r for r in recs if r["condition"] not in TEST_ONLY_CONDITIONS]
    latencies = [r["latency_ms"] for r in calls]
    return {
        "present": True,
        "complete": complete,
        "model_versions": versions,
        "records": len(recs),
        "groups": out_groups,
        "cost_usd_total": round(sum(r["cost_usd"] for r in calls), 6),
        "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
        "first_ts": min(r["ts"] for r in recs) if recs else None,
        "last_ts": max(r["ts"] for r in recs) if recs else None,
        "runs": runs if isinstance(runs, list) else [runs],
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def recorded_failures(path: Path = QUANT_FAILURES) -> dict:
    """Post 4 pairs that couldn't run (a method that won't load on a model), as recorded by
    `analysis/quantization.py --record-failure`. They count as accounted for, never as done."""
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def record_addendum_failure(name: str, stage: str, error: str,
                            path: Path = ADDENDUM_FAILURES) -> dict:  # fmt: skip
    """Record that one addendum run couldn't run. It is reported, never replaced."""
    if name not in ARRANGE_GROUPS and name not in LAYOUT_GROUPS:
        raise SystemExit(f"{name!r} isn't an addendum run name")
    failures = recorded_failures(path)
    failures[name] = {"stage": stage, "error": error, "recorded_at": utc_now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2) + "\n", "utf-8")
    return failures[name]


def build_addendum(failures: dict | None = None) -> dict:
    """Both parts' coverage, each from its own directory."""
    failures = recorded_failures(ADDENDUM_FAILURES) if failures is None else failures
    a = build_manifest(ARRANGE_DIR, ARRANGE_GROUPS, failures)
    b = build_manifest(LAYOUT_DIR, LAYOUT_GROUPS, failures)
    out = {k: v for k, v in a.items() if k not in ("contestants", "recorded_failures", "complete")}
    out.update({"complete": a["complete"] and b["complete"],
                "contestants": {**a["contestants"], **b["contestants"]},
                "recorded_failures": a["recorded_failures"] + b["recorded_failures"]})  # fmt: skip
    return out


def build_manifest(directory: Path = PREDICTIONS_DIR, groups: dict | None = None,
                   failures: dict | None = None) -> dict:  # fmt: skip
    expected = expected_ids()
    groups = GROUPS if groups is None else groups
    contestants = {c: summarize(c, g, expected, directory) for c, g in groups.items()}
    for name, failure in (failures or {}).items():
        if name in contestants and not contestants[name]["complete"]:
            contestants[name]["recorded_failure"] = failure
    accounted = all(s["complete"] or "recorded_failure" in s for s in contestants.values())
    splits = json.loads((ROOT / "data" / "splits" / "manifest.json").read_text("utf-8"))
    rendered = load_requests("test", 1)[0]
    out = {
        "written_at": utc_now(),
        "complete": accounted,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },  # fmt: skip
        "data": {
            "source_sha256": splits["source_sha256"],
            "split_seed": splits["seed"],
            "base_rate_train": splits["base_rate_train"],
            "template_version": rendered["template_version"],
            "customers": {s: len(ids) for s, ids in expected.items()},
        },
        "contestants": contestants,
    }
    if failures is not None:  # post 4 only; the M3 and M6 manifests keep their format
        out["recorded_failures"] = sorted(c for c, v in contestants.items()
                                          if "recorded_failure" in v)  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    ap.add_argument("--m6", action="store_true", help="the M6 contestants (PLAN.md §11)")
    ap.add_argument("--quant", action="store_true", help="post 4's precisions (PLAN.md §12)")
    ap.add_argument("--addendum", action="store_true",
                    help="post 5's addendum, both parts (PLAN.md §12)")  # fmt: skip
    ap.add_argument("--record-failure", metavar="NAME",
                    help="addendum: record a run that couldn't run (--stage, --error)")  # fmt: skip
    ap.add_argument("--stage", default="run")
    ap.add_argument("--error", default="")
    args = ap.parse_args()
    if args.record_failure:
        print(record_addendum_failure(args.record_failure, args.stage, args.error))
        return
    if sum((args.m6, args.quant, args.addendum)) > 1:
        ap.error("pick one of --m6, --quant and --addendum")
    if args.addendum:
        manifest, target = build_addendum(), MANIFEST_ADDENDUM
    elif args.quant:
        manifest = build_manifest(args.predictions_dir, QUANT_GROUPS, recorded_failures())
        target = MANIFEST_QUANT
    else:
        manifest = build_manifest(args.predictions_dir, M6_GROUPS if args.m6 else None)
        target = MANIFEST_M6 if args.m6 else MANIFEST
    print(
        f"{'contestant':<16} {'records':>8} {'complete':>9} {'cost $':>9} {'median ms':>10}  model"
    )
    for name, s in manifest["contestants"].items():
        if "recorded_failure" in s:
            print(f"{name:<16} {'-':>8} {'failed':>9}  {s['recorded_failure'].get('error', '')}")
            continue
        if not s["present"]:
            print(f"{name:<16} {'-':>8} {'no':>9}")
            continue
        versions = "; ".join(s["model_versions"])
        done = "yes" if s["complete"] else "NO"
        median = s["latency_ms_median"] or 0
        print(
            f"{name:<16} {s['records']:>8} {done:>9} {s['cost_usd_total']:>9.4f} "
            f"{median:>10.0f}  {versions}"
        )
    if not manifest["complete"] and not args.allow_incomplete:
        sys.exit(
            "Incomplete: not writing the manifest (pass --allow-incomplete to write it anyway)."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(f"Wrote {target.relative_to(ROOT)} (complete: {manifest['complete']})")


if __name__ == "__main__":
    main()
