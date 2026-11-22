"""Check M3 coverage and write results/run_manifest.json.

    uv run python runners/manifest.py                     # exit 1 if anything is missing
    uv run python runners/manifest.py --allow-incomplete  # write it anyway (e.g. on the L4)

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

from harness import RUN_CONDITIONS, load_requests
from records import PREDICTIONS_DIR, TEST_ONLY_CONDITIONS, read_records, utc_now

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "run_manifest.json"
SPLITS = ("tune", "test")

AI_CONTESTANTS = ("jev", "kev-0.8b", "kev-4b", "laya", "qwen-logprobs", "frontier-verbal")
GROUPS = {
    **{c: [(cond, s) for cond in RUN_CONDITIONS for s in SPLITS] for c in AI_CONTESTANTS},
    "scorecard": [("trained", "tune"), ("trained", "test")],
    "gbm": [("trained", "tune"), ("trained", "test"), ("trained_calibrated", "test")],
}


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


def build_manifest(directory: Path = PREDICTIONS_DIR) -> dict:
    expected = expected_ids()
    contestants = {c: summarize(c, g, expected, directory) for c, g in GROUPS.items()}
    splits = json.loads((ROOT / "data" / "splits" / "manifest.json").read_text("utf-8"))
    rendered = load_requests("test", 1)[0]
    return {
        "written_at": utc_now(),
        "complete": all(s["complete"] for s in contestants.values()),
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    args = ap.parse_args()
    manifest = build_manifest(args.predictions_dir)
    print(
        f"{'contestant':<16} {'records':>8} {'complete':>9} {'cost $':>9} {'median ms':>10}  model"
    )
    for name, s in manifest["contestants"].items():
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
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} (complete: {manifest['complete']})")


if __name__ == "__main__":
    main()
