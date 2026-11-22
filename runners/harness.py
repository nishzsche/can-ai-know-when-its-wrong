"""Shared loop for every AI contestant runner.

A runner supplies a Predictor (contestant id, model version, predict(request) -> Prediction); the
harness feeds it the rendered System One requests, writes validated prediction records, keeps the
raw responses for audit, and resumes where it stopped.

    --smoke              20 tune customers, written to results/smoke/ (the M2 check)
    --split test         full split, written to results/predictions/ (M3)
    --condition both     out_of_box and base_rate (default); `recalibrated` is computed in M4

Every contestant gets the identical request from data/rendered/<split>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "analysis"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from env import load_env  # noqa: E402
from records import PREDICTIONS_DIR, make_record, utc_now, validate_record  # noqa: E402

RENDER_DIR = ROOT / "data" / "rendered"
SMOKE_DIR = ROOT / "results" / "smoke"
RUN_CONDITIONS = ("out_of_box", "base_rate")
L4_USD_PER_HOUR = 0.44  # JarvisLabs on-demand L4, checked 2026-11-22


@dataclass
class Prediction:
    p_default: float
    latency_ms: float
    cost_usd: float
    raw: dict = field(default_factory=dict)


class Predictor(Protocol):
    contestant: str
    model_version: str
    meta: dict

    def predict(self, request: dict) -> Prediction: ...


def gpu_cost(latency_ms: float, usd_per_hour: float) -> float:
    """Amortized GPU cost of one request when requests run one after another."""
    return latency_ms / 3_600_000 * usd_per_hour


def load_requests(split: str, limit: int | None = None) -> list[dict]:
    path = RENDER_DIR / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def build_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["tune", "test"], default="tune")
    ap.add_argument("--condition", choices=[*RUN_CONDITIONS, "both"], default="both")
    ap.add_argument("--limit", type=int, help="only the first N customers of the split")
    ap.add_argument("--smoke", action="store_true", help="20 tune customers into results/smoke/")
    ap.add_argument(
        "--out-dir", type=Path, help="default: results/predictions (smoke: results/smoke)"
    )
    return ap


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.split, args.limit = "tune", args.limit or 20
    if args.out_dir is None:
        args.out_dir = SMOKE_DIR if args.smoke else PREDICTIONS_DIR
    args.conditions = RUN_CONDITIONS if args.condition == "both" else (args.condition,)
    return args


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text("utf-8").splitlines():
        if line:
            rec = json.loads(line)
            validate_record(rec)
            keys.add((rec["condition"], rec["split"], rec["id"]))
    return keys


def run(predictor: Predictor, args: argparse.Namespace, log: Callable[[str], None] = print) -> Path:
    """Predict every (condition, customer) not already in the output file; append as we go."""
    load_env()
    requests = load_requests(args.split, args.limit)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    out_path = out_dir / f"{predictor.contestant}.jsonl"
    raw_path = out_dir / "raw" / f"{predictor.contestant}.jsonl"
    meta_path = out_dir / "raw" / f"{predictor.contestant}.meta.json"
    meta_path.write_text(
        json.dumps({"model_version": predictor.model_version, "started": utc_now(),
                    **predictor.meta}, indent=2) + "\n", "utf-8")  # fmt: skip
    done = _done_keys(out_path)
    log(f"{predictor.contestant}: {predictor.model_version}")
    log(f"{len(requests)} {args.split} customers x {len(args.conditions)} condition(s); "
        f"{len(done)} records already in {out_path}")  # fmt: skip

    with (
        out_path.open("a", encoding="utf-8", newline="\n") as out,
        raw_path.open("a", encoding="utf-8", newline="\n") as raw,
    ):
        for condition in args.conditions:
            ps, ys, start, n_new = [], [], time.perf_counter(), 0
            for row in requests:
                if (condition, args.split, row["id"]) in done:
                    continue
                pred = predictor.predict(row["requests"][condition])
                rec = make_record(
                    id=row["id"], split=args.split, contestant=predictor.contestant,
                    condition=condition, model_version=predictor.model_version,
                    p_default=pred.p_default, label=row["label"], latency_ms=pred.latency_ms,
                    cost_usd=pred.cost_usd, ts=utc_now(),
                )  # fmt: skip
                out.write(json.dumps(rec) + "\n")
                raw.write(json.dumps({"id": row["id"], "split": args.split,
                                      "condition": condition, "raw": pred.raw}) + "\n")  # fmt: skip
                out.flush()
                raw.flush()
                ps.append(pred.p_default)
                ys.append(row["label"])
                n_new += 1
            if n_new:
                secs = time.perf_counter() - start
                extremes = sum(p in (0.0, 1.0) for p in ps)
                log(f"  {condition}: {n_new} new in {secs:.1f}s | mean p {sum(ps) / n_new:.3f} "
                    f"vs actual {sum(ys) / n_new:.3f} | exactly 0 or 1: {extremes}")  # fmt: skip
                if extremes > 0.1 * n_new:
                    log("  WARNING: many probabilities are exactly 0 or 1; check the readout.")
            else:
                log(f"  {condition}: nothing new")
    return out_path
