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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "analysis"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from env import load_env  # noqa: E402
from records import (  # noqa: E402
    PREDICTIONS_DIR,
    TEST_ONLY_CONDITIONS,
    make_record,
    utc_now,
    validate_record,
)

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


class PredictionFailed(Exception):
    """No usable probability for this customer (refusal, malformed answer after retries).

    The harness logs it, writes nothing for that customer, and a later run retries it.
    """

    def __init__(self, reason: str, raw: dict | None = None):
        super().__init__(reason)
        self.raw = raw or {}


class Predictor(Protocol):
    contestant: str
    model_version: str
    meta: dict

    def predict(self, request: dict) -> Prediction: ...


class RateLimiter:
    """Spaces request starts so a run stays under a requests-per-minute cap (thread-safe)."""

    def __init__(self, per_minute: float | None):
        import threading

        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + self.interval
        time.sleep(max(0.0, start - time.monotonic()))


def http_client(timeout: float = 300):
    """One kept-alive HTTP client that connects over IPv4 only.

    On the laptop network, OpenRouter's IPv6 addresses time out (21 s each, then IPv4 works in
    20 ms), which added ~42 s to every fresh connection and would have been recorded as the
    model's latency. Forcing IPv4 and reusing the connection keeps latency honest everywhere.
    """
    import httpx2

    transport = httpx2.HTTPTransport(local_address="0.0.0.0", retries=2)
    return httpx2.Client(transport=transport, timeout=timeout)


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
    ap.add_argument(
        "--workers", type=int, default=1,
        help="parallel requests; API runners only (on one GPU they'd queue and skew latency)",
    )  # fmt: skip
    return ap


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.split, args.limit = "tune", args.limit or 20
    if args.out_dir is None:
        args.out_dir = SMOKE_DIR if args.smoke else PREDICTIONS_DIR
    args.conditions = RUN_CONDITIONS if args.condition == "both" else (args.condition,)
    return args


def _existing(path: Path) -> tuple[set[tuple[str, str, str]], set[str]]:
    """(condition, split, id) keys already written, and the model versions that wrote them.

    Recalibrated records are written later by analysis/recalibrate.py, under a version that
    names the calibrator, so they don't count as another model version here.
    """
    if not path.exists():
        return set(), set()
    keys, versions = set(), set()
    for line in path.read_text("utf-8").splitlines():
        if line:
            rec = json.loads(line)
            validate_record(rec)
            keys.add((rec["condition"], rec["split"], rec["id"]))
            if rec["condition"] not in TEST_ONLY_CONDITIONS:
                versions.add(rec["model_version"])
    return keys, versions


def environment() -> dict:
    """Software and hardware this run used (recorded in the run's metadata)."""
    import importlib.metadata
    import platform
    import subprocess

    packages = {}
    for name in ("torch", "transformers", "laya", "vegaml", "jevper", "openai", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "gpu": gpu, "packages": packages}  # fmt: skip


def _parallel(fn, items, workers):
    """Yield fn(item) results as they finish; on any error, cancel everything still queued."""
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(fn, item) for item in items]
    try:
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run(
    predictor: Predictor,
    args: argparse.Namespace,
    log: Callable[[str], None] = print,
    requests: list[dict] | None = None,
) -> Path:
    """Predict every (condition, customer) not already in the output file; append as we go.

    `requests` replaces the split's rendered customers (checks that use a sample pass theirs).
    """
    load_env()
    if requests is None:
        requests = load_requests(args.split, args.limit)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    out_path = out_dir / f"{predictor.contestant}.jsonl"
    raw_path = out_dir / "raw" / f"{predictor.contestant}.jsonl"
    meta_path = out_dir / "raw" / f"{predictor.contestant}.meta.json"
    done, versions = _existing(out_path)
    if versions - {predictor.model_version}:
        raise SystemExit(
            f"{out_path} already holds records from {sorted(versions)}, but this run is "
            f"{predictor.model_version!r}. One file must not mix model versions: move the old "
            f"file aside (or delete it) to start over with the new version."
        )
    # One metadata entry per run, so a later run doesn't erase an earlier one's details.
    runs = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else []
    if isinstance(runs, dict):  # older single-run format
        runs = [runs]
    runs.append({"model_version": predictor.model_version, "started": utc_now(),
                 "split": args.split, "conditions": list(args.conditions), "limit": args.limit,
                 "workers": getattr(args, "workers", 1),
                 "environment": environment(), **predictor.meta})  # fmt: skip
    meta_path.write_text(json.dumps(runs, indent=2) + "\n", "utf-8")
    log(f"{predictor.contestant}: {predictor.model_version}")
    log(f"{len(requests)} {args.split} customers x {len(args.conditions)} condition(s); "
        f"{len(done)} records already in {out_path}")  # fmt: skip

    with (
        out_path.open("a", encoding="utf-8", newline="\n") as out,
        raw_path.open("a", encoding="utf-8", newline="\n") as raw,
    ):
        for condition in args.conditions:
            ps, ys, start, n_new, cost, failed = [], [], time.perf_counter(), 0, 0.0, 0
            pending = [r for r in requests if (condition, args.split, r["id"]) not in done]

            def attempt(row, condition=condition):
                try:
                    return row, predictor.predict(row["requests"][condition]), None
                except PredictionFailed as failure:
                    return row, None, failure

            workers = getattr(args, "workers", 1) or 1
            if workers == 1:
                results = map(attempt, pending)
            else:  # API runners only: each request's latency is still measured on its own
                results = _parallel(attempt, pending, workers)
            for row, pred, exc in results:
                if exc is not None:
                    failed += 1
                    log(f"  {condition} {row['id']}: no usable answer ({exc}); will retry next run")
                    raw.write(json.dumps({"id": row["id"], "split": args.split,
                                          "condition": condition, "failed": str(exc),
                                          "raw": exc.raw}) + "\n")  # fmt: skip
                    raw.flush()
                    continue
                cost += pred.cost_usd
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
                    f"vs actual {sum(ys) / n_new:.3f} | exactly 0 or 1: {extremes} | "
                    f"cost ${cost:.4f}")  # fmt: skip
                if extremes > 0.1 * n_new:
                    log("  WARNING: many probabilities are exactly 0 or 1; check the readout.")
            else:
                log(f"  {condition}: nothing new")
            if failed:
                log(f"  {condition}: {failed} customer(s) without a usable answer; re-run to retry")
    return out_path
