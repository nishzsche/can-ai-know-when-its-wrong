"""Post 4 speed addendum: customers per second at 1, 8 and 32 at a time (PLAN.md §12).

Kev: the server runners/kev_quant_serve.py runs at one speed tag (or post 4 precision). The same N
tune customers go in with 1, 8 and 32 requests in flight; Kev's server batches whatever is waiting
when its model thread frees up (with CUDA graphs on; without them it takes one request at a time).
Qwen: in-process, the same N tune customers in batches of 1, 8 and 32
(QwenFTPredictor.predict_batch, one left-padded forward pass per batch).

Before timing, every batch size runs once untimed on the first customers (warm-up: CUDA graph
capture, torch.compile). Writes results/checks/speed/<name>.json: per batch size, the customers,
the seconds, customers per second, cost per 1,000 decisions at the L4 rate, and the largest
difference from the one-at-a-time answers (batching shouldn't change an answer beyond rounding).
Tune customers only: nothing here is scored for honesty, and nothing touches test.

    uv run python runners/throughput.py --kev 4b --finetuned /home/runs/kev-4b-ft \\
        --precision fp8-dynamic --base-url http://127.0.0.1:8009
    uv run --no-sync python runners/throughput.py --qwen /home/runs/qwen-ft --precision fp8-dynamic
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import L4_USD_PER_HOUR, ROOT, TRAINED, load_requests, request_for  # noqa: E402
from quantize import PRECISIONS  # noqa: E402
from speed import BATCHES, SPEED_TAGS  # noqa: E402

OUT_DIR = ROOT / "results" / "checks" / "speed"
WARMUP = 64  # customers run untimed at each batch size first


def summarize(n: int, seconds: float, usd_per_hour: float) -> dict:
    """Throughput and cost for n customers answered in `seconds` of GPU time."""
    per_s = n / seconds if seconds > 0 else float("inf")
    return {"customers": n, "seconds": seconds, "customers_per_second": per_s,
            "cost_per_1000_usd": 1000 / per_s / 3600 * usd_per_hour}  # fmt: skip


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    common = a.keys() & b.keys()
    return max((abs(a[k] - b[k]) for k in common), default=0.0)


def sweep(score_batch, rows: list[dict], batches=BATCHES, usd_per_hour=L4_USD_PER_HOUR,
          warmup: int = WARMUP, settle=None, log=print) -> dict:  # fmt: skip
    """score_batch(rows) -> (answers {id: p}, seconds, per-request ms or None). The same rows at
    every batch size; answers compared with batch size 1. `settle()` runs after each warm-up
    (Kev: wait for pending CUDA graph captures)."""
    out, first = {}, None
    for size in batches:
        for group in chunks(rows[:warmup], size):  # untimed warm-up
            score_batch(group)
        if settle is not None:
            settle()
        answers, seconds, per_request = {}, 0.0, []
        for group in chunks(rows, size):
            got, secs, ms = score_batch(group)
            answers.update(got)
            seconds += secs
            per_request += ms or []
        entry = summarize(len(answers), seconds, usd_per_hour)
        if per_request:
            entry["latency_ms_median"] = statistics.median(per_request)
        if first is None:
            first = answers
        entry["max_abs_diff_vs_batch_1"] = max_abs_diff(first, answers)
        out[str(size)] = entry
        log(f"  {size:>2} at a time: {entry['customers_per_second']:.1f} customers/s, "
            f"${entry['cost_per_1000_usd']:.4f} per 1,000, max |diff| vs one at a time "
            f"{entry['max_abs_diff_vs_batch_1']:.4f}")  # fmt: skip
    return out


def kev_scorer(predictor):
    """Send a group as concurrent requests; seconds = wall time for the whole group. Only the
    scoring call is made (KevPredictor.predict also reads memory after each answer)."""
    from kev import QUESTION_ID, _http_json

    url = f"{predictor.base_url}/v1/systemone"

    def score(group: list[dict]):
        def one(row):
            start = time.perf_counter()
            resp = _http_json(url, {"model": "kev-latest", **request_for(row, TRAINED)})
            ms = (time.perf_counter() - start) * 1000
            return row["id"], float(resp["answers"][QUESTION_ID]["noul"]), ms

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            done = list(pool.map(one, group))
        seconds = time.perf_counter() - start
        return {i: p for i, p, _ in done}, seconds, [ms for _, _, ms in done]

    return score


def kev_settle(base_url: str, timeout_s: float = 120, poll_s: float = 0.5):
    """Wait until the server reports no CUDA graph waiting for capture (it captures when idle)."""
    from kev import _http_json

    def settle() -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            graphs = _http_json(f"{base_url.rstrip('/')}/v1/quantization").get("cuda_graphs")
            if not isinstance(graphs, dict) or graphs.get("pending", 0) == 0:
                return
            time.sleep(poll_s)

    return settle


def qwen_scorer(predictor):
    """One left-padded forward pass per group; seconds = that pass."""

    def score(group: list[dict]):
        ps, ms = predictor.predict_batch([request_for(r, TRAINED) for r in group])
        return {r["id"]: p for r, p in zip(group, ps, strict=True)}, ms / 1000, None

    return score


def write(name: str, model_version: str, result: dict, meta: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"contestant": name, "model_version": model_version,
                                "split": "tune", "by_batch": result, **meta},
                               indent=2) + "\n", "utf-8")  # fmt: skip
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--kev", choices=("0.8b", "4b"), help="a Kev size, served by the launcher")
    who.add_argument("--qwen", type=Path, metavar="ADAPTER", help="Qwen in-process: the adapter")
    ap.add_argument("--finetuned", type=Path, help="Kev: the fine-tuned run directory served")
    ap.add_argument("--precision", choices=(*PRECISIONS, *SPEED_TAGS), required=True)
    ap.add_argument("--n", type=int, default=500, help="tune customers (default 500)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    args = ap.parse_args()
    rows = load_requests("tune", args.n)
    if args.kev:
        from kev import KevPredictor

        if args.finetuned is None:
            ap.error("--kev needs --finetuned")
        predictor = KevPredictor(args.kev, args.base_url, args.usd_per_hour,
                                 finetuned=args.finetuned, precision=args.precision)  # fmt: skip
        score, settle = kev_scorer(predictor), kev_settle(args.base_url)
    else:
        from qwen_ft import QwenFTPredictor

        predictor = QwenFTPredictor(args.qwen, args.device, args.usd_per_hour, args.precision)
        score, settle = qwen_scorer(predictor), None
    print(f"{predictor.contestant}: {len(rows)} tune customers at {BATCHES} at a time")
    result = sweep(score, rows, usd_per_hour=args.usd_per_hour, settle=settle)
    meta = {"quantization": predictor.meta.get("quantization"), "usd_per_hour": args.usd_per_hour,
            "warmup_customers": WARMUP, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                     time.gmtime())}  # fmt: skip
    print(f"Saved -> {write(predictor.contestant, predictor.model_version, result, meta)}")


if __name__ == "__main__":
    main()
