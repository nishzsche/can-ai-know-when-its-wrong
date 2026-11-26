"""Kev-0.8B / Kev-4B through a local Kev server (System One API).

Start the server on the L4 first (in a clone of github.com/jaredpalmer/kev, bf16 on CUDA), at the
exact v1.0 commit (see CHECKPOINTS and L4_RUNBOOK.md):

    uv run --extra serve python -m kev.serve \
        --run jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --port 8009

then run, from this repo:

    uv run python runners/kev.py --size 4b --smoke             # M2 check: 20 tune customers
    uv run python runners/kev.py --size 4b --split test        # M3

The runner refuses to start if the server isn't serving the pinned v1.0 checkpoint.

M6 (`kev-4b-ft`, `kev-0.8b-ft`, condition `trained`): serve the fine-tuned run directory at
temperature 1.0 (PLAN.md §11, choice 3) and point the runner at it:

    KEV_TEMPERATURE=1.0 uv run --extra serve python -m kev.serve \
        --run /home/runs/kev-4b-ft --port 8009
    uv run python runners/kev.py --size 4b --finetuned /home/runs/kev-4b-ft --split test

Post 4 (`kev-4b-ft-<precision>`, condition `trained`): serve the fine-tuned run through
runners/kev_quant_serve.py at one precision (PLAN.md §12) and pass the same --precision here. The
runner refuses a server that reports another precision, and after each request reads the server's
peak GPU memory (outside the timed call). `--reproduce-check 50` (bf16 only) compares the first 50
tune customers with M6's kev-<size>-ft predictions, writes nothing to results/predictions, and
stops on a mismatch:

    uv run python runners/kev.py --size 4b --finetuned /home/runs/kev-4b-ft --precision bf16 \\
        --reproduce-check 50
    uv run python runners/kev.py --size 4b --finetuned /home/runs/kev-4b-ft --precision bnb-nf4 \\
        --split tune
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness import (
    L4_USD_PER_HOUR,
    TRAINED,
    Prediction,
    build_parser,
    gpu_cost,
    reproduction_check,
    resolve_args,
    run,
)
from quantize import PRECISIONS, contestant_name
from speed import SPEED_TAGS, speed_name

# Kev 1.0: (Hub repo, commit the v1.0 tag points at, weights revision in the README's Kev 1.0
# table). The tag sits on a later model-card commit; on 2026-11-22 we checked that the adapter,
# head and tokenizer files (LFS hashes) are identical at both commits.
CHECKPOINTS = {
    "0.8b": ("jaredpalmer/kev-0.8b", "788ddbdd65715bb03a56788c822f6c632c9a551d", "9a45d25e"),
    "4b": ("jaredpalmer/kev-4b", "591dcb5bd6d05eb0b5131ea6608f93f10243335c", "139fdd94"),
}
QUESTION_ID = "will_default"


def _http_json(url: str, body: dict | None = None, timeout: float = 120, tries: int = 3) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local URL)
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _find_key(obj, key: str) -> list:
    """Every value stored under `key` anywhere in a JSON object."""
    if isinstance(obj, dict):
        return [v for k, v in obj.items() if k == key] + [
            x for v in obj.values() for x in _find_key(v, key)
        ]
    if isinstance(obj, list):
        return [x for v in obj for x in _find_key(v, key)]
    return []


class KevPredictor:
    def __init__(self, size: str, base_url: str, usd_per_hour: float, check_version: bool = True,
                 finetuned: Path | None = None, precision: str | None = None):  # fmt: skip
        repo, tag_commit, weights_rev = CHECKPOINTS[size]
        self.contestant = f"kev-{size}"
        self.model_version = f"{repo}@v1.0 ({tag_commit[:8]}; weights {weights_rev})"
        self.base_url = base_url.rstrip("/")
        self.usd_per_hour = usd_per_hour
        models = _http_json(f"{self.base_url}/v1/models")
        served = json.dumps(models)
        accepted = (tag_commit[:8], weights_rev, f"{repo}@v1.0")
        if finetuned is not None:  # M6: a run directory written by kev.train --init_from
            self.contestant = f"kev-{size}-ft"
            weights = sorted(finetuned.glob("*.safetensors")) + [finetuned / "head.pt"]
            digest = hashlib.sha256(b"".join(p.read_bytes() for p in weights)).hexdigest()
            self.model_version = (f"kev-{size}-ft: LoRA delta from {repo}@v1.0 ({tag_commit[:8]}), "
                                  f"trained on train; weights {digest[:12]}; T 1.0")  # fmt: skip
            accepted = (finetuned.name,)
            temps = _find_key(models, "temperature")
            if check_version and any(abs(float(t) - 1.0) > 1e-9 for t in temps):
                raise SystemExit(f"The server runs at temperature {temps}; start it with "
                                 f"KEV_TEMPERATURE=1.0 (PLAN.md §11, choice 3).")  # fmt: skip
        if check_version and not any(ident in served for ident in accepted):
            raise SystemExit(
                f"The server at {self.base_url} doesn't report {repo} v1.0 in /v1/models "
                f"(looked for any of {accepted}). Start it with --run {repo}@{tag_commit}, or "
                f"check /v1/models and pass --skip-version-check if it names the checkpoint "
                f"differently."
            )
        self.meta = {"server_models": models, "base_url": self.base_url,
                     "usd_per_hour": usd_per_hour}  # fmt: skip
        self.precision = precision
        if precision is not None:
            self._check_precision(size, finetuned, precision)

    def _check_precision(self, size: str, finetuned: Path | None, precision: str) -> None:
        """Post 4: the server must be runners/kev_quant_serve.py at exactly this precision."""
        if finetuned is None:
            raise SystemExit("--precision is for the fine-tuned models (pass --finetuned too)")
        try:
            served = _http_json(f"{self.base_url}/v1/quantization", tries=1)
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"{self.base_url} has no /v1/quantization ({exc}); serve the run through "
                f"runners/kev_quant_serve.py for --precision"
            ) from None
        # A speed tag's server reports its base precision under "precision" and the tag under
        # "tag" (speed.apply); post 4's servers have no "tag".
        reported = served.get("tag", served.get("precision"))
        if reported != precision:
            raise SystemExit(
                f"The server reports precision {reported!r}, not {precision!r}; "
                f"restart it with --precision {precision}."
            )
        if precision in SPEED_TAGS:  # the speed addendum: Kev's fast setup, as the server reports
            graphs = served.get("cuda_graphs")
            if not isinstance(graphs, dict):
                raise SystemExit("The speed tags run with Kev's CUDA graphs; the server has none.")
            self.contestant = speed_name(f"kev-{size}-ft", precision)
            setup = f"CUDA graphs on, fused kernels {'on' if served.get('fused') else 'off'}"
            if served.get("fast_layers_missing"):
                setup += f"; fast-layer kernels missing: {', '.join(served['fast_layers_missing'])}"
        else:
            if served.get("cuda_graphs") is not False or served.get("fused") is not False:
                raise SystemExit("Post 4 serves its precisions without CUDA graphs or fusion.")
            self.contestant = contestant_name(f"kev-{size}-ft", precision)
            setup = "CUDA graphs and fused kernels off"
        self.model_version = (
            f"{self.model_version}; served by runners/kev_quant_serve.py ({setup}); "
            f"{served['description']}"
        )
        self.meta["quantization"] = served

    def predict(self, request: dict) -> Prediction:
        body = {"model": "kev-latest", **request}
        start = time.perf_counter()
        resp = _http_json(f"{self.base_url}/v1/systemone", body)
        latency_ms = (time.perf_counter() - start) * 1000
        p = float(resp["answers"][QUESTION_ID]["noul"])
        raw = {
            "answer": resp["answers"][QUESTION_ID],
            "usage": resp.get("usage"),
            "server_latency_ms": resp.get("latency_ms"),
        }
        if self.precision is not None:  # after the timed call, so it never counts as latency
            served = _http_json(f"{self.base_url}/v1/quantization")
            raw["gpu_max_allocated"] = served["memory_now"].get("max_allocated")
            if isinstance(served.get("cuda_graphs"), dict):
                raw["cuda_graphs"] = served["cuda_graphs"]
        return Prediction(p_default=p, latency_ms=latency_ms,
                          cost_usd=gpu_cost(latency_ms, self.usd_per_hour), raw=raw)  # fmt: skip


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--size", choices=sorted(CHECKPOINTS), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    ap.add_argument("--skip-version-check", action="store_true")
    ap.add_argument("--finetuned", type=Path, help="M6: the fine-tuned run directory being served")
    ap.add_argument("--precision", choices=(*PRECISIONS, *SPEED_TAGS),
                    help="post 4: the precision or speed tag the launcher serves")  # fmt: skip
    ap.add_argument("--reproduce-check", type=int, metavar="N",
                    help="bf16: compare N tune customers with M6; bf16-tuned / bnb-nf4-tuned: "
                         "with post 4's bf16 / bnb-nf4 run; then stop")  # fmt: skip
    args = ap.parse_args()
    if args.precision and args.finetuned is None:
        ap.error("--precision needs --finetuned")
    reference = reproduce_reference(f"kev-{args.size}-ft", args.precision)
    if args.reproduce_check and reference is None:
        ap.error("--reproduce-check runs with --precision bf16, bf16-tuned or bnb-nf4-tuned")
    if args.finetuned is not None:
        args.condition = TRAINED
    args = resolve_args(args)
    predictor = KevPredictor(
        args.size,
        args.base_url,
        args.usd_per_hour,
        check_version=not args.skip_version_check,
        finetuned=args.finetuned,
        precision=args.precision,
    )
    if args.reproduce_check:
        reproduction_check(predictor, reference, args.reproduce_check)
        return
    run(predictor, args)


def reproduce_reference(model: str, precision: str | None) -> str | None:
    """The run a reproduction check compares with: M6's for bf16, post 4's for the two tuned tags
    whose arithmetic doesn't change (runners/speed.py REPRODUCE); None if there isn't one."""
    from speed import REPRODUCE

    if precision == "bf16":
        return model
    if precision in REPRODUCE:
        return contestant_name(model, REPRODUCE[precision])
    return None


if __name__ == "__main__":
    main()
