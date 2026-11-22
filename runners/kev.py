"""Kev-0.8B / Kev-4B through a local Kev server (System One API).

Start the server on the L4 first (in a clone of github.com/jaredpalmer/kev, bf16 on CUDA), at the
exact v1.0 commit (see CHECKPOINTS and L4_RUNBOOK.md):

    uv run --extra serve python -m kev.serve \
        --run jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --port 8009

then run, from this repo:

    uv run python runners/kev.py --size 4b --smoke             # M2 check: 20 tune customers
    uv run python runners/kev.py --size 4b --split test        # M3

The runner refuses to start if the server isn't serving the pinned v1.0 checkpoint.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from harness import L4_USD_PER_HOUR, Prediction, build_parser, gpu_cost, resolve_args, run

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


class KevPredictor:
    def __init__(self, size: str, base_url: str, usd_per_hour: float, check_version: bool = True):
        repo, tag_commit, weights_rev = CHECKPOINTS[size]
        self.contestant = f"kev-{size}"
        self.model_version = f"{repo}@v1.0 ({tag_commit[:8]}; weights {weights_rev})"
        self.base_url = base_url.rstrip("/")
        self.usd_per_hour = usd_per_hour
        models = _http_json(f"{self.base_url}/v1/models")
        served = json.dumps(models)
        accepted = (tag_commit[:8], weights_rev, f"{repo}@v1.0")
        if check_version and not any(ident in served for ident in accepted):
            raise SystemExit(
                f"The server at {self.base_url} doesn't report {repo} v1.0 in /v1/models "
                f"(looked for any of {accepted}). Start it with --run {repo}@{tag_commit}, or "
                f"check /v1/models and pass --skip-version-check if it names the checkpoint "
                f"differently."
            )
        self.meta = {"server_models": models, "base_url": self.base_url,
                     "usd_per_hour": usd_per_hour}  # fmt: skip

    def predict(self, request: dict) -> Prediction:
        body = {"model": "kev-latest", **request}
        start = time.perf_counter()
        resp = _http_json(f"{self.base_url}/v1/systemone", body)
        latency_ms = (time.perf_counter() - start) * 1000
        p = float(resp["answers"][QUESTION_ID]["noul"])
        return Prediction(
            p_default=p,
            latency_ms=latency_ms,
            cost_usd=gpu_cost(latency_ms, self.usd_per_hour),
            raw={
                "answer": resp["answers"][QUESTION_ID],
                "usage": resp.get("usage"),
                "server_latency_ms": resp.get("latency_ms"),
            },  # fmt: skip
        )


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--size", choices=sorted(CHECKPOINTS), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    ap.add_argument("--skip-version-check", action="store_true")
    args = resolve_args(ap.parse_args())
    predictor = KevPredictor(
        args.size, args.base_url, args.usd_per_hour, check_version=not args.skip_version_check
    )
    run(predictor, args)


if __name__ == "__main__":
    main()
