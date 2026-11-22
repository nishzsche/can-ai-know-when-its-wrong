"""Qwen3.5-4B through vLLM, with P(yes) read from token logprobs by jevper (method="logprobs").

Start vLLM on the L4 first, with exactly the command in STATUS.md (thinking off, pinned revision),
then:

    uv run --group runners python runners/qwen_logprobs.py --smoke
    uv run --group runners python runners/qwen_logprobs.py --split test

jevper turns the System One request into a chat prompt and reads P(yes) as the softmax over the
answer labels' logprobs of the first answer token. Temperature is left at 1.0 so the logprobs are
the model's own distribution even if the server reports processed rather than raw logprobs.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
import urllib.request

from harness import L4_USD_PER_HOUR, Prediction, build_parser, gpu_cost, resolve_args, run

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERVED_NAME = "qwen3.5-4b"
QUESTION_ID = "will_default"


class QwenPredictor:
    contestant = "qwen-logprobs"

    def __init__(self, base_url: str, served_name: str, usd_per_hour: float):
        from jevper import SystemOneClient
        from openai import OpenAI

        base_url = base_url.rstrip("/")
        with urllib.request.urlopen(f"{base_url}/models", timeout=30) as resp:  # noqa: S310
            models = json.loads(resp.read().decode("utf-8"))
        served = {m["id"]: m for m in models.get("data", [])}
        if served_name not in served:
            raise SystemExit(f"{served_name!r} isn't served at {base_url}; got {sorted(served)}")
        root = served[served_name].get("root", "")
        if MODEL not in root:
            raise SystemExit(f"{served_name!r} is {root!r}, expected {MODEL}")

        key = os.environ.get("VLLM_API_KEY", "EMPTY")
        self.client = SystemOneClient(
            OpenAI(base_url=base_url, api_key=key),
            model=served_name,
            method="logprobs",
            temperature=1.0,
            max_concurrency=1,
        )
        jevper_version = importlib.metadata.version("jevper")
        self.model_version = (
            f"{MODEL}@{REVISION[:8]}; jevper {jevper_version} logprobs; thinking off"
        )
        self.usd_per_hour = usd_per_hour
        self.meta = {"server_model": served[served_name], "base_url": base_url,
                     "jevper": jevper_version, "usd_per_hour": usd_per_hour}  # fmt: skip

    def predict(self, request: dict) -> Prediction:
        start = time.perf_counter()
        resp = self.client.system_one(state=request["state"], questions=request["questions"])
        latency_ms = (time.perf_counter() - start) * 1000
        answer = resp.answers[QUESTION_ID]
        raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {"repr": repr(resp)}
        return Prediction(
            p_default=float(answer.noul),
            latency_ms=latency_ms,
            cost_usd=gpu_cost(latency_ms, self.usd_per_hour),
            raw=raw,
        )


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--served-name", default=SERVED_NAME)
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    args = resolve_args(ap.parse_args())
    run(QwenPredictor(args.base_url, args.served_name, args.usd_per_hour), args)


if __name__ == "__main__":
    main()
