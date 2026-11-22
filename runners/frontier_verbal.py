"""Frontier chat model asked for its probability in words: GPT-6.2 Sol through OpenRouter.

    uv run --group runners python runners/frontier_verbal.py --smoke
    uv run --group runners python runners/frontier_verbal.py --split test

The "ask ChatGPT how sure it is" contestant. It gets the same account summary and question as
every other AI contestant, plus one fixed line asking for a probability from 0 to 100 as JSON.

- Model: `openai/gpt-6.2-sol`, what ChatGPT starts new chats on (user's choice, 2026-11-22).
- Reasoning: Sol always reasons; its published default effort is "medium", sent explicitly so a
  later change of default can't silently change the contestant.
- Sampling: Sol accepts no temperature; `seed` is fixed for best-effort repeatability.
- Cost: OpenRouter reports each request's real cost in `usage.cost`; that's what we record.
- Needs OPENROUTER_API_KEY (environment, or the git-ignored .env).
"""

from __future__ import annotations

import json
import math
import os
import time

from harness import (
    Prediction,
    PredictionFailed,
    build_parser,
    http_client,
    resolve_args,
    run,
)

MODEL = "openai/gpt-6.2-sol"
REASONING_EFFORT = "medium"  # Sol's default_effort in OpenRouter's model metadata (2026-11-22)
SEED = 0
PROMPT_VERSION = "v1"
BASE_URL = "https://openrouter.ai/api/v1"
ANSWER_LINE = (
    "Answer with the probability, from 0 to 100, that this customer misses next month's "
    'payment. Reply as JSON: {"probability": <number>}'
)
SCHEMA = {
    "type": "object",
    "properties": {"probability": {"type": "number"}},
    "required": ["probability"],
    "additionalProperties": False,
}
QUESTION_ID = "will_default"


def build_prompt(request: dict) -> str:
    """Summary, then the contestant-wide question (with the base-rate line if any), then the
    answer format. The first two parts are byte-identical to what every other contestant gets."""
    instructions = request["questions"][QUESTION_ID]["instructions"]
    return f"{request['state']}\n\n{instructions}\n\n{ANSWER_LINE}"


def parse_probability(content: str | None) -> float:
    """Return P(default) in [0, 1] from the model's JSON, or raise ValueError."""
    if not content:
        raise ValueError("empty answer")
    value = json.loads(content)["probability"]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"probability is not a number: {value!r}")
    if not 0 <= value <= 100:
        raise ValueError(f"probability out of range: {value!r}")
    return float(value) / 100


class FrontierVerbalPredictor:
    contestant = "frontier-verbal"

    def __init__(self, base_url: str = BASE_URL, api_key: str | None = None, retries: int = 2):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY (environment or the git-ignored .env).")
        self.client = OpenAI(
            base_url=base_url, api_key=key, max_retries=5, timeout=300,
            http_client=http_client(timeout=300),  # kept alive, IPv4 (see harness.http_client)
        )  # fmt: skip
        self.retries = retries
        # OpenRouter serves Sol without a dated snapshot; every record's `ts` carries the date.
        self.model_version = f"{MODEL} (effort {REASONING_EFFORT}; prompt {PROMPT_VERSION})"
        self.meta = {"model": MODEL, "reasoning_effort": REASONING_EFFORT, "seed": SEED,
                     "prompt_version": PROMPT_VERSION, "answer_line": ANSWER_LINE,
                     "base_url": base_url}  # fmt: skip

    def _ask(self, prompt: str):
        return self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "default_probability", "strict": True, "schema": SCHEMA},
            },
            seed=SEED,
            extra_body={"reasoning": {"effort": REASONING_EFFORT}},
        )

    def predict(self, request: dict) -> Prediction:
        prompt = build_prompt(request)
        attempts, total_cost, start = [], 0.0, time.perf_counter()
        for _ in range(self.retries + 1):
            resp = self._ask(prompt)
            body = resp.model_dump(mode="json")
            usage = body.get("usage") or {}
            total_cost += float(usage.get("cost") or 0.0)
            choice = (body.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            raw = {
                "model": body.get("model"),
                "provider": body.get("provider"),
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "usage": usage,
            }
            attempts.append(raw)
            try:
                p = parse_probability(content)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raw["error"] = str(exc)
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            return Prediction(
                p_default=p,
                latency_ms=latency_ms,
                cost_usd=total_cost,  # includes any retried attempts: the real spend
                raw={"attempts": attempts},
            )
        raise PredictionFailed(
            attempts[-1].get("error", "no usable answer"), {"attempts": attempts}
        )


def main() -> None:
    args = resolve_args(build_parser(__doc__).parse_args())
    from env import load_env

    load_env()
    run(FrontierVerbalPredictor(), args)


if __name__ == "__main__":
    main()
