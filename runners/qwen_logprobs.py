"""Qwen3.5-4B through vLLM, with P(yes) read from token logprobs by jevper (method="logprobs").

Start vLLM on the L4 first, with exactly the command in STATUS.md (thinking off, pinned revision),
then:

    uv run --group runners python runners/qwen_logprobs.py --smoke
    uv run --group runners python runners/qwen_logprobs.py --split test

jevper turns the System One request into a chat prompt ("Options: A: Yes / B: No") and sends it.
Temperature is left at 1.0 so the logprobs are the model's own distribution even if the server
reports processed rather than raw logprobs.

P(yes) is computed here, not taken from jevper's `noul`. jevper 0.7.12's readout strips whitespace
from the top-20 tokens and lets the *last* match overwrite the rival label's logprob, so a
low-ranked " A" (-10.5) replaces "A" (-1.2) and pushes every probability towards 0 or 1 (found in
the 2026-11-22 L4 smoke test; jevper's value is kept as `jevper_noul` in the raw record). Here each
label's logprob is the logsumexp of every top-20 variant that strips to it, for both labels alike,
and P(yes) is the softmax over the two.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import time
import urllib.request

from harness import (
    L4_USD_PER_HOUR,
    Prediction,
    PredictionFailed,
    build_parser,
    gpu_cost,
    resolve_args,
    run,
)

READOUT = "label-variant logsumexp over first-token top-20"
_OPTION_LINE = re.compile(r"^([A-Z]):\s*(Yes|No)\s*$", re.MULTILINE)


def option_letters(request: dict) -> tuple[str, str]:
    """(yes_letter, no_letter) from the options block jevper put in the prompt it sent."""
    found: dict[str, str] = {}
    for message in request.get("input", []):
        content = message.get("content")
        if isinstance(content, str):
            for letter, word in _OPTION_LINE.findall(content):
                found.setdefault(word, letter)
    if set(found) != {"Yes", "No"}:
        raise ValueError(f"couldn't find 'X: Yes' and 'Y: No' option lines in the request: {found}")
    return found["Yes"], found["No"]


def p_yes_from_top_logprobs(
    top_logprobs: list[dict], yes_letter: str, no_letter: str
) -> tuple[float, list[str]]:
    """Softmax over the two labels, each the logsumexp of its whitespace/case variants.

    Returns (P(yes), labels missing from the top list). A missing label gets probability 0.
    """
    per_label: dict[str, list[float]] = {yes_letter: [], no_letter: []}
    for entry in top_logprobs:
        lp = float(entry["logprob"])
        if math.isnan(lp) or lp > 0:
            raise ValueError(f"impossible logprob {lp!r} for token {entry.get('token')!r}")
        label = entry["token"].strip().upper()
        if label in per_label:
            per_label[label].append(lp)
    missing = [label for label, lps in per_label.items() if not lps]
    if len(missing) == 2:
        raise ValueError("neither label is among the top logprobs")
    lse = {
        label: (max(lps) + math.log(sum(math.exp(x - max(lps)) for x in lps))) if lps else -math.inf
        for label, lps in per_label.items()
    }
    top = max(lse.values())
    weights = {label: math.exp(v - top) for label, v in lse.items()}
    return weights[yes_letter] / (weights[yes_letter] + weights[no_letter]), missing


def p_yes_from_raw(raw: dict, question_id: str) -> tuple[float, list[str]]:
    """Recompute P(yes) from the request and first answer token that jevper recorded."""
    attempts = [
        a for a in raw.get("debug", {}).get("llm_attempts", [])
        if a.get("question_id") == question_id and a.get("response")
    ]  # fmt: skip
    if not attempts:
        raise ValueError("no recorded LLM attempt to read logprobs from")
    attempt = attempts[-1]  # the attempt whose answer jevper accepted
    yes_letter, no_letter = option_letters(attempt["request"])
    for item in attempt["response"].get("output", []):
        for part in item.get("content") or []:
            for token in part.get("logprobs") or []:
                if token["token"].strip():  # first non-whitespace token, as jevper reads it
                    if token["token"].strip().upper() not in (yes_letter, no_letter):
                        raise ValueError(f"first answer token {token['token']!r} isn't a label")
                    return p_yes_from_top_logprobs(token["top_logprobs"], yes_letter, no_letter)
    raise ValueError("no answer token with logprobs in the response")


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
            f"{MODEL}@{REVISION[:8]}; jevper {jevper_version} logprobs; thinking off; "
            f"readout: {READOUT}"
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
        try:
            p_yes, missing = p_yes_from_raw(raw, QUESTION_ID)
        except (ValueError, KeyError, TypeError) as exc:
            raise PredictionFailed(f"logprob readout failed: {exc}", raw) from exc
        raw["jevper_noul"] = float(answer.noul)
        raw["readout"] = {"method": READOUT, "p_yes": p_yes, "missing_labels": missing}
        return Prediction(
            p_default=p_yes,
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
