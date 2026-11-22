"""Does the frontier contestant recognize our account summaries as the UCI credit default data?

Why it matters: the summaries show NT$ amounts and April-September 2005. A model that recognizes
the dataset could recall its ~22% default rate, which would flatter its out-of-the-box
calibration: it would get, from memory, the number the `base_rate` condition hands everyone.

Two probes, same model settings as the contestant (reasoning effort medium, seed 0, strict JSON):

1. Recognition: the exact `state` the contestant saw (identical in both conditions), followed by
   RECOGNITION_QUESTION instead of the default question. Run on --n random test customers
   (fixed seed) plus every test customer the memorization tests found Sol reproducing. It names
   "public dataset" as one possibility, so it measures whether Sol *can* place a summary when
   asked, an upper bound on what it might do unprompted.
2. Knowledge: KNOWLEDGE_QUESTION, no summary, asked --repeats times. Does Sol know the rate?

    uv run --group runners python data/recognition.py --model openai/gpt-6.2-sol
    uv run --group runners python data/recognition.py --dry-run

Results: data/memcheck/results/<model>__recognition.json (prompts verbatim, every answer, usage).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent / "runners"))

RESULTS_DIR = HERE / "memcheck" / "results"
RENDERED = HERE / "rendered" / "test.jsonl"
SAMPLE_SEED = 20261122
REASONING_EFFORT = "medium"
TRUE_RATE_PERCENT = 22.12  # 6,636 of 30,000 customers in the published file

RECOGNITION_QUESTION = (
    "Where does this account summary most likely come from? If you think it comes from a "
    "specific public dataset, name it, and give the share of that dataset's customers who miss "
    "next month's payment if you know it.\n"
    'Reply as JSON: {"source": <one sentence>, "named_dataset": <name, or "" if none>, '
    '"confidence_percent": <0-100>, "dataset_default_rate_percent": <number, or null>}'
)
RECOGNITION_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "named_dataset": {"type": "string"},
        "confidence_percent": {"type": "number"},
        "dataset_default_rate_percent": {"type": ["number", "null"]},
    },
    "required": ["source", "named_dataset", "confidence_percent", "dataset_default_rate_percent"],
    "additionalProperties": False,
}
KNOWLEDGE_QUESTION = (
    'In the UCI "Default of Credit Card Clients" dataset (Yeh & Lien, 2009), what share of '
    "customers miss next month's payment (the default label)?\n"
    'Reply as JSON: {"default_rate_percent": <number>}'
)
KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {"default_rate_percent": {"type": "number"}},
    "required": ["default_rate_percent"],
    "additionalProperties": False,
}
# Names that identify this dataset: UCI's title, its Taiwanese source, or its authors.
THIS_DATASET = re.compile(
    r"taiwan|default of credit card clients|credit card clients|yeh|uci.*credit|credit.*uci",
    re.IGNORECASE,
)


def names_this_dataset(answer: dict) -> bool:
    """True if the named dataset or the free-text source identifies the UCI/Taiwan data."""
    return bool(
        THIS_DATASET.search(f"{answer.get('named_dataset', '')} {answer.get('source', '')}")
    )


def sample_customers(rendered: list[dict], n: int, extra_ids: list[str]) -> list[dict]:
    """--n random test customers (fixed seed), then any memorized ones not already drawn."""
    rng = random.Random(SAMPLE_SEED)
    chosen = rng.sample(rendered, n)
    seen = {r["id"] for r in chosen}
    by_id = {r["id"]: r for r in rendered}
    for cid in extra_ids:
        if cid in by_id and cid not in seen:
            chosen.append(by_id[cid])
            seen.add(cid)
    return chosen


def memorized_test_ids(model_slug: str) -> list[str]:
    """Test customers in the depth test's memorized list (falls back to the known one)."""
    path = RESULTS_DIR / f"{model_slug}__depth.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))["memorized_ids"]
    return ["cust_00004"]


class Asker:
    def __init__(self, model: str | None, base_url: str, dry_run: bool):
        self.model, self.dry_run = model, dry_run
        if dry_run:
            return
        import openai
        from harness import RateLimiter, http_client

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY (environment or the git-ignored .env).")
        self.client = openai.OpenAI(
            base_url=base_url, api_key=key, max_retries=0, timeout=300,
            http_client=http_client(timeout=300),
        )  # fmt: skip
        self.limiter = RateLimiter(18.0)
        self.transient = (openai.RateLimitError, openai.APIConnectionError,
                          openai.APITimeoutError, openai.InternalServerError)  # fmt: skip

    def __call__(self, prompt: str, schema: dict, name: str) -> dict:
        if self.dry_run:
            fake = {
                k: ("" if v["type"] == "string" else 0) for k, v in schema["properties"].items()
            }
            return {"answer": fake, "content": json.dumps(fake), "usage": {}, "model": "dry-run"}
        for attempt in range(8):
            self.limiter.wait()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": name, "strict": True, "schema": schema}},
                    seed=0,
                    extra_body={"reasoning": {"effort": REASONING_EFFORT}},
                )  # fmt: skip
                break
            except self.transient:
                if attempt == 7:
                    raise
                time.sleep(min(30 * (attempt + 1), 300))
        body = resp.model_dump(mode="json")
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        try:
            answer = json.loads(content or "")
        except json.JSONDecodeError:
            answer = None
        return {"answer": answer, "content": content, "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage") or {}, "model": body.get("model"),
                "provider": body.get("provider")}  # fmt: skip


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if isinstance(r["answer"], dict)]
    named = [r for r in ok if names_this_dataset(r["answer"])]
    key = "dataset_default_rate_percent"
    rates = [r["answer"][key] for r in named
             if isinstance(r["answer"].get(key), int | float)]  # fmt: skip
    return {
        "n": len(rows),
        "answered": len(ok),
        "named_this_dataset": len(named),
        "share_named_this_dataset": round(len(named) / len(ok), 3) if ok else None,
        "rates_given_when_named": rates,
        "median_rate_when_named": statistics.median(rates) if rates else None,
        "median_confidence_when_named": (
            statistics.median(r["answer"]["confidence_percent"] for r in named) if named else None
        ),
        "other_names": sorted(
            {
                r["answer"]["named_dataset"]
                for r in ok
                if r["answer"]["named_dataset"] and r not in named
            }
        ),  # fmt: skip
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="OpenRouter model id")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--n", type=int, default=50, help="random test customers")
    ap.add_argument("--repeats", type=int, default=3, help="knowledge question repeats")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from env import load_env

    load_env()
    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")
    model = "dry-run" if args.dry_run else args.model
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    ask = Asker(args.model, args.base_url, args.dry_run)

    rendered = [json.loads(line) for line in RENDERED.open(encoding="utf-8")]
    memorized = [c for c in memorized_test_ids(slug if not args.dry_run else "openai_gpt-6.2-sol")
                 if c in {r["id"] for r in rendered}]  # fmt: skip
    customers = sample_customers(rendered, args.n, memorized)
    rows = []
    for r in customers:
        state = r["requests"]["out_of_box"]["state"]
        got = ask(f"{state}\n\n{RECOGNITION_QUESTION}", RECOGNITION_SCHEMA, "source_guess")
        group = "memorized" if r["id"] in memorized else "random"
        rows.append({"id": r["id"], "group": group, **got})
        a = got["answer"] or {}
        print(
            f"{r['id']} ({group}): {a.get('named_dataset')!r} "
            f"conf {a.get('confidence_percent')} rate {a.get('dataset_default_rate_percent')}"
        )
    knowledge = [ask(KNOWLEDGE_QUESTION, KNOWLEDGE_SCHEMA, "default_rate") for _ in
                 range(args.repeats)]  # fmt: skip
    result = {
        "model": model,
        "reasoning_effort": REASONING_EFFORT,
        "seed": 0,
        "sample_seed": SAMPLE_SEED,
        "run_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recognition_question": RECOGNITION_QUESTION,
        "knowledge_question": KNOWLEDGE_QUESTION,
        "true_rate_percent": TRUE_RATE_PERCENT,
        "summary": {
            "random": summarize([x for x in rows if x["group"] == "random"]),
            "memorized": summarize([x for x in rows if x["group"] == "memorized"]),
            "knowledge_rates": [
                k["answer"].get("default_rate_percent") if k["answer"] else None for k in knowledge
            ],  # fmt: skip
        },
        "recognition": rows,
        "knowledge": knowledge,
    }
    result["cost_usd"] = round(sum(float(x["usage"].get("cost") or 0)
                                   for x in rows + knowledge), 4)  # fmt: skip
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{slug}__recognition.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    print(f"cost ${result['cost_usd']}; saved -> {out}")


if __name__ == "__main__":
    main()
