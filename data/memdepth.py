"""How deep does an LLM's memory of the data file go? A longer, deterministic header test.

tabmemcheck's header test (data/memcheck.py) cuts the file at random inside rows 2-8 and caps
each answer at 1,000 tokens, which for a reasoning model includes its hidden reasoning. So when
GPT-6.2 Sol stopped after 622 / 764 characters, still matching, we couldn't tell whether its
memory ended there or the budget did. This walks further and records why every answer ended:

- It cuts the file partway through customer k, for k = 2, 12, 22, ... (--start, --step), and asks
  the model to continue. The prompt format is tabmemcheck's header test: its system prompt, then
  one few-shot example from each of its default CSV files cut at the same character offset, then
  our file up to the cut. Few-shot continuations are --length characters (1,500 by default, about
  18 rows), so the model is shown that a long continuation is wanted.
- The token budget (--max-tokens, 16,000) leaves room for reasoning and the full answer.
- Same sampling settings as the contestant: reasoning effort medium, seed 0, no temperature.
- A customer counts as memorized when the model writes their whole row exactly (all 25 fields,
  label included) and that row isn't in the prompt; or when it completes the cut row k exactly
  and the completed part holds a non-zero bill or payment (an all-zero tail is easy to guess).
- The walk stops after --patience consecutive cuts with no memorized customer, or past --max-row.

    uv run --group memcheck python data/memdepth.py --model openai/gpt-6.2-sol
    uv run --group memcheck python data/memdepth.py --dry-run   # offline: a dummy model

Results: data/memcheck/results/<model>__depth.json. Its `memorized_ids` is the union over both
file versions plus the customers the original header test reproduced (from <model>.json);
analysis/summarize.py uses it for the with/without comparison.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent / "runners"))
from memcheck import RESULTS_DIR, exact_prefix_chars, prepare_csvs  # noqa: E402

N_FIELDS = 25  # ID, 23 features, label
SEED = 20261122
REASONING_EFFORT = "medium"  # the contestant's setting (runners/frontier_verbal.py)
FEW_SHOT_FILES = ["iris.csv", "adult-train.csv", "openml-diabetes.csv", "uci-wine.csv",
                  "california-housing.csv"]  # tabmemcheck's DEFAULT_FEW_SHOT_CSV_FILES  # fmt: skip


def row_starts(data: str) -> list[int]:
    """Character offset where each line starts; index 0 is the header, index i is customer i."""
    starts, pos = [], 0
    for line in data.split("\n"):
        starts.append(pos)
        pos += len(line) + 1
    return starts


def cut_offset(data: str, starts: list[int], k: int, seed: int = SEED) -> int:
    """A reproducible cut between a third and two thirds of the way through customer k's row."""
    length = len(data.split("\n")[k])
    rng = np.random.default_rng(seed + k)
    return starts[k] + int(rng.integers(length // 3, 2 * length // 3))


def file_rows(data: str) -> dict[int, str]:
    """Customer ID -> that customer's line, as written in the file."""
    rows = {}
    for line in data.split("\n")[1:]:
        if line:
            rows[int(line.split(",")[0])] = line
    return rows


def has_nonzero_money(fields: list[str], money_from: int) -> bool:
    """True if any bill or payment field (all but the label) is non-zero."""
    money = fields[money_from:-1] if len(fields) > 1 else []
    for v in money:
        try:
            if float(v) != 0:
                return True
        except ValueError:
            continue
    return False


def strip_fences(text: str) -> str:
    """Drop a Markdown code fence around the answer, if the model added one."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def score(data: str, rows: dict[int, str], k: int, offset: int, row_start: int, response: str,
          length: int) -> dict:  # fmt: skip
    """Compare a continuation with the file. See the module docstring for what counts."""
    response = strip_fences(response)
    window = max(length, len(response))
    expected = data[offset : offset + window]
    tail = rows[k][offset - row_start :]
    lines = response.split("\n")
    first = lines[0].strip()
    tail_exact = first == tail
    tail_fields = tail.split(",")
    # Bills and payments are the last 13 fields of a row (12 money columns + label); in the
    # tail, money fields are whichever of those it still contains.
    money_from = max(0, len(tail_fields) - N_FIELDS + 12)
    tail_informative = tail_exact and has_nonzero_money(tail_fields, money_from)

    exact_ids, near = [], []
    for line in lines[1:]:
        line = line.strip()
        fields = line.split(",")
        if len(fields) != N_FIELDS or not fields[0].isdigit():
            continue
        cid = int(fields[0])
        if cid not in rows:
            continue
        true = rows[cid].split(",")
        matches = sum(a == b for a, b in zip(fields[1:], true[1:], strict=True))
        if line == rows[cid] and cid > k:
            exact_ids.append(cid)
        elif cid > k:
            near.append({"id": cid, "fields_matched": matches, "of": N_FIELDS - 1})
    return {
        "exact_prefix_chars": exact_prefix_chars(expected, response),
        "response_chars": len(response),
        "cut_row_tail_exact": tail_exact,
        "cut_row_tail_informative": tail_informative,
        "exact_rows": exact_ids,
        "inexact_rows": near,
        "memorized": sorted(set(exact_ids) | ({k} if tail_informative else set())),
    }


@functools.cache
def few_shot_data(name: str) -> str:
    import tabmemcheck

    return tabmemcheck.utils.load_csv_string(name, header=True)


def few_shot_examples(offset: int, length: int) -> tuple[list[dict], list[str]]:
    """tabmemcheck's few-shot block: one (prefix, continuation) per default file long enough."""
    messages, used = [], []
    for name in FEW_SHOT_FILES:
        fs = few_shot_data(name)
        if len(fs) < offset + length:
            continue  # too short to give a full example at this offset
        messages += [{"role": "user", "content": fs[:offset]},
                     {"role": "assistant", "content": fs[offset : offset + length]}]  # fmt: skip
        used.append(name)
    return messages, used


def build_messages(data: str, offset: int, length: int) -> tuple[list[dict], list[str]]:
    import tabmemcheck

    system = tabmemcheck.config.system_prompts["header"]
    shots, used = few_shot_examples(offset, length)
    return [{"role": "system", "content": system}, *shots,
            {"role": "user", "content": data[:offset]}], used  # fmt: skip


class Asker:
    """One chat call: the model's text, why it stopped, and usage. Retries transient errors."""

    def __init__(self, model: str | None, base_url: str, max_tokens: int, dry_run: bool):
        self.model, self.max_tokens, self.dry_run = model, max_tokens, dry_run
        if dry_run:
            return
        import openai
        from harness import RateLimiter, http_client

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY (environment or the git-ignored .env).")
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=600,
                                    http_client=http_client(timeout=600))  # fmt: skip
        self.limiter = RateLimiter(18.0)  # OpenRouter's 20/min cap for Sol on this account
        self.transient = (openai.RateLimitError, openai.APIConnectionError,
                          openai.APITimeoutError, openai.InternalServerError)  # fmt: skip

    def __call__(self, messages: list[dict]) -> dict:
        if self.dry_run:
            return {"content": "0,0,0", "finish_reason": "stop", "usage": {}, "model": "dry-run"}
        for attempt in range(8):
            self.limiter.wait()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_tokens, seed=0,
                    extra_body={"reasoning": {"effort": REASONING_EFFORT}},
                )  # fmt: skip
                break
            except self.transient:
                if attempt == 7:
                    raise
                time.sleep(min(30 * (attempt + 1), 300))
        body = resp.model_dump(mode="json")
        choice = (body.get("choices") or [{}])[0]
        return {
            "content": (choice.get("message") or {}).get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") or {},
            "model": body.get("model"),
            "provider": body.get("provider"),
        }


def walk(name: str, csv_path: Path, ask: Asker, args) -> dict:
    data = csv_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    starts, rows = row_starts(data), file_rows(data)
    queries, misses, k = [], 0, args.start
    while k <= args.max_row and misses < args.patience:
        offset = cut_offset(data, starts, k)
        messages, used = build_messages(data, offset, args.length)
        answer = ask(messages)
        s = score(data, rows, k, offset, starts[k], answer["content"], args.length)
        queries.append({"cut_customer": k, "offset": offset, "few_shot_files": used, **s,
                        "finish_reason": answer["finish_reason"], "usage": answer["usage"],
                        "served_model": answer.get("model"), "provider": answer.get("provider"),
                        "response": answer["content"]})  # fmt: skip
        misses = 0 if s["memorized"] else misses + 1
        print(f"[{name}] cut in customer {k}: {len(s['memorized'])} memorized "
              f"{s['memorized'][:20]}, exact prefix {s['exact_prefix_chars']}/"
              f"{s['response_chars']} chars, finish {answer['finish_reason']}")  # fmt: skip
        k += args.step
    memorized = sorted({c for q in queries for c in q["memorized"]})
    return {
        "queries": queries,
        "memorized": memorized,
        "deepest_memorized": max(memorized, default=None),
        "stopped_because": "patience" if misses >= args.patience else "max_row",
    }


def earlier_header_test_ids(model_slug: str, csvs: dict[str, Path]) -> list[int]:
    """Customers the original tabmemcheck header test reproduced exactly (whole rows)."""
    path = RESULTS_DIR / f"{model_slug}.json"
    if not path.exists():
        return []
    found = set()
    for name, v in json.loads(path.read_text("utf-8"))["variants"].items():
        rows = file_rows(csvs[name].read_text(encoding="utf-8").replace("\r\n", "\n"))
        for line in strip_fences(v["header_test"]["response"]).split("\n")[1:]:
            fields = line.strip().split(",")
            if len(fields) == N_FIELDS and fields[0].isdigit():
                cid = int(fields[0])
                if rows.get(cid) == line.strip():
                    found.add(cid)
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="OpenRouter model id")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--variants", nargs="+", default=["uci", "kaggle_style"])
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--length", type=int, default=1500, help="continuation length, characters")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-row", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from env import load_env

    load_env()
    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")
    csvs = prepare_csvs()
    ask = Asker(args.model, args.base_url, args.max_tokens, args.dry_run)
    model = "dry-run" if args.dry_run else args.model
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    result = {
        "model": model,
        "settings": {k: getattr(args, k) for k in
                     ("start", "step", "length", "max_tokens", "patience", "max_row")},
        "reasoning_effort": REASONING_EFFORT, "seed": 0, "cut_seed": SEED,
        "tabmemcheck": importlib.metadata.version("tabmemcheck"),
        "run_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variants": {},
    }  # fmt: skip
    for name in args.variants:
        result["variants"][name] = walk(name, csvs[name], ask, args)
    earlier = earlier_header_test_ids(slug, csvs)
    ids = sorted({c for v in result["variants"].values() for c in v["memorized"]} | set(earlier))
    result["earlier_header_test_ids"] = earlier
    result["memorized_ids"] = [f"cust_{c:05d}" for c in ids]
    result["cost_usd"] = round(sum(float(q["usage"].get("cost") or 0)
                                   for v in result["variants"].values()
                                   for q in v["queries"]), 4)  # fmt: skip
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{slug}__depth.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"memorized customers ({len(ids)}): {ids}")
    print(f"cost ${result['cost_usd']}; saved -> {out}")


if __name__ == "__main__":
    main()
