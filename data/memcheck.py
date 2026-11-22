"""LLM memorization tests for the UCI credit default data (Bordt et al., "Elephants Never Forget").

Runs tabmemcheck's header test and row-completion test against any OpenAI-compatible endpoint
(OpenRouter, a vLLM server on the L4, ...). Decision models (Jev, Kev, Laya) can't generate
text, so they can't be probed this way.

    # general LLM and frontier model through OpenRouter
    uv run --group memcheck python data/memcheck.py --model <openrouter-model-id>
    # the Qwen contestant on the GPU box (vLLM OpenAI server; no key needed on localhost)
    uv run --group memcheck python data/memcheck.py --model <served-name> \
        --base-url http://127.0.0.1:8000/v1
    # check the pipeline offline (no API calls; a dummy model answers "0,0,0")
    uv run --group memcheck python data/memcheck.py --dry-run

The tests send rows of the *published* file, including its demographic columns, because the
question is whether the file itself was memorized. Nothing here is a model input for the
experiment. CSVs are cached in data/memcheck/csv/ (not committed); results in
data/memcheck/results/<model>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

HERE = Path(__file__).resolve().parent / "memcheck"
CSV_DIR = HERE / "csv"
RESULTS_DIR = HERE / "results"

UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
UCI_CSV_SHA256 = "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"

# Header of the widely mirrored Kaggle copy ("UCI_Credit_Card.csv"), with money columns as
# floats. This variant is our reconstruction of that file's format, not a download of it.
KAGGLE_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
    + ["default.payment.next.month"]
)
KAGGLE_FLOAT_COLUMNS = ["LIMIT_BAL"] + KAGGLE_COLUMNS[12:24]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_csvs() -> dict[str, Path]:
    """Download the published CSV (checksum-verified) and build the Kaggle-style variant."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    uci = CSV_DIR / "uci-350.csv"
    if not uci.exists():
        urllib.request.urlretrieve(UCI_CSV_URL, uci)  # noqa: S310 (fixed https URL)
    digest = sha256_file(uci)
    if digest != UCI_CSV_SHA256:
        raise RuntimeError(f"{uci} sha256 {digest} != pinned {UCI_CSV_SHA256}; UCI file changed?")

    kaggle = CSV_DIR / "kaggle-style.csv"
    if not kaggle.exists():
        df = pd.read_csv(uci)
        df.columns = KAGGLE_COLUMNS
        df[KAGGLE_FLOAT_COLUMNS] = df[KAGGLE_FLOAT_COLUMNS].astype(float)
        df.to_csv(kaggle, index=False, lineterminator="\n")
    return {"uci": uci, "kaggle_style": kaggle}


def similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance (1.0 = identical)."""
    import jellyfish

    if not a and not b:
        return 1.0
    return 1 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b))


def exact_prefix_chars(expected: str, response: str) -> int:
    n = 0
    for x, y in zip(expected, response, strict=False):
        if x != y:
            break
        n += 1
    return n


def is_trivial_row(row: str) -> bool:
    """A row whose bills and payments are all zero is easy to guess without memorization."""
    fields = row.strip().split(",")
    if len(fields) != 25:
        return False
    return all(float(v) == 0 for v in fields[12:24])


def make_llm(args):
    import tabmemcheck
    from openai import OpenAI

    if args.dry_run:

        class DummyLLM(tabmemcheck.LLM_Interface):
            """Always answers "0,0,0": exercises the pipeline, can't match any real row."""

            chat_mode = True

            def chat_completion(self, messages, temperature, max_tokens):
                return "0,0,0"

            def completion(self, prompt, temperature, max_tokens):
                return "0,0,0"

        return DummyLLM()
    key = os.environ.get(args.api_key_env)
    if not key and urlparse(args.base_url).hostname in ("127.0.0.1", "localhost"):
        key = "EMPTY"  # a local vLLM server started without --api-key accepts any key
    if not key:
        raise SystemExit(f"Set {args.api_key_env} (or pass --api-key-env) to run against a model.")
    client = OpenAI(base_url=args.base_url, api_key=key)
    return tabmemcheck.llm.OpenAILLM(client, args.model, chat_mode=not args.completion)


def run(args) -> dict:
    import tabmemcheck

    csvs = prepare_csvs()
    llm = make_llm(args)
    result = {
        "model": "dry-run" if args.dry_run else args.model,
        "endpoint": "none" if args.dry_run else urlparse(args.base_url).netloc,
        "mode": "completion" if args.completion else "chat",
        "tabmemcheck": importlib.metadata.version("tabmemcheck"),
        "uci_csv_sha256": UCI_CSV_SHA256,
        "run_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variants": {},
    }
    for name in args.variants:
        csv_path = str(csvs[name])
        try:
            prompt, expected, response = tabmemcheck.header_test(
                csv_path, llm, completion_length=args.completion_length, verbose=False
            )
        except UnboundLocalError:
            # tabmemcheck 0.1.6 never sets its result when every response is empty.
            prompt, expected, response = "", "", ""
        # tabmemcheck trims `expected` to the response's length, which flatters short answers.
        # Score against the file's true continuation of the full requested length instead.
        data = tabmemcheck.utils.load_csv_string(csv_path, header=True)
        expected = data[len(prompt) : len(prompt) + args.completion_length] if prompt else ""
        rows, responses = tabmemcheck.row_completion_test(
            csv_path, llm, num_queries=args.num_queries, print_levenshtein=False
        )
        exact = [r.strip() == s.strip() for r, s in zip(rows, responses, strict=True)]
        nontrivial = [e for e, r in zip(exact, rows, strict=True) if not is_trivial_row(r)]
        result["variants"][name] = {
            "header_test": {
                "prompt_tail": prompt[-300:],
                "expected": expected,
                "response": response,
                "similarity": round(similarity(expected, response), 4),
                "exact_prefix_chars": exact_prefix_chars(expected, response),
                "expected_chars": len(expected),
            },
            "row_completion_test": {
                "num_queries": len(rows),
                "exact_matches": sum(exact),
                "exact_matches_nontrivial_rows": sum(nontrivial),
                "nontrivial_rows": len(nontrivial),
                "mean_similarity": round(
                    sum(
                        similarity(r.strip(), s.strip())
                        for r, s in zip(rows, responses, strict=True)
                    )
                    / max(len(rows), 1),
                    4,
                ),
                "rows": rows,
                "responses": responses,
            },
        }
        h = result["variants"][name]["header_test"]
        r = result["variants"][name]["row_completion_test"]
        print(
            f"[{name}] header: similarity {h['similarity']:.2f}, exact prefix "
            f"{h['exact_prefix_chars']}/{h['expected_chars']} chars | rows: "
            f"{r['exact_matches']}/{r['num_queries']} exact "
            f"({r['exact_matches_nontrivial_rows']}/{r['nontrivial_rows']} non-trivial), "
            f"mean similarity {r['mean_similarity']:.2f}"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="model id at the endpoint")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--completion", action="store_true", help="use /completions, not chat")
    ap.add_argument("--variants", nargs="+", default=["uci", "kaggle_style"])
    ap.add_argument("--num-queries", type=int, default=25)
    ap.add_argument("--completion-length", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")

    result = run(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", result["model"])
    if args.completion:
        slug += "__completion"  # keep chat and raw-completion runs of one model apart
    out = RESULTS_DIR / f"{slug}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
