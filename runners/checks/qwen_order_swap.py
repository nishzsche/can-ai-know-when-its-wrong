"""Robustness check: does Qwen3.5-4B's P(yes) depend on which letter "Yes" gets?

For each customer, send the normal jevper request ("A: Yes / B: No"), then replay jevper's exact
recorded request with the two option lines swapped ("A: No / B: Yes"), and read P(yes) from both
with the runner's own readout. Same prompt, same settings; only the letter order differs.

Label-free on purpose: it reports only how P(yes) moves between orders (no labels, no
calibration), so it can run before PREDICTIONS.md without revealing results. Output goes to
results/checks/, never to results/predictions/, and the contestant itself is unchanged.

    uv run --group runners python runners/checks/qwen_order_swap.py --n 200
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qwen_logprobs as ql  # noqa: E402
from harness import ROOT, load_requests  # noqa: E402

OUT = ROOT / "results" / "checks" / "qwen_order_swap.jsonl"


def swap_options(request: dict) -> dict:
    swapped = copy.deepcopy(request)
    n = 0
    for message in swapped["input"]:
        content = message.get("content")
        if isinstance(content, str) and "Options:\nA: Yes\nB: No" in content:
            message["content"] = content.replace(
                "Options:\nA: Yes\nB: No", "Options:\nA: No\nB: Yes"
            )
            n += 1
    if n != 1:
        raise ValueError(f"expected one 'A: Yes / B: No' options block, found {n}")
    return swapped


def first_token(response: dict) -> dict:
    for item in response.get("output", []):
        for part in item.get("content") or []:
            for token in part.get("logprobs") or []:
                if token["token"].strip():
                    return token
    raise ValueError("no answer token with logprobs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="first N tune customers")
    ap.add_argument("--condition", default="out_of_box")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()

    predictor = ql.QwenPredictor(args.base_url, ql.SERVED_NAME, ql.L4_USD_PER_HOUR)
    from openai import OpenAI

    openai_client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUT.open("w", encoding="utf-8") as fh:
        for item in load_requests("tune", args.n):
            req = item["requests"][args.condition]
            pred = predictor.predict(req)
            attempt = [
                a for a in pred.raw["debug"]["llm_attempts"]
                if a.get("question_id") == ql.QUESTION_ID and a.get("response")
            ][-1]  # fmt: skip
            swapped_req = swap_options(attempt["request"])
            resp = openai_client.responses.create(**swapped_req)
            # Read the first answer position's top-20 even when the sampled token is a word
            # ("No") rather than a letter: the A/B distribution is still there.
            first = first_token(resp.model_dump(mode="json"))
            yes_letter, no_letter = ql.option_letters(swapped_req)
            p_ba, missing = ql.p_yes_from_top_logprobs(first["top_logprobs"], yes_letter, no_letter)
            missing = pred.raw["readout"]["missing_labels"] + missing
            row = {"id": item["id"], "condition": args.condition,
                   "p_yes_A": pred.p_default, "p_yes_B": p_ba, "missing": missing,
                   "sampled_B": first["token"]}  # fmt: skip
            fh.write(json.dumps(row) + "\n")
            rows.append(row)

    diffs = [r["p_yes_A"] - r["p_yes_B"] for r in rows]
    a = [r["p_yes_A"] for r in rows]
    b = [r["p_yes_B"] for r in rows]
    print(f"{len(rows)} tune customers ({args.condition}); Yes as A vs Yes as B")
    print(f"  mean P(yes|Yes=A) - P(yes|Yes=B): {statistics.mean(diffs):+.3f}")
    print(f"  mean |difference|: {statistics.mean(abs(d) for d in diffs):.3f}")
    print(f"  correlation between orders: {statistics.correlation(a, b):.3f}")
    print(f"  rank agreement (Spearman): {statistics.correlation(a, b, method='ranked'):.3f}")
    print(f"  labels missing from top-20: {sum(bool(r['missing']) for r in rows)}")
    words = sum(r["sampled_B"].strip().upper() not in ("A", "B") for r in rows)
    print(f"  swapped order: answered with a word, not a letter: {words}/{len(rows)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
