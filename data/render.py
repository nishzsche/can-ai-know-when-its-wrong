"""Render each customer as a short account summary: the one input every AI contestant reads.

    uv run python data/render.py --split tune        # write data/rendered/tune.jsonl
    uv run python data/render.py --split test
    uv run python data/render.py --show 10            # 10 tune summaries next to their raw rows
    uv run python data/render.py --token-report       # token counts per contestant tokenizer

One fixed template (TEMPLATE_VERSION). Change the wording -> bump the version and re-render.

Each JSONL record carries a ready-to-send System One request per condition:
  requests.out_of_box = {"state": <summary>, "questions": {"will_default": {...}}}
  requests.base_rate  = same state; the question is prefixed with one base-rate sentence.
Keeping the state identical across conditions means the base-rate sentence is the only change.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from load import BILL, DEMOGRAPHIC, FEATURES, PAID, PAY_STATUS, TARGET
from splits import MANIFEST, SPLIT_DIR

TEMPLATE_VERSION = "v2"  # v1 used the "common interpretation" of codes -2/-1/0; see README
RENDER_DIR = Path(__file__).resolve().parent / "rendered"

MONTHS = ("Sep 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005")
QUESTION_ID = "will_default"
QUESTION = "Based on this account summary, will this customer miss next month's payment?"
BASE_RATE_SENTENCE = "In this portfolio, about {pct}% of customers miss next month's payment."

# Yeh & Lien document -1 = "pay duly" and 1..9 = months of payment delay. Codes -2 and 0 are
# undocumented but sit on the not-late side of that scale. We state only that, and always show the
# raw code, so the AI contestants get exactly the information the trained baselines get.
# (v1's "common interpretation" was contradicted by the data: e.g. "no balance" for -2 while
# 40.6% of those months carry a positive bill; see README, "Known data quirks".)
STATUS_TEXT = {
    -2: "not late (code -2)",
    -1: "paid on time (code -1)",
    0: "not late (code 0)",
    1: "1 month late (code 1)",
    **{k: f"{k} months late (code {k})" for k in range(2, 9)},
    9: "9+ months late (code 9)",
}

# Tokenizers pinned to the revisions the contestants load.
TOKENIZERS = {
    # English Laya: 512-token context, of which ~320 tokens are left for the state.
    "laya (ModernBERT)": (
        "convaiinnovations/laya",
        "tokenizer/tokenizer.json",
        "fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1",
    ),
    # Kev-0.8B / Kev-4B are adapters on Qwen3.5 base models (revision from the Kev README).
    "kev (Qwen3.5)": (
        "Qwen/Qwen3.5-4B-Base",
        "tokenizer.json",
        "1001bb4d826a52d1f399e183466143f4da7b741b",
    ),
}
LAYA_STATE_BUDGET = 320


def money(amount: int) -> str:
    """NT$ with thousands separators. Negative bills are credit balances."""
    if amount < 0:
        return f"-NT${-amount:,} (credit)"
    return f"NT${amount:,}"


def status_text(code: int) -> str:
    try:
        return STATUS_TEXT[int(code)]
    except KeyError:
        raise ValueError(f"unknown repayment status code {code!r}") from None


def render_summary(row: Mapping) -> str:
    """The account summary for one customer. Uses only the financial fields."""
    leaked = [c for c in DEMOGRAPHIC if c in row]
    if leaked:
        raise ValueError(f"demographic fields must be dropped before rendering: {leaked}")
    lines = [
        "Credit card account summary",
        f"Credit limit: {money(int(row['LIMIT_BAL']))}",
        "",
        "Month (most recent first) | Bill amount | Amount paid | Repayment status",
    ]
    for month, s, b, p in zip(MONTHS, PAY_STATUS, BILL, PAID, strict=True):
        lines.append(
            f"{month} | {money(int(row[b]))} | {money(int(row[p]))} | {status_text(row[s])}"
        )
    return "\n".join(lines)


def base_rate_pct(rate: float) -> str:
    return f"{rate * 100:.1f}"


def build_record(row: Mapping, split: str, base_rate: float) -> dict:
    state = render_summary({k: row[k] for k in FEATURES})
    base_q = f"{BASE_RATE_SENTENCE.format(pct=base_rate_pct(base_rate))} {QUESTION}"
    return {
        "id": row["id"],
        "split": split,
        "label": int(row[TARGET]),
        "template_version": TEMPLATE_VERSION,
        "requests": {
            "out_of_box": {
                "state": state,
                "questions": {QUESTION_ID: {"type": "noul", "instructions": QUESTION}},
            },
            "base_rate": {
                "state": state,
                "questions": {QUESTION_ID: {"type": "noul", "instructions": base_q}},
            },
        },
    }


# ---- parsing back (used by tests to check every summary against its raw row) -----------------

_MONEY = r"(-?)NT\$([\d,]+)(?: \(credit\))?"
_ROW = re.compile(rf"^(\w{{3}} 2005) \| {_MONEY} \| {_MONEY} \| (.+)$")
_TEXT_TO_STATUS = {v: k for k, v in STATUS_TEXT.items()}


def _money_value(sign: str, digits: str) -> int:
    return -int(digits.replace(",", "")) if sign else int(digits.replace(",", ""))


def parse_summary(text: str) -> dict:
    """Inverse of render_summary: recover every field from the text."""
    lines = text.split("\n")
    limit = re.fullmatch(rf"Credit limit: {_MONEY}", lines[1])
    out = {"LIMIT_BAL": _money_value(*limit.groups())}
    for i, line in enumerate(lines[4:]):
        m = _ROW.match(line)
        if not m or m.group(1) != MONTHS[i]:
            raise ValueError(f"unexpected line {line!r}")
        out[BILL[i]] = _money_value(m.group(2), m.group(3))
        out[PAID[i]] = _money_value(m.group(4), m.group(5))
        out[PAY_STATUS[i]] = _TEXT_TO_STATUS[m.group(6)]
    return out


# ---- CLI -------------------------------------------------------------------------------------


def load_split(split: str) -> pd.DataFrame:
    return pd.read_parquet(SPLIT_DIR / f"{split}.parquet")


def base_rate_from_manifest() -> float:
    return json.loads(MANIFEST.read_text("utf-8"))["base_rate_train"]


def write_split(split: str) -> Path:
    df = load_split(split)
    rate = base_rate_from_manifest()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    path = RENDER_DIR / f"{split}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(build_record(row, split, rate), ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(df):,} records -> {path.relative_to(Path.cwd())} (template {TEMPLATE_VERSION})"
    )
    return path


def show(n: int, split: str = "tune", seed: int = 0) -> None:
    df = load_split(split)
    rows = df.sample(n=n, random_state=seed).to_dict("records")
    for row in rows:
        raw = " ".join(f"{k}={row[k]}" for k in FEATURES)
        print(f"=== {row['id']}  label={row[TARGET]}\nRAW: {raw}\n")
        print(render_summary({k: row[k] for k in FEATURES}))
        print()


def load_tokenizer(name: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    repo, filename, revision = TOKENIZERS[name]
    return Tokenizer.from_file(hf_hub_download(repo, filename, revision=revision))


def token_report(splits: tuple[str, ...] = ("train", "tune", "test", "reserve")) -> dict:
    """Token counts of every summary (all 30,000 customers) with each contestant's tokenizer."""
    texts = []
    for split in splits:
        for row in load_split(split).to_dict("records"):
            texts.append(render_summary({k: row[k] for k in FEATURES}))
    rate = base_rate_from_manifest()
    question_base = f"{BASE_RATE_SENTENCE.format(pct=base_rate_pct(rate))} {QUESTION}"
    report = {"template_version": TEMPLATE_VERSION, "customers": len(texts), "tokenizers": {}}
    for name in TOKENIZERS:
        tok = load_tokenizer(name)
        counts = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
        report["tokenizers"][name] = {
            "revision": TOKENIZERS[name][2],
            "state_tokens_mean": round(statistics.fmean(counts), 1),
            "state_tokens_p99": sorted(counts)[int(0.99 * len(counts))],
            "state_tokens_max": max(counts),
            "question_tokens_out_of_box": len(tok.encode(QUESTION, add_special_tokens=False).ids),
            "question_tokens_base_rate": len(
                tok.encode(question_base, add_special_tokens=False).ids
            ),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["train", "tune", "test", "reserve"], action="append")
    ap.add_argument("--show", type=int, metavar="N")
    ap.add_argument("--token-report", action="store_true")
    args = ap.parse_args()
    if not (args.split or args.show or args.token_report):
        ap.error("nothing to do: pass --split, --show or --token-report")
    for split in args.split or []:
        write_split(split)
    if args.show:
        show(args.show, seed=random.Random(TEMPLATE_VERSION).randrange(10_000))
    if args.token_report:
        report = token_report()
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        (RENDER_DIR / "token_report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        print(json.dumps(report, indent=2))
        laya_max = report["tokenizers"]["laya (ModernBERT)"]["state_tokens_max"]
        verdict = "fits" if laya_max <= LAYA_STATE_BUDGET else "DOES NOT FIT"
        print(
            f"Laya English state budget {LAYA_STATE_BUDGET}: "
            f"longest summary {laya_max} -> {verdict}"
        )


if __name__ == "__main__":
    main()
