"""Post 5: the same account summary laid out four other ways (PLAN.md §12, post 5 spec).

A diagnostic, not a template change: template v2 (data/render.py) stays the one input every
contestant's headline results use. Each format here takes a v2 summary apart and lays out the same
values differently. Every value string (the credit limit, each month's name, bill, payment and
repayment label with its raw code) is copied byte for byte from v2, so the numbers and labels a
model reads are identical; only the arrangement changes. Each format has a parser, and
tests/test_formats.py reads every customer back and checks the values against v2's.

    prose     one sentence per month
    markdown  a markdown table: v2's pipe table plus the header separator and outer pipes
    csv       a CSV table (amounts with thousands separators are quoted, as CSV requires)
    json      one compact JSON object holding the credit limit and the months

Every format keeps v2's first line, "Credit card account summary".

    uv run python data/formats.py --show cust_00005     # one customer in every format
    uv run python data/formats.py --token-report        # tokens per format and tokenizer (test)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
from pathlib import Path

FORMAT_VERSION = "f1"
FORMATS = ("prose", "markdown", "csv", "json")
TITLE = "Credit card account summary"
LIMIT_PREFIX = "Credit limit: "
COLUMNS = ("Month (most recent first)", "Bill amount", "Amount paid", "Repayment status")
JSON_KEYS = ("month", "bill_amount", "amount_paid", "repayment_status")
ROOT = Path(__file__).resolve().parents[1]
TOKEN_REPORT = ROOT / "results" / "checks" / "formats_tokens.json"

Values = tuple[str, list[tuple[str, str, str, str]]]  # (credit limit, rows of 4 strings)


def values(state: str) -> Values:
    """(credit limit, [(month, bill, paid, status), ...]) from a template v2 summary."""
    lines = state.split("\n")
    if lines[0] != TITLE or not lines[1].startswith(LIMIT_PREFIX) or lines[2] != "" \
            or lines[3] != " | ".join(COLUMNS):  # fmt: skip
        raise ValueError("not a template v2 summary")
    rows = [tuple(line.split(" | ")) for line in lines[4:]]
    if len(rows) != 6 or any(len(r) != 4 for r in rows):
        raise ValueError("expected six months of four fields")
    return lines[1].removeprefix(LIMIT_PREFIX), rows


# ---- the four layouts ------------------------------------------------------------------------

_SENTENCE = ("In {}, the bill amount was {}, the amount paid was {}, "
             "and the repayment status was {}.")  # fmt: skip
_SENTENCE_RE = re.compile(r"^In (.+?), the bill amount was (.+?), the amount paid was (.+?), "
                          r"and the repayment status was (.+)\.$")  # fmt: skip
PROSE_INTRO = "The last six months, most recent first:"


def to_prose(state: str) -> str:
    limit, rows = values(state)
    return "\n".join([TITLE, LIMIT_PREFIX + limit, "", PROSE_INTRO,
                      *(_SENTENCE.format(*r) for r in rows)])  # fmt: skip


def to_markdown(state: str) -> str:
    limit, rows = values(state)
    table = ["| " + " | ".join(r) + " |" for r in (COLUMNS, *rows)]
    table.insert(1, "|" + "---|" * len(COLUMNS))
    return "\n".join([TITLE, LIMIT_PREFIX + limit, "", *table])


def to_csv(state: str) -> str:
    limit, rows = values(state)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([COLUMNS, *rows])
    return "\n".join([TITLE, LIMIT_PREFIX + limit, "", buf.getvalue().rstrip("\n")])


def to_json(state: str) -> str:
    limit, rows = values(state)
    obj = {"credit_limit": limit,
           "months_most_recent_first": [dict(zip(JSON_KEYS, r, strict=True)) for r in rows]}
    return "\n".join([TITLE, json.dumps(obj, ensure_ascii=False)])


RENDER = {"prose": to_prose, "markdown": to_markdown, "csv": to_csv, "json": to_json}


def render(state: str, fmt: str) -> str:
    return RENDER[fmt](state)


# ---- parsers (the tests' round trip) ---------------------------------------------------------


def _header(lines: list[str]) -> str:
    if lines[0] != TITLE or not lines[1].startswith(LIMIT_PREFIX) or lines[2] != "":
        raise ValueError("missing the title or the credit limit")
    return lines[1].removeprefix(LIMIT_PREFIX)


def parse(text: str, fmt: str) -> Values:
    """The values a formatted summary carries, to compare with values(v2 summary)."""
    lines = text.split("\n")
    if fmt == "json":
        if lines[0] != TITLE or len(lines) != 2:
            raise ValueError("expected the title and one JSON line")
        obj = json.loads(lines[1])
        return obj["credit_limit"], [tuple(month[k] for k in JSON_KEYS)
                                     for month in obj["months_most_recent_first"]]  # fmt: skip
    limit = _header(lines)
    if fmt == "prose":
        if lines[3] != PROSE_INTRO:
            raise ValueError("missing the prose intro")
        return limit, [_SENTENCE_RE.fullmatch(line).groups() for line in lines[4:]]
    if fmt == "markdown":
        if lines[4] != "|" + "---|" * len(COLUMNS):
            raise ValueError("missing the markdown separator")
        table = [tuple(line[2:-2].split(" | ")) for line in [lines[3], *lines[5:]]]
    elif fmt == "csv":
        table = [tuple(r) for r in csv.reader(lines[3:])]
    else:
        raise ValueError(f"unknown format {fmt!r}")
    if table[0] != COLUMNS:
        raise ValueError("unexpected column header")
    return limit, table[1:]


# ---- CLI ---------------------------------------------------------------------------------------


def _test_states() -> list[str]:
    path = ROOT / "data" / "rendered" / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return [r["requests"]["out_of_box"]["state"] for r in rows]


def token_report() -> dict:
    """State tokens per format with each contestant tokenizer, on the 2,000 test customers, and
    whether each format fits Laya's state budget (data/render.py)."""
    from render import LAYA_STATE_BUDGET, TOKENIZERS, load_tokenizer

    states = _test_states()
    report = {"format_version": FORMAT_VERSION, "customers": len(states),
              "laya_state_budget": LAYA_STATE_BUDGET, "tokenizers": {}}  # fmt: skip
    for name in TOKENIZERS:
        tok = load_tokenizer(name)
        by_format = {}
        for fmt in ("v2", *FORMATS):
            texts = states if fmt == "v2" else [render(s, fmt) for s in states]
            counts = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
            by_format[fmt] = {"mean": round(statistics.fmean(counts), 1), "max": max(counts)}
            if name.startswith("laya"):
                by_format[fmt]["over_laya_budget"] = sum(c > LAYA_STATE_BUDGET for c in counts)
        report["tokenizers"][name] = by_format
    laya = report["tokenizers"]["laya (ModernBERT)"]
    report["laya_fits"] = {fmt: laya[fmt]["over_laya_budget"] == 0 for fmt in FORMATS}
    report["note"] = ("The budget is the model card's (512 minus a 192-token question). Laya's "
                      "real room is 512 minus the question header minus 1 (laya.common), larger "
                      "for our short question; runners/laya_classic.py stops if Laya ever "
                      "reports truncating a summary.")  # fmt: skip
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--show", metavar="ID", help="one test or tune customer in every format")
    ap.add_argument("--token-report", action="store_true")
    args = ap.parse_args()
    if args.show:
        for split in ("tune", "test"):
            path = ROOT / "data" / "rendered" / f"{split}.jsonl"
            for line in path.read_text("utf-8").splitlines():
                r = json.loads(line)
                if r["id"] == args.show:
                    state = r["requests"]["out_of_box"]["state"]
                    print(f"=== v2\n{state}\n")
                    for fmt in FORMATS:
                        print(f"=== {fmt}\n{render(state, fmt)}\n")
                    return
        raise SystemExit(f"{args.show} isn't in tune or test")
    if args.token_report:
        report = token_report()
        TOKEN_REPORT.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_REPORT.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        print(json.dumps(report, indent=2))
        return
    ap.error("nothing to do: pass --show ID or --token-report")


if __name__ == "__main__":
    main()
