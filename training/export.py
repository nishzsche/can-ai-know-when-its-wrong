"""M6: the 20,000 `train` customers as training files for each trainable model.

Every file holds the same template v2 summaries and the same question the zero-shot runs used,
with the customer's real outcome as the label. Demographic fields never appear (the rendered
summaries don't contain them). Natural default rate, no reweighting (PLAN.md §11, choice 5).

    uv run python data/render.py --split train     # once: data/rendered/train.jsonl
    uv run python training/export.py               # -> training/data/{kev,laya,qwen}_train.jsonl

- kev:  Kev's labelled-request JSONL (kev.data.load_records): the System One request plus
        `"label": true|false` on the question.
- laya: laya.train's `{state, questions, gold}` rows, gold as probabilities {"true", "false"}.
- qwen: the exact chat messages jevper sent Qwen zero-shot (system prompt, question block,
        then the summary as a document), plus the answer letter ("A" = Yes, "B" = No).
        `qwen_messages` is checked against the requests recorded in M3 (tests/test_training.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERED = ROOT / "data" / "rendered" / "train.jsonl"
OUT_DIR = ROOT / "training" / "data"
QUESTION_ID = "will_default"
YES, NO = "A", "B"  # jevper's fixed option order: "A: Yes", "B: No" (checked on 3,000 M3 requests)


def load_rows(path: Path = RENDERED) -> list[dict]:
    if not path.exists():
        raise SystemExit("Render train first: uv run python data/render.py --split train")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def kev_row(row: dict) -> dict:
    req = row["requests"]["out_of_box"]
    q = dict(req["questions"][QUESTION_ID])
    return {"state": req["state"],
            "questions": {QUESTION_ID: {**q, "label": bool(row["label"])}},
            "_meta": {"id": row["id"]}}  # fmt: skip


def laya_row(row: dict) -> dict:
    req = row["requests"]["out_of_box"]
    y = float(row["label"])
    return {"id": row["id"], "state": req["state"], "questions": req["questions"],
            "gold": {QUESTION_ID: {"probabilities": {"true": y, "false": 1.0 - y}}}}  # fmt: skip


def qwen_messages(request: dict) -> list[dict[str, str]]:
    """The chat messages jevper 0.7.12 (method="logprobs") builds for a System One request."""
    from jevper.prompts import SYSTEM_PROMPT, render_state_messages

    q = request["questions"][QUESTION_ID]
    question_block = f"Question:\n{q['instructions']}\n\nOptions:\n{YES}: Yes\n{NO}: No"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_block},
            *render_state_messages(request["state"])]  # fmt: skip


def qwen_row(row: dict) -> dict:
    return {"id": row["id"], "messages": qwen_messages(row["requests"]["out_of_box"]),
            "answer": YES if row["label"] else NO}  # fmt: skip


WRITERS = {"kev": kev_row, "laya": laya_row, "qwen": qwen_row}


def main(argv: list[str] | None = None) -> None:
    which = (argv if argv is not None else sys.argv[1:]) or list(WRITERS)
    rows = load_rows()
    rate = sum(r["label"] for r in rows) / len(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in which:
        path = OUT_DIR / f"{name}_train.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(WRITERS[name](row), ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows):,} rows (default rate {rate:.4f}) -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
