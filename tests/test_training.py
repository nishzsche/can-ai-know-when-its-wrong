"""M6 (PLAN.md §11): training files, the rebuilt Qwen prompt, the examples request, the
neighbours, and the `trained` condition in the harness. No model weights needed."""

import json
import sys
from pathlib import Path

import harness
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import examples  # noqa: E402
import export  # noqa: E402

TUNE = harness.load_requests("tune", 30)
DEMOGRAPHIC_WORDS = ("sex", "gender", "marri", "age ", "educat")


def test_kev_and_laya_rows_carry_the_label_and_the_zero_shot_request():
    for row in TUNE:
        k, la = export.kev_row(row), export.laya_row(row)
        req = row["requests"]["out_of_box"]
        assert k["state"] == la["state"] == req["state"]
        assert k["questions"]["will_default"]["label"] is bool(row["label"])
        assert (
            k["questions"]["will_default"]["instructions"]
            == (req["questions"]["will_default"]["instructions"])
        )
        probs = la["gold"]["will_default"]["probabilities"]
        assert probs == {"true": float(row["label"]), "false": 1.0 - row["label"]}


def test_qwen_answer_letter_follows_jevper_option_order():
    for row in TUNE:
        q = export.qwen_row(row)
        assert q["answer"] == ("A" if row["label"] else "B")
        assert "A: Yes\nB: No" in q["messages"][1]["content"]


def test_qwen_prompt_matches_what_jevper_sent_in_m3():
    raw = ROOT / "results" / "predictions" / "raw" / "qwen-logprobs.jsonl"
    if not raw.exists():
        pytest.skip("M3 Qwen raw responses not present")
    rendered = {}
    for split in ("tune", "test"):
        for row in harness.load_requests(split):
            rendered[(split, row["id"])] = row["requests"]
    checked = 0
    with raw.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            first = r["raw"]["debug"]["llm_attempts"][0]["request"]["input"]
            sent = [{"role": m["role"], "content": m["content"]} for m in first]
            assert sent == export.qwen_messages(rendered[(r["split"], r["id"])][r["condition"]])
            checked += 1
            if checked == 300:
                break
    assert checked == 300


def test_examples_state_has_20_labelled_neighbours_then_the_customer():
    train = {f"cust_{i:05d}": (f"summary {i}", i % 2) for i in range(25)}
    ids = list(train)[:20]
    state = examples.examples_state("THE TARGET", ids, train)
    assert state.count("Example ") == 20
    assert state.index("Example 20.") < state.index(examples.TARGET_HEADER)
    assert state.endswith(f"{examples.TARGET_HEADER}\nTHE TARGET")
    assert "Example 1. Missed next month's payment: no" in state  # summary 0 -> label 0
    assert "Example 2. Missed next month's payment: yes" in state


def test_neighbours_come_from_train_only_and_hold_no_demographics():
    path = ROOT / "data" / "neighbours" / "test.json"
    if not path.exists():
        pytest.skip("run data/neighbours.py first")
    lists = json.loads(path.read_text("utf-8"))
    train_ids = set(pd.read_parquet(ROOT / "data" / "splits" / "train.parquet")["id"])
    test_ids = set(pd.read_parquet(ROOT / "data" / "splits" / "test.parquet")["id"])
    assert set(lists["neighbours"]) == test_ids
    for nbrs in lists["neighbours"].values():
        assert len(nbrs) == lists["k"] == 20
        assert set(nbrs) <= train_ids


def test_examples_requests_keep_the_question_and_add_no_demographics():
    rows = (
        examples.with_examples(TUNE[:3], "tune")
        if (ROOT / "data" / "rendered" / "train.jsonl").exists()
        else None
    )
    if rows is None:
        pytest.skip("render train first")
    for row in rows:
        req = row["requests"][harness.TRAINED]
        assert req["questions"] == row["requests"]["out_of_box"]["questions"]
        assert req["state"].endswith(row["requests"]["out_of_box"]["state"])
        assert not any(w in req["state"].lower() for w in DEMOGRAPHIC_WORDS)


def test_trained_condition_reuses_out_of_box_unless_the_row_has_its_own():
    row = TUNE[0]
    assert harness.request_for(row, "trained") is row["requests"]["out_of_box"]
    own = {**row, "requests": {**row["requests"], "trained": {"state": "x", "questions": {}}}}
    assert harness.request_for(own, "trained")["state"] == "x"
    assert harness.request_for(row, "base_rate") is row["requests"]["base_rate"]
