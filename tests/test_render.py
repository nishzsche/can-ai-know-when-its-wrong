"""Tests for data/render.py: the one template every AI contestant reads."""

import json
import re

import pandas as pd
import pytest
import render
from load import BILL, DEMOGRAPHIC, FEATURES, PAID, PAY_STATUS, TARGET
from splits import MANIFEST, SPLIT_DIR

ROW = {
    "LIMIT_BAL": 200_000,
    "PAY_0": 0, "PAY_2": -1, "PAY_3": -2, "PAY_4": 1, "PAY_5": 2, "PAY_6": 9,
    "BILL_AMT1": 45_210, "BILL_AMT2": -1_234, "BILL_AMT3": 0,
    "BILL_AMT4": 1_664_089, "BILL_AMT5": 7, "BILL_AMT6": 30_000,
    "PAY_AMT1": 2_000, "PAY_AMT2": 0, "PAY_AMT3": 1_234,
    "PAY_AMT4": 50, "PAY_AMT5": 0, "PAY_AMT6": 896_040,
}  # fmt: skip

GOLDEN = """\
Credit card account summary
Credit limit: NT$200,000

Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$45,210 | NT$2,000 | not late (code 0)
Aug 2005 | -NT$1,234 (credit) | NT$0 | paid on time (code -1)
Jul 2005 | NT$0 | NT$1,234 | not late (code -2)
Jun 2005 | NT$1,664,089 | NT$50 | 1 month late (code 1)
May 2005 | NT$7 | NT$0 | 2 months late (code 2)
Apr 2005 | NT$30,000 | NT$896,040 | 9+ months late (code 9)"""


def test_golden_summary():
    # If this fails because you changed the wording on purpose, bump TEMPLATE_VERSION too.
    assert render.TEMPLATE_VERSION == "v2"
    assert render.render_summary(ROW) == GOLDEN


def test_months_run_most_recent_first_and_line_up_with_columns():
    lines = render.render_summary(ROW).split("\n")[4:]
    assert [line.split(" | ")[0] for line in lines] == list(render.MONTHS)
    assert render.MONTHS[0] == "Sep 2005"  # PAY_0 / BILL_AMT1 / PAY_AMT1 = September 2005


@pytest.mark.parametrize(
    ("code", "text"),
    [(-2, "not late (code -2)"), (-1, "paid on time (code -1)"), (0, "not late (code 0)"),
     (1, "1 month late (code 1)"), (2, "2 months late (code 2)"), (8, "8 months late (code 8)"),
     (9, "9+ months late (code 9)")],
)  # fmt: skip
def test_status_codes(code, text):
    assert render.status_text(code) == text


def test_every_status_shows_its_raw_code():
    """AI contestants must see the same information the trained baselines see: the code itself."""
    for code, text in render.STATUS_TEXT.items():
        assert text.endswith(f"(code {code})")


def test_status_text_makes_no_undocumented_claims():
    """v1 asserted meanings the data contradicts ("no balance" while a bill is due, etc.)."""
    banned = re.compile(r"balance|in full|minimum|revolving|not used", re.I)
    for text in render.STATUS_TEXT.values():
        assert not banned.search(text), text


@pytest.mark.parametrize("code", [-3, 10, 99])
def test_unknown_status_code_raises(code):
    with pytest.raises(ValueError):
        render.status_text(code)


@pytest.mark.parametrize(
    ("amount", "text"),
    [(0, "NT$0"), (7, "NT$7"), (45_210, "NT$45,210"), (1_000_000, "NT$1,000,000"),
     (-1_234, "-NT$1,234 (credit)")],
)  # fmt: skip
def test_money(amount, text):
    assert render.money(amount) == text


@pytest.mark.parametrize("field", DEMOGRAPHIC)
def test_demographic_fields_are_rejected(field):
    with pytest.raises(ValueError, match="demographic"):
        render.render_summary({**ROW, field: 1})


def test_parse_inverts_render():
    assert render.parse_summary(render.render_summary(ROW)) == ROW


def test_request_shape_is_system_one():
    rec = render.build_record({"id": "cust_00001", **ROW, TARGET: 1}, "tune", 0.2212)
    assert set(rec) == {"id", "split", "label", "template_version", "requests"}
    assert set(rec["requests"]) == {"out_of_box", "base_rate"}
    for req in rec["requests"].values():
        assert set(req) == {"state", "questions"}
        assert list(req["questions"]) == ["will_default"]
        assert req["questions"]["will_default"]["type"] == "noul"
    oob, br = rec["requests"]["out_of_box"], rec["requests"]["base_rate"]
    assert oob["state"] == br["state"]  # the base-rate sentence is the only difference
    assert br["questions"]["will_default"]["instructions"] == (
        "In this portfolio, about 22.1% of customers miss next month's payment. "
        + oob["questions"]["will_default"]["instructions"]
    )


# ---- checks over the real data ---------------------------------------------------------------


@pytest.fixture(scope="module")
def splits():
    if not MANIFEST.exists():
        pytest.skip("run `uv run python data/load.py && uv run python data/splits.py` first")
    return {
        s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet") for s in ("train", "tune", "test", "reserve")
    }


def test_every_customer_renders_and_reads_back_exactly(splits):
    """All 30,000 summaries: parsing the text recovers every raw field (automated hand-check)."""
    n = 0
    for df in splits.values():
        for row in df[list(FEATURES)].to_dict("records"):
            assert render.parse_summary(render.render_summary(row)) == row
            n += 1
    assert n == 30_000


@pytest.mark.parametrize("split", ["tune", "test"])
def test_rendered_files_are_current(splits, split):
    path = render.RENDER_DIR / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"run `uv run python data/render.py --split {split}` first")
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    df = splits[split]
    assert [r["id"] for r in records] == df["id"].tolist()
    assert [r["label"] for r in records] == df[TARGET].tolist()
    base_rate = json.loads(MANIFEST.read_text("utf-8"))["base_rate_train"]
    for rec, row in zip(records, df.to_dict("records"), strict=True):
        assert rec["split"] == split
        assert rec["template_version"] == render.TEMPLATE_VERSION
        assert rec == render.build_record(row, split, base_rate)


def test_no_demographic_words_in_any_summary(splits):
    pattern = re.compile(r"\b(sex|gender|male|female|age|educat\w*|marri\w*|single)\b", re.I)
    for split in ("tune", "test"):
        for row in splits[split][list(FEATURES)].to_dict("records"):
            assert not pattern.search(render.render_summary(row))


def test_summaries_fit_the_smallest_context(splits):
    """English Laya leaves ~320 tokens for the state; every one of the 30,000 summaries must fit."""
    try:
        tok = render.load_tokenizer("laya (ModernBERT)")
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer unavailable: {exc}")
    texts = [
        render.render_summary(row)
        for df in splits.values()
        for row in df[list(FEATURES)].to_dict("records")
    ]
    longest = max(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    assert longest <= render.LAYA_STATE_BUDGET


def test_field_groups_are_complete():
    assert len(PAY_STATUS) == len(BILL) == len(PAID) == len(render.MONTHS) == 6
