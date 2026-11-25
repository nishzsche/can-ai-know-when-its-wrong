"""Post 5 (PLAN.md §12): the four formats carry exactly v2's values, only laid out differently,
and the disguise script runs them as diagnostics into results/checks/formats/."""

import json
from pathlib import Path

import formats as f
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _states(split):
    path = ROOT / "data" / "rendered" / f"{split}.jsonl"
    return {r["id"]: r["requests"]["out_of_box"]["state"]
            for r in map(json.loads, path.read_text("utf-8").splitlines())}  # fmt: skip


@pytest.fixture(scope="module")
def states():
    return {**_states("tune"), **_states("test")}


@pytest.mark.parametrize("fmt", f.FORMATS)
def test_every_customer_reads_back_to_the_same_values(states, fmt):
    for cid, state in states.items():
        text = f.render(state, fmt)
        assert f.parse(text, fmt) == f.values(state), cid
        assert text.split("\n")[0] == f.TITLE


def test_formats_differ_from_v2_and_from_each_other(states):
    state = states["cust_00005"]
    texts = {fmt: f.render(state, fmt) for fmt in f.FORMATS}
    assert state not in texts.values() and len(set(texts.values())) == len(f.FORMATS)


def test_golden_lines_for_one_customer(states):
    state = states["cust_00005"]
    assert f.render(state, "prose").split("\n")[4] == (
        "In Sep 2005, the bill amount was NT$8,617, the amount paid was NT$2,000, "
        "and the repayment status was paid on time (code -1).")  # fmt: skip
    md = f.render(state, "markdown").split("\n")
    assert md[3:6] == [
        "| Month (most recent first) | Bill amount | Amount paid | Repayment status |",
        "|---|---|---|---|",
        "| Sep 2005 | NT$8,617 | NT$2,000 | paid on time (code -1) |",
    ]
    csv_row = f.render(state, "csv").split("\n")[4]
    assert csv_row == 'Sep 2005,"NT$8,617","NT$2,000",paid on time (code -1)'
    obj = json.loads(f.render(state, "json").split("\n")[1])
    assert obj["credit_limit"] == "NT$50,000"
    assert obj["months_most_recent_first"][0] == {
        "month": "Sep 2005", "bill_amount": "NT$8,617", "amount_paid": "NT$2,000",
        "repayment_status": "paid on time (code -1)"}  # fmt: skip


def test_credit_balances_and_late_codes_survive_every_format(states):
    picked = [s for s in states.values() if "(credit)" in s and "months late" in s][:5]
    assert picked, "expected customers with a credit balance and a late month"
    for state in picked:
        for fmt in f.FORMATS:
            assert f.parse(f.render(state, fmt), fmt) == f.values(state)


def test_values_refuses_anything_but_a_v2_summary():
    with pytest.raises(ValueError):
        f.values("Credit card account summary\nsomething else")


def test_disguise_script_runs_formats_as_named_diagnostics():
    import sys

    sys.path.insert(0, str(ROOT / "runners" / "checks"))
    import disguise

    for fmt in f.FORMATS:
        transform, tag = disguise.VARIANTS[fmt]
        assert tag == f"format {fmt} {f.FORMAT_VERSION}"

    class Base:
        contestant = "jev"
        model_version = "jev-1.14"

        def __init__(self):
            self.meta = {}

        def predict(self, request):
            return request["state"]

    wrapped = disguise.wrap(Base, variant="json")()
    assert wrapped.contestant == "jev-json"
    assert wrapped.model_version == "jev-1.14; format json f1"
    state = next(iter(_states("test").values()))
    assert wrapped.predict({"state": state}) == f.render(state, "json")
    assert disguise.FORMAT_DIR.name == "formats"
