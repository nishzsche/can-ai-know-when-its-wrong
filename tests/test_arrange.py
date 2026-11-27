"""Post 5 addendum, part A (PLAN.md §12): every arrangement holds exactly the original 19 fields,
each column keeping its name, and undoes back to the original table."""

from pathlib import Path

import arrange as A
import pandas as pd
import pytest
from load import BILL, FEATURES, PAID, PAY_STATUS

SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


@pytest.fixture(scope="module", params=("train", "tune", "test"))
def X(request):
    return pd.read_parquet(SPLIT_DIR / f"{request.param}.parquet")[list(FEATURES)]


@pytest.mark.parametrize("name", A.ARRANGEMENTS)
def test_every_arrangement_undoes_to_the_original_table_exactly(X, name):
    arranged = A.arrange(X, name)
    assert sorted(arranged.columns) == sorted(FEATURES)  # all 19, each once, names kept
    pd.testing.assert_frame_equal(A.undo(arranged, name), X)


@pytest.mark.parametrize("name", A.ARRANGEMENTS)
def test_each_column_keeps_its_own_values(X, name):
    arranged = A.arrange(X, name)
    for col in FEATURES:
        if name == "thousands" and col in A.MONEY:
            assert (arranged[col] * 1000).round().astype("int64").equals(X[col])
        else:
            assert arranged[col].equals(X[col]), col


def test_column_arrangements_all_differ_from_the_original_and_each_other():
    orders = {a: A.columns(a) for a in (*A.STRUCTURED, *A.SHUFFLED)}
    assert len(set(orders.values())) == len(orders)
    assert all(A.columns(a) == FEATURES for a in (*A.ROWS, "thousands"))


def test_by_month_and_realigned_mirror_the_text_rows():
    assert A.columns("by-month")[:4] == ("LIMIT_BAL", BILL[0], PAID[0], PAY_STATUS[0])
    realigned = A.columns("realigned")
    # Sep: bill, no payment yet, status; Aug: bill, the payment made in Sep, status; ...
    assert realigned[:6] == ("LIMIT_BAL", BILL[0], PAY_STATUS[0], BILL[1], PAID[0], PAY_STATUS[1])
    assert realigned[-1] == PAID[5]  # the earliest payment last, as disguise.align does


def test_rows_arrangements_reorder_only_the_context():
    orders = [A.context_order(20_000, a) for a in A.ROWS]
    assert all(sorted(o) == list(range(20_000)) for o in orders)
    assert len({tuple(o[:10]) for o in orders}) == len(orders)
    assert A.context_order(20_000, "original") is None
    assert A.context_order(20_000, "shuffled-1") is None


def test_unknown_arrangement_is_refused():
    with pytest.raises(ValueError):
        A.columns("sideways")
