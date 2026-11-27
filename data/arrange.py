"""Post 5 addendum, part A (PLAN.md §12): the table models' 19 fields, laid out another way.

Kumo-Tabular and TabICL-v2 never see text, so their "layout" is the table itself: the order of
its columns, the order of the context rows, and the units. Each arrangement holds exactly the
original 19 fields, each column keeping its name; `undo` gives the original table back exactly,
which tests/test_arrange.py checks on train, tune and test.

    uv run python data/arrange.py   # print each arrangement's column order

| name               | what changes                                                         |
|--------------------|----------------------------------------------------------------------|
| original           | nothing: M6's order (limit, 6 statuses, 6 bills, 6 payments)         |
| months-reversed    | the months in each group run oldest first                            |
| by-month           | each month's bill, payment and status side by side (the text rows)   |
| realigned          | each bill beside the following month's payment, the earliest payment |
|                    | last, as runners/checks/disguise.py `align` does it                  |
| shuffled-1 ... -5  | five random column orders (fixed seeds)                              |
| rows-1 ... -5      | the context rows in five random orders; columns unchanged            |
| thousands          | the limit, bills and payments in thousands of NT$ (statuses as they are) |

Rows arrangements reorder only the context (`train`); the customers being answered keep their
order, so each one's answer can be compared one to one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from load import BILL, FEATURES, PAID, PAY_STATUS

ARRANGE_DIR = Path(__file__).resolve().parents[1] / "results" / "checks" / "arrangements"
ARRANGE_VERSION = "r1"
ARRANGE_SEED = 20261127  # the shuffles' seeds; fixed before any run
N_SHUFFLES = 5
MONEY = ("LIMIT_BAL", *BILL, *PAID)

STRUCTURED = ("original", "months-reversed", "by-month", "realigned")
SHUFFLED = tuple(f"shuffled-{i}" for i in range(1, N_SHUFFLES + 1))
ROWS = tuple(f"rows-{i}" for i in range(1, N_SHUFFLES + 1))
ARRANGEMENTS = (*STRUCTURED, *SHUFFLED, *ROWS, "thousands")


def _by_month() -> tuple[str, ...]:
    cols = ["LIMIT_BAL"]
    for k in range(6):  # Sep 2005 ... Apr 2005, as in the text table's rows
        cols += [BILL[k], PAID[k], PAY_STATUS[k]]
    return tuple(cols)


def _realigned() -> tuple[str, ...]:
    """Row k: the month's bill, the payment made the following month (none for the latest
    month), its status; the earliest payment last, so every field appears once."""
    cols = ["LIMIT_BAL"]
    for k in range(6):
        cols += [BILL[k], *([PAID[k - 1]] if k else []), PAY_STATUS[k]]
    return (*cols, PAID[5])


def _shuffled(i: int) -> tuple[str, ...]:
    order = np.random.default_rng(ARRANGE_SEED + i).permutation(len(FEATURES))
    return tuple(FEATURES[j] for j in order)


def columns(name: str) -> tuple[str, ...]:
    """The column order an arrangement uses."""
    if name in ("original", "thousands") or name in ROWS:
        return FEATURES
    if name == "months-reversed":
        return ("LIMIT_BAL", *PAY_STATUS[::-1], *BILL[::-1], *PAID[::-1])
    if name == "by-month":
        return _by_month()
    if name == "realigned":
        return _realigned()
    if name in SHUFFLED:
        return _shuffled(SHUFFLED.index(name) + 1)
    raise ValueError(f"unknown arrangement {name!r}")


def arrange(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """The feature table in an arrangement's column order and units (rows unchanged)."""
    assert tuple(X.columns) == FEATURES, "expects the original 19 fields in M6's order"
    out = X.loc[:, list(columns(name))].copy()
    if name == "thousands":
        out[list(MONEY)] = out[list(MONEY)].astype(float) / 1000
    return out


def undo(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """The original table back from an arranged one."""
    out = X.loc[:, list(FEATURES)].copy()
    if name == "thousands":
        out[list(MONEY)] = (out[list(MONEY)] * 1000).round().astype("int64")
    return out


def context_order(n: int, name: str) -> np.ndarray | None:
    """For a rows arrangement, the order of the n context rows; None otherwise."""
    if name not in ROWS:
        return None
    return np.random.default_rng(ARRANGE_SEED + 100 + ROWS.index(name) + 1).permutation(n)


def describe(name: str) -> str:
    """The tag each record's model_version carries."""
    return f"arrangement {name} {ARRANGE_VERSION}"


def run_name(model: str, arrangement: str, seed_offset: int = 0, estimators: int | None = None):
    """A run's record name: <model>-<arrangement>, plus -s<k> for another seed and -e1 for the
    single-estimator diagnostic, e.g. kumo-tabular-original-s2 or tabicl-v2-by-month-e1."""
    return (f"{model}-{arrangement}" + (f"-s{seed_offset}" if seed_offset else "")
            + (f"-e{estimators}" if estimators is not None else ""))  # fmt: skip


if __name__ == "__main__":
    for a in ARRANGEMENTS:
        extra = " (context rows reordered)" if a in ROWS else ""
        extra += " (amounts in thousands)" if a == "thousands" else ""
        print(f"{a:16} {', '.join(columns(a))}{extra}", file=sys.stdout)
