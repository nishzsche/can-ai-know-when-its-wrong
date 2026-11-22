"""Fetch UCI "Default of Credit Card Clients" (id 350), drop demographic fields, save a local copy.

    uv run python data/load.py

Writes:
  data/raw/credit_default.parquet  financial fields + target only (no SEX/MARRIAGE/AGE/EDUCATION)
  data/raw/source.json             where it came from, versions, content hash, data-quirk counts
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

UCI_ID = 350
RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_PARQUET = RAW_DIR / "credit_default.parquet"
SOURCE_JSON = RAW_DIR / "source.json"

DEMOGRAPHIC = ("SEX", "EDUCATION", "MARRIAGE", "AGE")
PAY_STATUS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")  # Sep 2005 ... Apr 2005
BILL = tuple(f"BILL_AMT{i}" for i in range(1, 7))  # Sep 2005 ... Apr 2005
PAID = tuple(f"PAY_AMT{i}" for i in range(1, 7))  # Sep 2005 ... Apr 2005
FEATURES = ("LIMIT_BAL",) + PAY_STATUS + BILL + PAID
TARGET = "default_next_month"  # UCI "Y": default payment next month (1 = yes)
EXPECTED_ROWS = 30_000

CITATION = (
    "Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the "
    "predictive accuracy of probability of default of credit card clients. Expert Systems "
    "with Applications, 36(2), 2473-2480."
)


def content_hash(df: pd.DataFrame) -> str:
    """Stable hash of the table's values (column order and dtypes included)."""
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fetch() -> tuple[pd.DataFrame, dict]:
    """Return the full UCI table with readable column names, plus the ucimlrepo metadata."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    # ucimlrepo names columns X1..X23 / Y; the readable names live in the variables table.
    names = {
        row["name"]: row["description"]
        for _, row in ds.variables.iterrows()
        if row["role"] == "Feature"
    }
    features = ds.data.features.rename(columns=names)
    target = ds.data.targets.rename(columns={"Y": TARGET})
    df = pd.concat([ds.data.ids, features, target], axis=1)
    return df, ds.metadata


def drop_demographics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=list(DEMOGRAPHIC))
    assert not set(DEMOGRAPHIC) & set(out.columns)
    return out[["ID", *FEATURES, TARGET]]


def profile(df: pd.DataFrame) -> dict:
    """Counts behind the data quirks the write-up has to mention."""
    pay = df[list(PAY_STATUS)]
    bill = df[list(BILL)]
    codes = pd.concat([pay[c] for c in PAY_STATUS]).value_counts().sort_index()
    per_month_code_1 = {c: int((pay[c] == 1).sum()) for c in PAY_STATUS}
    return {
        "rows": len(df),
        "defaults": int(df[TARGET].sum()),
        "default_rate": float(df[TARGET].mean()),
        "repayment_status_code_counts": {str(k): int(v) for k, v in codes.items()},
        "repayment_status_code_1_by_month": per_month_code_1,
        "undocumented_codes_rows": {
            "-2": int((pay == -2).any(axis=1).sum()),
            "0": int((pay == 0).any(axis=1).sum()),
        },
        "negative_bill_cells": int((bill < 0).sum().sum()),
        "negative_bill_rows": int((bill < 0).any(axis=1).sum()),
        "min_bill": int(bill.min().min()),
        "max_bill": int(bill.max().max()),
        "max_paid": int(df[list(PAID)].max().max()),
        "bill_over_limit_cells": int((bill.to_numpy() > df[["LIMIT_BAL"]].to_numpy()).sum()),
        "limit_range": [int(df["LIMIT_BAL"].min()), int(df["LIMIT_BAL"].max())],
        "duplicate_input_rows": int(df[list(FEATURES)].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }


def validate(df: pd.DataFrame) -> None:
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert df["ID"].is_unique
    assert df.isna().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}
    for col in FEATURES:
        assert pd.api.types.is_integer_dtype(df[col]), f"{col} is not integer"


def main() -> None:
    full, meta = fetch()
    df = drop_demographics(full)
    validate(df)
    prof = profile(df)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET, index=False)
    source = {
        "dataset": meta["name"],
        "uci_id": meta["uci_id"],
        "repository_url": meta["repository_url"],
        "data_url": meta["data_url"],
        "doi": f"https://doi.org/{meta['dataset_doi']}",
        "uci_last_updated": meta["last_updated"],
        "license": "CC BY 4.0 (stated on the UCI dataset page)",
        "citation": CITATION,
        "loader": f"ucimlrepo=={importlib.metadata.version('ucimlrepo')}",
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dropped_columns": list(DEMOGRAPHIC),
        "columns": list(df.columns),
        "sha256_full_table": content_hash(full),
        "sha256_saved_table": content_hash(df),
        "profile": prof,
    }
    SOURCE_JSON.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    print(
        f"Saved {len(df):,} rows x {df.shape[1]} columns -> {RAW_PARQUET.relative_to(Path.cwd())}"
    )
    print(f"Dropped (never used as model input): {', '.join(DEMOGRAPHIC)}")
    print(f"Default rate: {prof['default_rate']:.4f} ({prof['defaults']:,} / {prof['rows']:,})")
    print(
        f"UCI last updated: {meta['last_updated']}; "
        f"sha256 (saved table): {source['sha256_saved_table'][:16]}..."
    )


if __name__ == "__main__":
    main()
