"""Stratified train / tune / test / reserve splits with a fixed seed.

    uv run python data/splits.py

Reads data/raw/credit_default.parquet (written by load.py) and writes data/splits/*.parquet plus
data/splits/manifest.json. `test` is for scoring only; `tune` is the only data used for
recalibration and threshold choices; `reserve` is never used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from load import FEATURES, RAW_PARQUET, SOURCE_JSON, TARGET, content_hash
from sklearn.model_selection import train_test_split

SEED = 20261122
SIZES = {"train": 20_000, "tune": 2_000, "test": 2_000, "reserve": 6_000}
SPLIT_DIR = Path(__file__).resolve().parent / "splits"
MANIFEST = SPLIT_DIR / "manifest.json"


def customer_id(uci_id: int) -> str:
    return f"cust_{uci_id:05d}"


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split stratified on the target. Deterministic for a given seed and input order."""
    assert len(df) == sum(SIZES.values()), "split sizes must cover every row exactly once"
    df = df.sort_values("ID").reset_index(drop=True)
    df.insert(0, "id", df.pop("ID").map(customer_id))

    train, rest = train_test_split(
        df, train_size=SIZES["train"], stratify=df[TARGET], random_state=seed
    )
    tune, rest = train_test_split(
        rest, train_size=SIZES["tune"], stratify=rest[TARGET], random_state=seed + 1
    )
    test, reserve = train_test_split(
        rest, train_size=SIZES["test"], stratify=rest[TARGET], random_state=seed + 2
    )
    out = {"train": train, "tune": tune, "test": test, "reserve": reserve}
    return {name: part.sort_values("id").reset_index(drop=True) for name, part in out.items()}


def ids_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    df = pd.read_parquet(RAW_PARQUET)
    source = json.loads(SOURCE_JSON.read_text("utf-8"))
    assert content_hash(df) == source["sha256_saved_table"], "raw parquet doesn't match source.json"

    splits = make_splits(df)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "stratified_on": TARGET,
        "source_sha256": source["sha256_saved_table"],
        "columns": ["id", *FEATURES, TARGET],
        "base_rate_train": float(splits["train"][TARGET].mean()),
        "splits": {},
    }
    print(f"{'split':<8} {'rows':>7} {'defaults':>9} {'rate':>8}")
    for name, part in splits.items():
        part.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        n, k = len(part), int(part[TARGET].sum())
        manifest["splits"][name] = {
            "rows": n,
            "defaults": k,
            "default_rate": k / n,
            "ids_sha256": ids_hash(part["id"]),
        }
        print(f"{name:<8} {n:>7,} {k:>9,} {k / n:>8.4f}")
    total_k = int(df[TARGET].sum())
    print(f"{'all':<8} {len(df):>7,} {total_k:>9,} {total_k / len(df):>8.4f}")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Base rate for the 'told the base rate' condition (train): "
        f"{manifest['base_rate_train']:.4f}"
    )


if __name__ == "__main__":
    main()
