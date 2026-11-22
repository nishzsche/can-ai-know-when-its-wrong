"""Tests for load.py and splits.py output: sizes, stratification, no leakage, no demographics."""

import json

import pandas as pd
import pytest
import splits as sp
from load import DEMOGRAPHIC, FEATURES, RAW_PARQUET, SOURCE_JSON, TARGET, content_hash


@pytest.fixture(scope="module")
def raw():
    if not RAW_PARQUET.exists():
        pytest.skip("run `uv run python data/load.py` first")
    return pd.read_parquet(RAW_PARQUET)


@pytest.fixture(scope="module")
def parts():
    if not sp.MANIFEST.exists():
        pytest.skip("run `uv run python data/splits.py` first")
    return {name: pd.read_parquet(sp.SPLIT_DIR / f"{name}.parquet") for name in sp.SIZES}


@pytest.fixture(scope="module")
def manifest():
    return json.loads(sp.MANIFEST.read_text("utf-8"))


def test_raw_matches_its_recorded_hash(raw):
    assert content_hash(raw) == json.loads(SOURCE_JSON.read_text("utf-8"))["sha256_saved_table"]


def test_no_demographic_columns_anywhere(raw, parts):
    for df in [raw, *parts.values()]:
        assert not set(DEMOGRAPHIC) & set(df.columns)


def test_split_sizes(parts):
    assert {name: len(df) for name, df in parts.items()} == sp.SIZES


def test_splits_are_disjoint_and_cover_every_customer(raw, parts):
    ids = [i for df in parts.values() for i in df["id"]]
    assert len(ids) == len(set(ids)) == 30_000
    assert set(ids) == set(raw["ID"].map(sp.customer_id))


def test_test_and_tune_never_overlap_train(parts):
    train = set(parts["train"]["id"])
    assert not train & set(parts["tune"]["id"])
    assert not train & set(parts["test"]["id"])
    assert not set(parts["tune"]["id"]) & set(parts["test"]["id"])


def test_stratified_default_rate(raw, parts):
    overall = raw[TARGET].mean()
    for df in parts.values():
        assert abs(df[TARGET].mean() - overall) < 0.002


def test_columns(parts, manifest):
    for df in parts.values():
        assert list(df.columns) == manifest["columns"] == ["id", *FEATURES, TARGET]


def test_saved_splits_match_the_code(raw, parts):
    """Re-running make_splits with the recorded seed reproduces the saved files exactly."""
    again = sp.make_splits(raw, seed=sp.SEED)
    for name in sp.SIZES:
        pd.testing.assert_frame_equal(again[name], parts[name])


def test_manifest_is_consistent(parts, manifest):
    assert manifest["base_rate_train"] == pytest.approx(parts["train"][TARGET].mean())
    for name, df in parts.items():
        assert manifest["splits"][name]["ids_sha256"] == sp.ids_hash(df["id"])
        assert manifest["splits"][name]["defaults"] == int(df[TARGET].sum())


def test_different_seed_gives_different_test_set(raw):
    a = sp.make_splits(raw, seed=1)["test"]["id"]
    b = sp.make_splits(raw, seed=2)["test"]["id"]
    assert set(a) != set(b)
