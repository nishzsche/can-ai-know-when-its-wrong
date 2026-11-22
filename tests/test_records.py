"""Tests for analysis/records.py: the prediction-record schema every contestant writes."""

import pytest
import records as rc

GOOD = dict(
    id="cust_01234",
    split="test",
    contestant="kev-4b",
    condition="out_of_box",
    model_version="jaredpalmer/kev-4b@abc123",
    p_default=0.31,
    label=1,
    latency_ms=184,
    cost_usd=0.0,
    ts="2026-11-23T10:00:00Z",
)


def test_valid_record_passes_and_keeps_schema_order():
    rec = rc.make_record(**GOOD)
    assert tuple(rec) == rc.FIELDS


@pytest.mark.parametrize(
    "change",
    [
        {"id": "cust_1234"},
        {"id": "customer_01234"},
        {"split": "train"},
        {"contestant": "Kev 4B"},
        {"condition": "zero_shot"},
        {"model_version": " "},
        {"p_default": 1.01},
        {"p_default": -0.01},
        {"p_default": float("nan")},
        {"label": 2},
        {"latency_ms": -1},
        {"cost_usd": float("inf")},
        {"ts": "2026-11-23 10:00:00"},
        {"condition": "recalibrated", "split": "tune"},
        {"condition": "trained_calibrated", "split": "tune"},
    ],
)
def test_invalid_records_are_rejected(change):
    with pytest.raises(ValueError):
        rc.make_record(**{**GOOD, **change})


def test_bool_label_and_bool_probability_are_rejected():
    rec = rc.make_record(**GOOD)
    with pytest.raises(ValueError):
        rc.validate_record({**rec, "label": True})
    with pytest.raises(ValueError):
        rc.validate_record({**rec, "p_default": True})


def test_missing_or_extra_fields_are_rejected():
    rec = rc.make_record(**GOOD)
    with pytest.raises(ValueError):
        rc.validate_record({k: v for k, v in rec.items() if k != "cost_usd"})
    with pytest.raises(ValueError):
        rc.validate_record({**rec, "raw": "extra"})


def test_write_and_read_round_trip(tmp_path):
    a = rc.make_record(**{**GOOD, "id": "cust_00002"})
    b = rc.make_record(**{**GOOD, "id": "cust_00001"})
    c = rc.make_record(**{**GOOD, "id": "cust_00001", "condition": "base_rate"})
    rc.write_records("kev-4b", [a, b, c], directory=tmp_path)
    back = rc.read_records("kev-4b", directory=tmp_path)
    assert back == [b, a, c]  # sorted by condition, split, id


def test_duplicates_and_wrong_contestant_are_rejected(tmp_path):
    rec = rc.make_record(**GOOD)
    with pytest.raises(ValueError, match="duplicate"):
        rc.write_records("kev-4b", [rec, rec], directory=tmp_path)
    with pytest.raises(ValueError, match="file"):
        rc.write_records("jev", [rec], directory=tmp_path)
