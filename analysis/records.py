"""Prediction records: the one format every contestant writes and the analysis reads.

Schema (CLAUDE.md), one JSON object per line in results/predictions/<contestant>.jsonl:
  id, split, contestant, condition, model_version, p_default, label, latency_ms, cost_usd, ts

Every write goes through `validate_record`, so a malformed record fails at write time rather
than in the analysis.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "results" / "predictions"

AI_CONDITIONS = ("out_of_box", "base_rate", "recalibrated")
BASELINE_CONDITIONS = ("trained", "trained_calibrated")
CONDITIONS = AI_CONDITIONS + BASELINE_CONDITIONS
# Conditions whose calibrator was fitted on `tune`: their tune predictions would be in-sample,
# so they are only ever written for `test`.
TEST_ONLY_CONDITIONS = ("recalibrated", "trained_calibrated")
SPLITS = ("tune", "test")
FIELDS = (
    "id",
    "split",
    "contestant",
    "condition",
    "model_version",
    "p_default",
    "label",
    "latency_ms",
    "cost_usd",
    "ts",
)

_ID = re.compile(r"^cust_\d{5}$")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CONTESTANT = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def validate_record(rec: dict) -> None:
    """Raise ValueError if `rec` doesn't match the schema exactly."""
    if not isinstance(rec, dict):
        raise ValueError(f"record must be a dict, got {type(rec).__name__}")
    if tuple(rec) != FIELDS:
        raise ValueError(f"fields must be exactly {FIELDS} in order, got {tuple(rec)}")
    if not (isinstance(rec["id"], str) and _ID.match(rec["id"])):
        raise ValueError(f"bad id {rec['id']!r}")
    if rec["split"] not in SPLITS:
        raise ValueError(f"bad split {rec['split']!r}")
    if not (isinstance(rec["contestant"], str) and _CONTESTANT.match(rec["contestant"])):
        raise ValueError(f"bad contestant {rec['contestant']!r}")
    if rec["condition"] not in CONDITIONS:
        raise ValueError(f"bad condition {rec['condition']!r}")
    if rec["condition"] in TEST_ONLY_CONDITIONS and rec["split"] != "test":
        raise ValueError(
            f"{rec['condition']} is fitted on tune, so it may only be written for test"
        )
    if not (isinstance(rec["model_version"], str) and rec["model_version"].strip()):
        raise ValueError("model_version must be a non-empty string")
    p = rec["p_default"]
    if not (_is_number(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"p_default must be a number in [0, 1], got {p!r}")
    if rec["label"] not in (0, 1) or isinstance(rec["label"], bool):
        raise ValueError(f"label must be 0 or 1, got {rec['label']!r}")
    for key in ("latency_ms", "cost_usd"):
        if not (_is_number(rec[key]) and rec[key] >= 0):
            raise ValueError(f"{key} must be a non-negative number, got {rec[key]!r}")
    if not (isinstance(rec["ts"], str) and _TS.match(rec["ts"])):
        raise ValueError(f"ts must look like 2026-11-23T10:00:00Z, got {rec['ts']!r}")


def make_record(
    *,
    id: str,
    split: str,
    contestant: str,
    condition: str,
    model_version: str,
    p_default: float,
    label: int,
    latency_ms: float,
    cost_usd: float,
    ts: str,
) -> dict:
    rec = {
        "id": id,
        "split": split,
        "contestant": contestant,
        "condition": condition,
        "model_version": model_version,
        "p_default": float(p_default),
        "label": int(label),
        "latency_ms": round(float(latency_ms), 4),
        "cost_usd": float(cost_usd),
        "ts": ts,
    }
    validate_record(rec)
    return rec


def write_records(
    contestant: str, records: Iterable[dict], directory: Path = PREDICTIONS_DIR
) -> Path:
    """Replace results/predictions/<contestant>.jsonl with `records` (validated, sorted)."""
    records = list(records)
    seen = set()
    for rec in records:
        validate_record(rec)
        if rec["contestant"] != contestant:
            raise ValueError(f"record for {rec['contestant']!r} in {contestant!r} file")
        key = (rec["condition"], rec["split"], rec["id"])
        if key in seen:
            raise ValueError(f"duplicate record {key}")
        seen.add(key)
    records.sort(
        key=lambda r: (CONDITIONS.index(r["condition"]), SPLITS.index(r["split"]), r["id"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{contestant}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
    return path


def read_records(contestant: str, directory: Path = PREDICTIONS_DIR) -> list[dict]:
    path = directory / f"{contestant}.jsonl"
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    for rec in records:
        validate_record(rec)
    return records
