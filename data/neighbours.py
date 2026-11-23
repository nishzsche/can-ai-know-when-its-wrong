"""M6: the 20 most similar `train` customers for every `tune` and `test` customer.

Jev and GPT-6.2 Sol can't be fine-tuned, so in M6 they learn from `train` the only way they can:
each customer is shown with its 20 nearest `train` customers and what happened to them
(runners/examples.py). This script picks those neighbours, once, so both models see the same ones.

Similarity (PLAN.md §11, flagged choice 4): the 19 financial fields, amounts as a signed log
(sign(x) * log(1 + |x|)), repayment codes as numbers, each standardized with `train`'s mean and
standard deviation, then Euclidean distance. Most similar first. Demographic fields never appear.

The retrieval is itself a small model fitted on `train`, so its own answer, the neighbours'
default rate, is written as a reference contestant (`neighbours-20`, condition `trained`) to show
whether Jev and Sol add anything beyond counting their examples.

    uv run python data/neighbours.py

Writes data/neighbours/{tune,test}.json and results/predictions/neighbours-20.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "analysis"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from load import DEMOGRAPHIC, FEATURES, TARGET  # noqa: E402
from records import make_record, utc_now, write_records  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from splits import MANIFEST, SPLIT_DIR  # noqa: E402

K = 20
OUT_DIR = ROOT / "data" / "neighbours"
CONTESTANT = "neighbours-20"
CODES = tuple(f for f in FEATURES if f.startswith("PAY_") and not f.startswith("PAY_AMT"))
TRANSFORM = "amounts: sign(x)*log1p(|x|); codes as numbers; z-scored on train; Euclidean"


def features(df: pd.DataFrame) -> np.ndarray:
    assert not set(DEMOGRAPHIC) & set(df.columns), "demographic columns in a split file"
    cols = []
    for f in FEATURES:
        x = df[f].to_numpy(dtype=float)
        cols.append(x if f in CODES else np.sign(x) * np.log1p(np.abs(x)))
    return np.column_stack(cols)


def train_fingerprint() -> str:
    return json.loads(MANIFEST.read_text("utf-8"))["splits"]["train"]["ids_sha256"][:12]


def find(k: int = K) -> tuple[dict[str, dict], list[dict]]:
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    x_train = features(train)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=k).fit((x_train - mean) / std)
    train_ids = train["id"].to_numpy()
    train_y = train[TARGET].to_numpy()
    version = f"{k} nearest train customers ({TRANSFORM}); train {train_fingerprint()}"

    lists, records = {}, []
    for split in ("tune", "test"):
        df = pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
        start = time.perf_counter()
        dist, idx = nn.kneighbors((features(df) - mean) / std)
        ms = (time.perf_counter() - start) * 1000 / len(df)
        lists[split] = {
            "k": k,
            "transform": TRANSFORM,
            "train_fingerprint": train_fingerprint(),
            "neighbours": {
                cid: [str(t) for t in train_ids[row]]
                for cid, row in zip(df["id"], idx, strict=True)
            },
        }
        p = train_y[idx].mean(axis=1)
        for cid, pi, yi in zip(df["id"], p, df[TARGET], strict=True):
            records.append(make_record(
                id=cid, split=split, contestant=CONTESTANT, condition="trained",
                model_version=version, p_default=float(pi), label=int(yi), latency_ms=ms,
                cost_usd=0.0, ts=utc_now(),
            ))  # fmt: skip
        y = df[TARGET].to_numpy()
        print(f"{split}: mean neighbours' default rate {p.mean():.3f} vs actual {y.mean():.3f}; "
              f"median distance to the 20th {np.median(dist[:, -1]):.2f}")  # fmt: skip
    return lists, records


def main() -> None:
    lists, records = find()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, payload in lists.items():
        path = OUT_DIR / f"{split}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    path = write_records(CONTESTANT, records)
    print(f"wrote {len(records)} records -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
