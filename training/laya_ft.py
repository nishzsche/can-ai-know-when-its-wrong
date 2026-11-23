"""M6: fine-tune Laya (the 421M classic encoder) on the 20,000 `train` customers (`laya-ft`).

    uv run --group laya python training/laya_ft.py --out /home/runs/laya-ft --device cuda
    uv run --group laya python training/laya_ft.py --out /home/runs/laya-smoke --limit 64 --epochs 1

The maker's own routine, `laya.train.finetune`, with its defaults (PLAN.md §11): 4 epochs, the
"rlcd" objective (a proper-scoring reward plus soft cross-entropy), effective batch 64, and its
temperature fit on a calibration slice it holds out of the training file (10%, at most 400
rows, so these come from `train` too). It starts from the same pinned checkpoint the zero-shot
contestant used. The result loads with runners/laya_classic.py --checkpoint DIR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "runners"))

from export import OUT_DIR  # noqa: E402

REPO = "convaiinnovations/laya"
REVISION = "fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1"  # = runners/laya_classic.py
DATA = OUT_DIR / "laya_train.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, help="default: laya's (4)")
    ap.add_argument("--limit", type=int, help="first N customers only (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from laya.train import TrainConfig, finetune

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} isn't empty; pick a new --out")
    data = args.data
    if args.limit:
        lines = [x for x in data.read_text("utf-8").splitlines() if x][: args.limit]
        data = args.out.parent / f"{args.out.name}-data.jsonl"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("\n".join(lines) + "\n", "utf-8")
    base = snapshot_download(REPO, revision=REVISION)
    config = TrainConfig(seed=args.seed)
    if args.epochs:
        config.epochs = args.epochs
    start = time.time()
    summary = finetune(str(data), base, str(args.out), config=config, device=args.device)
    record = {"base": f"{REPO}@{REVISION}", "data": str(args.data), "limit": args.limit,
              "config": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in vars(config).items()},
              "train_minutes": round((time.time() - start) / 60, 1),
              "summary": summary}  # fmt: skip
    (args.out / "m6_training.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", "utf-8"
    )
    print(f"saved -> {args.out} ({record['train_minutes']} min)")


if __name__ == "__main__":
    main()
