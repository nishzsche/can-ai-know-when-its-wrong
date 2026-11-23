"""M6: the fine-tuned Qwen3.5-4B (`qwen-ft`), scored in-process on the L4.

    # 1. First, the readout check: the base model in-process must reproduce M3's zero-shot
    #    numbers (vLLM + top-20 logprobs) on 50 tune customers before anything is scored.
    uv run --group runners --group training python runners/qwen_ft.py --zero-shot-check 50
    # 2. Then the trained adapter, condition `trained`:
    G='--group runners --group training'
    uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split tune
    uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split test

Why not vLLM: serving a LoRA on the multimodal Qwen3.5 through vLLM is the least certain step,
and the readout doesn't need it. The prompt is byte-identical to what jevper sent (checked on all
8,000 M3 requests, tests/test_training.py), and P(yes) is the same quantity: each label's
probability summed over the tokens that strip to it, then the two renormalized. One customer per
forward pass, as the M3 runs were sequential, so latencies stay comparable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from harness import (
    L4_USD_PER_HOUR,
    PREDICTIONS_DIR,
    ROOT,
    TRAINED,
    Prediction,
    build_parser,
    gpu_cost,
    load_requests,
    resolve_args,
    run,
)

if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from export import qwen_messages  # noqa: E402
from qwen_lora import (  # noqa: E402
    MODEL,
    REVISION,
    file_sha256,
    label_token_ids,
    last_logits,
    load_base,
    p_yes,
    prompt_ids,
)

READOUT = "first answer token, full-vocabulary logsumexp per label (in-process, bf16)"
CHECK_PATH = ROOT / "results" / "checks" / "qwen_inprocess_check.json"
CHECK_MAX_MEAN_ABS = 0.02


class QwenFTPredictor:
    contestant = "qwen-ft"

    def __init__(self, adapter: Path | None, device: str, usd_per_hour: float):
        import torch

        self.torch = torch
        self.tok, model = load_base(device)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            cfg = json.loads((adapter / "training_config.json").read_text("utf-8"))
            sha = file_sha256(adapter / "adapter_model.safetensors")
            rank = json.loads((adapter / "adapter_config.json").read_text("utf-8"))["r"]
            self.model_version = (
                f"qwen-ft: LoRA r{rank} "
                f"on {MODEL}@{REVISION[:8]}, trained on train (data {cfg['data_sha256'][:12]}); "
                f"adapter {sha[:12]}; readout: {READOUT}"
            )
            self.meta = {"adapter": str(adapter), "adapter_sha256": sha, "training_config": cfg}
        else:
            self.model_version = f"{MODEL}@{REVISION[:8]} (base, in-process); readout: {READOUT}"
            self.meta = {}
        self.model = model.eval()
        self.yes_ids, self.no_ids, _, _ = label_token_ids(self.tok)
        self.usd_per_hour = usd_per_hour
        self.meta.update({"device": device, "usd_per_hour": usd_per_hour, "readout": READOUT,
                          "yes_variants": len(self.yes_ids),
                          "no_variants": len(self.no_ids)})  # fmt: skip

    def predict(self, request: dict) -> Prediction:
        ids = prompt_ids(self.tok, qwen_messages(request))
        start = time.perf_counter()
        with self.torch.inference_mode():
            logits = last_logits(self.model, ids)
            p = p_yes(logits, self.yes_ids, self.no_ids)
        if self.model.device.type == "cuda":
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        return Prediction(p_default=p, latency_ms=latency_ms,
                          cost_usd=gpu_cost(latency_ms, self.usd_per_hour),
                          raw={"prompt_tokens": len(ids), "p_yes": p})  # fmt: skip


def zero_shot_check(n: int, device: str) -> None:
    """Base model in-process vs M3's recorded zero-shot P(yes) on the first n tune customers."""
    recorded = {}
    for line in (PREDICTIONS_DIR / "qwen-logprobs.jsonl").read_text("utf-8").splitlines():
        if line:
            r = json.loads(line)
            if r["split"] == "tune" and r["condition"] == "out_of_box":
                recorded[r["id"]] = r["p_default"]
    predictor = QwenFTPredictor(None, device, L4_USD_PER_HOUR)
    rows, diffs = [], []
    for row in load_requests("tune", n):
        p = predictor.predict(row["requests"]["out_of_box"]).p_default
        diffs.append(abs(p - recorded[row["id"]]))
        rows.append({"id": row["id"], "in_process": p, "m3_vllm": recorded[row["id"]]})
    mean_abs, max_abs = sum(diffs) / len(diffs), max(diffs)
    ok = mean_abs <= CHECK_MAX_MEAN_ABS
    CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text(json.dumps({"n": n, "mean_abs_diff": mean_abs, "max_abs_diff": max_abs,
                                      "threshold_mean_abs": CHECK_MAX_MEAN_ABS, "passed": ok,
                                      "rows": rows}, indent=1) + "\n", "utf-8")  # fmt: skip
    print(f"in-process vs M3 zero-shot on {n} tune customers: mean |diff| {mean_abs:.4f}, "
          f"max {max_abs:.4f} -> {'PASS' if ok else 'FAIL'} ({CHECK_PATH.name})")  # fmt: skip
    if not ok:
        raise SystemExit("The in-process readout doesn't reproduce M3; don't score the adapter.")


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument(
        "--adapter", type=Path, help="trained adapter directory (training/qwen_lora.py)"
    )
    ap.add_argument("--zero-shot-check", type=int, metavar="N")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    args = ap.parse_args()
    if args.zero_shot_check:
        zero_shot_check(args.zero_shot_check, args.device)
        return
    if args.adapter is None:
        ap.error("--adapter is required (or --zero-shot-check N)")
    args.condition = TRAINED
    args = resolve_args(args)
    run(QwenFTPredictor(args.adapter, args.device, args.usd_per_hour), args)


if __name__ == "__main__":
    main()
