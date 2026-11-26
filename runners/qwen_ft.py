"""M6: the fine-tuned Qwen3.5-4B (`qwen-ft`), scored in-process on the L4.

    # 1. First, the readout check: the base model in-process must reproduce M3's zero-shot
    #    numbers (vLLM + top-20 logprobs) on 50 tune customers before anything is scored.
    uv run --group runners --group training python runners/qwen_ft.py --zero-shot-check 50
    # 2. Then the trained adapter, condition `trained`:
    G='--group runners --group training'
    uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split tune
    uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split test

Post 4 (`qwen-ft-<precision>`, PLAN.md §12): `--precision` merges the adapter first, as Kev's
loader does (the base loaded in fp32 on the CPU, the LoRA added in fp32, one rounding to bf16),
then quantizes every linear layer except lm_head and the vision tower (runners/quantize.py).
`--precision bf16` is the merged, unquantized reference; `--reproduce-check 50` compares it with
M6's qwen-ft predictions on 50 tune customers and stops on a mismatch:

    uv run $G --group quant python runners/qwen_ft.py --adapter /home/runs/qwen-ft \\
        --precision bf16 --reproduce-check 50
    uv run $G --group quant python runners/qwen_ft.py --adapter /home/runs/qwen-ft \\
        --precision hqq-4 --split tune

Post 4 speed addendum (runners/speed.py, PLAN.md §12): `--precision` also takes the speed tags. The
model is merged and shrunk exactly as above, then set up for speed (speed.apply) and its forward
pass compiled with torch.compile (dynamic shapes; `--no-compile` to compare). `--reproduce-check`
with bf16-tuned or bnb-nf4-tuned compares with post 4's bf16 / bnb-nf4 run. `predict_batch` scores
several customers in one left-padded pass, for runners/throughput.py.

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
    reproduction_check,
    resolve_args,
    run,
)
from quantize import PRECISIONS, after_load, contestant_name, quantize_linears
from speed import REPRODUCE, SPEED_TAGS, speed_name
from speed import apply as speed_apply

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

    def __init__(self, adapter: Path | None, device: str, usd_per_hour: float,
                 precision: str | None = None, compile_model: bool = True):  # fmt: skip
        import torch

        self.torch = torch
        self.precision = precision
        if precision is not None:
            if adapter is None:
                raise SystemExit("--precision needs --adapter")
            self.tok, model = self._merged(adapter, device)
        else:
            self.tok, model = load_base(device)
        if adapter is not None:
            from peft import PeftModel

            if precision is None:
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
        if precision in SPEED_TAGS:
            info = speed_apply(model, precision, device=device)
            self.contestant = speed_name("qwen-ft", precision)
            setup = "torch.compile (dynamic shapes)" if compile_model else "not compiled"
            self.model_version += (f"; merged in fp32 with one rounding to bf16 (post 4); "
                                   f"{info['description']}; {setup}")  # fmt: skip
            info["compiled"] = compile_model
            self.meta["quantization"] = after_load(info)
            if compile_model:  # the forward only, so the module keeps its attributes (device)
                model.forward = torch.compile(model.forward, dynamic=True)
        elif precision is not None:
            info = quantize_linears(model, precision, device=device)
            self.contestant = contestant_name("qwen-ft", precision)
            self.model_version += (f"; merged in fp32 with one rounding to bf16 (post 4); "
                                   f"{info['description']}")  # fmt: skip
            self.meta["quantization"] = after_load(info)
        self.model = model.eval()
        self.yes_ids, self.no_ids, _, _ = label_token_ids(self.tok)
        self.usd_per_hour = usd_per_hour
        self.meta.update({"device": device, "usd_per_hour": usd_per_hour, "readout": READOUT,
                          "yes_variants": len(self.yes_ids),
                          "no_variants": len(self.no_ids)})  # fmt: skip

    @staticmethod
    def _merged(adapter: Path, device: str):
        """(tokenizer, model): the adapter merged into the base the way Kev's loader merges it:
        fp32 weights (the bf16 checkpoint upcast exactly), W + delta in fp32, one rounding to
        bf16. On the CPU, so the fp32 copy never needs GPU memory."""
        import torch
        from peft import PeftModel

        tok, model = load_base("cpu", "fp32")
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        return tok, model.to(torch.bfloat16).to(device)

    def predict(self, request: dict) -> Prediction:
        ids = prompt_ids(self.tok, qwen_messages(request))
        start = time.perf_counter()
        with self.torch.inference_mode():
            logits = last_logits(self.model, ids)
            p = p_yes(logits, self.yes_ids, self.no_ids)
        if self.model.device.type == "cuda":
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        raw = {"prompt_tokens": len(ids), "p_yes": p}
        if self.precision is not None and self.model.device.type == "cuda":
            raw["gpu_max_allocated"] = self.torch.cuda.max_memory_allocated()
        return Prediction(p_default=p, latency_ms=latency_ms,
                          cost_usd=gpu_cost(latency_ms, self.usd_per_hour), raw=raw)  # fmt: skip

    def predict_batch(self, requests: list[dict]) -> tuple[list[float], float]:
        """(P(yes) per request, milliseconds for the whole batch): one left-padded forward pass.

        Left padding keeps every prompt's last token in the last column, which is where the
        readout looks; the attention mask hides the pads (rotary positions only shift, which
        attention doesn't see). runners/throughput.py compares the answers with one at a time."""
        torch = self.torch
        ids = [prompt_ids(self.tok, qwen_messages(r)) for r in requests]
        width = max(len(x) for x in ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        rows = [[pad] * (width - len(x)) + x for x in ids]
        mask = [[0] * (width - len(x)) + [1] * len(x) for x in ids]
        dev = self.model.device
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_ids=torch.tensor(rows, device=dev),
                                attention_mask=torch.tensor(mask, device=dev),
                                logits_to_keep=1).logits[:, -1].float()  # fmt: skip
            ps = [p_yes(row, self.yes_ids, self.no_ids) for row in logits]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return ps, (time.perf_counter() - start) * 1000


def reproduce_reference(precision: str | None) -> str | None:
    """The run a reproduction check compares with: M6's qwen-ft for bf16, post 4's run for the
    two tuned tags whose arithmetic doesn't change; None otherwise."""
    if precision == "bf16":
        return "qwen-ft"
    if precision in REPRODUCE:
        return contestant_name("qwen-ft", REPRODUCE[precision])
    return None


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
    ap.add_argument("--precision", choices=(*PRECISIONS, *SPEED_TAGS),
                    help="post 4: merge, then quantize (or a speed tag: speed.py)")  # fmt: skip
    ap.add_argument("--reproduce-check", type=int, metavar="N",
                    help="bf16: compare N tune customers with M6; bf16-tuned / bnb-nf4-tuned: "
                         "with post 4's bf16 / bnb-nf4 run; then stop")  # fmt: skip
    ap.add_argument("--no-compile", action="store_true",
                    help="speed tags: skip torch.compile (recorded in model_version)")  # fmt: skip
    ap.add_argument("--both-splits", action="store_true",
                    help="tune, then test, in one process (the model loads once)")  # fmt: skip
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--usd-per-hour", type=float, default=L4_USD_PER_HOUR)
    args = ap.parse_args()
    if args.zero_shot_check:
        zero_shot_check(args.zero_shot_check, args.device)
        return
    if args.adapter is None:
        ap.error("--adapter is required (or --zero-shot-check N)")
    reference = reproduce_reference(args.precision)
    if args.reproduce_check and reference is None:
        ap.error("--reproduce-check runs with --precision bf16, bf16-tuned or bnb-nf4-tuned")
    args.condition = TRAINED
    args = resolve_args(args)
    predictor = QwenFTPredictor(args.adapter, args.device, args.usd_per_hour, args.precision,
                                compile_model=not args.no_compile)  # fmt: skip
    if args.reproduce_check:
        reproduction_check(predictor, reference, args.reproduce_check)
        return
    for split in ("tune", "test") if args.both_splits else (args.split,):
        args.split = split
        run(predictor, args)


if __name__ == "__main__":
    main()
