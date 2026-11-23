"""M6: LoRA fine-tune of Qwen3.5-4B on the 20,000 `train` customers (contestant `qwen-ft`).

    uv run --group runners --group training python training/qwen_lora.py --out /home/runs/qwen-ft
    uv run --group runners --group training python training/qwen_lora.py \
        --out /home/runs/qwen-smoke --limit 64 --max-steps 4                     # L4 smoke, ~1 min

What it learns from: training/data/qwen_train.jsonl (training/export.py), the exact chat messages
jevper sent Qwen zero-shot, with the customer's outcome as the answer letter ("A" = Yes,
"B" = No). The loss is cross-entropy on that one answer token, so the model learns P(default)
directly at the position the zero-shot readout already reads.

Settings (PLAN.md §11; Qwen has no maker recipe for this, so common LoRA defaults): rank 16,
alpha 32, dropout 0.05, every linear layer of the language model, AdamW lr 1e-4 with 3% warmup
and linear decay, one customer per forward pass (no padding anywhere, as in the zero-shot runs)
and 16 per optimizer step, bf16, gradient checkpointing, one epoch, seed 0, natural default rate.
`--init-adapter DIR --epochs 1` continues a saved adapter for a second epoch (choice 1).

Writes the adapter (peft format) plus training_config.json and train_log.jsonl into --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("training", "data"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from export import NO, OUT_DIR, YES  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # the zero-shot contestant's revision
DATA = OUT_DIR / "qwen_train.jsonl"


# ---- shared with runners/qwen_ft.py ----------------------------------------------------------


def load_base(device: str = "cuda", dtype_name: str = "bf16"):
    """(tokenizer, model) at the pinned revision. Refuses a load that left weights uninitialized
    (a text-only class reading the multimodal checkpoint would do that silently)."""
    import torch
    import transformers

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name)
        try:
            model, info = cls.from_pretrained(
                MODEL, revision=REVISION, dtype=dtype, output_loading_info=True
            )
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"{cls_name}: {exc}")
            continue
        missing = [k for k in info.get("missing_keys", []) if "visual" not in k]
        if missing:
            errors.append(
                f"{cls_name}: {len(missing)} weights not in the checkpoint, e.g. {missing[:3]}"
            )
            del model
            continue
        return tok, model.to(device)
    raise SystemExit("Couldn't load Qwen3.5-4B cleanly:\n  " + "\n  ".join(errors))


def prompt_ids(tok, messages: list[dict]) -> list[int]:
    """Token ids of the prompt as vLLM rendered it zero-shot (thinking off, generation prompt)."""
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def label_token_ids(tok) -> tuple[list[int], list[int], int, int]:
    """(every token that strips to "A", every token that strips to "B", "A" itself, "B" itself).

    The zero-shot readout summed each label's variants among the top 20 (runners/qwen_logprobs.py);
    here the sum runs over the whole vocabulary, which includes those 20.
    """
    yes, no = [], []
    for i in range(len(tok)):
        s = tok.decode([i]).strip().upper()
        if s == YES:
            yes.append(i)
        elif s == NO:
            no.append(i)
    y_id, n_id = tok.convert_tokens_to_ids(YES), tok.convert_tokens_to_ids(NO)
    assert y_id in yes and n_id in no, (y_id, n_id)
    return yes, no, y_id, n_id


def last_logits(model, ids: list[int]):
    import torch

    x = torch.tensor([ids], device=model.device)
    return model(input_ids=x, logits_to_keep=1).logits[0, -1].float()


def p_yes(logits, yes_ids, no_ids) -> float:
    import torch

    lp = torch.log_softmax(logits, dim=-1)
    return float(torch.sigmoid(torch.logsumexp(lp[yes_ids], 0) - torch.logsumexp(lp[no_ids], 0)))


def linear_suffixes(model) -> str:
    """A peft target regex: every nn.Linear in the language model (not the vision tower, not
    lm_head)."""
    import torch.nn as nn

    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    lm = [n for n in names if "language_model" in n] or [
        n for n in names if "visual" not in n and "lm_head" not in n
    ]
    suffixes = sorted({n.rsplit(".", 1)[-1] for n in lm})
    prefix = r".*language_model\..*\." if any("language_model" in n for n in lm) else r".*\."
    return prefix + "(" + "|".join(suffixes) + ")"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- training ----------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=16, help="customers per optimizer step")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="first N customers only (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke)")
    ap.add_argument("--init-adapter", type=Path, help="continue this adapter (a second epoch)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} isn't empty; pick a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in args.data.read_text("utf-8").splitlines() if line]
    rows = rows[: args.limit] if args.limit else rows
    tok, model = load_base(args.device)
    _, _, y_id, n_id = label_token_ids(tok)
    items = [(prompt_ids(tok, r["messages"]), y_id if r["answer"] == YES else n_id) for r in rows]
    lengths = sorted(len(ids) for ids, _ in items)
    rate = sum(t == y_id for _, t in items) / len(items)
    print(f"{len(items):,} customers, prompt tokens median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}; default rate {rate:.4f}")  # fmt: skip

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        target = "from " + str(args.init_adapter)
    else:
        target = linear_suffixes(model)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=target, task_type="CAUSAL_LM",
        ))  # fmt: skip
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / args.accum)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    )

    config = {"args": {k: str(v) for k, v in vars(args).items()}, "model": MODEL,
              "revision": REVISION, "target_modules": target, "steps": total,
              "customers": len(items), "data_sha256": file_sha256(args.data),
              "torch": torch.__version__,
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    (args.out / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    log = (args.out / "train_log.jsonl").open("w", encoding="utf-8")

    model.train()
    step, start, run_loss, run_n = 0, time.time(), 0.0, 0
    for _epoch in range(args.epochs):
        order = list(range(len(items)))
        random.shuffle(order)
        for i in range(0, len(order), args.accum):
            batch = order[i : i + args.accum]
            for j in batch:
                ids, target_id = items[j]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    logits = last_logits(model, ids)
                loss = torch.nn.functional.cross_entropy(
                    logits[None], torch.tensor([target_id], device=logits.device)
                )
                (loss / len(batch)).backward()
                run_loss += float(loss)
                run_n += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total:
                el = time.time() - start
                eta = el / step * (total - step)
                entry = {"step": step, "of": total, "loss": run_loss / run_n,
                         "lr": sched.get_last_lr()[0], "elapsed_s": round(el),
                         "eta_s": round(eta)}  # fmt: skip
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(f"step {step}/{total} loss {entry['loss']:.4f} "
                      f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)  # fmt: skip
                run_loss, run_n = 0.0, 0
            if step >= total:
                break
        if step >= total:
            break

    model.save_pretrained(args.out)
    config["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config["train_minutes"] = round((time.time() - start) / 60, 1)
    config["adapter_sha256"] = file_sha256(args.out / "adapter_model.safetensors")
    (args.out / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    print(f"saved adapter -> {args.out} ({config['train_minutes']} min)")


if __name__ == "__main__":
    main()
