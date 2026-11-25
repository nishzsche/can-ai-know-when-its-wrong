"""Post 4: shrink a loaded model's linear layers to 8, 4, 3 or 2 bits (PLAN.md §12, post 4 spec).

One function for every model, so Kev (inside its own server, runners/kev_quant_serve.py) and Qwen
(in-process, runners/qwen_ft.py --precision) are quantized the same way. The fine-tuned LoRA is
already merged into bf16 weights (merged in fp32, one rounding to bf16), then every nn.Linear of
the backbone is replaced in place. The readout (Kev's pointer head, which isn't in the backbone;
Qwen's lm_head), the embeddings and the norms stay in bf16, as both libraries leave them by default.

Precisions. None needs calibration data, so no customer is used to quantize:

  bf16       no quantization: the reference, served through the same code path
  bnb-int8   bitsandbytes LLM.int8 with outlier threshold 6.0, as transformers' load_in_8bit sets it
             (the bare Linear8bitLt default is 0.0, which is plain int8, not LLM.int8)
  bnb-nf4    bitsandbytes NF4, blocksize 64, no double quantization, bf16 compute, as transformers'
             load_in_4bit with bnb_4bit_quant_type="nf4" sets it (the bare Linear4bit defaults are
             FP4 with double quantization)
  hqq-4/3/2  HQQ at 4, 3 and 2 bits with its defaults (group size 64, axis 1), bf16 compute. One
             method all the way down, so the curve shows the bit width rather than the method

Only the standard library is imported at module level: analysis scripts import the names, and
the launcher imports this file inside Kev's own environment.
"""

from __future__ import annotations

import importlib.metadata

PRECISIONS = ("bf16", "bnb-int8", "bnb-nf4", "hqq-4", "hqq-3", "hqq-2")
QUANTIZED = PRECISIONS[1:]
QUANT_MODELS = ("kev-0.8b-ft", "kev-4b-ft", "qwen-ft")  # the M6 trained models that need a GPU
LIBRARY = {"bf16": None, "bnb-int8": "bitsandbytes", "bnb-nf4": "bitsandbytes",
           "hqq-4": "hqq", "hqq-3": "hqq", "hqq-2": "hqq"}  # fmt: skip
PINNED = {"bitsandbytes": "0.50.2", "hqq": "0.2.8.post1"}  # pyproject's `quant` group pins these
SETTINGS = {
    "bf16": {},
    "bnb-int8": {"threshold": 6.0, "has_fp16_weights": False},
    "bnb-nf4": {"quant_type": "nf4", "blocksize": 64, "compress_statistics": False,
                "compute_dtype": "bfloat16"},
    **{f"hqq-{n}": {"nbits": n, "group_size": 64, "axis": 1, "compute_dtype": "bfloat16"}
       for n in (4, 3, 2)},
}  # fmt: skip
# Dotted-name components whose linear layers are never quantized: Qwen's readout and vision tower.
DEFAULT_SKIP = ("lm_head", "visual")


def contestant_name(model: str, precision: str) -> str:
    """The record name for one model at one precision, e.g. kev-0.8b-ft-bnb-nf4."""
    if model not in QUANT_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {QUANT_MODELS}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {PRECISIONS}")
    return f"{model}-{precision}"


def all_contestants() -> tuple[str, ...]:
    return tuple(contestant_name(m, p) for m in QUANT_MODELS for p in PRECISIONS)


def split_name(contestant: str) -> tuple[str, str]:
    """Inverse of contestant_name: (model, precision)."""
    for model in QUANT_MODELS:
        for precision in PRECISIONS:
            if contestant == f"{model}-{precision}":
                return model, precision
    raise ValueError(f"{contestant!r} isn't a post 4 record name")


def library_version(precision: str) -> str | None:
    lib = LIBRARY[precision]
    if lib is None:
        return None
    try:
        return importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        return None


def describe(precision: str, version: str | None = None) -> str:
    """One line for model_version, e.g. 'bnb-nf4 (bitsandbytes 0.50.2: nf4, blocksize 64, no
    double quantization, bf16 compute)'."""
    if precision == "bf16":
        return "bf16 (not quantized)"
    lib = LIBRARY[precision]
    version = version or library_version(precision) or "not installed"
    s = SETTINGS[precision]
    if precision == "bnb-int8":
        detail = f"LLM.int8, outlier threshold {s['threshold']}"
    elif precision == "bnb-nf4":
        detail = f"nf4, blocksize {s['blocksize']}, no double quantization, bf16 compute"
    else:
        detail = f"{s['nbits']}-bit, group size {s['group_size']}, axis {s['axis']}, bf16 compute"
    return f"{precision} ({lib} {version}: {detail})"


def skipped(name: str, skip: tuple[str, ...] = DEFAULT_SKIP) -> bool:
    """Whether the module at this dotted name stays in bf16."""
    return any(part in skip for part in name.split("."))


def linear_names(model, skip: tuple[str, ...] = DEFAULT_SKIP) -> tuple[list[str], list[str]]:
    """(the nn.Linear layers to quantize, the ones kept in bf16), by dotted name."""
    import torch.nn as nn

    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    return [n for n in names if not skipped(n, skip)], [n for n in names if skipped(n, skip)]


def _replace(linear, precision: str, device):
    """A quantized copy of one bf16 nn.Linear, on `device`."""
    import torch
    import torch.nn as nn

    if linear.weight.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 weights (the merged model), got {linear.weight.dtype}")
    has_bias = linear.bias is not None
    s = SETTINGS[precision]
    if precision.startswith("hqq-"):
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

        cfg = BaseQuantizeConfig(nbits=s["nbits"], group_size=s["group_size"], axis=s["axis"])
        return HQQLinear(linear, cfg, del_orig=True, compute_dtype=torch.bfloat16,
                         device=str(device))  # fmt: skip
    import bitsandbytes as bnb

    # bitsandbytes quantizes a weight when it moves to the GPU, and its int8 parameter only from
    # the CPU, so the bf16 weight goes through the CPU first.
    w = linear.weight.data.to("cpu")
    if precision == "bnb-int8":
        q = bnb.nn.Linear8bitLt(linear.in_features, linear.out_features, bias=has_bias,
                                has_fp16_weights=s["has_fp16_weights"],
                                threshold=s["threshold"], device="meta")  # fmt: skip
        q.weight = bnb.nn.Int8Params(w, requires_grad=False, has_fp16_weights=False)
    elif precision == "bnb-nf4":
        q = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=has_bias,
                              compute_dtype=torch.bfloat16,
                              compress_statistics=s["compress_statistics"],
                              quant_type=s["quant_type"], device="meta")  # fmt: skip
        q.weight = bnb.nn.Params4bit(w, requires_grad=False, blocksize=s["blocksize"],
                                     compress_statistics=s["compress_statistics"],
                                     quant_type=s["quant_type"], module=q)  # fmt: skip
    else:
        raise ValueError(f"unknown precision {precision!r}")
    if has_bias:
        q.bias = nn.Parameter(linear.bias.data.to("cpu"), requires_grad=False)
    return q.to(device)


def _count_by_part(names: list[str], skip: tuple[str, ...]) -> str:
    """'lm_head 1, visual 24' for the kept layers, or 'none'."""
    counts = {s: sum(s in n.split(".") for n in names) for s in skip}
    return ", ".join(f"{s} {c}" for s, c in counts.items() if c) or "none"


def _parent(model, name: str):
    parent_name, _, attr = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attr


def quantize_linears(model, precision: str, skip: tuple[str, ...] = DEFAULT_SKIP,
                     device="cuda") -> dict:  # fmt: skip
    """Replace every nn.Linear of `model` (except `skip`) at this precision, in place.

    Returns what was done, for model_version and the run metadata. `bf16` changes nothing.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {PRECISIONS}")
    targets, kept = linear_names(model, skip)
    info = {"precision": precision, "description": describe(precision),
            "library": LIBRARY[precision], "version": library_version(precision),
            "settings": SETTINGS[precision], "linear_layers": len(targets) + len(kept),
            "quantized_layers": 0 if precision == "bf16" else len(targets),
            "kept_bf16": "the readout head, the embeddings and the norms; linear layers kept: "
                         + _count_by_part(kept, skip)}  # fmt: skip
    if precision == "bf16":
        return info
    if info["version"] is None:
        raise SystemExit(f"{LIBRARY[precision]} isn't installed (pin {PINNED[LIBRARY[precision]]})")
    if info["version"] != PINNED[LIBRARY[precision]]:
        raise SystemExit(f"{LIBRARY[precision]} {info['version']} is installed; the spec pins "
                         f"{PINNED[LIBRARY[precision]]} (PLAN.md §12, post 4)")  # fmt: skip
    for name in targets:
        parent, attr = _parent(model, name)
        setattr(parent, attr, _replace(getattr(parent, attr), precision, device))
    return info


def gpu_memory() -> dict:
    """Bytes of GPU memory held by tensors now, at peak since the last reset, and reserved."""
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"allocated": torch.cuda.memory_allocated(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.memory_reserved()}  # fmt: skip


def after_load(info: dict) -> dict:
    """Free the staging copies, record the memory the model now holds, and restart the peak."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        info["memory_after_load"] = gpu_memory()
        torch.cuda.reset_peak_memory_stats()
    return info
