"""Post 4 speed addendum (PLAN.md §12): the same fine-tuned models, set up to run fast.

Post 4's first runs served every precision on an untuned path: two kinds of layers on slow
reference code (causal_conv1d; Qwen's gated delta rule without flash-linear-attention), the shrunk
versions on the libraries' generic paths, no CUDA graphs, one customer at a time. These tags
measure the tuned setup. Each record name is `<model>-<tag>`, e.g. `kev-4b-ft-fp8-dynamic`.

  bf16-tuned     full size on the fast setup: causal-conv1d and flash-linear-attention installed,
                 Kev with CUDA graphs (and its fused kernels if its pinned FLA version imports),
                 Qwen through torch.compile. The speed reference.
  bnb-nf4-tuned  post 4's NF4 (bitsandbytes) on the same fast setup, CUDA graphs if they capture.
  hqq-4-aoint4   post 4's HQQ 4-bit weights on PyTorch's fused int4 matmul
                 (aten._weight_int4pack_mm, HQQ's prepare_for_inference backend "torchao_int4";
                 bf16, axis 1). Built for few rows at a time.
  hqq-4-gemlite  the same weights on GemLite's Triton kernels (fp16 compute), which handle more
                 rows at a time.
  fp8-dynamic    weights and activations in FP8 (e4m3) with dynamic per-row scales (torchao),
                 which the L4's FP8 tensor cores run natively. The scales come from the weights and
                 from each request at run time, so no data is used. Static FP8 and SmoothQuant would
                 need calibration data and are out.

Honesty: bf16-tuned and bnb-nf4-tuned must reproduce post 4's bf16 / bnb-nf4 predictions on 50
tune customers (harness.reproduction_check); the arithmetic of the other three changes, so they get
full tune and test runs. The same layers as post 4 stay in bf16 (Qwen's lm_head and vision tower;
everything outside Kev's backbone).

Only the standard library is imported at module level: the analysis imports the names, and Kev's
launcher imports this file inside Kev's own environment.
"""

from __future__ import annotations

import importlib.metadata

from quantize import DEFAULT_SKIP, QUANT_MODELS, after_load, quantize_linears, skipped

SPEED_TAGS = ("bf16-tuned", "bnb-nf4-tuned", "hqq-4-aoint4", "hqq-4-gemlite", "fp8-dynamic")
BASE = {"bf16-tuned": "bf16", "bnb-nf4-tuned": "bnb-nf4", "hqq-4-aoint4": "hqq-4",
        "hqq-4-gemlite": "hqq-4", "fp8-dynamic": "bf16"}  # fmt: skip
FULL_RUN = ("hqq-4-aoint4", "hqq-4-gemlite", "fp8-dynamic")  # new arithmetic: tune and test
REPRODUCE = {"bf16-tuned": "bf16", "bnb-nf4-tuned": "bnb-nf4"}  # vs post 4's run, 50 tune
HQQ_BACKEND = {"hqq-4-aoint4": "torchao_int4", "hqq-4-gemlite": "gemlite"}
# The libraries each tag needs beyond post 4's pinned ones. Their versions depend on each
# environment's torch, so L4_RUNBOOK.md §8 pins them per environment; every record carries the
# installed version.
EXTRA_LIBS = {"bf16-tuned": (), "bnb-nf4-tuned": (), "hqq-4-aoint4": (),
              "hqq-4-gemlite": ("gemlite",), "fp8-dynamic": ("torchao",)}  # fmt: skip
FAST_LAYERS = ("causal-conv1d", "flash-linear-attention")  # the two layers post 4 ran slowly
BATCHES = (1, 8, 32)  # customers at a time in the throughput sweep


def speed_name(model: str, tag: str) -> str:
    if model not in QUANT_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {QUANT_MODELS}")
    if tag not in SPEED_TAGS:
        raise ValueError(f"unknown speed tag {tag!r}; choose from {SPEED_TAGS}")
    return f"{model}-{tag}"


def all_speed_contestants(tags: tuple[str, ...] = SPEED_TAGS) -> tuple[str, ...]:
    return tuple(speed_name(m, t) for m in QUANT_MODELS for t in tags)


def split_speed_name(name: str) -> tuple[str, str]:
    for model in QUANT_MODELS:
        for tag in SPEED_TAGS:
            if name == f"{model}-{tag}":
                return model, tag
    raise ValueError(f"{name!r} isn't a speed-addendum record name")


def version(lib: str) -> str | None:
    try:
        return importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions(tag: str) -> dict:
    """The versions that make this tag what it is: extra libraries and fast-layer kernels."""
    libs = (*EXTRA_LIBS[tag], *FAST_LAYERS, "torch")
    if BASE[tag].startswith("hqq"):
        libs = ("hqq", *libs)
    if BASE[tag].startswith("bnb"):
        libs = ("bitsandbytes", *libs)
    return {lib: version(lib) for lib in libs}


def describe(tag: str, found: dict | None = None) -> str:
    """One line for model_version."""
    found = versions(tag) if found is None else found
    what = {
        "bf16-tuned": "bf16, not quantized, fast setup",
        "bnb-nf4-tuned": "post 4's bnb-nf4, fast setup",
        "hqq-4-aoint4": "post 4's hqq-4 weights on aten._weight_int4pack_mm (HQQ torchao_int4 "
                        "backend, bf16)",
        "hqq-4-gemlite": "post 4's hqq-4 weights on GemLite (HQQ gemlite backend, fp16 compute)",
        "fp8-dynamic": "FP8 e4m3 weights and activations, dynamic per-row scales (torchao), "
                       "no data used",
    }[tag]  # fmt: skip
    libs = ", ".join(f"{k} {v or 'not installed'}" for k, v in found.items())
    return f"{tag} ({what}; {libs})"


def missing(tag: str) -> list[str]:
    """Libraries this tag needs that aren't installed (the fast-layer kernels included)."""
    return [lib for lib in (*EXTRA_LIBS[tag], *FAST_LAYERS) if version(lib) is None]


def _fp8(model, skip: tuple[str, ...]) -> int:
    """FP8 dynamic quantization of every nn.Linear not skipped (torchao). Returns the count."""
    import torch.nn as nn
    from torchao.quantization import PerRow, quantize_

    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as Config

        config = Config(granularity=PerRow())
    except ImportError:  # older torchao names it as a function
        from torchao.quantization import float8_dynamic_activation_float8_weight

        config = float8_dynamic_activation_float8_weight(granularity=PerRow())
    count = 0

    def take(module, fqn: str) -> bool:
        nonlocal count
        ok = isinstance(module, nn.Linear) and not skipped(fqn, skip)
        count += ok
        return ok

    quantize_(model, config, filter_fn=take)
    return count


def apply(model, tag: str, skip: tuple[str, ...] = DEFAULT_SKIP, device="cuda") -> dict:
    """Set up `model` (bf16, merged) for this tag, in place. Returns what was done.

    The shrinking itself is post 4's quantize_linears, so hqq-4-* and bnb-nf4-tuned hold exactly
    post 4's quantized weights; only the kernels that multiply them change."""
    if tag not in SPEED_TAGS:
        raise ValueError(f"unknown speed tag {tag!r}; choose from {SPEED_TAGS}")
    if lacking := [lib for lib in EXTRA_LIBS[tag] if version(lib) is None]:
        raise SystemExit(f"{tag} needs {', '.join(lacking)} (L4_RUNBOOK.md §8)")
    info = quantize_linears(model, BASE[tag], skip=skip, device=device)
    found = versions(tag)
    info.update({"tag": tag, "base_precision": BASE[tag], "description": describe(tag, found),
                 "versions": found, "fast_layers_missing": missing(tag)})  # fmt: skip
    if tag in HQQ_BACKEND:
        from hqq.utils.patching import prepare_for_inference

        prepare_for_inference(model, backend=HQQ_BACKEND[tag])
        info["hqq_backend"] = HQQ_BACKEND[tag]
    elif tag == "fp8-dynamic":
        info["quantized_layers"] = _fp8(model, skip)
        info["library"], info["version"] = "torchao", found.get("torchao")
    return info


__all__ = ["BATCHES", "BASE", "FULL_RUN", "REPRODUCE", "SPEED_TAGS", "after_load", "apply",
           "all_speed_contestants", "describe", "missing", "speed_name", "split_speed_name",
           "versions"]  # fmt: skip
