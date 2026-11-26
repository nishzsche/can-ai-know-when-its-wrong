"""Post 4: Kev's own server, with the fine-tuned backbone quantized (PLAN.md §12, post 4 spec).

Kev's loader offers bf16, fp16 and fp32 only (LoadOptions in kev/checkpoint.py at 65b1e763), so
this launcher wraps it (the author's decision, 2026-11-25). It loads the run through Kev's own
Checkpoint.load (the LoRA merged in fp32 with one rounding to bf16, Kev's serving default), replaces
the backbone's linear layers with runners/quantize.py, then starts Kev's own server. Request
handling, the pointer head (outside the backbone, kept as Kev loads it) and the readout stay the
maker's.

Every precision, bf16 included, is served the same way: CUDA graphs and the fused kernels off
(both need plain weights), temperature 1.0, one request at a time. M6 served bf16 with CUDA graphs
on, so post 4's bf16 reference is re-run here instead of borrowed: latency stays like for like.
`runners/kev.py --reproduce-check` compares that bf16 run with M6's predictions first.

Run inside Kev's environment on the L4 (L4_RUNBOOK.md §7), with the pinned libraries installed
there (`uv pip install bitsandbytes==0.50.2 hqq==0.2.8.post1`):

    cd /home/kev
    KEV_TEMPERATURE=1.0 uv run --extra serve python \\
        /home/can-ai-know-when-its-wrong/runners/kev_quant_serve.py --precision bnb-nf4 \\
        --run /home/runs/kev-0.8b-ft --port 8009

Every other argument goes to kev.serve unchanged. GET /v1/quantization reports the precision, the
library version, the settings and GPU memory; `runners/kev.py --precision` refuses a server that
reports a different precision, and reads the peak memory from it after each request.

Post 4 speed addendum (runners/speed.py, PLAN.md §12): `--precision` also takes the speed tags.
Those run Kev's fast setup: CUDA graphs on (KEV_CUDA_GRAPHS unset or 1), captured lazily by Kev
over whatever layers the backbone holds by then, so shrunk layers are captured if their kernels
allow it (a bucket that fails to capture keeps running eagerly, and /v1/quantization reports Kev's
graph stats). Kev's fused kernels rewrite plain weights only, so they're allowed for bf16-tuned
(Kev turns them on if its pinned flash-linear-attention imports) and off for every shrunk tag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quantize as q  # noqa: E402
import speed as sp  # noqa: E402

# Kev's environment knobs that would change the served model; each must be unset or this value.
DEFAULT_ONLY ={"KEV_DTYPE": "bf16", "KEV_MERGE": "1", "KEV_BACKEND": "torch", "KEV_LORA_SCALE": "1"}
CHOICES = (*q.PRECISIONS, *sp.SPEED_TAGS)


def _force(env: dict, name: str, value: str, why: str) -> None:
    if env.get(name, value) != value:
        raise SystemExit(f"{name} must be {value} or unset: {why}.")
    env[name] = value


def check_env(env: dict, precision: str = "bf16") -> dict:
    """The environment kev.serve should see, or SystemExit naming the setting that's off.

    Post 4's precisions: CUDA graphs and fused kernels off. Speed tags: CUDA graphs on; fused
    kernels left to Kev for bf16-tuned (on when its FLA version imports), off for shrunk weights."""
    env = dict(env)
    if env.get("KEV_TEMPERATURE") != "1.0":
        raise SystemExit("Start it with KEV_TEMPERATURE=1.0 (PLAN.md §11, choice 3).")
    if precision in q.PRECISIONS:
        for name in ("KEV_CUDA_GRAPHS", "KEV_FUSED"):
            _force(env, name, "0", "post 4's precisions are served without it")
    elif precision in sp.SPEED_TAGS:
        _force(env, "KEV_CUDA_GRAPHS", "1", "the speed tags run Kev's CUDA graphs")
        if precision != "bf16-tuned":
            _force(env, "KEV_FUSED", "0", "Kev's fused kernels rewrite plain weights only")
    else:
        raise SystemExit(f"unknown precision {precision!r}; choose from {CHOICES}")
    for name, value in DEFAULT_ONLY.items():
        if env.get(name, value) != value:
            raise SystemExit(f"{name}={env[name]} changes the served model; leave it unset.")
    return env


def features(model, opts) -> dict:
    """What the served model actually runs with: CUDA graphs (Kev's stats) and fused kernels."""
    graphs = getattr(model, "graphs", None)
    return {"cuda_graphs": graphs.stats() if graphs is not None else False,
            "fused": bool(getattr(opts, "fused", False))}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--precision", choices=CHOICES, required=True)
    args, rest = ap.parse_known_args()
    os.environ.update(check_env(os.environ, args.precision))
    tuned = args.precision in sp.SPEED_TAGS

    import torch

    # runners/ is on sys.path (for quantize, and as the script's directory), where runners/kev.py
    # would shadow Kev's own `kev` package: drop it before importing Kev.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    from kev import serve
    from kev.checkpoint import Checkpoint

    state: dict = {}
    original_load = Checkpoint.load

    def load(self, device, *a, **kw):
        if not str(device).startswith("cuda"):
            raise SystemExit(f"post 4 serves on CUDA only (got {device})")
        tok, model = original_load(self, device, *a, **kw)
        opts = a[0] if a else kw.get("opts")
        graphs = getattr(model, "graphs", None)
        if model.backend != "torch" or (graphs is None) == tuned:
            want = "with" if tuned else "without"
            raise SystemExit(f"expected Kev's torch backend {want} CUDA graphs, got "
                             f"{model.backend} / graphs {graphs}")  # fmt: skip
        if model.dtype != "bfloat16":
            raise SystemExit(f"expected the merged bf16 backbone, got {model.dtype}")
        if tuned:  # graphs are captured lazily, after this, over the layers the backbone holds
            info = sp.apply(model.lm, args.precision, skip=(), device=device)
        else:
            info = q.quantize_linears(model.lm, args.precision, skip=(), device=device)
        state["opts"] = opts
        # Kev takes its masks' dtype from the backbone's first parameter (the embeddings, which
        # stay bf16); make sure quantizing didn't change that.
        if model.dtype != "bfloat16":
            raise SystemExit(f"the backbone's first parameter became {model.dtype}")
        model.backend = f"torch; precision {args.precision}"  # reported in /v1/models
        state["model"] = model
        state["info"] = q.after_load({**info, "run": self.requested, "device": str(device),
                                      "torch": torch.__version__})  # fmt: skip
        print(f"post 4: {info['description']}; {info['quantized_layers']} of "
              f"{info['linear_layers']} linear layers quantized; GPU memory after load "
              f"{state['info'].get('memory_after_load')}")  # fmt: skip
        return tok, model

    Checkpoint.load = load

    @serve.app.get("/v1/quantization")
    def quantization():
        return {**state["info"], "memory_now": q.gpu_memory(),
                **features(state["model"], state.get("opts"))}  # fmt: skip

    sys.argv = [sys.argv[0], *rest]
    serve.main()


if __name__ == "__main__":
    main()
