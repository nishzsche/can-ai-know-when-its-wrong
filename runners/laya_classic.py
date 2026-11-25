"""Laya: the classic 421M English encoder (ModernBERT-large + decision head), in-process.

    uv run --group laya python runners/laya_classic.py --smoke                  # CPU works too
    uv run --group laya python runners/laya_classic.py --split test --device cuda
    uv run --group laya python runners/laya_classic.py --checkpoint /home/runs/laya-ft \
        --split test --device cuda                          # M6: laya-ft, condition trained

Since laya 0.5, `laya.Agent()` defaults to a different engine (vegaml: a frozen LLM reader). The
contestant in PLAN.md is the 421M classic encoder, so this runner asks for it explicitly
(`vegaml=False`) at a pinned Hub revision and refuses to run if it gets anything else.

The script isn't named laya.py, because that would shadow the `laya` package it imports.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from harness import L4_USD_PER_HOUR, TRAINED, Prediction, build_parser, gpu_cost, resolve_args, run

REPO = "convaiinnovations/laya"
REVISION = "fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1"
QUESTION_ID = "will_default"
STATE_BUDGET_TOKENS = 320  # 512 context minus the 192-token question budget (model card)


class LayaPredictor:
    contestant = "laya"

    def __init__(self, device: str | None, usd_per_hour: float, checkpoint: Path | None = None):
        import laya

        if checkpoint is None:
            self.agent = laya.Agent(REPO, revision=REVISION, device=device, vegaml=False)
        else:  # M6: a checkpoint written by training/laya_ft.py
            self.contestant = "laya-ft"
            self.agent = laya.load(str(checkpoint), device=device, vegaml=False)
        kind = type(self.agent).__name__
        if "vega" in kind.lower():
            raise SystemExit(f"laya returned {kind}, not the classic encoder")
        if checkpoint is None:
            resolved = getattr(self.agent, "revision", None) or REVISION
            if not resolved.startswith(REVISION[:8]):
                raise SystemExit(f"laya loaded revision {resolved}, expected {REVISION}")
            self.model_version = f"{REPO}@{resolved}"
        else:
            sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
            self.model_version = (f"laya-ft: {REPO}@{REVISION[:8]} fine-tuned on train "
                                  f"(laya.train defaults); weights {sha[:12]}")  # fmt: skip
        self.usd_per_hour = usd_per_hour
        self.meta = {"agent_class": kind, "laya_version": getattr(laya, "__version__", None),
                     "device": str(getattr(self.agent, "device", device)),
                     "usd_per_hour": usd_per_hour}  # fmt: skip

    def predict(self, request: dict) -> Prediction:
        start = time.perf_counter()
        out = self.agent.predict(request["state"], request["questions"])
        latency_ms = (time.perf_counter() - start) * 1000
        answers = out.get("answers", out)
        answer = answers[QUESTION_ID]
        usage = out.get("usage") if isinstance(out.get("usage"), dict) else None
        # Laya clamps a state that doesn't fit its context silently (laya.common); never score a
        # cut summary. Its own count, not an estimate (post 5's formats are longer than v2).
        if usage and usage.get("truncated"):
            raise SystemExit(f"Laya truncated this summary ({usage.get('state_tokens_dropped')} "
                             f"tokens dropped); this input doesn't fit its context.")  # fmt: skip
        raw = {"answer": answer}
        if usage:
            raw["usage"] = usage
        return Prediction(
            p_default=float(answer["noul"]),
            latency_ms=latency_ms,
            cost_usd=gpu_cost(latency_ms, self.usd_per_hour),
            raw=raw,
        )


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: laya's choice)")
    ap.add_argument("--checkpoint", type=Path, help="M6: a fine-tuned checkpoint (laya-ft)")
    ap.add_argument(
        "--usd-per-hour", type=float, default=None,
        help=f"compute price for cost_usd (default: {L4_USD_PER_HOUR} on cuda, 0 on cpu)",
    )  # fmt: skip
    args = ap.parse_args()
    if args.checkpoint is not None:
        args.condition = TRAINED
    args = resolve_args(args)
    if args.usd_per_hour is None:
        args.usd_per_hour = L4_USD_PER_HOUR if args.device == "cuda" else 0.0
    run(LayaPredictor(args.device, args.usd_per_hour, args.checkpoint), args)


if __name__ == "__main__":
    main()
