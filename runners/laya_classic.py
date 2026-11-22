"""Laya: the classic 421M English encoder (ModernBERT-large + decision head), in-process.

    uv run --group laya python runners/laya_classic.py --smoke                  # CPU works too
    uv run --group laya python runners/laya_classic.py --split test --device cuda

Since laya 0.5, `laya.Agent()` defaults to a different engine (vegaml: a frozen LLM reader). The
contestant in PLAN.md is the 421M classic encoder, so this runner asks for it explicitly
(`vegaml=False`) at a pinned Hub revision and refuses to run if it gets anything else.

The script isn't named laya.py, because that would shadow the `laya` package it imports.
"""

from __future__ import annotations

import time

from harness import L4_USD_PER_HOUR, Prediction, build_parser, gpu_cost, resolve_args, run

REPO = "convaiinnovations/laya"
REVISION = "fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1"
QUESTION_ID = "will_default"
STATE_BUDGET_TOKENS = 320  # 512 context minus the 192-token question budget (model card)


class LayaPredictor:
    contestant = "laya"

    def __init__(self, device: str | None, usd_per_hour: float):
        import laya

        self.agent = laya.Agent(REPO, revision=REVISION, device=device, vegaml=False)
        kind = type(self.agent).__name__
        if "vega" in kind.lower():
            raise SystemExit(f"laya returned {kind}, not the classic encoder")
        resolved = getattr(self.agent, "revision", None) or REVISION
        if not resolved.startswith(REVISION[:8]):
            raise SystemExit(f"laya loaded revision {resolved}, expected {REVISION}")
        self.model_version = f"{REPO}@{resolved}"
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
        return Prediction(
            p_default=float(answer["noul"]),
            latency_ms=latency_ms,
            cost_usd=gpu_cost(latency_ms, self.usd_per_hour),
            raw={"answer": answer},
        )


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: laya's choice)")
    ap.add_argument(
        "--usd-per-hour", type=float, default=None,
        help=f"compute price for cost_usd (default: {L4_USD_PER_HOUR} on cuda, 0 on cpu)",
    )  # fmt: skip
    args = resolve_args(ap.parse_args())
    if args.usd_per_hour is None:
        args.usd_per_hour = L4_USD_PER_HOUR if args.device == "cuda" else 0.0
    run(LayaPredictor(args.device, args.usd_per_hour), args)


if __name__ == "__main__":
    main()
