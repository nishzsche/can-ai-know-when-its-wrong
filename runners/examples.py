"""M6: Jev and GPT-6.2 Sol learn from `train` the only way they can, from examples in the prompt.

Neither can be fine-tuned: TypeSafe says "Jev is not fine-tuned or LoRA-adapted with customer
data", and OpenAI's fine-tuning is closed to new users and never covered GPT-6.2. So each customer
is shown with its 20 most similar `train` customers (data/neighbours.py) and whether each missed
next month's payment, then asked the usual question. Every summary, examples included, is the
same template v2 text the other contestants read; the question is unchanged.

    uv run --group runners python runners/examples.py --contestant jev --show 1    # print, no call
    uv run --group runners python runners/examples.py --contestant jev --smoke     # 20 tune
    uv run --group runners python runners/examples.py --contestant sol --split test --workers 4

Contestants `jev-examples` and `sol-examples`, condition `trained` (it used the 20,000 `train`
customers), so they line up with the fine-tuned models and the baselines. Needs
OPENROUTER_API_KEY. Sol's runs are paced like M3 (--rpm).
"""

from __future__ import annotations

import json

from harness import RENDER_DIR, ROOT, TRAINED, build_parser, load_requests, resolve_args, run

EXAMPLES_VERSION = "v1"
K = 20
NEIGHBOUR_DIR = ROOT / "data" / "neighbours"
QUESTION_ID = "will_default"
INTRO = (
    "Examples: the {k} customers from the same portfolio whose accounts are most similar to the "
    "one below, most similar first, and whether each missed next month's payment."
)
EXAMPLE_HEADER = "Example {n}. Missed next month's payment: {outcome}"
TARGET_HEADER = "The account summary to assess:"
CONTESTANTS = {"jev": "jev-examples", "sol": "sol-examples"}


def load_train() -> dict[str, tuple[str, int]]:
    """id -> (template v2 summary, label) for every `train` customer."""
    path = RENDER_DIR / "train.jsonl"
    if not path.exists():
        raise SystemExit("Render train first: uv run python data/render.py --split train")
    out = {}
    for line in path.read_text("utf-8").splitlines():
        if line:
            row = json.loads(line)
            out[row["id"]] = (row["requests"]["out_of_box"]["state"], int(row["label"]))
    return out


def examples_state(state: str, neighbour_ids: list[str], train: dict, k: int = K) -> str:
    parts = [INTRO.format(k=k)]
    for n, nid in enumerate(neighbour_ids[:k], start=1):
        summary, label = train[nid]
        parts.append(f"{EXAMPLE_HEADER.format(n=n, outcome='yes' if label else 'no')}\n{summary}")
    parts.append(f"{TARGET_HEADER}\n{state}")
    return "\n\n".join(parts)


def with_examples(rows: list[dict], split: str, k: int = K) -> list[dict]:
    """Each row gets a `trained` request: its summary after its k nearest train customers."""
    lists = json.loads((NEIGHBOUR_DIR / f"{split}.json").read_text("utf-8"))
    if lists["k"] < k:
        raise SystemExit(f"data/neighbours/{split}.json has {lists['k']} neighbours, need {k}")
    train = load_train()
    out = []
    for row in rows:
        base = row["requests"]["out_of_box"]
        request = {"state": examples_state(base["state"], lists["neighbours"][row["id"]], train, k),
                   "questions": base["questions"]}  # fmt: skip
        out.append({**row, "requests": {**row["requests"], TRAINED: request}})
    return out


class ExamplesPredictor:
    """Wraps the M3 predictor; only the contestant id and model version change."""

    def __init__(self, inner, contestant: str, k: int = K):
        self.inner = inner
        self.contestant = contestant
        self.model_version = (
            f"{inner.model_version}; {k} train examples (examples {EXAMPLES_VERSION})"
        )
        self.meta = {**inner.meta, "examples_version": EXAMPLES_VERSION, "k": k,
                     "intro": INTRO, "example_header": EXAMPLE_HEADER,
                     "target_header": TARGET_HEADER}  # fmt: skip

    def predict(self, request: dict):
        return self.inner.predict(request)


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--contestant", choices=sorted(CONTESTANTS), required=True)
    ap.add_argument("--rpm", type=float, default=18.0, help="Sol only: requests-per-minute cap")
    ap.add_argument("--show", type=int, metavar="N", help="print N requests and stop (no calls)")
    args = ap.parse_args()
    args.condition = TRAINED
    args = resolve_args(args)
    rows = with_examples(load_requests(args.split, args.limit), args.split)

    if args.show:
        for row in rows[: args.show]:
            req = row["requests"][TRAINED]
            print(req["state"])
            print(f"\n[question] {req['questions'][QUESTION_ID]['instructions']}")
            print(f"[{row['id']}, label {row['label']}, {len(req['state']):,} characters, "
                  f"~{len(req['state']) // 4:,} tokens]\n")  # fmt: skip
        return

    from env import load_env

    load_env()
    if args.contestant == "jev":
        from jev import JevPredictor

        inner = JevPredictor()
    else:
        from frontier_verbal import FrontierVerbalPredictor

        inner = FrontierVerbalPredictor(rpm=args.rpm)
    run(ExamplesPredictor(inner, CONTESTANTS[args.contestant]), args, requests=rows)


if __name__ == "__main__":
    main()
