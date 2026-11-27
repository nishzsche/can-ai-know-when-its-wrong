"""Post 5 addendum, part B (PLAN.md §12): a trained model's predictor on post 5's layouts.

Post 3's fine-tuned models learned from the original table (template v2). With `--layout`, the
trained runners (kev.py --finetuned, qwen_ft.py, laya_classic.py --checkpoint) send the same
customers' summaries rewritten in one of post 5's layouts, through exactly post 5's renderers
(data/formats.py and runners/checks/disguise.py `align`, values copied character for character),
into results/checks/layouts_trained/<contestant>-<layout>.jsonl. `again` sends the original
summaries again in the same session: the run-to-run noise floor.

    uv run --group runners python runners/qwen_ft.py --adapter DIR --precision bf16 \\
        --split test --layout json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "checks"))

from disguise import VARIANTS  # noqa: E402
from harness import ROOT, TRAINED, Prediction  # noqa: E402

LAYOUT_DIR = ROOT / "results" / "checks" / "layouts_trained"
LAYOUTS = ("markdown", "csv", "json", "prose", "aligned")
AGAIN = "again"
AGAIN_TAG = "again: the original summaries, same session"


class LayoutPredictor:
    """Wraps a trained predictor: same model, the summary rewritten in one layout."""

    def __init__(self, inner, layout: str):
        if layout == AGAIN:
            self.transform, tag = None, AGAIN_TAG
        else:
            self.transform, tag = VARIANTS[layout]
        self.inner, self.layout = inner, layout
        self.contestant = f"{inner.contestant}-{layout}"
        self.model_version = f"{inner.model_version}; {tag}"
        self.meta = {**getattr(inner, "meta", {}), "check": f"post 5 addendum: {layout}"}

    def predict(self, request: dict) -> Prediction:
        if self.transform is not None:
            request = {**request, "state": self.transform(request["state"])}
        return self.inner.predict(request)


ALL = "all"  # every layout, then `again`, in one process (the model loads once)


def add_layout_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--layout", choices=(*LAYOUTS, AGAIN, ALL),
                    help="post 5 addendum: the trained model on one of post 5's layouts (again: "
                         "the original, asked again; all: the five, then again), test only, "
                         "into results/checks/layouts_trained/")  # fmt: skip


def layouts_to_run(layout: str | None) -> tuple[str | None, ...]:
    """What one invocation runs: the original (None), one layout, or all six in order."""
    return (*LAYOUTS, AGAIN) if layout == ALL else (layout,)


def check_layout_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse what the addendum doesn't run, and send its records to LAYOUT_DIR. Call before
    harness.resolve_args, which would otherwise default the output to results/predictions."""
    if not args.layout:
        return
    if getattr(args, "reproduce_check", None):
        ap.error("--reproduce-check runs on the original layout: drop --layout")
    if args.condition != TRAINED:
        ap.error("--layout is for the trained models (post 3's fine-tuned runs)")
    if args.split != "test" and not args.smoke:
        ap.error("the layout runs answer test only (the reference's tune answers set the cut-off)")
    if args.out_dir is None and not args.smoke:
        args.out_dir = LAYOUT_DIR


def with_layout(predictor, layout: str | None):
    return LayoutPredictor(predictor, layout) if layout else predictor
