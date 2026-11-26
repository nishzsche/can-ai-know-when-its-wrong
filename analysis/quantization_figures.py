"""Post 4 figures: the fine-tuned models shrunk to 8, 4, 3 and 2 bits (PLAN.md §12, post 4 spec).

    uv run python analysis/quantization_figures.py   # results/figures/post4_*.png, light and dark

- post4_ranking.png: ranking (AUC) at each size, per model, with the scorecard and a coin flip.
- post4_honesty.png: calibration error at each size, with the band a perfectly honest model stays
  in by chance; the 2-bit values are off the scale and written out.
- post4_drift.png: Kev-0.8B's answer for each test customer, full size against four shrunk
  versions, with how far the answers moved and the ranking and honesty that hid it.
- post4_cost.png: GPU memory at peak and time per answer on the tuned setup (the speed addendum,
  results/speed_throughput.csv), with the first runs' times for the same weights as hollow dots.

    uv run python analysis/quantization_figures.py drift cost   # only these

Reads results/quantization.csv (analysis/quantization.py) and the prediction records. Same palette,
style and dark variant as posts 1-3 (analysis/figures.py): each model keeps its colour from the
series. Every mark is named in text, so colour never carries identity alone.
"""

from __future__ import annotations

import csv
import sys
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures as F  # noqa: E402
import metrics as m  # noqa: E402
import quantization as Q  # noqa: E402
from robustness import load  # noqa: E402

MODELS = (  # (record prefix, palette key from post 1, label)
    ("kev-4b-ft", "kev-4b", "Kev-4B"),
    ("kev-0.8b-ft", "kev-0.8b", "Kev-0.8B"),
    ("qwen-ft", "qwen-logprobs", "Qwen3.5-4B"),
)
SIZES = ("bf16", "bnb-int8", "bnb-nf4", "hqq-4", "hqq-3", "hqq-2")
TICK = {"bf16": "Full size\n16-bit", "bnb-int8": "8-bit", "bnb-nf4": "4-bit\nNF4",
        "hqq-4": "4-bit\nHQQ", "hqq-3": "3-bit\nHQQ", "hqq-2": "2-bit\nHQQ"}  # fmt: skip
ROW = {"bf16": "Full size (16-bit)", "bnb-int8": "8-bit", "bnb-nf4": "4-bit, NF4",
       "hqq-4": "4-bit, HQQ", "hqq-3": "3-bit, HQQ", "hqq-2": "2-bit, HQQ"}  # fmt: skip
CHANCE = 0.025  # ECE a perfectly honest model stays below in 95 of 100 samples here (0.023-0.026)
TEST_LINE = "Test set: 2,000 real credit-card customers (UCI, Taiwan 2005), 22.1% defaulted."
TRAINED_LINE = ("All three were fine-tuned on the 20,000 customers the scorecard learned from "
                "(post 3).")  # fmt: skip
METHOD_LINE = ("8-bit and NF4: bitsandbytes (LLM.int8; NF4, the common 4-bit recipe). HQQ: one "
               "method from 4 bits to 2. No data used.")  # fmt: skip
GB = 1024**3 / 1e9  # the CSV's memory columns are GiB; the charts show decimal GB, like the post
POINTS = FuncFormatter(lambda v, _: f"{100 * v:.0f}")


@cache
def table() -> dict[tuple[str, str], dict]:
    out = {}
    with Q.OUT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["model"], r["precision"]] = {
                k: (float(v) if k not in ("model", "precision", "contestant", "model_version")
                    and v not in ("", "None") else v)
                for k, v in r.items()
            }  # fmt: skip
    return out


def value(model: str, size: str, key: str) -> float:
    return table()[model, size][key]


@cache
def scorecard_auc() -> float:
    y, p, _ = load("scorecard", "trained")
    return m.auc(y, p)


def method_brackets(ax) -> None:
    """'bitsandbytes' over 8-bit and NF4, 'HQQ' over the last three, just above the plot."""
    tr = ax.get_xaxis_transform()
    for lo, hi, label in ((0.8, 2.2, "bitsandbytes"), (2.8, 5.2, "HQQ")):
        ax.plot([lo, hi], [1.02, 1.02], color=F.AXIS, lw=1.2, transform=tr, clip_on=False)
        ax.text((lo + hi) / 2, 1.035, label, transform=tr, ha="center", va="bottom", fontsize=9,
                color=F.INK_2)  # fmt: skip


def size_axis(ax) -> None:
    ax.set_xticks(range(len(SIZES)), [TICK[s] for s in SIZES])
    ax.tick_params(axis="x", length=0, labelsize=9.5)
    ax.set_xlim(-0.35, len(SIZES) - 0.65)
    ax.grid(axis="x", visible=False)


# ---- 1. ranking by size --------------------------------------------------------------------------

RANK_DY = {"kev-4b-ft": 0, "kev-0.8b-ft": 7, "qwen-ft": -7}  # end-label offsets (points)


def ranking(dark: bool = False) -> Path:
    F.set_theme(dark)
    plt = F.plt
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=0.11, right=0.8, top=0.74, bottom=0.21)
    x = np.arange(len(SIZES))
    sc = scorecard_auc()
    ax.axhline(sc, color=F.COLOR["scorecard"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.3, sc - 0.008, f"Scorecard {sc:.3f}", fontsize=9, color=F.INK_2, va="top")
    ax.axhline(0.5, color=F.MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.text(-0.3, 0.508, "Coin flip 0.5", fontsize=9, color=F.INK_2, va="bottom")
    for model, key, name in MODELS:
        col = F.COLOR[key]
        ys = [value(model, s, "auc") for s in SIZES]
        ax.plot(x, ys, color=col, lw=2.2, zorder=3)
        ax.plot(x, ys, "o", ms=8, color=col, mec=F.SURFACE, mew=1.5, zorder=4)
        ax.annotate(f"{name}  {ys[-1]:.2f}", (x[-1], ys[-1]), xytext=(11, RANK_DY[model]),
                    textcoords="offset points", va="center", fontsize=10, color=F.INK,
                    annotation_clip=False)  # fmt: skip
    size_axis(ax)
    method_brackets(ax)
    ax.set_ylim(0.38, 0.86)
    ax.set_ylabel("Ranking (AUC): how often a defaulter is rated riskier")
    F.header(fig, "Ranking held to 4 bits, and collapsed at 2",
             "Each trained model's ranking at each size, on the same 2,000 customers.\n"
             "Down to 4 bits, all three stay within 0.005 of full size. At 3 bits only\n"
             "Kev-4B slips, to 0.776. At 2 bits, two rank worse than a coin flip.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, TRAINED_LINE, METHOD_LINE)))
    return F.save(fig, "post4_ranking.png")


# ---- 2. honesty by size --------------------------------------------------------------------------

HONEST_DY = {"kev-4b-ft": -9, "kev-0.8b-ft": 12, "qwen-ft": -9}  # label above or below its line


def honesty(dark: bool = False) -> Path:
    F.set_theme(dark)
    plt = F.plt
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.74, bottom=0.21)
    x = np.arange(len(SIZES))
    top = 0.1
    ax.axhspan(0, CHANCE, color=F.GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(0.05, 0.003, "Shaded: within chance for a perfectly honest model", fontsize=8.5,
            color=F.INK_2, va="bottom")  # fmt: skip
    off = []
    for model, key, name in MODELS:
        col = F.COLOR[key]
        ys = [value(model, s, "ece") for s in SIZES]
        ax.plot(x, ys, color=col, lw=2.2, zorder=3)
        shown = [(i, v) for i, v in zip(x, ys, strict=True) if v <= top]
        ax.plot([i for i, _ in shown], [v for _, v in shown], "o", ms=8, color=col,
                mec=F.SURFACE, mew=1.5, zorder=4)  # fmt: skip
        off.append((ys[-1], name))
        dy = HONEST_DY[model]
        ax.annotate(f"{name}, full size {100 * ys[0]:.1f}", (x[0], ys[0]), xytext=(-8, dy),
                    textcoords="offset points", ha="left", va="bottom" if dy > 0 else "top",
                    fontsize=9.5, color=F.INK)  # fmt: skip
    # The one change above 2 bits that the paired test (Holm over 15) finds bigger than chance.
    sig = [(m_, s) for m_, _k, _n in MODELS for s in SIZES[1:-1]
           if value(m_, s, "ece_vs_bf16_p_holm") < 0.05]  # fmt: skip
    for m_, s in sig:
        ax.annotate("bigger than\nchance", (SIZES.index(s), value(m_, s, "ece")), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top", fontsize=8.5,
                    color=F.INK_2, linespacing=1.1)  # fmt: skip
    off_text = "\n".join(f"{name}: {100 * v:.0f} points off" for v, name in sorted(off))
    ax.text(0.05, top * 0.97, f"Off the scale at 2 bits:\n{off_text}", ha="left", va="top",
            fontsize=9, color=F.INK_2, linespacing=1.35)  # fmt: skip
    size_axis(ax)
    method_brackets(ax)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(POINTS)
    ax.set_ylabel("Calibration error: points off, on average")
    F.header(fig, "Honesty held, with one change bigger than chance",
             "Each trained model's calibration error at each size. Above 2 bits, only one\n"
             "change is bigger than chance, and it's an improvement: 4-bit HQQ made\n"
             "Kev-0.8B more honest. At 2 bits, all three broke.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, TRAINED_LINE,
                             "Raw output. Calibration error (ECE) × 100, ten equal-width risk "
                             "groups. Changes: paired bootstrap, Holm over 15.")))  # fmt: skip
    return F.save(fig, "post4_honesty.png")


# ---- 3. per-customer drift for Kev-0.8B ----------------------------------------------------------

DRIFT_MODEL = ("kev-0.8b-ft", "kev-0.8b", "Kev-0.8B")
DRIFT_SIZES = ("bnb-nf4", "hqq-4", "hqq-3", "hqq-2")


def share_text(share: float) -> str:
    if share < 0.02:
        return f"{1000 * share:.0f} in 1,000"
    return f"{100 * share:.0f} in 100"


def drift(dark: bool = False) -> Path:
    F.set_theme(dark)
    plt = F.plt
    model, key, name = DRIFT_MODEL
    col = F.COLOR[key]
    ids0, _, p0 = Q.split_arrays(f"{model}-bf16", "trained", "test")
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.79, bottom=0.14, hspace=0.32, wspace=0.1)
    for ax, size in zip(axes.flat, DRIFT_SIZES, strict=True):
        ids, _, p = Q.split_arrays(f"{model}-{size}", "trained", "test")
        assert ids == ids0
        ax.scatter(p0, p, s=6, color=col, alpha=0.3, lw=0, zorder=2)
        ax.plot([0, 1], [0, 1], color=F.INK_2, lw=1.1, ls=(0, (4, 3)), zorder=3)
        moved = value(model, size, "mean_abs_change_vs_bf16")
        big = value(model, size, "moved_5pts_share")
        ax.set_title(ROW[size], loc="left", fontsize=11.5, pad=6)
        ax.text(0.03, 0.97, f"Average move: {100 * moved:.1f} pts\n"
                f"Moved 5+ pts: {share_text(big)}\n"
                f"Ranking {value(model, size, 'auc'):.3f}\n"
                f"{100 * value(model, size, 'ece'):.1f} pts off",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color=F.INK,
                linespacing=1.35,
                bbox={"boxstyle": "round,pad=0.3", "fc": F.SURFACE, "ec": "none",
                      "alpha": 0.85})  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Full-size answer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Shrunk answer")
    F.header(fig, "The scores held while the customers moved",
             f"Each dot is one test customer: {name}'s full-size answer against its shrunk\n"
             f"answer. At 3 bits, 6 in 10 customers moved by 5 points or more, yet its\n"
             f"ranking and honesty looked as good as full size ({value(model, 'hqq-3', 'auc'):.3f} "
             f"and {100 * value(model, 'hqq-3', 'ece'):.1f} off,\nagainst "
             f"{value(model, 'bf16', 'auc'):.3f} and {100 * value(model, 'bf16', 'ece'):.1f} at "
             f"full size).")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE,
                             "Dashed line: no change. Raw output, before any correction.")))
    return F.save(fig, "post4_drift.png")


# ---- 4. memory and time per answer, on the tuned setup (the speed addendum) ---------------------

COST_DY = {"kev-4b-ft": -0.2, "kev-0.8b-ft": 0.0, "qwen-ft": 0.2}  # row offsets, so dots don't hide
SPEED_ROWS = (  # (speed tag, post 4 weights it uses, row label)
    ("bf16-tuned", "bf16", "Full size (16-bit)"),
    ("bnb-nf4-tuned", "bnb-nf4", "4-bit, NF4"),
    ("hqq-4-gemlite", "hqq-4", "4-bit, HQQ\non GemLite"),
    ("hqq-4-aoint4", "hqq-4", "4-bit, HQQ on\nPyTorch int4"),
    ("fp8-dynamic", None, "FP8"),
)
SPEED_THROUGHPUT = Q.ROOT / "results" / "speed_throughput.csv"
SPEED_CSV = Q.ROOT / "results" / "speed.csv"


@cache
def speed_ms() -> dict[str, float]:
    """Average time per answer, one customer at a time, on the tuned setup: {contestant: ms}."""
    with SPEED_THROUGHPUT.open(encoding="utf-8") as f:
        return {r["contestant"]: 1000 / float(r["customers_per_second"])
                for r in csv.DictReader(f) if r["batch"] == "1"}  # fmt: skip


@cache
def speed_peak_gib() -> dict[str, float]:
    with SPEED_CSV.open(encoding="utf-8") as f:
        return {r["contestant"]: float(r["gpu_peak_gb"]) for r in csv.DictReader(f)}


def cost(dark: bool = False) -> Path:
    F.set_theme(dark)
    plt = F.plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 6.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.71, bottom=0.2, wspace=0.12)
    ms, peak = speed_ms(), speed_peak_gib()
    for model, key, _name in MODELS:
        col = F.COLOR[key]
        for y, (tag, base, _label) in enumerate(SPEED_ROWS):
            yy = y + COST_DY[model]
            # Memory: the first runs' measurement of the same weights (every version served the
            # same way there); FP8 has no first run, so its own run (Qwen, in-process).
            gib = value(model, base, "gpu_peak_gb") if base else peak.get(f"{model}-{tag}")
            if gib is not None:
                left.plot(GB * gib, yy, "o", ms=8, color=col, mec=F.SURFACE, mew=1.2, zorder=3)
            t = ms.get(f"{model}-{tag}")
            if t is not None:
                right.plot(t, yy, "o", ms=8, color=col, mec=F.SURFACE, mew=1.2, zorder=3)
            if tag in ("bf16-tuned", "bnb-nf4-tuned"):  # the same weights, timed in the first runs
                t0 = value(model, base, "latency_ms_median")
                right.plot(t0, yy, "o", ms=7, mfc="none", mec=col, mew=1.3, zorder=3)
                right.annotate("", (t, yy), xytext=(t0, yy), zorder=2,
                               arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1,
                                           "alpha": 0.6, "shrinkA": 4, "shrinkB": 5})  # fmt: skip
    failed = {"hqq-4-aoint4": (COST_DY["qwen-ft"], "Qwen3.5-4B: wouldn't load"),
              "fp8-dynamic": (-0.1, "Both Kev models: no answers (NaN)")}  # fmt: skip
    for y, (tag, _base, _label) in enumerate(SPEED_ROWS):
        if tag in failed:
            dy, text = failed[tag]
            x0 = 5 if tag == "fp8-dynamic" else 52
            right.text(x0, y + dy, text, fontsize=8.5, color=F.INK_2, va="center")
    left.text(0.4, len(SPEED_ROWS) - 1 - 0.1, "Kev: no FP8 run", fontsize=8.5, color=F.INK_2,
              va="center")  # fmt: skip
    for ax in (left, right):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=F.INK)
    left.set_yticks(range(len(SPEED_ROWS)), [label for _t, _b, label in SPEED_ROWS])
    left.set_ylim(len(SPEED_ROWS) - 0.5, -0.6)
    left.set_xlim(0, 10)
    left.set_xlabel("GPU memory at peak (GB)")
    right.set_xlim(0, 260)
    right.set_xlabel("Time per answer (milliseconds)")
    left.set_title("Memory", loc="left", fontsize=11.5)
    right.set_title("Time, tuned setup", loc="left", fontsize=11.5)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=8, color=F.COLOR[key], label=name)
               for _model, key, name in MODELS]  # fmt: skip
    handles.append(plt.Line2D([], [], marker="o", ls="none", ms=7, mfc="none", mec=F.INK_2,
                              mew=1.3, label="first runs"))  # fmt: skip
    left.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.36, 1.08), ncol=4,
                fontsize=9.5, handletextpad=0.3, columnspacing=1.1)  # fmt: skip
    gain = ms["kev-4b-ft-hqq-4-gemlite"] / ms["kev-4b-ft-bf16-tuned"]
    F.header(fig, "Shrinking saved memory, not time",
             "GPU memory and time per answer, one customer at a time on an NVIDIA L4,\n"
             "with the setup tuned. 4 bits cut the 4B models' memory by more than half.\n"
             "Only one shrunk version answered faster than its full size: Kev-4B on\n"
             f"GemLite, by {100 * (1 - gain):.0f}%. Hollow dots: my first, untuned runs.",
             )  # fmt: skip
    F.footer(fig, "\n".join((
        "Time: average over 500 customers (first runs: median over 2,000). Tuned: fast kernels, "
        "Kev's CUDA graphs on",
        "(its fused kernels need full-size weights), Qwen compiled. Memory: the first runs; FP8 "
        "from its own run.",
        "At $0.44 per GPU hour, 100 ms per answer is about 1.2 cents per 1,000 answers.",
    )))  # fmt: skip
    return F.save(fig, "post4_cost.png")


FIGURES = (ranking, honesty, drift, cost)


def main() -> None:
    names = set(sys.argv[1:])  # optional: only these, e.g. `drift cost`
    for dark in (False, True):
        for fig in FIGURES:
            if not names or fig.__name__ in names:
                print(f"Saved -> {fig(dark)}")


if __name__ == "__main__":
    main()
