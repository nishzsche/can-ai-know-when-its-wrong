"""Post 5 figures: the same customers in other layouts (PLAN.md §12, post 5 spec).

    uv run python analysis/layouts_figures.py   # results/figures/post5_*.png, light and dark

- post5_ranking.png: each zero-shot model's ranking (AUC) with the original table and the five
  other layouts, with the scorecard and a coin flip; changes within chance are faded.
- post5_moved.png: how far each model's answers moved per customer in each layout, against its
  own run-to-run noise and post 4's biggest move from shrinking to 3 bits.
- post5_customers.png: four models-and-layouts, each test customer's answer to the original table
  against the new layout.
- post5_promised.png: the average risk each model promised in each layout, against the 22.1% who
  defaulted.
- post5_pieces.png: one row of the original table as a program reads it (four values) and as the
  models' tokenizers split it (data/render.py; downloads the two tokenizer files if not cached).

    uv run python analysis/layouts_figures.py moved customers   # only these (row_pieces: the last)

Reads results/layouts.csv (analysis/layouts.py) and the prediction records. Same palette, style
and dark variant as posts 1-4 (analysis/figures.py): each model keeps its colour, the original
table is an open dot and the other layouts filled ones, and every mark is named in text.
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
import layouts as L  # noqa: E402
import metrics as m  # noqa: E402
from robustness import load  # noqa: E402

# Rows top to bottom, the model whose answers moved most first.
ORDER = ("laya", "kev-4b", "qwen-logprobs", "jev", "kev-0.8b", "frontier-verbal")
NAME = dict(F.NAME)
LAYOUT = {"v2": "Original table", "markdown": "Markdown", "csv": "CSV", "json": "JSON",
          "prose": "Sentences", "aligned": "Realigned table"}  # fmt: skip
TEST_LINE = "Test set: 2,000 real credit-card customers (UCI, Taiwan 2005), 22.1% defaulted."
ZERO_SHOT_LINE = ("All six AI models answered cold (zero-shot); the scorecard learned from 20,000 "
                  "labelled customers.")  # fmt: skip
SAME_VALUES_LINE = ("Every layout holds the same values, character for character; only the layout "
                    "changes.")  # fmt: skip
SHRINK_3BIT = 0.066  # post 4: Kev-0.8B at 3 bits, the biggest average move from shrinking


@cache
def table() -> dict[tuple[str, str], dict]:
    out = {}
    with L.OUT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["model"], r["layout"]] = {
                k: (v if k in ("model", "layout", "label", "model_version") else float(v))
                for k, v in r.items()
            }  # fmt: skip
    return out


def value(model: str, layout: str, key: str) -> float:
    """A layout's value; for "v2", the original table's (stored on every row as *_v2)."""
    if layout == "v2":
        return table()[model, "json"][f"{key}_v2"]
    return table()[model, layout][key]


@cache
def scorecard_auc() -> float:
    y, p, _ = load("scorecard", "trained")
    return m.auc(y, p)


def tint(color: str, share: float) -> str:
    """`color` mixed with the page colour: a solid tint, so overlaps and the PNG's 256-colour
    palette can't shift its hue the way transparency did."""
    a = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(F.SURFACE[i : i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{round(v):02x}" for v in share * a + (1 - share) * b)


TOP = -0.85  # y for reference labels, above the first row


def rows_axis(ax) -> None:
    ax.set_yticks(range(len(ORDER)), [NAME[c] for c in ORDER])
    ax.set_ylim(len(ORDER) - 0.5, -1.1)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.grid(axis="y", visible=False)


def strip(ax, key: str, fmt, label_rows: dict, faded=lambda c, lay: False) -> None:
    """One row per model: a thin line over its range, filled dots for the five other layouts and
    an open dot for the original table. `label_rows` names the dots to label in each row."""
    for i, c in enumerate(ORDER):
        col = F.COLOR[c]
        vals = {lay: value(c, lay, key) for lay in LAYOUT}
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=tint(col, 0.45), lw=2,
                zorder=2, solid_capstyle="round")  # fmt: skip
        for lay in sorted(L.LAYOUTS, key=lambda k: not faded(c, k)):  # full colour on top
            ax.plot(vals[lay], i, "o", ms=8.5, color=tint(col, 0.35) if faded(c, lay) else col,
                    mec=F.SURFACE, mew=1.5, zorder=3)  # fmt: skip
        ax.plot(vals["v2"], i, "o", ms=10, mfc=F.SURFACE, mec=col, mew=2.2, zorder=4)
        for lay, side in label_rows.get(c, ()):
            ax.annotate(f"{LAYOUT[lay]} {fmt(vals[lay])}", (vals[lay], i),
                        xytext=(0, 10 if side == "above" else -10), textcoords="offset points",
                        ha="center", va="bottom" if side == "above" else "top", fontsize=10,
                        color=F.INK_2)  # fmt: skip


# ---- 1. ranking by layout ------------------------------------------------------------------------

RANK_LABELS = {"laya": (("json", "above"), ("prose", "above")),
               "kev-4b": (("v2", "above"), ("json", "above"))}  # fmt: skip


def ranking(dark: bool = False) -> Path:
    F.set_theme(dark)
    fig, ax = F.plt.subplots(figsize=(7.6, 6.8))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.78, bottom=0.22)
    sc = scorecard_auc()
    ax.axvline(0.5, color=F.MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.text(0.503, TOP, "Coin flip", fontsize=10, color=F.INK_2, va="center")
    ax.axvline(sc, color=F.COLOR["scorecard"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(sc - 0.004, TOP, f"Scorecard {sc:.3f}", fontsize=10, color=F.INK_2, ha="right",
            va="center")  # fmt: skip
    strip(ax, "auc", lambda v: f"{v:.3f}", RANK_LABELS,
          faded=lambda c, lay: value(c, lay, "auc_vs_v2_p_holm") >= 0.05)  # fmt: skip
    rows_axis(ax)
    ax.set_xlim(0.44, 0.8)
    ax.set_xlabel("Ranking (AUC): how often a defaulter is rated riskier")
    F.header(fig, "Same numbers, new layout, new ranking",
             "Each AI model's ranking of the same 2,000 customers: open dot, the original\n"
             "table; filled dots, five other layouts of the same numbers; faded, within\n"
             "chance of the original. Laya went from 0.716 to worse than a coin flip, and\n"
             "Kev-4B from 0.594 to 0.729.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, ZERO_SHOT_LINE,
                             "Other layouts: Markdown, CSV, JSON, sentences, a realigned table; "
                             "the same values in each.",
                             "Within chance: paired bootstrap against the original table, Holm "
                             "over 30.")))  # fmt: skip
    return F.save(fig, "post5_ranking.png")


# ---- 2. how far the answers moved, per customer --------------------------------------------------

MOVED_LABELS = {"laya": ("csv", "json"), "kev-4b": ("json",), "qwen-logprobs": ("json",),
                "jev": ("aligned",), "kev-0.8b": ("csv",)}  # fmt: skip


def moved(dark: bool = False) -> Path:
    F.set_theme(dark)
    fig, ax = F.plt.subplots(figsize=(7.6, 6.8))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.78, bottom=0.2)
    ax.axvline(SHRINK_3BIT, color=F.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(SHRINK_3BIT + 0.004, TOP, "Post 4: shrinking to 3 bits, at most", fontsize=10,
            color=F.INK_2, va="center")  # fmt: skip
    for i, c in enumerate(ORDER):
        col = F.COLOR[c]
        noise = table()[c, "json"]["noise_mean_abs_change"]
        if noise > 0.001:
            ax.barh(i, noise, height=0.42, color=F.GRID, zorder=1)
        vals = {lay: value(c, lay, "mean_abs_change") for lay in L.LAYOUTS}
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=tint(col, 0.45), lw=2,
                zorder=2, solid_capstyle="round")  # fmt: skip
        for v in vals.values():
            ax.plot(v, i, "o", ms=8.5, color=col, mec=F.SURFACE, mew=1.5, zorder=3)
        if c in MOVED_LABELS:
            keys = MOVED_LABELS[c]
            x = max(vals[k] for k in keys)
            pts = f"{100 * x:.0f}" if x >= 0.1 else f"{100 * x:.1f}"
            ax.annotate(f"{' and '.join(LAYOUT[k] for k in keys)} {pts}", (x, i),
                        xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color=F.INK_2)  # fmt: skip
        else:  # the page colour behind the text, so post 4's line passes under it
            ax.annotate(f"All five: {100 * min(vals.values()):.1f} to "
                        f"{100 * max(vals.values()):.1f}", (max(vals.values()), i),
                        xytext=(12, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=10, color=F.INK_2, zorder=5,
                        bbox={"fc": F.SURFACE, "ec": "none", "pad": 1.5})  # fmt: skip
    rows_axis(ax)
    ax.set_xlim(0, 0.29)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax.set_xlabel("Average move per customer, in percentage points")
    F.header(fig, "Laya's answers moved 26 points, Sol's 3",
             "How far each model's answer moved per customer when the same numbers came in\n"
             "another layout. Each dot is one of the five layouts. Grey bar: how far the answers\n"
             "move when the original table is simply asked again. Sol moved no more than "
             "that.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, "All six AI models answered cold (zero-shot). Grey bar: "
                             "200 test customers asked again.",
                             "Kev gives the same answer every time; Laya's re-run was on another "
                             "machine.",
                             "Dashed line: post 4's trained Kev-0.8B at 3 bits, "
                             "6.6 points.")))  # fmt: skip
    return F.save(fig, "post5_moved.png")


# ---- 3. each customer, original table against a new layout ---------------------------------------

PANELS = (("laya", "json"), ("laya", "prose"), ("kev-4b", "json"), ("frontier-verbal", "json"))


def share_text(share: float) -> str:
    return f"{100 * share:.0f} in 100"


def customers(dark: bool = False) -> Path:
    F.set_theme(dark)
    fig, axes = F.plt.subplots(2, 2, figsize=(7.6, 9.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.745, bottom=0.12, hspace=0.62, wspace=0.1)
    for ax, (c, lay) in zip(axes.flat, PANELS, strict=True):
        ref = L.answers(c, L.PREDICTIONS_DIR)
        directory, suffix, _ = L.LAYOUTS[lay]
        other = L.answers(f"{c}-{suffix}", directory)
        ids = sorted(ref)
        _, p0, p1 = L.arrays(ref, other, ids)
        ax.scatter(p0, p1, s=6, color=F.COLOR[c], alpha=0.3, lw=0, zorder=2)
        ax.plot([0, 1], [0, 1], color=F.INK_2, lw=1.1, ls=(0, (4, 3)), zorder=3)
        r = table()[c, lay]
        ax.set_title(f"{NAME[c]}, {LAYOUT[lay]}", loc="left", fontsize=11.5, pad=52)
        ax.text(0, 1.03, f"Average move: {100 * r['mean_abs_change']:.1f} pts\n"
                f"Moved 5+ pts: {share_text(r['moved_5pts_share'])}\n"
                f"Ranking {r['auc_v2']:.3f} → {r['auc']:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=F.INK_2,
                linespacing=1.3)  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Answer to the original table")
    for ax in axes[:, 0]:
        ax.set_ylabel("Answer to the new layout")
    laya = table()["laya", "json"]
    F.header(fig, "Same customers, same numbers, different answers",
             "Each dot is one test customer: a model's answer to the original table against\n"
             "its answer to a new layout. Laya's JSON answers barely relate to its table\n"
             f"answers (rank correlation {laya['spearman']:.2f}). Sol's stay on the line, within "
             "its usual\nrun-to-run wobble.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, "Dashed line: no change. Sol recognizes this dataset "
                             "and knows its default rate.")))  # fmt: skip
    return F.save(fig, "post5_customers.png")


# ---- 4. the average risk promised, by layout -----------------------------------------------------

PROMISED_LABELS = {"laya": (("prose", "above"), ("csv", "above")),
                   "kev-4b": (("json", "above"), ("prose", "above"))}  # fmt: skip


def promised(dark: bool = False) -> Path:
    F.set_theme(dark)
    fig, ax = F.plt.subplots(figsize=(7.6, 6.8))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.8, bottom=0.2)
    rate = table()["laya", "json"]["observed_rate"]
    ax.axvline(rate, color=F.INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(rate + 0.006, TOP, f"Actually defaulted: {100 * rate:.1f}%", fontsize=10,
            color=F.INK_2, va="center")  # fmt: skip
    strip(ax, "mean", lambda v: f"{100 * v:.0f}%", PROMISED_LABELS)
    rows_axis(ax)
    ax.set_xlim(0.08, 0.6)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.set_xlabel("Average risk the model promised")
    F.header(fig, "The risk each model promised moved with the layout",
             "Each AI model's average answer for the same 2,000 customers: open dot, the\n"
             "original table; filled dots, the five other layouts. Laya promised 14% for\n"
             "sentences and 44% for CSV. Sol stayed within a point of 26%.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, "All six AI models answered cold (zero-shot). Raw "
                             "answers, before any correction.",
                             "Sol recognizes this dataset and knows its default "
                             "rate.")))  # fmt: skip
    return F.save(fig, "post5_promised.png")


# ---- 5. one row, as a program reads it and as the models read it ---------------------------------

ROW_TEXT = "Sep 2005 | NT$46,990 | NT$2,000 | not late (code 0)"  # a test customer's September row
PROGRAM = (("Sep 2005", "month"), ("46990", "bill"), ("2000", "payment"),
           ("not late (code 0)", "status"))  # fmt: skip


def pieces(name: str) -> list[str]:
    """ROW_TEXT split by one contestant tokenizer (data/render.py), as the model reads it."""
    sys.path.insert(0, str(L.ROOT / "data"))
    from render import load_tokenizer

    tok = load_tokenizer(name)
    enc = tok.encode(ROW_TEXT, add_special_tokens=False)
    return [tok.decode([i]) for i in enc.ids]


def chips(ax, y: float, texts: list[str], fill: str, width: float = 100.0) -> float:
    """Draw each piece as a box, wrapping at `width`; returns the y below the last line."""
    char, pad, gap, line = 1.32, 1.3, 0.9, 7.5
    x = 0.0
    for t in texts:
        shown = t.strip() or "␣"
        w = len(shown) * char + 2 * pad
        if x + w > width:
            x, y = 0.0, y - line
        ax.add_patch(F.plt.Rectangle((x, y - 2.6), w, 5.2, fc=fill, ec=F.SURFACE, lw=1.5))
        ax.text(x + w / 2, y, shown, ha="center", va="center", family="monospace", fontsize=11,
                color=F.INK)  # fmt: skip
        x += w + gap
    return y - line


def row_pieces(dark: bool = False) -> Path:
    F.set_theme(dark)
    fig, ax = F.plt.subplots(figsize=(7.6, 6.2))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.8, bottom=0.12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-63, 4)
    ax.axis("off")
    kev, laya = pieces("kev (Qwen3.5)"), pieces("laya (ModernBERT)")
    ax.text(0, 2, "A program: four values", fontsize=12, fontweight="bold", color=F.INK)
    x = 0.0
    for text, what in PROGRAM:
        w = len(text) * 1.32 + 6
        ax.add_patch(F.plt.Rectangle((x, -9.6), w, 5.2, fc=F.GRID, ec=F.SURFACE, lw=1.5))
        ax.text(x + w / 2, -7, text, ha="center", va="center", family="monospace", fontsize=11,
                color=F.INK)  # fmt: skip
        ax.text(x + w / 2, -12.5, what, ha="center", va="top", fontsize=10, color=F.INK_2)
        x += w + 4
    ax.text(0, -21, f"Kev and Qwen: {len(kev)} pieces", fontsize=12, fontweight="bold",
            color=F.INK)  # fmt: skip
    y = chips(ax, -28, kev, tint(F.COLOR["kev-4b"], 0.3))
    ax.text(0, y - 1, f"Laya: {len(laya)} pieces", fontsize=12, fontweight="bold",
            color=F.INK)  # fmt: skip
    chips(ax, y - 8, laya, tint(F.COLOR["laya"], 0.3))
    F.header(fig, "A program reads four values; a model reads 31 pieces",
             "One row of the original table, as a program that splits it at the bars sees it,\n"
             "and as the models read it. Nothing in the pieces says which number is the bill\n"
             "and which the payment: a model has to work that out from where they "
             "sit.")  # fmt: skip
    row = ROW_TEXT.replace("$", r"\$")  # a bare pair of $ would be read as maths
    F.footer(fig, "\n".join((f"The row: {row}",
                             "Pieces from each model's own tokenizer; Kev and Qwen share one. "
                             "␣: a piece that is only a space.")))  # fmt: skip
    return F.save(fig, "post5_pieces.png")


def social_card() -> Path:
    """1200x630 link-preview image in the style of posts 1-4: title and hook on the left, and on
    the right each model's ranking with the original table (open) and the other layouts (filled),
    light only like theirs."""
    F.set_theme(False)
    fig = F.plt.figure(figsize=(6, 3.15))
    t = fig.text
    dot = "·"
    t(0.055, 0.89, f"LEARNING CURVE  {dot}  nishzsche.github.io", fontsize=7.5, color=F.MUTED,
      fontweight="bold", va="center")  # fmt: skip
    t(0.055, 0.80, "Why AI can write\ncode but struggle\nto read a table", fontsize=17,
      fontweight="bold", color=F.INK, va="top", linespacing=1.05)  # fmt: skip
    t(0.055, 0.425, "Six AI models read the same customers\nas a table, sentences, CSV and JSON.\n"
      "Not one number changed.", fontsize=8.5, color=F.INK_2, va="top",
      linespacing=1.25)  # fmt: skip
    t(0.055, 0.2, "The answers did.", fontsize=12, fontweight="bold", color=F.INK, va="top")
    t(0.055, 0.075, f"6 AI models, answering cold  {dot}  2,000 real credit-card customers",
      fontsize=7.5, color=F.MUTED, va="center")  # fmt: skip
    t(0.56, 0.875, "Ranking (AUC) by layout", fontsize=9, fontweight="bold", color=F.INK,
      va="center")  # fmt: skip
    t(0.56, 0.815, "open: the original table  ·  filled: other layouts", fontsize=6.5,
      color=F.MUTED, va="center")  # fmt: skip
    ax = fig.add_axes((0.71, 0.2, 0.25, 0.54))
    ax.axvline(0.5, color=F.MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(0.505, len(ORDER) - 0.35, "coin flip", fontsize=6, color=F.MUTED, va="center")
    for i, c in enumerate(ORDER):
        col = F.COLOR[c]
        vals = {lay: value(c, lay, "auc") for lay in LAYOUT}
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=tint(col, 0.45), lw=1.6,
                zorder=2, solid_capstyle="round")  # fmt: skip
        for lay in L.LAYOUTS:
            ax.plot(vals[lay], i, "o", ms=4.5, color=col, mec=F.SURFACE, mew=0.6, zorder=3)
        ax.plot(vals["v2"], i, "o", ms=5.5, mfc=F.SURFACE, mec=col, mew=1.4, zorder=4)
    ax.set_yticks(range(len(ORDER)), [NAME[c] for c in ORDER], fontsize=7.5)
    ax.set_ylim(len(ORDER) - 0.1, -0.6)
    ax.set_xlim(0.44, 0.8)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0, labelcolor=F.INK)
    ax.grid(axis="y", visible=False)
    return F.save(fig, "post5_social_card.png", card=True)


FIGURES = (ranking, moved, customers, promised, row_pieces)


def main() -> None:
    wanted = set(sys.argv[1:])  # optional: only these, e.g. `moved row_pieces` or `social_card`
    for fig in FIGURES:
        if wanted and fig.__name__ not in wanted:
            continue
        for dark in (False, True):
            print(fig(dark))
    if not wanted or "social_card" in wanted:
        print(social_card())


if __name__ == "__main__":
    main()
