"""Build post 2's LinkedIn carousel PDF from the slide text in writeup/accuracy_linkedin.md.

    uv run python writeup/build_accuracy_carousel.py            # writeup/accuracy_carousel.pdf
    uv run python writeup/build_accuracy_carousel.py --png DIR  # also one PNG per slide

Same size, type and page furniture as post 1's carousel (writeup/build_carousel.py, whose helpers
it reuses). Charts are redrawn at phone size from results/accuracy.json (run analysis/accuracy.py
first), in the palette and per-model colours of analysis/figures.py. Slides 3 and 6 share one row
order (by accuracy) so they can be read side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_carousel as C  # noqa: E402  (also puts analysis/ on the path)
import figures as F  # noqa: E402
from accuracy import pct1  # noqa: E402
from build_carousel import M, note, text_block, wrap  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

TITLE = "Why 99% accuracy can still be a disaster: 8 models tested"  # LinkedIn's limit: 58 chars
SLIDES_MD = HERE / "accuracy_linkedin.md"
OUT_PDF = HERE / "accuracy_carousel.pdf"
RESULTS = HERE.parent / "results" / "accuracy.json"
FAIR = "* Learned from 20,000 past customers; the AI models answered cold, with no examples."
ROWS = ("gbm", "qwen-logprobs", "scorecard", "frontier-verbal", "kev-0.8b", "laya", "jev", "kev-4b")
NOTHING = "Doing nothing"
NEAR_ZERO = ("kev-0.8b", "qwen-logprobs", "jev", "kev-4b")
C.SERIES = "Why 99% accuracy can still be a disaster"


def name(c: str) -> str:
    return NOTHING if c == NOTHING else F.NAME[c] + ("*" if c in F.BASELINES else "")


def by_accuracy(res: dict) -> list[str]:
    m, floor = res["models"], res["always_no_accuracy"]
    return sorted([*ROWS, NOTHING],
                  key=lambda c: -(floor if c == NOTHING else m[c]["accuracy_at_50"]))  # fmt: skip


def rows_axis(ax, keys: list[str], labels: list[str] | None = None) -> None:
    ax.set_yticks(range(len(keys)), labels or [name(c) for c in keys])
    ax.set_ylim(len(keys) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", visible=False)


def slide_hook(fig, s, n, total, res):
    lines = wrap(fig, s["headline"], 38, fontweight="bold")
    fig.text(M, 0.74, "\n".join(lines), fontsize=38, fontweight="bold", color=F.INK, va="top",
             linespacing=1.12)  # fmt: skip
    fig.text(M, 0.42, "\n".join(wrap(fig, s["body"], 24)), fontsize=24, color=F.INK_2,
             va="top", linespacing=1.25)  # fmt: skip
    fig.text(M, 0.18, s["footer"], fontsize=15, color=F.INK_2)
    fig.text(1 - M, 0.035, "Swipe →", fontsize=12, color=F.INK_2, ha="right")
    fig.text(M, 0.035, f"{n} / {total}", fontsize=10, color=F.MUTED)


def slide_fraud(fig, s, n, total, res):
    """1,000 payments as a grid, one of them fraud."""
    y = C.frame(fig, n, total, s)
    cols, rows, fraud = 50, 20, 637
    ax = fig.add_axes([M, 0.2, 1 - 2 * M, y - 0.26])
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        ax.add_patch(F.plt.Rectangle((c + 0.12, rows - 1 - r + 0.12), 0.76, 0.76, lw=0,
                                     color=F.INK if i == fraud else F.GRID))  # fmt: skip
    r, c = divmod(fraud, cols)
    ax.annotate("the 1 fraud", (c + 0.5, rows - 1 - r + 0.5), xytext=(c + 6, rows + 1.6),
                fontsize=13, color=F.INK, ha="left", va="center", annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": F.INK, "lw": 1})  # fmt: skip
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal", anchor="W")
    ax.axis("off")
    text_block(fig, 0.15, s["footer"], 14, color=F.INK_2)


def slide_floor(fig, s, n, total, res):
    """Accuracy at 50% with the do-nothing floor."""
    y = C.frame(fig, n, total, s)
    m, floor = res["models"], res["always_no_accuracy"]
    keys = by_accuracy(res)
    ax = fig.add_axes([M + 0.27, 0.16, 0.56, y - 0.22])
    ax.axvspan(0.5, floor, color=F.GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(floor, color=F.MUTED, lw=1.5, zorder=1)
    for i, c in enumerate(keys):
        x = floor if c == NOTHING else m[c]["accuracy_at_50"]
        if c == NOTHING:
            ax.plot(x, i, "X", ms=14, color=F.MUTED, zorder=3)
        else:
            ax.plot(x, i, "o", ms=14, color=F.COLOR[c], mec=F.SURFACE, mew=2, zorder=3)
        ax.text(x + 0.013, i, pct1(x), va="center", fontsize=13, color=F.INK)
    ax.text(0.505, -0.5, "less accurate than\ndoing nothing", fontsize=11, color=F.INK_2,
            va="top")  # fmt: skip
    rows_axis(ax, keys)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.set_xlim(0.5, 0.9)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Accuracy at the usual 50% cut-off", fontsize=12)
    note(fig, FAIR)


def slide_mistakes(fig, s, n, total, res):
    """Where the 2,000 landed at 50%: the 442 defaulters and the 1,558 payers."""
    y = C.frame(fig, n, total, s)
    m = res["models"]
    rows = ((NOTHING, res["always_no_confusion"], F.MUTED),
            ("Scorecard*", m["scorecard"]["confusion"]["50"], F.COLOR["scorecard"]),
            ("Jev", m["jev"]["confusion"]["50"], F.COLOR["jev"]))  # fmt: skip
    groups = (("The 442 who defaulted", "defaulters_missed", "missed",
               "defaulters_caught", "caught"),
              ("The 1,558 who paid", "payers_turned_away", "turned away",
               "payers_approved", "approved"))  # fmt: skip
    top, bottom, gap = y - 0.06, 0.17, 0.075
    h = (top - bottom - gap) / 2
    for k, (title, bad, bad_word, good, good_word) in enumerate(groups):
        y0 = top - (k + 1) * h - k * gap
        ax = fig.add_axes([M + 0.22, y0, 1 - 2 * M - 0.22, h])
        width = rows[0][1][bad] + rows[0][1][good]
        for i, (_, cm, col) in enumerate(rows):
            ax.barh(i, cm[bad], color=col, height=0.36)
            ax.barh(i, cm[good], left=cm[bad], color=col, alpha=0.25, height=0.36)
            kw = {"va": "bottom", "fontsize": 12.5, "color": F.INK}
            ax.text(0, i - 0.24, f"{cm[bad]:,} {bad_word}", ha="left", **kw)
            ax.text(width, i - 0.24, f"{cm[good]:,} {good_word}", ha="right", **kw)
        ax.set_xlim(0, width)
        ax.set_ylim(len(rows) - 0.6, -0.95)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
        ax.set_xticks([])
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(M, y0 + h + 0.006, title, fontsize=14.5, fontweight="bold", color=F.INK)
    fig.text(M, 0.115, "Solid: a mistake.  Pale: a right call.  At the usual 50% cut-off.",
             fontsize=11.5, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_lend(fig, s, n, total, res):
    y = C.frame(fig, n, total, s, headline_size=30) - 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, 20, bullet=True, color=F.INK) - 0.03
    fig.text(M, 0.16, s["footer"], fontsize=24, fontweight="bold", color=F.INK)


def slide_profit(fig, s, n, total, res, loss: str = "4"):
    """Profit per 1,000 on own percentages (open) and with a learned cut-off (filled)."""
    y = C.frame(fig, n, total, s)
    m = res["models"]
    keys = by_accuracy(res)
    ax = fig.add_axes([M + 0.36, 0.16, 1 - 2 * M - 0.36, y - 0.29])
    ax.axvline(0, color=F.AXIS, lw=1.2, zorder=1)
    kw = {"va": "center", "fontsize": 12.5, "color": F.INK}
    for i, c in enumerate(keys):
        if c == NOTHING:
            v = res["approve_everyone"][loss]
            ax.plot(v, i, "X", ms=13, color=F.MUTED, zorder=3)
            ax.text(v + 22, i, f"loses {-v:.0f}", ha="left", **kw)
            continue
        d = m[c]["decisions"][loss]
        own, cut = d["own_percentages"]["profit"], d["tune_cutoff"]["profit"]
        col = F.COLOR[c]
        ax.plot([own, cut], [i, i], color=col, lw=3, alpha=0.35, zorder=2)
        ax.plot(cut, i, "o", ms=13, color=col, zorder=3)
        ax.plot(own, i, "o", ms=13, color=F.SURFACE, mec=col, mew=2.5, zorder=4)
        if abs(own - cut) < 30:
            ax.text(max(own, cut) + 22, i, f"{own:.0f}", ha="left", **kw)
        else:
            ax.text(own - 22, i, f"{own:.0f}", ha="right", **kw)
            ax.text(cut + 22, i, f"{cut:.0f}", ha="left", **kw)
    floor = res["always_no_accuracy"]
    labels = [f"{name(c)}  {pct1(floor if c == NOTHING else m[c]['accuracy_at_50'])}"
              for c in keys]  # fmt: skip
    rows_axis(ax, keys, labels)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-190, 420)
    ax.set_xlabel("Profit per 1,000 customers", fontsize=12)
    ax.text(-0.02, 1.01, "Model, accuracy", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color=F.MUTED)  # fmt: skip
    handles = [
        F.plt.Line2D([], [], marker="o", ls="none", ms=11, color=F.INK_2, mfc=F.SURFACE, mew=2.5,
                     label="Lending on its own percentages"),
        F.plt.Line2D([], [], marker="o", ls="none", ms=11, color=F.INK_2,
                     label="Its ranking, with a cut-off learned from past outcomes"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M - 0.01, y + 0.005),
               fontsize=12, handletextpad=0.3, frameon=False)  # fmt: skip
    note(fig, FAIR)


def slide_costs(fig, s, n, total, res):
    """Profit at own percentages as the cost of a default rises from 1 to 10."""
    y = C.frame(fig, n, total, s)
    sw = res["sweep_own_percentages"]
    xs = np.array(sw["loss_ratios"])
    ax = fig.add_axes([M + 0.1, 0.17, 0.48, y - 0.24])
    ax.axhline(0, color=F.AXIS, lw=1.2, zorder=1)
    ev = np.array(sw["approve_everyone"])
    ax.plot(xs, ev, color=F.MUTED, lw=2, ls=(0, (1, 2)), zorder=2)
    whole = xs == xs.astype(int)
    ax.plot(xs[whole], ev[whole], "X", ms=9, color=F.MUTED, zorder=2)
    ax.text(4.9, -330, "Approve\neveryone", fontsize=12.5, color=F.INK, va="center")
    ends = []
    for c in ROWS:
        ys = np.array(sw["models"][c])
        lw = 2 if c in F.BASELINES else (1.8 if c in NEAR_ZERO else 2.8)
        ax.plot(xs, ys, color=F.COLOR[c], lw=lw, ls=(0, (4, 3)) if c == "scorecard" else "-",
                zorder=3)  # fmt: skip
        if c not in NEAR_ZERO:
            ends.append((ys[-1], name(c)))
    ends.append((float(np.mean([sw["models"][c][-1] for c in NEAR_ZERO])),
                 "Jev, Qwen, Kevs:\nalmost no approvals"))  # fmt: skip
    placed, prev = None, 1
    for v, label in sorted(ends, reverse=True):  # push labels down so none overlap
        lines = label.count("\n") + 1
        placed = v if placed is None else min(v, placed - (48 * (prev + lines) + 8))
        prev = lines
        ax.annotate(label, (10, v), xytext=(10.4, placed), fontsize=11.5, color=F.INK,
                    va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": F.AXIS, "lw": 0.9})  # fmt: skip
    ax.set_xlim(1, 10)
    ax.set_ylim(-620, 660)
    ax.set_xticks(range(1, 11))
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Cost of one default, in units a payer earns", fontsize=12)
    ax.set_ylabel("Profit per 1,000 customers", fontsize=12)
    note(fig, FAIR)


def slide_takeaways(fig, s, n, total, res):
    y = C.frame(fig, n, total, s, headline_size=30) - 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, 20, bullet=True, color=F.INK) - 0.03
    text_block(fig, 0.17, s["footer"], 19, fontweight="bold", color=F.INK)


def slide_next(fig, s, n, total, res):
    y = C.frame(fig, n, total, s, headline_size=32)
    text_block(fig, y - 0.08, s["footer"], 24, fontweight="bold", color=F.INK)
    fig.text(M, 0.12, "Write-up, code and every number: link in the first comment.",
             fontsize=14, color=F.INK_2)  # fmt: skip


MAKERS = {  # by slide title in accuracy_linkedin.md, so slides can be reordered there
    "Hook": slide_hook,
    "The fraud detector": slide_fraud,
    "The floor": slide_floor,
    "Which mistakes": slide_mistakes,
    "Make them lend": slide_lend,
    "More accurate isn't worth more": slide_profit,
    "When mistakes cost more": slide_costs,
    "What to look at": slide_takeaways,
    "Next": slide_next,
}


def build(png_dir: Path | None = None) -> Path:
    F.set_theme(False)
    res = json.loads(RESULTS.read_text("utf-8"))
    slides = C.parse(SLIDES_MD.read_text(encoding="utf-8"))
    total = len(slides)
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF, metadata={"Title": TITLE}) as pdf:
        for n in sorted(slides):
            fig = C.new_slide()
            MAKERS[slides[n]["title"]](fig, slides[n], n, total, res)
            pdf.savefig(fig)
            if png_dir:
                fig.savefig(png_dir / f"slide_{n:02d}.png", dpi=C.DPI)
            F.plt.close(fig)
    return OUT_PDF


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--png", type=Path, help="also write one PNG per slide here")
    print(f"Saved -> {build(ap.parse_args().png)}")


if __name__ == "__main__":
    main()
