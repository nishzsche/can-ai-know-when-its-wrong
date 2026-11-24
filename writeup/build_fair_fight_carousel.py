"""Build post 3's LinkedIn carousel PDF from the slide text in writeup/fair_fight_linkedin.md.

    uv run python writeup/build_fair_fight_carousel.py            # writeup/fair_fight_carousel.pdf
    uv run python writeup/build_fair_fight_carousel.py --png DIR  # also one PNG per slide

Same size, type and page furniture as posts 1 and 2 (writeup/build_carousel.py, whose helpers it
reuses). Charts are redrawn at phone size from the M6 predictions through
analysis/fair_fight_figures.py, so every number matches the blog's charts: the same row order
(FF.MODELS), colours, correction and widened intervals. The paired resampling test behind slide 3
takes a minute or two.
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
import fair_fight_figures as FF  # noqa: E402
import figures as F  # noqa: E402
from build_carousel import PT, M, note, text_block, wrap  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

TITLE = "I gave AI the same data as a bank's model. It caught up."  # LinkedIn's limit: 58 chars
SLIDES_MD = HERE / "fair_fight_linkedin.md"
OUT_PDF = HERE / "fair_fight_carousel.pdf"
FAIR = "Every model learned from the same 20,000 past customers (Jev and Sol: the 20 most similar)."
SOL_NOTE = "† GPT-6.2 Sol recognizes this public dataset and knows its default rate."
GROUPS = {"tabular": "Built for tables: read all 20,000", "fine-tuned": "Trained on the 20,000",
          "20 examples": "Shown its 20 most similar"}  # fmt: skip
C.SERIES = TITLE
LABEL_BOX = {"boxstyle": "square,pad=0.15", "fc": None, "ec": "none"}


def col(c: str) -> str:
    return F.COLOR[c] if c in F.BASELINES else (F.MUTED if c == "neighbours-20" else
                                                FF.colour(c, False))  # fmt: skip


def label_box() -> dict:
    return {**LABEL_BOX, "fc": F.SURFACE}


def model_rows(ax, labels: list[str], size: float = 13) -> None:
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=size, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)


def dots(open_label: str, filled_label: str) -> list:
    return [
        F.plt.Line2D([], [], marker="o", ls="none", ms=10, color=F.INK_2, mfc=F.SURFACE, mew=2,
                     label=open_label),
        F.plt.Line2D([], [], marker="o", ls="none", ms=10, color=F.INK_2, label=filled_label),
    ]  # fmt: skip


def refs(neighbours: bool = True, gbm: str = "Gradient boosting") -> list:
    out = [F.plt.Line2D([], [], color=F.COLOR["gbm"], lw=1.6, label=gbm),
           F.plt.Line2D([], [], color=F.COLOR["scorecard"], lw=1.6, ls=(0, (4, 3)),
                        label="Scorecard")]  # fmt: skip
    if neighbours:
        out.append(F.plt.Line2D([], [], color=F.MUTED, lw=1.4, ls=(0, (1, 2)),
                                label="20 look-alikes, averaged"))  # fmt: skip
    return out


def legend_rows(fig, y: float, *rows: list) -> float:
    """One legend per row of handles, from y down; return the y below them."""
    for handles in rows:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M - 0.012, y),
                   ncol=len(handles), fontsize=12, frameon=False, handletextpad=0.35,
                   columnspacing=1.1, handlelength=1.8, borderaxespad=0)  # fmt: skip
        y -= 0.045
    return y


def slide_hook(fig, s, n, total):
    lines = wrap(fig, s["headline"], 38, fontweight="bold")
    top = 0.80
    fig.text(M, top, "\n".join(lines), fontsize=38, fontweight="bold", color=F.INK, va="top",
             linespacing=1.12)  # fmt: skip
    y = top - len(lines) * 38 * 1.12 * PT - 0.07
    fig.text(M, y, "\n".join(wrap(fig, s["body"], 23)), fontsize=23, color=F.INK_2, va="top",
             linespacing=1.25)  # fmt: skip
    fig.text(M, 0.15, s["footer"], fontsize=15, color=F.INK_2)
    fig.text(1 - M, 0.035, "Swipe →", fontsize=12, color=F.INK_2, ha="right")
    fig.text(M, 0.035, f"{n} / {total}", fontsize=10, color=F.MUTED)


def slide_catch_up(fig, s, n, total):
    """Ranking cold (open) and after learning from the 20,000 (filled), grouped by how."""
    y = C.frame(fig, n, total, s)
    y = legend_rows(fig, y, dots("Answering cold (my first test)", "After the 20,000"), refs())
    ax = fig.add_axes([M + 0.25, 0.16, 1 - 2 * M - 0.25, y - 0.18])
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    ticks, labels, row, last = [], [], 0, None
    for c, zs, name, how in FF.MODELS:
        if how != last:  # a group heading takes a row of its own
            ax.text(M, row, GROUPS[how], transform=trans, fontsize=12, fontweight="bold",
                    color=F.INK_2, va="center")  # fmt: skip
            row, last = row + 1, how
        tv = FF.metric(c, "trained")[0]
        if zs:
            zv = FF.metric(zs, "out_of_box")[0]
            ax.plot([zv, tv], [row, row], color=col(c), lw=3, alpha=0.35, zorder=2)
            ax.plot(zv, row, "o", ms=12, color=F.SURFACE, mec=col(c), mew=2.2, zorder=3)
        ax.plot(tv, row, "o", ms=12, color=col(c), zorder=4)
        ax.text(tv + 0.006, row, f"{tv:.3f}", va="center", fontsize=12, color=F.INK, zorder=5,
                bbox=label_box())  # fmt: skip
        ticks.append(row)
        labels.append(name)
        row += 1
    ax.axvline(FF.metric("gbm", "trained")[0], color=F.COLOR["gbm"], lw=1.4, zorder=1)
    ax.axvline(FF.metric("scorecard", "trained")[0], color=F.COLOR["scorecard"], lw=1.4,
               ls=(0, (4, 3)), zorder=1)  # fmt: skip
    ax.axvline(FF.metric("neighbours-20", "trained")[0], color=F.MUTED, lw=1.2, ls=(0, (1, 2)),
               zorder=1)  # fmt: skip
    ax.set_yticks(ticks, labels)
    ax.set_ylim(row - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.57, 0.835)
    ax.set_xticks([0.6, 0.65, 0.7, 0.75, 0.8])
    ax.set_xlabel("Ranking (AUC)", fontsize=12)
    note(fig, FAIR)


def slide_how_close(fig, s, n, total):
    """Paired AUC gaps to gradient boosting and the scorecard, intervals widened for eight."""
    y = C.frame(fig, n, total, s)
    left, gap_, bottom = M + 0.2, 0.035, 0.26
    width = (1 - M - left - gap_) / 2
    names = [name for _, _, name, _ in FF.MODELS]
    for k, (ref, title, ls) in enumerate((("gbm", "vs gradient boosting", "-"),
                                          ("scorecard", "vs the scorecard",
                                           (0, (4, 3))))):  # fmt: skip
        ax = fig.add_axes([left + k * (width + gap_), bottom, width, y - 0.07 - bottom])
        ax.axvline(0, color=F.COLOR[ref], lw=1.6, ls=ls, zorder=1)
        for i, (c, _zs, _name, _how) in enumerate(FF.MODELS):
            est, lo, hi = FF.gap(c, ref)
            ax.plot([lo, hi], [i, i], color=col(c), lw=4, alpha=0.45, solid_capstyle="round",
                    zorder=2)  # fmt: skip
            ax.plot(est, i, "o", ms=11, color=col(c), mec=F.SURFACE, mew=1.5, zorder=3)
        model_rows(ax, names)
        ax.set_ylim(len(names) - 0.05, -0.6)
        if k:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(-0.088, 0.048)
        ax.set_xticks([-0.06, -0.03, 0, 0.03])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
        ax.set_title(title, loc="left", fontsize=13.5, color=F.INK, pad=8)
        for x, ha, text in ((-0.004, "right", "worse"), (0.004, "left", "better")):
            ax.text(x, len(names) - 0.4, text, ha=ha, va="center", fontsize=11, color=F.INK_2)
    fig.text(left + width + gap_ / 2, bottom - 0.065, "Difference in ranking (AUC)", ha="center",
             fontsize=12, color=F.INK)  # fmt: skip
    text_block(fig, 0.155, "Dot: the difference. Bar: the range a resampling test can't rule out, "
               "widened for comparing eight models. A bar that crosses the line can't be told "
               "apart from it.", 11.5, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_honesty(fig, s, n, total):
    """Calibration error raw (open) and after the same correction (filled), with chance shaded."""
    y = C.frame(fig, n, total, s)
    y = legend_rows(fig, y, dots("Raw output", "After the same correction"),
                    refs(neighbours=False, gbm="Gradient boosting, corrected"))  # fmt: skip
    rows = sorted(FF.MODELS, key=lambda r: FF.metric(r[0], "trained_calibrated")[1])
    ax = fig.add_axes([M + 0.235, 0.21, 1 - 2 * M - 0.235 - 0.085, y - 0.24])
    ax.axvspan(0, FF.CHANCE, color=F.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(FF.CHANCE + 0.002, -0.85, "Shaded: within chance for\na perfectly honest model",
            fontsize=11, color=F.INK_2, va="center", linespacing=1.15)  # fmt: skip
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(1 - M, -1.0, "Corrected", transform=trans, ha="right", va="center", fontsize=11,
            color=F.MUTED)  # fmt: skip
    for i, (c, _zs, _name, _how) in enumerate(rows):
        raw, cal = FF.metric(c, "trained")[1], FF.metric(c, "trained_calibrated")[1]
        ax.plot([raw, cal], [i, i], color=col(c), lw=3, alpha=0.35, zorder=2)
        ax.plot(raw, i, "o", ms=12, color=F.SURFACE, mec=col(c), mew=2.2, zorder=3)
        ax.plot(cal, i, "o", ms=12, color=col(c), zorder=4)
        ax.text(1 - M, i, f"{cal * 100:.1f}", transform=trans, ha="right", va="center",
                fontsize=13, color=F.INK,
                fontweight="bold" if cal > FF.CHANCE else "normal")  # fmt: skip
    ax.axvline(FF.metric("gbm", "trained_calibrated")[1], color=F.COLOR["gbm"], lw=1.4, zorder=1)
    ax.axvline(FF.metric("scorecard", "trained")[1], color=F.COLOR["scorecard"], lw=1.4,
               ls=(0, (4, 3)), zorder=1)  # fmt: skip
    labels = [name + ("†" if c == "sol-examples" else "") for c, _zs, name, _how in rows]
    model_rows(ax, labels)
    ax.set_ylim(len(rows) - 0.4, -1.35)
    ax.set_xlim(0, 0.115)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("Points off, on average: lower is more honest", fontsize=12)
    fig.text(M, 0.105, SOL_NOTE, fontsize=11, color=F.INK_2)
    note(fig, FAIR)


def slide_kev(fig, s, n, total):
    """Kev-0.8B and Kev-4B promised vs actual, straight out of training and corrected."""
    y = C.frame(fig, n, total, s)
    y = legend_rows(fig, y, dots("Straight out of training", "After the same correction"))
    left, gap_ = M + 0.075, 0.08
    size = min((1 - M - left - gap_) / 2, y - 0.11 - 0.19)
    for k, (c, name) in enumerate((("kev-0.8b-ft", "Kev-0.8B"), ("kev-4b-ft", "Kev-4B"))):
        ax = fig.add_axes([left + k * (size + gap_), 0.19, size, size])
        rx, ry, rs = FF.deciles(c, "trained")
        cx, cy, cs = FF.deciles(c, "trained_calibrated")
        ax.plot([0, 0.8], [0, 0.8], color=F.AXIS, lw=1.2, zorder=1)
        ax.plot(rx, ry, color=col(c), lw=1.6, alpha=0.5, ls=(0, (3, 2)), zorder=2)
        ax.plot(rx, ry, "o", ms=9, color=F.SURFACE, mec=col(c), mew=2, zorder=3)
        ax.plot(cx, cy, color=col(c), lw=2.2, zorder=4)
        ax.plot(cx, cy, "o", ms=9, color=col(c), mec=F.SURFACE, mew=1.2, zorder=5)
        ax.set_title(name, loc="left", fontsize=14, color=F.INK, pad=30)
        ax.text(0, 1.035, f"{rs['ece'] * 100:.1f} points off → {cs['ece'] * 100:.1f} corrected",
                transform=ax.transAxes, fontsize=12, color=F.INK_2)  # fmt: skip
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Promised", fontsize=12)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Actually defaulted", fontsize=12)
            ax.annotate(f"Promised {rx[5]:.0%},\n{ry[5]:.0%} defaulted", (rx[5], ry[5]),
                        xytext=(0.03, 0.5), fontsize=12, color=F.INK, ha="left",
                        arrowprops={"arrowstyle": "-", "color": F.INK_2, "lw": 1})  # fmt: skip
            ax.text(0.66, 0.6, "honest", rotation=45, rotation_mode="anchor", ha="right",
                    va="top", fontsize=11, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_cost(fig, s, n, total):
    """Running cost per 1,000 decisions (log scale), cheapest first, with each model's ranking."""
    y = C.frame(fig, n, total, s)
    points = [(c, name) for c, _zs, name, _how in FF.MODELS] + [
        ("gbm", "Gradient boosting"), ("scorecard", "Scorecard"),
        ("neighbours-20", "20 look-alikes, averaged")]  # fmt: skip
    rows = sorted(points, key=lambda p: (FF.cost_per_1000(p[0]), -FF.metric(p[0], "trained")[0]))
    ax = fig.add_axes([M + 0.32, 0.25, 1 - 2 * M - 0.32 - 0.1, y - 0.28])
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.5)
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(1 - M, -0.95, "Ranking", transform=trans, ha="right", va="center", fontsize=11,
            color=F.MUTED)  # fmt: skip
    for i, (c, _name) in enumerate(rows):
        x = FF.cost_per_1000(c)
        ax.plot([0, x], [i, i], color=col(c), lw=2, alpha=0.3, zorder=2)
        ax.plot(x, i, "o", ms=12, color=col(c), mec=F.SURFACE, mew=1.5, zorder=3)
        ax.text(1 - M, i, f"{FF.metric(c, 'trained')[0]:.3f}", transform=trans, ha="right",
                va="center", fontsize=13, color=F.INK)  # fmt: skip
    model_rows(ax, [name for _, name in rows])
    ax.set_ylim(len(rows) - 0.4, -1.4)
    ax.set_xlim(-0.00002, 30)
    ax.set_xticks([0, 0.001, 0.01, 0.1, 1, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}"))
    ax.set_xlabel("Running cost per 1,000 decisions, US dollars (log scale)", fontsize=12)
    text_block(fig, 0.165, "Open models at rented GPU rates; Sol and Jev at their API prices. "
               "Fine-tuning cost about a dollar of GPU time at most, once per model.", 11,
               color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_sol(fig, s, n, total):
    """Sol's and Jev's answers against how many of their 20 look-alikes defaulted."""
    y = C.frame(fig, n, total, s)
    handles = [
        F.plt.Line2D([], [], marker="o", ls="none", ms=6, color=F.INK_2, alpha=0.5,
                     label="One customer"),
        F.plt.Line2D([], [], marker="o", ms=8, lw=2.2, color=F.INK_2, mec=F.SURFACE,
                     label="Middle answer"),
        F.plt.Line2D([], [], color=F.INK_2, lw=1.4, ls=(0, (4, 3)), label="Just counting"),
    ]  # fmt: skip
    y = legend_rows(fig, y, handles)
    _, p_nb, ids = FF.load("neighbours-20", "trained")
    k = np.rint(p_nb * 20).astype(int)
    left, gap_ = M + 0.085, 0.045
    width = (1 - M - left - gap_) / 2
    rng = np.random.default_rng(0)  # horizontal jitter only, as in the blog's chart
    for j, (c, name) in enumerate((("sol-examples", "GPT-6.2 Sol"), ("jev-examples", "Jev"))):
        ax = fig.add_axes([left + j * (width + gap_), 0.23, width, y - 0.09 - 0.23])
        _, p, ids_c = FF.load(c, "trained")
        assert ids_c == ids
        ax.scatter(k + rng.uniform(-0.3, 0.3, len(k)), p, s=10, color=col(c), alpha=0.22, lw=0,
                   zorder=2)  # fmt: skip
        ks = [v for v in range(21) if (k == v).sum() >= 10]
        med = [float(np.median(p[k == v])) for v in ks]
        ax.plot(ks, med, color=col(c), lw=2.4, zorder=4)
        ax.plot(ks, med, "o", ms=8, color=col(c), mec=F.SURFACE, mew=1.2, zorder=5)
        ax.plot([0, 20], [0, 1], color=F.INK_2, lw=1.4, ls=(0, (4, 3)), zorder=3)
        ax.set_title(f"{name}: ranking {FF.metric(c, 'trained')[0]:.3f}", loc="left",
                     fontsize=13.5, color=F.INK, pad=8)  # fmt: skip
        ax.set_xlim(-0.7, 20.7)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.yaxis.set_major_formatter(F.PCT)
        ax.tick_params(labelsize=11)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Its answer: chance of default", fontsize=12)
    fig.text(left + width + gap_ / 2, 0.16, "How many of its 20 look-alikes defaulted",
             ha="center", fontsize=12, color=F.INK)  # fmt: skip
    fig.text(M, 0.105, SOL_NOTE[2:], fontsize=11, color=F.INK_2)
    note(fig, FAIR)


def slide_ceiling(fig, s, n, total):
    """Every model that learned from the 20,000 on the full scale, plus the 2,000-customer
    scorecard from the first post's checks (results/robustness.json)."""
    y = C.frame(fig, n, total, s)
    small = json.loads((FF.ROOT / "results" / "robustness.json").read_text("utf-8"))
    small = small["scorecard_on_2000"]["auc"]
    rows = sorted(FF.CEILING, key=lambda c: -FF.metric(c, "trained")[0])
    aucs = {c: FF.metric(c, "trained")[0] for c in rows}
    ax = fig.add_axes([M + 0.3, 0.215, 1 - 2 * M - 0.3, y - 0.25])
    lo, hi = min(aucs.values()), max(aucs.values())
    ax.axvspan(lo, hi, color=F.GRID, alpha=0.9, lw=0, zorder=0)
    ax.text((lo + hi) / 2, -0.95, f"all eight: {lo:.3f} to {hi:.3f}", ha="center", va="bottom",
            fontsize=11, color=F.INK_2)  # fmt: skip
    for i, c in enumerate(rows):
        ax.plot(aucs[c], i, "o", ms=12, color=col(c), mec=F.SURFACE, mew=1.5, zorder=3)
        ax.text(hi + 0.014, i, f"{aucs[c]:.3f}", va="center", fontsize=12, color=F.INK)
    ys = len(rows) + 0.6
    ax.plot(small, ys, "o", ms=12, color=F.SURFACE, mec=F.COLOR["scorecard"], mew=2.2, zorder=3)
    ax.text(small - 0.014, ys, f"{small:.3f}", ha="right", va="center", fontsize=12, color=F.INK)
    ax.annotate("", (aucs["scorecard"], len(rows) - 1 + 0.3), (small, ys - 0.3),
                arrowprops={"arrowstyle": "->", "color": F.INK_2, "lw": 1.2})  # fmt: skip
    ax.text(hi + 0.014, ys - 0.55, f"10 times the data:\n+{aucs['scorecard'] - small:.3f}",
            va="center", fontsize=11.5, color=F.INK_2, linespacing=1.15)  # fmt: skip
    ax.text(0.515, 3, "None of the eight\ngets past 0.80", ha="left", va="center",
            fontsize=12.5, color=F.INK_2, linespacing=1.2)  # fmt: skip
    ax.set_yticks([*range(len(rows)), ys], [FF.CEILING_NAME[c] for c in rows]
                  + ["Scorecard on 2,000"])  # fmt: skip
    ax.set_ylim(ys + 0.7, -1.2)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  ["0.5\ncoin flip", "0.6", "0.7", "0.8", "0.9", "1.0\nperfect"])  # fmt: skip
    ax.set_xlabel("Ranking (AUC)", fontsize=12)
    note(fig, "The scorecard on 2,000 learned from a tenth of the data; the rest from the same "
         "20,000.")  # fmt: skip


def slide_bullets(fig, s, n, total, size: float = 20, footer_size: float = 22):
    y = C.frame(fig, n, total, s, headline_size=30) - 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, size, bullet=True, color=F.INK) - 0.025
    return text_block(fig, y - 0.03, s["footer"], footer_size, fontweight="bold", color=F.INK)


def slide_bank(fig, s, n, total):
    slide_bullets(fig, s, n, total, size=19, footer_size=21)


def slide_next(fig, s, n, total):
    slide_bullets(fig, s, n, total, size=18, footer_size=20)
    fig.text(M, 0.08, "Write-up, code and every number: link in the first comment.",
             fontsize=13, color=F.INK_2)  # fmt: skip


MAKERS = {  # by slide title in fair_fight_linkedin.md, so slides can be reordered there
    "Hook": slide_hook,
    "It caught up": slide_catch_up,
    "How close": slide_how_close,
    "Honest after the same correction": slide_honesty,
    "Training is not calibration": slide_kev,
    "Paying more": slide_cost,
    "The closed models can only look": slide_sol,
    "The ceiling": slide_ceiling,
    "What I'd tell a bank": slide_bank,
    "Next": slide_next,
}


def build(png_dir: Path | None = None) -> Path:
    F.set_theme(False)
    slides = C.parse(SLIDES_MD.read_text(encoding="utf-8"))
    total = len(slides)
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF, metadata={"Title": TITLE}) as pdf:
        for n in sorted(slides):
            fig = C.new_slide()
            MAKERS[slides[n]["title"]](fig, slides[n], n, total)
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
