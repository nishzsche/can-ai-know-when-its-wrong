"""Build post 5's LinkedIn carousel PDF from the slide text in writeup/formats_linkedin.md.

    uv run python writeup/build_formats_carousel.py            # formats_carousel.pdf
    uv run python writeup/build_formats_carousel.py --png DIR  # also one PNG per slide

Same size, type and page furniture as posts 1-4 (writeup/build_carousel.py, whose helpers it
reuses). Charts are redrawn at phone size from the same results as the blog's charts, through
analysis/layouts_figures.py: results/layouts.csv and the prediction records. Each model keeps its
colour from the series; the original table is an open dot and the other layouts filled ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_carousel as C  # noqa: E402  (also puts analysis/ on the path)
import figures as F  # noqa: E402
import layouts as L  # noqa: E402
import layouts_figures as LF  # noqa: E402
from build_carousel import PT, M, note, text_block, wrap  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TITLE = "Why AI can write code but struggle to read a table"  # LinkedIn's limit: 58 characters
SLIDES_MD = HERE / "formats_linkedin.md"
OUT_PDF = HERE / "formats_carousel.pdf"
FAIR = "All six AI models answered cold; the scorecard learned from 20,000 past customers."
C.SERIES = TITLE

ORDER, NAME, LAYOUT = LF.ORDER, LF.NAME, LF.LAYOUT
OTHERS = tuple(L.LAYOUTS)  # the five other layouts, in the blog's order


def rows_axis(ax, labelsize: float = 13.5) -> None:
    ax.set_yticks(range(len(ORDER)), [NAME[c] for c in ORDER])
    ax.set_ylim(len(ORDER) - 0.45, -1.4)
    ax.tick_params(axis="y", length=0, labelsize=labelsize, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", visible=False)


def strip(ax, key: str, fmt, label_rows: dict, faded=lambda c, lay: False) -> None:
    """The blog's strip at phone size: a line over each model's range, filled dots for the five
    other layouts, an open dot for the original table, and the named dots labelled."""
    for i, c in enumerate(ORDER):
        col = F.COLOR[c]
        vals = {lay: LF.value(c, lay, key) for lay in LAYOUT}
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=LF.tint(col, 0.45),
                lw=2.6, zorder=2, solid_capstyle="round")  # fmt: skip
        for lay in sorted(OTHERS, key=lambda k: not faded(c, k)):  # full colour on top
            ax.plot(vals[lay], i, "o", ms=11, color=LF.tint(col, 0.35) if faded(c, lay) else col,
                    mec=F.SURFACE, mew=1.6, zorder=3)  # fmt: skip
        ax.plot(vals["v2"], i, "o", ms=12.5, mfc=F.SURFACE, mec=col, mew=2.6, zorder=4)
        for lay, side in label_rows.get(c, ()):
            ax.annotate(f"{LAYOUT[lay]} {fmt(vals[lay])}", (vals[lay], i),
                        xytext=(0, 11 if side == "above" else -11), textcoords="offset points",
                        ha="center", va="bottom" if side == "above" else "top", fontsize=12,
                        color=F.INK_2)  # fmt: skip


def key_row(fig, y: float, faded: bool = False) -> float:
    """Open dot = the original table, filled = the other layouts (and faded = within chance)."""
    ink = F.INK_2
    handles = [
        F.plt.Line2D([], [], marker="o", ls="none", ms=11, mfc=F.SURFACE, mec=ink, mew=2.4,
                     label="Original table"),
        F.plt.Line2D([], [], marker="o", ls="none", ms=10, color=ink, label="Other layouts"),
    ]  # fmt: skip
    if faded:
        handles.append(F.plt.Line2D([], [], marker="o", ls="none", ms=10,
                                    color=LF.tint(ink, 0.35), label="Within chance"))  # fmt: skip
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M - 0.012, y),
               ncol=len(handles), fontsize=12.5, frameon=False, handletextpad=0.35,
               columnspacing=1.3, borderaxespad=0)  # fmt: skip
    return y - 0.05


# ---- slides --------------------------------------------------------------------------------------


def slide_hook(fig, s, n, total):
    lines = wrap(fig, s["headline"], 40, fontweight="bold")
    top = 0.80
    fig.text(M, top, "\n".join(lines), fontsize=40, fontweight="bold", color=F.INK, va="top",
             linespacing=1.12)  # fmt: skip
    y = top - len(lines) * 40 * 1.12 * PT - 0.07
    fig.text(M, y, "\n".join(wrap(fig, s["body"], 23)), fontsize=23, color=F.INK_2, va="top",
             linespacing=1.25)  # fmt: skip
    fig.text(M, 0.15, s["footer"], fontsize=15, color=F.INK_2)
    fig.text(1 - M, 0.035, "Swipe →", fontsize=12, color=F.INK_2, ha="right")
    fig.text(M, 0.035, f"{n} / {total}", fontsize=10, color=F.MUTED)


EXAMPLE = (  # the September row of the test customer the blog post shows, in four layouts
    ("Original table", ["Sep 2005 | NT$46,990 | NT$2,000 | not late (code 0)"]),
    ("CSV", ['Sep 2005,"NT$46,990","NT$2,000",not late (code 0)']),
    ("JSON", ['{"month": "Sep 2005", "bill_amount": "NT$46,990",',
              ' "amount_paid": "NT$2,000",',
              ' "repayment_status": "not late (code 0)"}']),
    ("Sentences", ["In Sep 2005, the bill amount was NT$46,990,",
                   "the amount paid was NT$2,000, and the",
                   "repayment status was not late (code 0)."]),
)  # fmt: skip


def slide_layouts(fig, s, n, total):
    y = C.frame(fig, n, total, s) - 0.02
    for label, lines in EXAMPLE:
        fig.text(M, y, label, fontsize=13.5, fontweight="bold", color=F.INK, va="top")
        y -= 0.037
        block = "\n".join(line.replace("$", r"\$") for line in lines)  # $ pairs would be maths
        fig.text(M + 0.02, y, block, family="monospace", fontsize=12.5, color=F.INK, va="top",
                 linespacing=1.25)  # fmt: skip
        y -= len(lines) * 12.5 * 1.25 * PT + 0.026
    assert y > 0.2, f"slide 2 overflows: {y:.3f}"
    text_block(fig, y - 0.005, "Also tested: a Markdown table, and the table with each bill "
               "beside the payment made the next month.", 12.5, color=F.INK_2)  # fmt: skip
    note(fig, "The models: Jev, Kev-4B, Kev-0.8B, Laya, Qwen3.5-4B and GPT-6.2 Sol, all answering "
         "cold.")  # fmt: skip


RANK_LABELS = {"laya": (("json", "above"), ("prose", "above")),
               "kev-4b": (("v2", "above"), ("json", "above"))}  # fmt: skip


def slide_ranking(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    y = key_row(fig, y - 0.005, faded=True)
    ax = fig.add_axes([M + 0.17, 0.235, 1 - 2 * M - 0.17, y - 0.285])
    sc = LF.scorecard_auc()
    ax.axvline(0.5, color=F.MUTED, lw=1.3, ls=(0, (1, 2)), zorder=1)
    ax.text(0.504, -1.13, "Coin flip", fontsize=12, color=F.INK_2, va="center")
    ax.axvline(sc, color=F.COLOR["scorecard"], lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(sc - 0.005, -1.13, f"Scorecard {sc:.3f}", fontsize=12, color=F.INK_2, ha="right",
            va="center")  # fmt: skip
    strip(ax, "auc", lambda v: f"{v:.3f}", RANK_LABELS,
          faded=lambda c, lay: LF.value(c, lay, "auc_vs_v2_p_holm") >= 0.05)  # fmt: skip
    rows_axis(ax)
    ax.set_xlim(0.44, 0.8)
    ax.set_xlabel("Ranking (AUC)", fontsize=12.5)
    text_block(fig, 0.145, "Faded: within chance of the original table, by a paired test.", 12,
               color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


MOVED_LABELS = {"laya": ("csv", "json"), "kev-4b": ("json",), "kev-0.8b": ("csv",)}


def slide_moved(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.17, 0.25, 1 - 2 * M - 0.17, y - 0.3])
    ax.axvline(LF.SHRINK_3BIT, color=F.MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(LF.SHRINK_3BIT + 0.004, -1.13, "Last post: shrinking to 3 bits", fontsize=12,
            color=F.INK_2, va="center")  # fmt: skip
    for i, c in enumerate(ORDER):
        col = F.COLOR[c]
        noise = LF.table()[c, "json"]["noise_mean_abs_change"]
        if noise > 0.001:
            ax.barh(i, noise, height=0.46, color=F.GRID, zorder=1)
        vals = {lay: LF.value(c, lay, "mean_abs_change") for lay in OTHERS}
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=LF.tint(col, 0.45),
                lw=2.6, zorder=2, solid_capstyle="round")  # fmt: skip
        for v in vals.values():
            ax.plot(v, i, "o", ms=11, color=col, mec=F.SURFACE, mew=1.6, zorder=3)
        hi = max(vals.values())
        if c in MOVED_LABELS:
            keys = MOVED_LABELS[c]
            x = max(vals[k] for k in keys)
            pts = f"{100 * x:.0f}" if x >= 0.1 else f"{100 * x:.1f}"
            ax.annotate(f"{' and '.join(LAYOUT[k] for k in keys)} {pts}", (x, i),
                        xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, color=F.INK_2)  # fmt: skip
        else:
            lo = min(vals.values())
            ax.annotate(f"{100 * lo:.1f} to {100 * hi:.1f}", (hi, i), xytext=(12, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=12,
                        color=F.INK_2, zorder=5,
                        bbox={"fc": F.SURFACE, "ec": "none", "pad": 1.5})  # fmt: skip
    rows_axis(ax)
    ax.set_xlim(0, 0.29)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax.set_xlabel("Average move per customer, in percentage points", fontsize=12.5)
    text_block(fig, 0.16, "Each dot is one layout. Grey bar: how far the answers move when the "
               "original table is simply asked again. Sol moved hardly more than that.", 12,
               color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


PANELS = (("laya", "json"), ("frontier-verbal", "json"))


def slide_customers(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    left, gap_, bottom = M + 0.075, 0.06, 0.24
    side = min((1 - M - left - gap_) / 2, y - 0.2 - bottom)
    for k, (c, lay) in enumerate(PANELS):
        ax = fig.add_axes([left + k * (side + gap_), bottom, side, side])
        ref = L.answers(c, L.PREDICTIONS_DIR)
        directory, suffix, _ = L.LAYOUTS[lay]
        ids = sorted(ref)
        _, p0, p1 = L.arrays(ref, L.answers(f"{c}-{suffix}", directory), ids)
        ax.scatter(p0, p1, s=10, color=F.COLOR[c], alpha=0.3, lw=0, zorder=2)
        ax.plot([0, 1], [0, 1], color=F.INK_2, lw=1.3, ls=(0, (4, 3)), zorder=3)
        r = LF.table()[c, lay]
        ax.set_title(f"{NAME[c]}, {LAYOUT[lay]}", loc="left", fontsize=14, color=F.INK, pad=62)
        ax.text(0, 1.025, f"Average move: {100 * r['mean_abs_change']:.1f} pts\n"
                f"5+ pts: {LF.share_text(r['moved_5pts_share'])}\n"
                f"Ranking {r['auc_v2']:.3f} → {r['auc']:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=12, color=F.INK_2,
                linespacing=1.3)  # fmt: skip
        for setter in (ax.set_xlim, ax.set_ylim):
            setter(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Original table", fontsize=12.5)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("JSON", fontsize=12.5)
    laya = LF.table()["laya", "json"]
    text_block(fig, 0.155, f"Dashed line: no change. Laya's two sets of answers barely relate "
               f"(rank correlation {laya['spearman']:.2f}). Sol knows this dataset.", 12,
               color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


PROMISED_LABELS = {"laya": (("prose", "above"), ("csv", "above")),
                   "kev-4b": (("json", "above"), ("prose", "above"))}  # fmt: skip


def slide_promised(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    y = key_row(fig, y - 0.005)
    ax = fig.add_axes([M + 0.17, 0.225, 1 - 2 * M - 0.17, y - 0.275])
    rate = LF.table()["laya", "json"]["observed_rate"]
    ax.axvline(rate, color=F.INK_2, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(rate + 0.007, -1.13, f"Defaulted: {100 * rate:.1f}%", fontsize=12,
            color=F.INK_2, va="center")  # fmt: skip
    strip(ax, "mean", lambda v: f"{100 * v:.0f}%", PROMISED_LABELS)
    rows_axis(ax)
    ax.set_xlim(0.08, 0.6)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.set_xlabel("Average risk the model promised", fontsize=12.5)
    text_block(fig, 0.14, "Raw answers, before any correction. Sol stayed within a point of "
               "26%.", 12, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


DECISION_COLS = (("again", "Asked\nagain"), ("markdown", "Mark-\ndown"), ("csv", "CSV"),
                 ("json", "JSON"), ("prose", "Sen-\ntences"),
                 ("aligned", "Re-\naligned"))  # fmt: skip


def pct(v: float) -> str:
    """Rounded exactly as the post's table was: plain formatting, so 197 of 2,000 shows as 9.8%."""
    if v == 0:
        return "0%"
    return f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"


def slide_decisions(fig, s, n, total):
    """The blog's decisions table as a shaded grid: darker, more decisions changed."""
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.2, 0.2, 1 - 2 * M - 0.2, y - 0.27])
    hi = max(LF.value(c, lay, "decisions_changed") for c in ORDER for lay in OTHERS)
    for i, c in enumerate(ORDER):
        for j, (lay, _label) in enumerate(DECISION_COLS):
            v = (LF.table()[c, "json"]["noise_decisions_changed"] if lay == "again"
                 else LF.value(c, lay, "decisions_changed"))  # fmt: skip
            share = v / hi
            fill = F.GRID if lay == "again" else LF.tint(F.INK_2, 0.05 + 0.4 * share)
            ax.add_patch(F.plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, fc=fill, ec="none"))
            ax.text(j, i, pct(v), ha="center", va="center", fontsize=14, color=F.INK,
                    fontweight="bold" if v >= 0.2 else "normal")  # fmt: skip
    ax.set_xlim(-0.5, len(DECISION_COLS) - 0.5)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_yticks(range(len(ORDER)), [NAME[c] for c in ORDER])
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(DECISION_COLS)), [label for _l, label in DECISION_COLS])
    ax.tick_params(axis="both", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.grid(visible=False)
    for side in ax.spines.values():
        side.set_visible(False)
    laya_json = LF.value("laya", "json", "approve")
    text_block(fig, 0.155, f"Asked again: the original table, a second time (200 customers). With "
               f"JSON, Laya's lender approves {100 * laya_json:.0f}% of customers, against "
               f"{100 * LF.value('laya', 'v2', 'approve'):.0f}%.", 12, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def chips(ax, y: float, texts: list[str], fill: str, width: float = 100.0) -> float:
    """Each piece as a box, wrapping at `width` (the blog's chips at phone size)."""
    char, pad, gap, line = 1.95, 1.4, 1.0, 9.5
    x = 0.0
    for t in texts:
        shown = t.strip() or "␣"
        w = len(shown) * char + 2 * pad
        if x + w > width:
            x, y = 0.0, y - line
        ax.add_patch(F.plt.Rectangle((x, y - 3.4), w, 6.8, fc=fill, ec=F.SURFACE, lw=1.6))
        ax.text(x + w / 2, y, shown, ha="center", va="center", family="monospace", fontsize=14,
                color=F.INK)  # fmt: skip
        x += w + gap
    return y - line


def slide_pieces(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M, 0.13, 1 - 2 * M, y - 0.15])
    ax.set_xlim(0, 100)
    ax.set_ylim(-88, 4)
    ax.axis("off")
    kev, laya = LF.pieces("kev (Qwen3.5)"), LF.pieces("laya (ModernBERT)")
    ax.text(0, 1, "A program: four values", fontsize=14, fontweight="bold", color=F.INK, va="top")
    x, yy = 0.0, -10.5
    for text, what in LF.PROGRAM:
        w = len(text) * 1.95 + 5
        if x + w > 100:
            x, yy = 0.0, yy - 15
        ax.add_patch(F.plt.Rectangle((x, yy - 3.4), w, 6.8, fc=F.GRID, ec=F.SURFACE, lw=1.6))
        ax.text(x + w / 2, yy, text, ha="center", va="center", family="monospace", fontsize=14,
                color=F.INK)  # fmt: skip
        ax.text(x + w / 2, yy - 4.6, what, ha="center", va="top", fontsize=12, color=F.INK_2)
        x += w + 3
    ax.text(0, yy - 15, f"Kev and Qwen: {len(kev)} pieces", fontsize=14, fontweight="bold",
            color=F.INK, va="top")  # fmt: skip
    yy = chips(ax, yy - 25, kev, LF.tint(F.COLOR["kev-4b"], 0.3))
    ax.text(0, yy + 1.5, f"Laya: {len(laya)} pieces", fontsize=14, fontweight="bold",
            color=F.INK, va="top")  # fmt: skip
    chips(ax, yy - 8.5, laya, LF.tint(F.COLOR["laya"], 0.3))
    note(fig, "Pieces from each model's own tokenizer; Kev and Qwen share one. ␣: a piece that "
         "is only a space.")  # fmt: skip


def slide_bullets(fig, s, n, total, size: float = 16.5, footer_size: float = 19):
    y = C.frame(fig, n, total, s, headline_size=30) - 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, size, bullet=True, color=F.INK) - 0.02
    return text_block(fig, y - 0.025, s["footer"], footer_size, fontweight="bold", color=F.INK)


def slide_next(fig, s, n, total):
    slide_bullets(fig, s, n, total)
    fig.text(M, 0.08, "Write-up, code and every number: link in the first comment.",
             fontsize=13, color=F.INK_2)  # fmt: skip


MAKERS = {  # by slide title in formats_linkedin.md, so slides can be reordered there
    "Hook": slide_hook,
    "Six layouts": slide_layouts,
    "Ranking": slide_ranking,
    "Answers moved": slide_moved,
    "Customers": slide_customers,
    "Promised": slide_promised,
    "Lending decisions": slide_decisions,
    "Why": slide_pieces,
    "What I'd tell a bank": slide_bullets,
    "Fine print": slide_next,
}


def build(png_dir: Path | None = None) -> Path:
    F.set_theme(False)
    slides = C.parse(SLIDES_MD.read_text(encoding="utf-8"))
    total = len(slides)
    assert len(TITLE) <= 58 and slides[1]["headline"] == TITLE, "title and hook must match"
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
