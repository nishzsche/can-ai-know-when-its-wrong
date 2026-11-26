"""Build post 4's LinkedIn carousel PDF from the slide text in writeup/quantization_linkedin.md.

    uv run python writeup/build_quantization_carousel.py            # quantization_carousel.pdf
    uv run python writeup/build_quantization_carousel.py --png DIR  # also one PNG per slide

Same size, type and page furniture as posts 1-3 (writeup/build_carousel.py, whose helpers it
reuses). Charts are redrawn at phone size from the same results as the blog's charts, through
analysis/quantization_figures.py: results/quantization.csv, the prediction records and the speed
addendum's results/speed_throughput.csv. Each model keeps its colour from the series.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_carousel as C  # noqa: E402  (also puts analysis/ on the path)
import figures as F  # noqa: E402
import quantization as Q  # noqa: E402
import quantization_figures as QF  # noqa: E402
from build_carousel import PT, M, note, text_block, wrap  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

TITLE = "I shrank AI models. It saved memory, not time."  # LinkedIn's limit: 58 characters
SLIDES_MD = HERE / "quantization_linkedin.md"
OUT_PDF = HERE / "quantization_carousel.pdf"
FAIR = "All three models were trained on the same 20,000 past customers as the scorecard."
TICKS = {"bf16": "Full\nsize", "bnb-int8": "8-bit", "bnb-nf4": "4-bit\nNF4", "hqq-4": "4-bit\nHQQ",
         "hqq-3": "3-bit\nHQQ", "hqq-2": "2-bit\nHQQ"}  # fmt: skip
C.SERIES = TITLE


def name_of(model: str) -> str:
    return next(n for m_, _k, n in QF.MODELS if m_ == model)


def colour(model: str) -> str:
    return F.COLOR[next(k for m_, k, _n in QF.MODELS if m_ == model)]


def model_handles() -> list:
    return [F.plt.Line2D([], [], marker="o", ls="none", ms=10, color=F.COLOR[key], label=name)
            for _m, key, name in QF.MODELS]  # fmt: skip


def legend_row(fig, y: float, handles: list) -> float:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M - 0.012, y),
               ncol=len(handles), fontsize=12, frameon=False, handletextpad=0.35,
               columnspacing=1.1, handlelength=1.8, borderaxespad=0)  # fmt: skip
    return y - 0.045


def size_axis(ax) -> None:
    ax.set_xticks(range(len(QF.SIZES)), [TICKS[s] for s in QF.SIZES])
    ax.tick_params(axis="x", length=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlim(-0.3, len(QF.SIZES) - 0.7)
    ax.grid(axis="x", visible=False)


@cache
def same_model_twice(model: str) -> float:
    """Decisions that differ between two full-size runs (post 3's against post 4's bf16), at the
    first run's tune cut-off: the floor for the decisions slide, as in the blog's table."""
    _, y, p_ref = Q.split_arrays(model, "trained", "test")
    _, y_tu, p_tu_ref = Q.split_arrays(model, "trained", "tune")
    _, _, p = Q.split_arrays(f"{model}-bf16", "trained", "test")
    _, y_tu2, p_tu = Q.split_arrays(f"{model}-bf16", "trained", "tune")
    d = Q.decisions(y, p, p_ref, Q.tune_cutoff(y_tu2, p_tu, Q.LOSS),
                    Q.tune_cutoff(y_tu, p_tu_ref, Q.LOSS))  # fmt: skip
    return d["decisions_changed_bf16_cutoff"]


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


SHRINK_ROWS = (
    ("bf16", "16 bits", "about 65,000"),
    ("bnb-int8", "8 bits", "256"),
    ("hqq-4", "4 bits", "16"),
    ("hqq-3", "3 bits", "8"),
    ("hqq-2", "2 bits", "4"),
)


def slide_shrink(fig, s, n, total):
    """Values per number and memory for the two 4B models' numbers, at each size."""
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.3, 0.27, 1 - 2 * M - 0.3 - 0.17, y - 0.37])
    fig.text(M, y - 0.06, "Each number", fontsize=12, fontweight="bold", color=F.INK_2)
    fig.text(M + 0.3, y - 0.06, "Memory for a 4B model's numbers", fontsize=12,
             fontweight="bold", color=F.INK_2)  # fmt: skip
    for i, (size, bits, values) in enumerate(SHRINK_ROWS):
        sizes = (size, "bnb-nf4") if size == "hqq-4" else (size,)  # 4 bits: either method
        gb = [QF.GB * QF.value(m_, s_, "gpu_weights_gb") for m_ in ("kev-4b-ft", "qwen-ft")
              for s_ in sizes]  # fmt: skip
        lo, hi = min(gb), max(gb)
        ax.barh(i, (lo + hi) / 2, height=0.56, color=F.INK_2, alpha=0.75, zorder=2)
        label = f"{lo:.1f} GB" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}–{hi:.1f} GB"
        ax.text(hi + 0.15, i, label, va="center", fontsize=14, color=F.INK)
        ax.text(-0.03, i, f"{bits}\n{values} values", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13.5, color=F.INK, linespacing=1.15)  # fmt: skip
    ax.set_yticks([])
    ax.set_ylim(len(SHRINK_ROWS) - 0.4, -0.6)
    ax.set_xlim(0, 9.5)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)
    caption = ("Each small block of numbers keeps its own scale, and some parts stay full size, "
               "so 4 bits saves less than three-quarters.")  # fmt: skip
    text_block(fig, 0.2, caption, 12, color=F.INK_2)
    note(fig, FAIR)


def slide_ranking(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.085, 0.255, 1 - 2 * M - 0.085 - 0.21, y - 0.31])
    x = np.arange(len(QF.SIZES))
    sc = QF.scorecard_auc()
    ax.axhline(sc, color=F.COLOR["scorecard"], lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.22, sc - 0.009, f"Scorecard {sc:.3f}", fontsize=12, color=F.INK_2, va="top")
    ax.axhline(0.5, color=F.MUTED, lw=1.3, ls=(0, (1, 2)), zorder=1)
    ax.text(-0.22, 0.508, "Coin flip 0.5", fontsize=12, color=F.INK_2, va="bottom")
    dy = {"kev-4b-ft": 0, "kev-0.8b-ft": 9, "qwen-ft": -9}
    for model, key, name in QF.MODELS:
        ys = [QF.value(model, s_, "auc") for s_ in QF.SIZES]
        ax.plot(x, ys, color=F.COLOR[key], lw=2.8, zorder=3)
        ax.plot(x, ys, "o", ms=10, color=F.COLOR[key], mec=F.SURFACE, mew=1.6, zorder=4)
        ax.annotate(f"{name}  {ys[-1]:.2f}", (x[-1], ys[-1]), xytext=(12, dy[model]),
                    textcoords="offset points", va="center", fontsize=13, color=F.INK,
                    annotation_clip=False)  # fmt: skip
    size_axis(ax)
    ax.set_ylim(0.38, 0.86)
    ax.set_ylabel("Ranking (AUC)", fontsize=12)
    text_block(fig, 0.155, "Down to 4 bits, all three stay within 0.005 of full size. At 3 bits "
               "only Kev-4B slips, to 0.776.", 12.5, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_honesty(fig, s, n, total):
    y = C.frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.085, 0.19, 1 - 2 * M - 0.085, y - 0.25])
    x = np.arange(len(QF.SIZES))
    top = 0.1
    ax.axhspan(0, QF.CHANCE, color=F.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.02, 0.003, "Shaded: within chance for a perfectly honest model", fontsize=11.5,
            color=F.INK_2, va="bottom")  # fmt: skip
    dy = {"kev-4b-ft": -11, "kev-0.8b-ft": 14, "qwen-ft": -11}
    off = []
    for model, key, name in QF.MODELS:
        ys = [QF.value(model, s_, "ece") for s_ in QF.SIZES]
        ax.plot(x, ys, color=F.COLOR[key], lw=2.8, zorder=3)
        shown = [(i, v) for i, v in zip(x, ys, strict=True) if v <= top]
        ax.plot([i for i, _ in shown], [v for _, v in shown], "o", ms=10, color=F.COLOR[key],
                mec=F.SURFACE, mew=1.6, zorder=4)  # fmt: skip
        off.append((ys[-1], name))
        ax.annotate(f"{name}  {100 * ys[0]:.1f}", (x[0], ys[0]), xytext=(-8, dy[model]),
                    textcoords="offset points", ha="left",
                    va="bottom" if dy[model] > 0 else "top", fontsize=13, color=F.INK)  # fmt: skip
    for m_, _k, _n in QF.MODELS:
        for s_ in QF.SIZES[1:-1]:
            if QF.value(m_, s_, "ece_vs_bf16_p_holm") < 0.05:
                ax.annotate("bigger than chance", (QF.SIZES.index(s_), QF.value(m_, s_, "ece")),
                            xytext=(0, -14), textcoords="offset points", ha="center", va="top",
                            fontsize=11.5, color=F.INK_2)  # fmt: skip
    off_text = "\n".join(f"{name}: {100 * v:.0f} points off" for v, name in sorted(off))
    ax.text(0.95, top * 0.97, f"Off the scale at 2 bits:\n{off_text}", ha="left", va="top",
            fontsize=12, color=F.INK_2, linespacing=1.3)  # fmt: skip
    size_axis(ax)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(QF.POINTS)
    ax.set_ylabel("Points off, on average: lower is more honest", fontsize=12)
    note(fig, FAIR)


def slide_drift(fig, s, n, total):
    """Kev-0.8B's answer for each test customer, full size against NF4 and against 3 bits."""
    y = C.frame(fig, n, total, s)
    model, key, _name = QF.DRIFT_MODEL
    ids0, _, p0 = Q.split_arrays(f"{model}-bf16", "trained", "test")
    left, gap_, bottom = M + 0.075, 0.06, 0.205
    side = min((1 - M - left - gap_) / 2, y - 0.165 - bottom)
    for k, size in enumerate(("bnb-nf4", "hqq-3")):
        ax = fig.add_axes([left + k * (side + gap_), bottom, side, side])
        ids, _, p = Q.split_arrays(f"{model}-{size}", "trained", "test")
        assert ids == ids0
        ax.scatter(p0, p, s=9, color=F.COLOR[key], alpha=0.3, lw=0, zorder=2)
        ax.plot([0, 1], [0, 1], color=F.INK_2, lw=1.3, ls=(0, (4, 3)), zorder=3)
        ax.set_title(QF.ROW[size], loc="left", fontsize=14, color=F.INK, pad=60)
        moved = 100 * QF.value(model, size, "mean_abs_change_vs_bf16")
        big = QF.share_text(QF.value(model, size, "moved_5pts_share"))
        ax.text(0, 1.025, f"Moved {moved:.1f} pts on average\n5+ pts: {big}\n"
                f"Ranking {QF.value(model, size, 'auc'):.3f}, "
                f"{100 * QF.value(model, size, 'ece'):.1f} pts off",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5, color=F.INK_2,
                linespacing=1.3)  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.tick_params(labelsize=11.5)
        ax.set_xlabel("Full-size answer", fontsize=12)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Shrunk answer", fontsize=12)
    auc0, ece0 = QF.value(model, "bf16", "auc"), 100 * QF.value(model, "bf16", "ece")
    full = f"Full size: ranking {auc0:.3f}, {ece0:.1f} pts off. Dashed line: no change."
    text_block(fig, 0.135, full, 12.5, color=F.INK_2)
    note(fig, FAIR)


DECISION_SIZES = ("bnb-int8", "bnb-nf4", "hqq-4", "hqq-3", "hqq-2")


def slide_decisions(fig, s, n, total):
    """Share of decisions that differ from full size at the full-size cut-off, by size."""
    y = C.frame(fig, n, total, s)
    y = legend_row(fig, y - 0.01, model_handles())
    rows = [("twice", "Same model twice")] + [(s_, QF.ROW[s_].replace(",", ""))
                                              for s_ in DECISION_SIZES]  # fmt: skip
    ax = fig.add_axes([M + 0.235, 0.255, 1 - 2 * M - 0.235 - 0.14, y - 0.3])
    for i, (size, _label) in enumerate(rows):
        vals = []
        for model, key, _name in QF.MODELS:
            v = (same_model_twice(model) if size == "twice"
                 else QF.value(model, size, "decisions_changed_bf16_cutoff"))  # fmt: skip
            vals.append(v)
            yy = i + QF.COST_DY[model] * 1.1
            ax.plot(v, yy, "o", ms=11, color=F.COLOR[key], mec=F.SURFACE, mew=1.4, zorder=3)
        lo, hi = min(vals), max(vals)
        digits = 1 if hi < 0.1 else 0  # counts out of 2,000 round half up, as in the post
        fmt = lambda v, d=digits: f"{100 * v + 1e-9:.{d}f}"  # noqa: E731
        ax.text(1.02, i, f"{fmt(lo)}–{fmt(hi)}%", transform=ax.get_yaxis_transform(),
                va="center", fontsize=13, color=F.INK)  # fmt: skip
    ax.set_yticks(range(len(rows)), [label for _s, label in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.8)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.set_xlabel("Approve-or-decline decisions that change", fontsize=12)
    text_block(fig, 0.155, "Each lender approves everyone below a risk cut-off, chosen on 2,000 "
               "other customers for the full-size model.", 12, color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_speed(fig, s, n, total):
    """Time per answer on the tuned setup (the speed addendum), first runs as hollow dots."""
    y = C.frame(fig, n, total, s)
    first = F.plt.Line2D([], [], marker="o", ls="none", ms=9, mfc="none", mec=F.INK_2, mew=1.8,
                         label="first runs")  # fmt: skip
    handles = [*model_handles(), first]
    y = legend_row(fig, y - 0.01, handles)
    ms = QF.speed_ms()
    ax = fig.add_axes([M + 0.235, 0.255, 1 - 2 * M - 0.235, y - 0.3])
    for model, key, _name in QF.MODELS:
        col = F.COLOR[key]
        for i, (tag, base, _label) in enumerate(QF.SPEED_ROWS):
            yy = i + QF.COST_DY[model] * 1.2
            t = ms.get(f"{model}-{tag}")
            if t is not None:
                ax.plot(t, yy, "o", ms=11, color=col, mec=F.SURFACE, mew=1.4, zorder=3)
            if tag in ("bf16-tuned", "bnb-nf4-tuned"):
                t0 = QF.value(model, base, "latency_ms_median")
                ax.plot(t0, yy, "o", ms=10, mfc="none", mec=col, mew=1.8, zorder=3)
                ax.annotate("", (t, yy), xytext=(t0, yy), zorder=2,
                            arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.3,
                                        "alpha": 0.6, "shrinkA": 5, "shrinkB": 6})  # fmt: skip
    ax.text(52, 3 + QF.COST_DY["qwen-ft"] * 1.2, "Qwen3.5-4B: wouldn't load", fontsize=12,
            color=F.INK_2, va="center")  # fmt: skip
    ax.text(5, 4 - 0.12, "Both Kev models: no answers", fontsize=12, color=F.INK_2,
            va="center")  # fmt: skip
    labels = ["Full size", "4-bit, NF4", "4-bit HQQ,\nGemLite", "4-bit HQQ,\nPyTorch int4", "FP8"]
    assert [t for t, _b, _l in QF.SPEED_ROWS] == ["bf16-tuned", "bnb-nf4-tuned", "hqq-4-gemlite",
                                                  "hqq-4-aoint4", "fp8-dynamic"]  # fmt: skip
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.45, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 260)
    ax.set_xlabel("Milliseconds per answer", fontsize=12)
    text_block(fig, 0.155, "Tuned: fast kernels, Kev's shortcut on, Qwen compiled. At $0.44 per "
               "GPU hour, 100 ms per answer is about 1.2 cents per 1,000.", 11.5,
               color=F.INK_2)  # fmt: skip
    note(fig, FAIR)


def slide_bullets(fig, s, n, total, size: float = 16.5, footer_size: float = 19):
    y = C.frame(fig, n, total, s, headline_size=30) - 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, size, bullet=True, color=F.INK) - 0.02
    return text_block(fig, y - 0.025, s["footer"], footer_size, fontweight="bold", color=F.INK)


def slide_next(fig, s, n, total):
    slide_bullets(fig, s, n, total, size=16.5, footer_size=19)
    fig.text(M, 0.08, "Write-up, code and every number: link in the first comment.",
             fontsize=13, color=F.INK_2)  # fmt: skip


MAKERS = {  # by slide title in quantization_linkedin.md, so slides can be reordered there
    "Hook": slide_hook,
    "How shrinking works": slide_shrink,
    "Ranking": slide_ranking,
    "Honesty": slide_honesty,
    "The customers moved": slide_drift,
    "Lending decisions": slide_decisions,
    "Speed": slide_speed,
    "What I'd tell a bank": slide_bullets,
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
