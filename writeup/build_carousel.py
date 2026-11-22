"""Build the LinkedIn carousel PDF from writeup/carousel.md and the results.

    uv run python writeup/build_carousel.py            # writeup/carousel.pdf
    uv run python writeup/build_carousel.py --png DIR  # also one PNG per slide, for checking

Slide text (headline, body, footer) is read from carousel.md, so that file stays the source.
Charts are drawn here from results/summary.csv and summary_detail.json, at phone size (1080 x
1080 px), in the same palette and per-contestant colours as analysis/figures.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))

import figures as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

CAROUSEL_MD = HERE / "carousel.md"
OUT_PDF = HERE / "carousel.pdf"
SIZE_IN, DPI = 7.5, 144  # 7.5 in x 144 dpi = 1080 px
M = 0.075  # side margin, figure fraction
PT = 1 / (SIZE_IN * 72)  # one point, as a figure fraction
FAIRNESS = "AI models answered cold; the classic models learned from 20,000 labelled customers."


def parse(md: str) -> dict[int, dict]:
    """Slide number -> headline, body (text or bullet list) and footer from carousel.md."""
    slides = {}
    for block in re.split(r"^### ", md.split("## Slides", 1)[1], flags=re.M)[1:]:
        number = int(block.split(".", 1)[0])
        fields = {}
        for name in ("Headline", "Body", "Footer"):
            m = re.search(rf"\*\*{name}:\*\*(.*?)(?=\n\*\*\w+:\*\*|\Z)", block, re.S)
            fields[name.lower()] = m.group(1).strip() if m else ""
        body = fields["body"]
        bullets = [b[2:].strip() for b in body.splitlines() if b.startswith("- ")]
        fields["bullets"] = bullets
        fields["body"] = "" if bullets else body
        slides[number] = {k: _plain(v) if isinstance(v, str) else [_plain(b) for b in v]
                          for k, v in fields.items()}  # fmt: skip
    return slides


def _plain(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text).replace("`", "")


def wrap(fig, text: str, size: float, first: str = "", rest: str = "", **kw) -> list[str]:
    """Greedy word wrap by rendered width, so every line fits between the side margins."""
    renderer = fig.canvas.get_renderer()
    limit = (1 - 2 * M) * fig.bbox.width

    def width(s: str) -> float:
        t = fig.text(0, 0, s, fontsize=size, **kw)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    lines, current, fresh = [], first, True  # fresh: no word on the current line yet
    for word in text.split():
        candidate = f"{current}{word}" if fresh else f"{current} {word}"
        if width(candidate) > limit and not fresh:
            lines.append(current)
            current = f"{rest}{word}"
        else:
            current = candidate
        fresh = False
    lines.append(current)
    return lines


def text_block(fig, y: float, text: str, size: float, bullet: bool = False, **kw) -> float:
    """Draw wrapped text from y downwards; return the y just below it."""
    first, rest = ("•  ", "    ") if bullet else ("", "")
    lines = wrap(fig, text, size, first, rest, **kw)
    fig.text(M, y, "\n".join(lines), fontsize=size, va="top", linespacing=1.25, **kw)
    return y - len(lines) * size * 1.25 * PT - 0.02


def frame(fig, n: int, total: int, slide: dict, headline_size: float = 27) -> float:
    """Headline, body and page furniture. Returns the y below the text."""
    y = text_block(fig, 1 - M, slide["headline"], headline_size, fontweight="bold", color=F.INK)
    if slide["body"]:
        y = text_block(fig, y - 0.005, slide["body"], 15.5, color=F.INK_2)
    fig.text(1 - M, 0.035, f"{n} / {total}", fontsize=10, color=F.MUTED, ha="right")
    fig.text(M, 0.035, "Can AI actually know when it's wrong?", fontsize=10, color=F.MUTED)
    return y


def new_slide():
    fig = plt.figure(figsize=(SIZE_IN, SIZE_IN), dpi=DPI)
    fig.patch.set_facecolor(F.SURFACE)
    return fig


def note(fig, text: str = FAIRNESS) -> None:
    fig.text(M, 0.075, text, fontsize=9.5, color=F.MUTED)


# ---- slides --------------------------------------------------------------------------------------


def slide_hook(fig, s, n, total):
    lines = wrap(fig, s["headline"], 36, fontweight="bold")
    fig.text(M, 0.70, "\n".join(lines), fontsize=36, fontweight="bold", color=F.INK, va="top",
             linespacing=1.15)  # fmt: skip
    fig.text(M, 0.40, s["body"], fontsize=24, color=F.COLOR["jev"], va="top")
    fig.text(M, 0.20, s["footer"], fontsize=15, color=F.INK_2)
    fig.text(1 - M, 0.035, "Swipe →", fontsize=12, color=F.INK_2, ha="right")
    fig.text(M, 0.035, f"{n} / {total}", fontsize=10, color=F.MUTED)


def slide_setup(fig, s, n, total):
    y = frame(fig, n, total, s)
    lead = [b for b in s["bullets"] if not b[0].isdigit() or b.startswith("2,000")]
    lead = [b for b in lead if not b.startswith("No ")]
    for b in lead:
        y = text_block(fig, y, b, 15.5, bullet=True, color=F.INK_2)
    cols = [("6 AI models, answering cold", F.AI),
            ("2 classic models, trained on\n20,000 customers", F.BASELINES)]  # fmt: skip
    top = y - 0.03
    for i, (title, keys) in enumerate(cols):
        x = M + i * 0.46
        fig.text(x, top, title, fontsize=14, fontweight="bold", color=F.INK, va="top")
        for j, key in enumerate(keys):
            yy = top - 0.105 - j * 0.058
            fig.patches.append(plt.Circle((x + 0.012, yy + 0.008), 0.011, color=F.COLOR[key],
                                          transform=fig.transFigure, figure=fig))  # fmt: skip
            fig.text(x + 0.035, yy, F.NAME[key], fontsize=15, color=F.INK, va="bottom")
    fig.text(M, 0.11, next(b for b in s["bullets"] if b.startswith("No ")), fontsize=14,
             color=F.INK_2)  # fmt: skip


def slide_honest(fig, s, n, total, detail):
    y = frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.09, 0.14, 0.76, y - 0.20])
    ax.plot([0, 0.8], [0, 0.8], color=F.AXIS, lw=2)
    dec = detail["scorecard"]["trained"]["deciles"]
    ax.plot(
        [d["promised"] for d in dec],
        [d["actual"] for d in dec],
        "o-",
        color=F.COLOR["scorecard"],
        lw=2,
        ms=6,
    )
    ax.text(0.47, 0.60, "the scorecard:\nhonest", fontsize=13, color=F.INK_2)
    ax.text(0.50, 0.18, "too pessimistic\n(promises more defaults\nthan happen)", fontsize=12,
            color=F.MUTED)  # fmt: skip
    ax.text(0.04, 0.55, "too optimistic", fontsize=12, color=F.MUTED)
    _reliability_axes(ax)


def _reliability_axes(ax):
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.yaxis.set_major_formatter(F.PCT)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Promised default rate", fontsize=13)
    ax.set_ylabel("Actual default rate", fontsize=13)


def slide_promised(fig, s, n, total, rows, detail):
    y = frame(fig, n, total, s)
    score = detail["scorecard"]["trained"]["deciles"]
    for i, key in enumerate(("jev", "kev-4b")):
        ax = fig.add_axes([M + 0.08 + i * 0.44, 0.23, 0.36, y - 0.31])
        ax.plot([0, 0.8], [0, 0.8], color=F.AXIS, lw=1.5)
        ax.plot(
            [d["promised"] for d in score],
            [d["actual"] for d in score],
            color=F.COLOR["scorecard"],
            lw=1.5,
            alpha=0.8,
        )
        dec = detail[key]["out_of_box"]["deciles"]
        ax.plot([d["promised"] for d in dec], [d["actual"] for d in dec], "o-", color=F.COLOR[key],
                lw=2.5, ms=6)  # fmt: skip
        mean = rows[(key, "out_of_box")]["mean_predicted"]
        ax.set_title(f"{F.NAME[key]}: says {mean:.0%}", loc="left", fontsize=14)
        _reliability_axes(ax)
        ax.set_xlabel("Promised", fontsize=12)
        ax.set_ylabel("Actual" if i == 0 else "", fontsize=12)
        if i:
            ax.tick_params(labelleft=False)
    fig.text(M, 0.115, "Grey line: the scorecard. On the diagonal = honest.", fontsize=11,
             color=F.INK_2)  # fmt: skip
    note(fig)


def slide_arc(fig, s, n, total, rows):
    y = frame(fig, n, total, s)
    ax = fig.add_axes([M + 0.17, 0.15, 0.66, y - 0.30])
    order = sorted(F.AI, key=lambda c: rows[(c, "out_of_box")]["ece"])
    for i, c in enumerate(order):
        xs = [rows[(c, st)]["ece"] for st, *_ in F.STAGES]
        ax.plot([min(xs), max(xs)], [i, i], color=F.COLOR[c], lw=2.5, alpha=0.35)
        for st, _, marker, face in F.STAGES:
            dy = 0.2 if st == "base_rate" else 0.0
            ax.plot(rows[(c, st)]["ece"], i + dy, marker=marker, ms=11, color=F.COLOR[c],
                    mfc=face or F.COLOR[c], mew=2, mec=F.COLOR[c], ls="none")  # fmt: skip
    ref = rows[("scorecard", "trained")]["ece"]
    ax.axvline(ref, color=F.COLOR["scorecard"], lw=1.2)
    ax.text(ref + 0.005, len(order) - 0.35, "scorecard", fontsize=11, color=F.INK_2)
    ax.set_yticks(range(len(order)), [F.NAME[c] for c in order])
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlim(-0.01, 0.3)
    ax.set_xlabel("Calibration error: lower is more honest", fontsize=13)
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=10, color=F.INK_2, mfc=f or F.INK_2,
                          mew=2, mec=F.INK_2, label=lab) for _, lab, mk, f in F.STAGES]  # fmt: skip
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M, y - 0.01), ncol=3,
               fontsize=11, frameon=False, handletextpad=0.3, columnspacing=1.0)  # fmt: skip


def slide_twist(fig, s, n, total):
    y = frame(fig, n, total, s, headline_size=25)
    stats = [("53/53", "summaries it named\nas this dataset"),
             ("22.12%", "its default rate,\nrecalled exactly"),
             ("70%", "of the way it moved\ntoward a false 10%")]  # fmt: skip
    for i, (big, small) in enumerate(stats):
        yy = y - 0.04 - i * 0.125
        fig.text(M, yy, big, fontsize=36, fontweight="bold", color=F.COLOR["frontier-verbal"],
                 va="top")  # fmt: skip
        fig.text(M + 0.40, yy - 0.012, small, fontsize=14, color=F.INK_2, va="top",
                 linespacing=1.3)  # fmt: skip
    fig.text(
        M,
        0.095,
        "GPT-6.2 Sol. It still recognised the data with the currency, dates\n"
        "and codes removed: 50 of 50.",
        fontsize=12,
        color=F.INK_2,
        linespacing=1.3,
    )


def slide_ranking(fig, s, n, total, rows):
    y = frame(fig, n, total, s)
    keys = sorted([(c, "trained") for c in F.BASELINES] + [(c, "out_of_box") for c in F.AI],
                  key=lambda k: rows[k]["auc"])  # fmt: skip
    ax = fig.add_axes([M + 0.24, 0.16, 0.62, y - 0.24])
    for i, k in enumerate(keys):
        r = rows[k]
        ax.barh(i, r["auc"] - 0.5, left=0.5, height=0.62, color=F.COLOR[k[0]])
        ax.plot([r["auc_lo"], r["auc_hi"]], [i, i], color=F.INK, lw=1.5)
        ax.text(r["auc_hi"] + 0.006, i, f"{r['auc']:.3f}", va="center", fontsize=12, color=F.INK)
    names = [F.NAME[c] + ("*" if c in F.BASELINES else "") for c, _ in keys]
    ax.set_yticks(range(len(keys)), names)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 0.86)
    ax.set_xlabel("AUC, out of the box (0.5 = coin flip), with 95% interval", fontsize=12)
    note(fig, "* Trained on 20,000 labelled customers. The AI models answered cold.")


def slide_takeaways(fig, s, n, total):
    y = frame(fig, n, total, s, headline_size=30)
    y -= 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, 20, bullet=True, color=F.INK) - 0.03
    fig.text(M, 0.13, s["footer"], fontsize=17, fontweight="bold", color=F.COLOR["jev"])


def build(png_dir: Path | None = None) -> Path:
    F.style()
    rows, detail = F.load()
    slides = parse(CAROUSEL_MD.read_text(encoding="utf-8"))
    total = len(slides)
    makers = {
        1: lambda fig, s: slide_hook(fig, s, 1, total),
        2: lambda fig, s: slide_setup(fig, s, 2, total),
        3: lambda fig, s: slide_honest(fig, s, 3, total, detail),
        4: lambda fig, s: slide_promised(fig, s, 4, total, rows, detail),
        5: lambda fig, s: slide_arc(fig, s, 5, total, rows),
        6: lambda fig, s: slide_twist(fig, s, 6, total),
        7: lambda fig, s: slide_ranking(fig, s, 7, total, rows),
        8: lambda fig, s: slide_takeaways(fig, s, 8, total),
    }
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF, metadata={"Title": "Can AI actually know when it's wrong?"}) as pdf:
        for n in sorted(slides):
            fig = new_slide()
            makers[n](fig, slides[n])
            pdf.savefig(fig)
            if png_dir:
                fig.savefig(png_dir / f"slide_{n:02d}.png", dpi=DPI)
            plt.close(fig)
    return OUT_PDF


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--png", type=Path, help="also write one PNG per slide here")
    args = ap.parse_args()
    print(f"Saved -> {build(args.png)}")


if __name__ == "__main__":
    main()
