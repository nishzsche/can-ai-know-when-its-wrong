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
from matplotlib.ticker import FuncFormatter  # noqa: E402

CAROUSEL_MD = HERE / "carousel.md"
OUT_PDF = HERE / "carousel.pdf"
SIZE_IN, DPI = 7.5, 144  # 7.5 in x 144 dpi = 1080 px
M = 0.075  # side margin, figure fraction
PT = 1 / (SIZE_IN * 72)  # one point, as a figure fraction
FAIRNESS = (
    "The AI models got no examples; the traditional models learned from 20,000 past customers."
)
# A few plain words on what each contestant is, for readers outside tech (slide 2).
WHAT = {"jev": "built for yes/no decisions", "kev-4b": "free, open-source",
        "kev-0.8b": "free, open-source, smaller", "laya": "open-source, tiny",
        "qwen-logprobs": "free, general-purpose", "frontier-verbal": "ChatGPT's model",
        "scorecard": "banks' classic", "gbm": "today's standard"}  # fmt: skip
# Plain-language legend for the three conditions (slide 6).
STAGE_LABEL = {"out_of_box": "Out of the box", "base_rate": "Told the overall rate",
               "recalibrated": "Corrected with past outcomes"}  # fmt: skip


def parse(md: str) -> dict[int, dict]:
    """Slide number -> headline, body (text or bullet list) and footer from carousel.md."""
    slides = {}
    for block in re.split(r"^### ", md.split("## Slides", 1)[1], flags=re.M)[1:]:
        number = int(block.split(".", 1)[0])
        fields = {"title": block.split("\n", 1)[0].split(".", 1)[1].strip()}
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
    cols = [
        ("6 AI models", "given no examples", F.AI),
        ("2 traditional models", "learned from 20,000 past customers", F.BASELINES),
    ]
    top = y - 0.03
    for i, (title, subtitle, keys) in enumerate(cols):
        x = M + i * 0.46
        fig.text(x, top, title, fontsize=14, fontweight="bold", color=F.INK, va="top")
        fig.text(x, top - 0.036, subtitle, fontsize=11, color=F.INK_2, va="top")
        for j, key in enumerate(keys):
            yy = top - 0.11 - j * 0.056
            fig.patches.append(plt.Circle((x + 0.012, yy + 0.008), 0.011, color=F.COLOR[key],
                                          transform=fig.transFigure, figure=fig))  # fmt: skip
            fig.text(x + 0.035, yy, F.NAME[key], fontsize=14.5, color=F.INK, va="bottom")
            fig.text(x + 0.035, yy - 0.004, WHAT[key], fontsize=10.5, color=F.MUTED, va="top")
    no = next(b for b in s["bullets"] if b.startswith("No "))
    text_block(fig, 0.10, no, 13, color=F.INK_2)


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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("Percentage points off, on average: lower is more honest", fontsize=13)
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=10, color=F.INK_2, mfc=f or F.INK_2,
                          mew=2, mec=F.INK_2, label=STAGE_LABEL[st])
               for st, _, mk, f in F.STAGES]  # fmt: skip
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(M, y - 0.01), ncol=3,
               fontsize=11, frameon=False, handletextpad=0.3, columnspacing=1.0)  # fmt: skip


def slide_twist(fig, s, n, total):
    y = frame(fig, n, total, s, headline_size=25)
    stats = [("53/53", "times it recognised\nthe public dataset"),
             ("22.12%", "its rate of missed\npayments, recalled exactly"),
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
    ax.set_xlabel("Ranking score (AUC): 0.5 = coin flip, 1 = perfect", fontsize=12)
    note(fig, "* Learned from 20,000 past customers; the AI models got no examples. "
         "Black line: likely range.")  # fmt: skip


def slide_takeaways(fig, s, n, total):
    y = frame(fig, n, total, s, headline_size=30)
    y -= 0.03
    for b in s["bullets"]:
        y = text_block(fig, y, b, 20, bullet=True, color=F.INK) - 0.03
    fig.text(M, 0.13, s["footer"], fontsize=17, fontweight="bold", color=F.COLOR["jev"])


def slide_safest(fig, s, n, total, explain):
    """The 'not late (code 0)' group: actual default rate vs each model's mean prediction."""
    y = frame(fig, n, total, s)
    group = next(r for r in explain["by_status"] if r["status"].startswith("not late (code 0)"))
    keys = sorted([*F.AI, "scorecard"], key=lambda k: group[k])
    ax = fig.add_axes([M + 0.24, 0.17, 0.62, y - 0.25])
    for i, k in enumerate(keys):
        ax.barh(i, group[k], height=0.62, color=F.COLOR[k])
        ax.text(group[k] + 0.012, i, f"{group[k]:.0%}", va="center", fontsize=13, color=F.INK)
    ax.axvline(group["actual"], color=F.INK, lw=1.5, ls="--")
    ax.text(group["actual"] + 0.008, len(keys) - 0.35, f"actually defaulted: {group['actual']:.0%}",
            fontsize=12, color=F.INK)  # fmt: skip
    names = [F.NAME[k] + ("*" if k in F.BASELINES else "") for k in keys]
    ax.set_yticks(range(len(keys)), names)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=F.INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.set_major_formatter(F.PCT)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.6, len(keys) - 0.1)
    ax.set_xlabel(f"Average predicted risk for these {group['n']:,} customers", fontsize=12)
    note(fig, "* Learned from 20,000 past customers; the AI models got no examples.")


def slide_portfolio(fig, s, n, total, explain):
    """Calibration error at 5%, 22% and 50% default rates; the most honest per column in bold."""
    y = frame(fig, n, total, s)
    rates = {r["default_rate"]: r for r in explain["base_rate_sensitivity"]}
    cols = [(0.05, "5%"), (0.221, "22%\n(actual)"), (0.50, "50%")]
    keys = [*F.AI, "scorecard"]
    best = {rate: min(keys, key=lambda k: rates[rate][k]["ece"]) for rate, _ in cols}
    x0, xs, top, step = M, [0.52, 0.69, 0.86], y - 0.04, 0.058
    fig.text(x0, top, "Points off, if this share\nmissed payments:", fontsize=12,
             color=F.MUTED, va="top", linespacing=1.2)  # fmt: skip
    for x, (_, label) in zip(xs, cols, strict=True):
        fig.text(x, top, label, fontsize=14, fontweight="bold", color=F.INK, ha="center",
                 va="top", linespacing=1.1)  # fmt: skip
    for i, k in enumerate(keys):
        yy = top - 0.10 - i * step
        fig.patches.append(plt.Circle((x0 + 0.012, yy + 0.008), 0.010, color=F.COLOR[k],
                                      transform=fig.transFigure, figure=fig))  # fmt: skip
        fig.text(x0 + 0.035, yy, F.NAME[k] + ("*" if k in F.BASELINES else ""), fontsize=14,
                 color=F.INK, va="bottom")  # fmt: skip
        for x, (rate, _) in zip(xs, cols, strict=True):
            v = rates[rate][k]["ece"] * 100
            text = f"{v:.1f}" if v < 2 else f"{v:.0f}"
            bold = k == best[rate]
            fig.text(x, yy, text, fontsize=15, ha="center", va="bottom",
                     fontweight="bold" if bold else "normal",
                     color=F.COLOR["jev"] if bold else F.INK_2)  # fmt: skip
    note(fig, "* The scorecard learned at 22%. Bold blue: the most honest model at that rate.")


def build(png_dir: Path | None = None) -> Path:
    import json

    F.style()
    rows, detail = F.load()
    explain = json.loads((HERE.parent / "results" / "explain.json").read_text("utf-8"))
    slides = parse(CAROUSEL_MD.read_text(encoding="utf-8"))
    total = len(slides)
    makers = {  # by slide title in carousel.md, so slides can be reordered or inserted there
        "Hook": lambda fig, s, n: slide_hook(fig, s, n, total),
        "The setup": lambda fig, s, n: slide_setup(fig, s, n, total),
        'What "honest" looks like': lambda fig, s, n: slide_honest(fig, s, n, total, detail),
        "Promised vs actual": lambda fig, s, n: slide_promised(fig, s, n, total, rows, detail),
        "Where it goes wrong": lambda fig, s, n: slide_safest(fig, s, n, total, explain),
        "The honesty arc": lambda fig, s, n: slide_arc(fig, s, n, total, rows),
        "The twist": lambda fig, s, n: slide_twist(fig, s, n, total),
        "The punchline": lambda fig, s, n: slide_ranking(fig, s, n, total, rows),
        "It depends on the portfolio": lambda fig, s, n: slide_portfolio(fig, s, n, total, explain),
        "What to take away": lambda fig, s, n: slide_takeaways(fig, s, n, total),
    }
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF, metadata={"Title": "Can AI actually know when it's wrong?"}) as pdf:
        for n in sorted(slides):
            fig = new_slide()
            makers[slides[n]["title"]](fig, slides[n], n)
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
