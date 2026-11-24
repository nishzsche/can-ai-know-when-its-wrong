"""Numbers for post 2, "Why 99% accuracy can still be a disaster" (PLAN.md §12, post 1 in the
next set). Reads the published M3 predictions only; no new model calls.

    uv run python analysis/accuracy.py      # results/accuracy.json, the figure, a printed table

For each model on the 2,000 test customers (the scorecard and LightGBM trained on 20,000; the AI
models zero-shot, out of the box):
- accuracy at the usual 50% cut-off, next to the "nobody defaults" model (always answer no),
  and the four-way count behind it (confusion matrix), at 50% and at the 20% lending cut-off;
- the share of defaulters caught and of customers flagged at that cut-off;
- ranking (AUC) and honesty (ECE), as in post 1;
- a simple lending decision. Each paying customer earns 1 unit and each default costs L units
  (illustrative; L = 4 in the post, 6 and 10 as a sensitivity check), so the break-even risk is
  1 / (1 + L) and a bank approves customers below it. Profit per 1,000 customers when the bank
  (a) uses the model's own percentages, and (b) picks the cut-off that maximizes profit on the
  2,000 *tune* customers, then applies it to test. Nothing is chosen on test.

Figure (results/figures/post2_profit.png): profit per 1,000 customers at a 4:1 loss ratio, each
model's own percentages (open dot) against a cut-off from known outcomes (filled dot), rows
sorted by accuracy. Same palette and style as the post 1 figures (analysis/figures.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as m  # noqa: E402
from records import ROOT, read_records  # noqa: E402

OUT = ROOT / "results" / "accuracy.json"
MODELS = (
    ("scorecard", "trained"),
    ("gbm", "trained"),
    ("frontier-verbal", "out_of_box"),
    ("kev-0.8b", "out_of_box"),
    ("qwen-logprobs", "out_of_box"),
    ("laya", "out_of_box"),
    ("jev", "out_of_box"),
    ("kev-4b", "out_of_box"),
)
LOSS_RATIOS = (4, 6, 10)
SWEEP = tuple(float(x) for x in np.arange(1, 10.001, 0.25))  # loss ratios for the cost chart


def get(contestant: str, condition: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    recs = sorted((r for r in read_records(contestant)
                   if r["split"] == split and r["condition"] == condition),
                  key=lambda r: r["id"])  # fmt: skip
    return (np.array([r["label"] for r in recs], float),
            np.array([r["p_default"] for r in recs], float))  # fmt: skip


def profit(y: np.ndarray, approve: np.ndarray, loss: float) -> float:
    """Profit per 1,000 customers: +1 per approved payer, -loss per approved defaulter."""
    return float(1000 * ((approve & (y == 0)).sum() - loss * (approve & (y == 1)).sum()) / len(y))


def tune_cutoff(y_tune: np.ndarray, p_tune: np.ndarray, loss: float) -> float:
    """The approval cut-off (approve if p <= t) that maximizes profit on tune."""
    return float(max(np.unique(p_tune), key=lambda t: profit(y_tune, p_tune <= t, loss)))


def confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    """The four-way count (confusion matrix), with "flagged" meaning "predicted to default"."""
    return {"defaulters_caught": int((flag & (y == 1)).sum()),
            "defaulters_missed": int((~flag & (y == 1)).sum()),
            "payers_turned_away": int((flag & (y == 0)).sum()),
            "payers_approved": int((~flag & (y == 0)).sum())}  # fmt: skip


def build() -> dict:
    y_test, _ = get("scorecard", "trained", "test")
    rate = float(y_test.mean())
    out = {"customers": len(y_test), "default_rate": rate, "always_no_accuracy": 1 - rate,
           "break_even_loss_ratio": (1 - rate) / rate,
           "approve_everyone": {str(L): profit(y_test, np.ones(len(y_test), bool), L)
                                for L in LOSS_RATIOS},
           "always_no_confusion": confusion(y_test, np.zeros(len(y_test), bool)),
           "sweep_own_percentages": {
               "loss_ratios": list(SWEEP),
               "approve_everyone": [profit(y_test, np.ones(len(y_test), bool), L)
                                    for L in SWEEP],
               "models": {}},
           "models": {}}  # fmt: skip
    for contestant, condition in MODELS:
        y, p = get(contestant, condition, "test")
        y_tu, p_tu = get(contestant, condition, "tune")
        assert np.array_equal(y, y_test)
        flag = p >= 0.5
        row = {
            "condition": condition,
            "accuracy_at_50": float((flag == y).mean()),
            "defaulters_caught_at_50": float((flag & (y == 1)).sum() / y.sum()),
            "flagged_at_50": float(flag.mean()),
            "auc": m.auc(y, p),
            "ece": m.ece(y, p),
            "mean_predicted": float(p.mean()),
            "decisions": {},
            # at the usual 50% cut-off, and at the 4:1 lending cut-off (flag at 20% or more)
            "confusion": {"50": confusion(y, flag), "20": confusion(y, p >= 0.2)},
        }
        for L in LOSS_RATIOS:
            own = p < 1 / (1 + L)
            t = tune_cutoff(y_tu, p_tu, L)
            tuned = p <= t
            row["decisions"][str(L)] = {
                "own_percentages": {"approve": float(own.mean()), "profit": profit(y, own, L)},
                "tune_cutoff": {"cutoff": t, "approve": float(tuned.mean()),
                                "profit": profit(y, tuned, L)},
            }  # fmt: skip
        out["models"][contestant] = row
        out["sweep_own_percentages"]["models"][contestant] = [
            profit(y, p < 1 / (1 + L), L) for L in SWEEP]  # fmt: skip
    return out


FIG_ROWS = (
    "gbm", "qwen-logprobs", "scorecard", "frontier-verbal", "kev-0.8b", "laya", "jev", "kev-4b",
)  # fmt: skip
LABELLED = ("kev-0.8b", "qwen-logprobs", "jev")  # the three big gaps the post discusses


def pct1(x: float) -> str:
    """A share as a percentage to one decimal, rounding half up (0.8185 -> 81.9%)."""
    from decimal import ROUND_HALF_UP, Decimal

    return f"{(Decimal(str(x)) * 100).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"


def figure(res: dict, loss: str = "4", dark: bool = False) -> Path:
    """The post's chart; dark=True draws the blog's dark-mode version (-dark.png)."""
    import figures as F

    F.set_theme(dark)
    plt = F.plt
    rows = sorted(FIG_ROWS, key=lambda c: -res["models"][c]["accuracy_at_50"])
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    fig.subplots_adjust(left=0.31, right=0.95, top=0.72, bottom=0.15)
    labels = []
    for y, c in enumerate(rows, start=1):
        r = res["models"][c]
        own = r["decisions"][loss]["own_percentages"]["profit"]
        cut = r["decisions"][loss]["tune_cutoff"]["profit"]
        col = F.COLOR[c]
        ax.plot([own, cut], [y, y], color=col, lw=2.5, alpha=0.35, zorder=2)
        ax.plot(cut, y, "o", ms=10, color=col, mec=col, zorder=3)
        ax.plot(own, y, "o", ms=10, color=F.SURFACE, mec=col, mew=2, zorder=4)
        if c in LABELLED:
            kw = {"va": "center", "fontsize": 9.5, "color": F.INK_2}
            ax.text(own - 9, y, f"{own:.0f}", ha="right", **kw)
            ax.text(cut + 9, y, f"{cut:.0f}", ha="left", **kw)
        labels.append(f"{F.NAME[c]}  {pct1(r['accuracy_at_50'])}")
    everyone = res["approve_everyone"][loss]
    ax.plot(everyone, 0, marker="X", ms=10, color=F.MUTED, zorder=3)
    ax.text(everyone + 12, 0, f"loses {-everyone:.0f}", va="center", fontsize=9.5, color=F.INK_2)
    labels.insert(0, f"Approve everyone  {pct1(res['always_no_accuracy'])}")
    ax.axvline(0, color=F.AXIS, lw=1.2, zorder=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.7)  # most accurate at the top
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=F.INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-160, 360)
    ax.set_xlabel("Profit per 1,000 customers")
    ax.text(-0.02, 1.01, "Model, accuracy", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=F.MUTED)  # fmt: skip
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=9, color=F.INK_2, mfc=F.SURFACE, mew=2,
                   label="Lending on its own percentages"),
        plt.Line2D([], [], marker="o", ls="none", ms=9, color=F.INK_2,
                   label="Its ranking, with a cut-off learned from known outcomes"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.43, 1.06), ncol=1,
              fontsize=9.5, handletextpad=0.3)  # fmt: skip
    F.header(fig, "More accurate isn't worth more",
             "What each model earns a lender: 1 unit per customer who pays, 4 lost per\n"
             "default. The gap between the dots is value lost to percentages that don't\n"
             "mean what they say. Where the dots overlap, they were already honest.")  # fmt: skip
    F.footer(fig, "Test set: 2,000 real credit-card customers (UCI, Taiwan 2005), 22.1% "
                  "defaulted.\nCut-offs chosen on 2,000 separate customers. The AI models "
                  "answered cold;\nthe scorecard and gradient boosting learned from 20,000 "
                  "labelled customers.")  # fmt: skip
    return F.save(fig, "post2_profit.png")


def social_card(res: dict, loss: str = "4") -> Path:
    """1200x630 link-preview image in the style of post 1's card: title and hook on the left, a
    mini chart on the right (three ways to lend, similar accuracy, very different profit)."""
    import figures as F

    F.set_theme(False)
    plt = F.plt
    fig = plt.figure(figsize=(6, 3.15))
    t = fig.text
    dot = "·"
    t(0.055, 0.89, f"LEARNING CURVE  {dot}  nishzsche.github.io", fontsize=7.5, color=F.MUTED,
      fontweight="bold", va="center")  # fmt: skip
    t(0.055, 0.80, "Why 99% accuracy\ncan still be\na disaster", fontsize=19, fontweight="bold",
      color=F.INK, va="top", linespacing=1.05)  # fmt: skip
    t(0.055, 0.36, "A model that says nobody defaults\nis 78% accurate.", fontsize=10.5,
      color=F.INK_2, va="top", linespacing=1.25)  # fmt: skip
    t(0.055, 0.205, "It loses money.", fontsize=12, fontweight="bold", color=F.INK, va="top")
    t(0.055, 0.075, f"8 models  {dot}  2,000 real credit-card customers", fontsize=7.5,
      color=F.MUTED, va="center")  # fmt: skip

    m = res["models"]
    q, sc = m["qwen-logprobs"], m["scorecard"]
    bars = [
        (f"Approve everyone\n{pct1(res['always_no_accuracy'])} accurate",
         res["approve_everyone"][loss], F.MUTED),
        (f"Qwen3.5-4B\n{pct1(q['accuracy_at_50'])} accurate",
         q["decisions"][loss]["own_percentages"]["profit"], F.COLOR["qwen-logprobs"]),
        (f"Scorecard\n{pct1(sc['accuracy_at_50'])} accurate",
         sc["decisions"][loss]["own_percentages"]["profit"], F.COLOR["scorecard"]),
    ]  # fmt: skip
    ax = fig.add_axes((0.67, 0.2, 0.29, 0.55))
    ys = range(len(bars))
    ax.barh(list(ys), [b[1] for b in bars], color=[b[2] for b in bars], height=0.55, zorder=2)
    for y, (_, v, _) in zip(ys, bars, strict=True):
        ax.text(v + (8 if v >= 0 else -8), y, f"{v:+.0f}", va="center", fontsize=8,
                ha="left" if v >= 0 else "right", color=F.INK_2)  # fmt: skip
    ax.set_yticks(list(ys), [b[0] for b in bars], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color=F.AXIS, lw=1)
    ax.set_xlim(-290, 390)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0, labelcolor=F.INK)
    ax.grid(axis="y", visible=False)
    ax.set_title("Profit per 1,000 customers", fontsize=9, loc="left", pad=8)
    ax.text(0, -0.2, "lending on each model's own percentages", transform=ax.transAxes,
            fontsize=6.5, color=F.MUTED)  # fmt: skip
    return F.save(fig, "post2_social_card.png")


FOOT = ("Test set: 2,000 real credit-card customers (UCI, Taiwan 2005), 22.1% defaulted.\n"
        "The AI models answered cold; the scorecard and gradient boosting learned from 20,000 "
        "labelled customers.")  # fmt: skip
NOTHING = "Doing nothing"  # the "nobody defaults" model: says "won't default" to everyone


def fig_accuracy(res: dict, dark: bool = False) -> Path:
    """Accuracy at 50% next to the share of defaulters caught, with the do-nothing floor."""
    import figures as F

    F.set_theme(dark)
    plt = F.plt
    m = res["models"]
    floor = res["always_no_accuracy"]
    rows = sorted([*FIG_ROWS, NOTHING],
                  key=lambda c: -(floor if c == NOTHING else m[c]["accuracy_at_50"]))  # fmt: skip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 6.6), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.74, bottom=0.17, wspace=0.12)
    ax1.axvspan(0.5, floor, color=F.GRID, alpha=0.55, lw=0, zorder=0)
    ax1.axvline(floor, color=F.MUTED, lw=1.2, zorder=1)
    ax1.text(0.505, -0.35, "Less accurate than doing nothing", ha="left", va="center",
             fontsize=8.5, color=F.INK_2)  # fmt: skip
    for y, c in enumerate(rows):
        if c == NOTHING:
            for ax, x in ((ax1, floor), (ax2, 0.0)):
                ax.plot(x, y, marker="X", ms=10, color=F.MUTED, zorder=3)
            continue
        col = F.COLOR[c]
        for ax, x in ((ax1, m[c]["accuracy_at_50"]), (ax2, m[c]["defaulters_caught_at_50"])):
            ax.plot(x, y, "o", ms=10, color=col, mec=F.SURFACE, mew=1.5, zorder=3)
    kw = {"va": "center", "fontsize": 9.5, "color": F.INK_2}
    best = rows[0]
    ax1.text(m[best]["accuracy_at_50"] + 0.009, 0, pct1(m[best]["accuracy_at_50"]), **kw)
    i = rows.index(NOTHING)
    ax1.text(floor + 0.011, i, pct1(floor), **kw)
    gap = (m[best]["accuracy_at_50"] - floor) * 100
    ax1.annotate("", (floor, 0.5), (m[best]["accuracy_at_50"], 0.5),
                 arrowprops={"arrowstyle": "|-|", "color": F.INK_2, "lw": 0.8,
                             "mutation_scale": 3})  # fmt: skip
    ax1.text(m[best]["accuracy_at_50"] + 0.005, 0.5, f"{gap:.1f} points", ha="left",
             fontsize=8.5, color=F.INK_2, va="center")  # fmt: skip
    for c in ("jev", "kev-4b"):
        ax2.text(m[c]["defaulters_caught_at_50"] + 0.025, rows.index(c),
                 f"{m[c]['defaulters_caught_at_50']:.0%}", **kw)  # fmt: skip
    ax2.text(0.025, i, "0%", **kw)
    for ax, title in ((ax1, "Accuracy"), (ax2, "Defaulters caught (of 442)")):
        ax.set_title(title, loc="left", pad=10)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(F.PCT)
        ax.tick_params(axis="y", length=0)
    ax1.set_xlim(0.5, 0.9)
    ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax2.set_xlim(-0.03, 0.85)
    ax1.set_yticks(range(len(rows)), [c if c == NOTHING else F.NAME[c] for c in rows])
    ax1.set_ylim(len(rows) - 0.3, -0.6)
    ax1.tick_params(axis="y", labelsize=10.5, labelcolor=F.INK)
    F.header(fig, "Accuracy barely beats doing nothing",
             "Accuracy at the usual 50% cut-off, and the share of the 442 defaulters each model "
             "caught.\nDoing nothing, saying \"won't default\" to everyone, scores 77.9% and "
             "catches no one.\nThe models that catch the most defaulters look worst by "
             "accuracy.")  # fmt: skip
    F.footer(fig, FOOT)
    return F.save(fig, "post2_accuracy.png")


def fig_confusion(res: dict, dark: bool = False) -> Path:
    """Where the 2,000 customers landed at 50%: defaulters and payers, mistakes solid."""
    import figures as F

    F.set_theme(dark)
    plt = F.plt
    rows = ((NOTHING, res["always_no_confusion"], F.MUTED),
            ("Scorecard", res["models"]["scorecard"]["confusion"]["50"], F.COLOR["scorecard"]),
            ("Jev", res["models"]["jev"]["confusion"]["50"], F.COLOR["jev"]))  # fmt: skip
    panels = (("The 442 who defaulted", "defaulters_missed", "missed",
               "defaulters_caught", "caught"),
              ("The 1,558 who paid", "payers_turned_away", "turned away",
               "payers_approved", "approved"))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.66, bottom=0.2, wspace=0.1)
    for ax, (title, bad, bad_word, good, good_word) in zip(axes, panels, strict=True):
        total = rows[0][1][bad] + rows[0][1][good]
        for y, (_name, cm, col) in enumerate(rows):
            ax.barh(y, cm[bad], color=col, height=0.5, zorder=2)
            ax.barh(y, cm[good], left=cm[bad], color=col, alpha=0.25, height=0.5, zorder=2)
            kw = {"va": "bottom", "fontsize": 9.5, "color": F.INK}
            ax.text(0, y - 0.3, f"{cm[bad]:,} {bad_word}", ha="left", **kw)
            ax.text(total, y - 0.3, f"{cm[good]:,} {good_word}", ha="right", **kw)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlim(0, total)
        ax.set_xticks(range(0, total + 1, 100 if total < 1000 else 400))
        ax.grid(visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(F.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].set_ylim(len(rows) - 0.55, -0.8)
    axes[0].tick_params(axis="y", labelsize=10.5, labelcolor=F.INK)
    axes[0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=F.INK_2, label="Mistake"),
        plt.Rectangle((0, 0), 1, 1, color=F.INK_2, alpha=0.25, label="Right call"),
    ], loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=9.5)  # fmt: skip
    F.header(fig, "Accuracy can't see which mistakes a model makes",
             "Where the 2,000 test customers landed at the usual 50% cut-off. Accuracy adds up "
             "only\nthe right calls, so it misses the trade: Jev catches twice as many defaulters "
             "as the\nscorecard, by turning away 656 people who would have paid.")  # fmt: skip
    F.footer(fig, FOOT)
    return F.save(fig, "post2_confusion.png")


NEAR_ZERO = ("kev-0.8b", "qwen-logprobs", "jev", "kev-4b")  # approve almost nobody at high cost


def fig_cost_ratio(res: dict, dark: bool = False) -> Path:
    """Profit at each model's own percentages as the cost of a default rises."""
    import figures as F

    F.set_theme(dark)
    plt = F.plt
    sw = res["sweep_own_percentages"]
    xs = np.array(sw["loss_ratios"])
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    fig.subplots_adjust(left=0.13, right=0.7, top=0.79, bottom=0.17)
    ax.axhline(0, color=F.AXIS, lw=1.2, zorder=1)
    ax.axvline(4, color=F.GRID, lw=6, zorder=0)
    ax.text(4, 640, "the example\nabove", ha="center", va="top", fontsize=8.5, color=F.INK_2)
    ev = np.array(sw["approve_everyone"])
    ax.plot(xs, ev, color=F.MUTED, lw=1.6, ls=(0, (1, 2)), zorder=2)
    ax.plot(xs[xs == xs.astype(int)], ev[xs == xs.astype(int)], "X", ms=7, color=F.MUTED,
            zorder=2)  # fmt: skip
    ax.text(5.25, -300, "Approve everyone", fontsize=9.5, color=F.INK, va="center")
    ax.text(6.3, -575, f"keeps falling: loses {-ev[-1]:,.0f} at 10", fontsize=8.5,
            color=F.INK_2, va="center")  # fmt: skip
    ends = []
    for c in FIG_ROWS:
        ys = np.array(sw["models"][c])
        style = {"scorecard": (0, (4, 3))}.get(c, "-")
        lw = 1.6 if c in BASE else (1.5 if c in NEAR_ZERO else 2.2)
        ax.plot(xs, ys, color=F.COLOR[c], lw=lw, ls=style, zorder=3)
        if c not in NEAR_ZERO:
            ends.append((ys[-1], F.NAME[c]))
    ends.append((float(np.mean([sw["models"][c][-1] for c in NEAR_ZERO])),
                 "Jev, Qwen and both Kevs:\napprove almost nobody"))  # fmt: skip
    placed, prev_lines = None, 1
    for v, name in sorted(ends, reverse=True):  # push labels down so none overlap
        lines = name.count("\n") + 1
        gap = 30 * (prev_lines + lines) + 6  # about half a line height per line, both labels
        placed = v if placed is None else min(v, placed - gap)
        prev_lines = lines
        ax.annotate(name, (10, v), xytext=(10.35, placed), fontsize=9.5, color=F.INK,
                    va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": F.AXIS, "lw": 0.8})  # fmt: skip
    ax.set_xlim(1, 10)
    ax.set_ylim(-620, 660)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Cost of one default, in units a paying customer earns")
    ax.set_ylabel("Profit per 1,000 customers")
    F.header(fig, "The costlier a mistake, the more honesty pays",
             "Profit when a lender takes each model at its word and approves\n"
             "customers whose risk is below break-even, as defaults get costlier.")  # fmt: skip
    F.footer(fig, FOOT)
    return F.save(fig, "post2_cost_ratio.png")


BASE = ("scorecard", "gbm")


def main() -> None:
    res = build()
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"default rate {res['default_rate']:.3f}; 'nobody defaults' accuracy "
          f"{res['always_no_accuracy']:.3f}; approving everyone pays only below a loss ratio of "
          f"{res['break_even_loss_ratio']:.2f}")  # fmt: skip
    print(f"{'model':<16}{'acc@50':>7}{'caught':>7}{'AUC':>7}{'ECE':>7}"
          f"{'own: approve':>13}{'profit':>7}{'tune: approve':>14}{'profit':>7} (4:1)")  # fmt: skip
    for name, r in res["models"].items():
        d = r["decisions"]["4"]
        print(f"{name:<16}{r['accuracy_at_50']:>7.3f}{r['defaulters_caught_at_50']:>7.2f}"
              f"{r['auc']:>7.3f}{r['ece']:>7.3f}{d['own_percentages']['approve']:>13.0%}"
              f"{d['own_percentages']['profit']:>7.0f}{d['tune_cutoff']['approve']:>14.0%}"
              f"{d['tune_cutoff']['profit']:>7.0f}")  # fmt: skip
    paths = [figure(res), figure(res, dark=True), social_card(res)]
    for fig in (fig_accuracy, fig_confusion, fig_cost_ratio):
        paths += [fig(res), fig(res, dark=True)]
    print(
        f"wrote {OUT.relative_to(ROOT)} and " + ", ".join(str(x.relative_to(ROOT)) for x in paths)
    )


if __name__ == "__main__":
    main()
