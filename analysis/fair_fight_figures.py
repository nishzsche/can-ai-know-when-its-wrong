"""M6 figures: the models that learned from the 20,000 `train` customers.

    uv run python analysis/fair_fight_figures.py   # results/figures/m6_*.png, light and dark

- m6_catch_up.png: ranking (AUC) for each AI model answering cold (post 1) and after learning from
  the 20,000, with the scorecard, gradient boosting and the 20-neighbours reference as lines.
- m6_honesty.png: calibration error raw and after the same correction learned on `tune`, the
  comparison the post leads with (author's decision, 2026-11-24: Laya's maker recipe ends with its
  own calibration step and was kept, Kev's ends with a temperature fit that was not applied, so
  raw numbers alone would tilt the comparison).
- m6_reliability.png: promised vs actual in ten risk groups, raw output (for the deep dive).
- m6_how_close.png: paired AUC gaps to gradient boosting and the scorecard, intervals widened
  for the eight comparisons (Bonferroni, so the chart matches the post's Holm wording).
- m6_kev_calibration.png: Kev-0.8B and Kev-4B promised vs actual, raw and after the correction.
- m6_cost.png: ranking against running cost per 1,000 decisions (from each record's cost_usd).
- m6_sol_neighbours.png: Sol's and Jev's answers against how many of the 20 look-alikes in
  their prompt defaulted (the neighbours-20 reference, times 20).

    uv run python analysis/fair_fight_figures.py cost how_close   # only these

Same palette, style and dark variant as the post 1 figures (analysis/figures.py). A trained model
keeps its cold colour. The two tabular models share palette slot 8 (red) as a family colour: slot
6 green fails the colour-blind check next to Kev-4B's orange (protan dE 3.2, the reason it was
dropped for Sol too). Every mark is named in text, so colour never carries identity alone.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures as F  # noqa: E402
import metrics as m  # noqa: E402
from records import ROOT, read_records  # noqa: E402
from robustness import load, paired_auc  # noqa: E402

MODELS = (  # (trained contestant, its cold self or None, label, how it learned)
    ("kumo-tabular", None, "Kumo-Tabular", "tabular"),
    ("tabicl-v2", None, "TabICLv2", "tabular"),
    ("kev-4b-ft", "kev-4b", "Kev-4B", "fine-tuned"),
    ("kev-0.8b-ft", "kev-0.8b", "Kev-0.8B", "fine-tuned"),
    ("qwen-ft", "qwen-logprobs", "Qwen3.5-4B", "fine-tuned"),
    ("laya-ft", "laya", "Laya", "fine-tuned"),
    ("sol-examples", "frontier-verbal", "GPT-6.2 Sol", "20 examples"),
    ("jev-examples", "jev", "Jev", "20 examples"),
)
EXTRA = {  # palette slot 8 (red), light and dark steps: the tabular family
    False: {"kumo-tabular": "#e34948", "tabicl-v2": "#e34948"},
    True: {"kumo-tabular": "#e66767", "tabicl-v2": "#e66767"},
}
COLD = {c: zs for c, zs, _, _ in MODELS if zs}
TEST_LINE = "Test set: 2,000 real credit-card customers (UCI, Taiwan 2005), 22.1% defaulted."
HOW_LINES = (
    "Fine-tuned: Kev and Qwen one pass (Qwen has no maker recipe), Laya its maker's four passes.",
    "Tabular: no training, the 20,000 read as examples. Jev and Sol: their 20 most similar "
    "customers in the prompt.",
)
CHANCE = 0.025  # ECE a perfectly honest model stays below in 95 of 100 samples here (0.024-0.027)


def colour(contestant: str, dark: bool) -> str:
    return EXTRA[dark].get(contestant) or F.COLOR[COLD[contestant]]


def metric(contestant: str, condition: str) -> tuple[float, float]:
    y, p, _ = load(contestant, condition)
    return m.auc(y, p), m.ece(y, p)


def reference_lines(ax, values: dict) -> None:
    ax.axvline(values["gbm"], color=F.COLOR["gbm"], lw=1.2, zorder=1)
    ax.axvline(values["scorecard"], color=F.COLOR["scorecard"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    if "neighbours" in values:
        ax.axvline(values["neighbours"], color=F.MUTED, lw=1, ls=(0, (1, 2)), zorder=1)


def reference_legend(ax, plt, with_neighbours: bool = True) -> list:
    refs = [
        plt.Line2D([], [], color=F.COLOR["gbm"], lw=1.4, label="Gradient boosting"),
        plt.Line2D([], [], color=F.COLOR["scorecard"], lw=1.4, ls=(0, (4, 3)), label="Scorecard"),
    ]
    if with_neighbours:
        refs.append(plt.Line2D([], [], color=F.MUTED, lw=1.2, ls=(0, (1, 2)),
                               label="20 neighbours, averaged"))  # fmt: skip
    return refs


def dot_handles(plt, open_label: str, filled_label: str) -> list:
    return [
        plt.Line2D([], [], marker="o", ls="none", ms=8, color=F.INK_2, mfc=F.SURFACE, mew=2,
                   label=open_label),
        plt.Line2D([], [], marker="o", ls="none", ms=8, color=F.INK_2, label=filled_label),
    ]  # fmt: skip


def catch_up(dark: bool = False) -> Path:
    """Ranking: cold (open dot) -> after learning from the 20,000 (filled dot)."""
    F.set_theme(dark)
    plt = F.plt
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.71, bottom=0.17)
    refs = {"scorecard": metric("scorecard", "trained")[0], "gbm": metric("gbm", "trained")[0],
            "neighbours": metric("neighbours-20", "trained")[0]}  # fmt: skip
    for y, (c, zs, _label, _how) in enumerate(MODELS):
        col = colour(c, dark)
        tv = metric(c, "trained")[0]
        if zs:
            zv = metric(zs, "out_of_box")[0]
            ax.plot([zv, tv], [y, y], color=col, lw=2.5, alpha=0.35, zorder=2)
            ax.plot(zv, y, "o", ms=9, color=F.SURFACE, mec=col, mew=2, zorder=3)
        ax.plot(tv, y, "o", ms=9, color=col, mec=col, zorder=4)
    reference_lines(ax, refs)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=F.INK)
    ax.set_yticks(range(len(MODELS)), [lab for _, _, lab, _ in MODELS])
    ax.set_ylim(len(MODELS) - 0.4, -0.6)
    ax.set_xlim(0.58, 0.81)
    ax.set_xlabel("Ranking (AUC): how often a defaulter is rated riskier than a payer")
    leg1 = ax.legend(handles=dot_handles(plt, "Answering cold (post 1)",
                                         "After learning from the 20,000"),
                     loc="lower left", bbox_to_anchor=(-0.27, 1.1), ncol=2, fontsize=9.5,
                     handletextpad=0.3, columnspacing=1.4)  # fmt: skip
    ax.add_artist(leg1)
    ax.legend(handles=reference_legend(ax, plt), loc="lower left", bbox_to_anchor=(-0.27, 1.01),
              ncol=3, fontsize=9.5, handlelength=2.2, columnspacing=1.2)  # fmt: skip
    F.header(fig, "Give AI the same data and it catches up",
             "Each AI model's ranking before and after learning from the 20,000 customers\n"
             "the scorecard and gradient boosting trained on. The tabular models need the\n"
             "examples to work at all, so they have no cold version.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE, *HOW_LINES)))
    return F.save(fig, "m6_catch_up.png")


def honesty(dark: bool = False) -> Path:
    """Calibration error: raw (open dot) -> after the same correction learned on tune (filled)."""
    F.set_theme(dark)
    plt = F.plt
    rows = sorted(MODELS, key=lambda r: metric(r[0], "trained_calibrated")[1])
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.71, bottom=0.18)
    refs = {"scorecard": metric("scorecard", "trained")[1],
            "gbm": metric("gbm", "trained_calibrated")[1]}  # fmt: skip
    ax.axvspan(0, CHANCE, color=F.GRID, alpha=0.55, lw=0, zorder=0)
    for y, (c, _zs, _label, _how) in enumerate(rows):
        col = colour(c, dark)
        raw, cal = metric(c, "trained")[1], metric(c, "trained_calibrated")[1]
        ax.plot([raw, cal], [y, y], color=col, lw=2.5, alpha=0.35, zorder=2)
        ax.plot(raw, y, "o", ms=9, color=F.SURFACE, mec=col, mew=2, zorder=3)
        ax.plot(cal, y, "o", ms=9, color=col, mec=col, zorder=4)
    reference_lines(ax, refs)
    ax.text(CHANCE + 0.002, -0.62, "Shaded: within chance for a perfectly honest model",
            fontsize=8, color=F.INK_2, va="center")  # fmt: skip
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=F.INK)
    ax.set_yticks(range(len(rows)), [lab for _, _, lab, _ in rows])
    ax.set_ylim(len(rows) - 0.4, -1.0)
    ax.set_xlim(0, 0.115)
    ax.set_xlabel("Calibration error (ECE): average gap between promised and actual risk")
    leg1 = ax.legend(handles=dot_handles(plt, "Raw output", "After the same correction"),
                     loc="lower left", bbox_to_anchor=(-0.27, 1.1), ncol=2, fontsize=9.5,
                     handletextpad=0.3, columnspacing=1.4)  # fmt: skip
    ax.add_artist(leg1)
    ax.legend(handles=reference_legend(ax, plt, with_neighbours=False), loc="lower left",
              bbox_to_anchor=(-0.27, 1.01), ncol=2, fontsize=9.5, handlelength=2.2,
              columnspacing=1.2)  # fmt: skip
    F.header(fig, "After the same correction, all but one are honest",
             "Calibration error before and after a correction learned from 2,000 separate\n"
             "customers, applied the same way to every model. Laya's raw output already\n"
             "includes its maker's own correction; Kev's maker step was not applied.")  # fmt: skip
    F.footer(fig, "\n".join((
        TEST_LINE,
        "The correction (Platt or isotonic, chosen by cross-validation) uses the 2,000 tune "
        "customers only.",
        "Lines: the scorecard as built, gradient boosting after the same correction.",
    )))  # fmt: skip
    return F.save(fig, "m6_honesty.png")


def deciles(contestant: str, condition: str = "trained") -> tuple[np.ndarray, np.ndarray, dict]:
    y, p, _ = load(contestant, condition)
    dec = m.reliability_by_decile(y, p)
    stats = {"auc": m.auc(y, p), "ece": m.ece(y, p), "mean": float(p.mean())}
    return np.array([d["promised"] for d in dec]), np.array([d["actual"] for d in dec]), stats


def reliability(dark: bool = False) -> Path:
    """Promised vs actual in ten risk groups, raw output, one panel per model (deep dive)."""
    F.set_theme(dark)
    plt = F.plt
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.17, wspace=0.12, hspace=0.42)
    gx, gy, _ = deciles("gbm")
    for ax, (c, _zs, name, how) in zip(axes.flat, MODELS, strict=True):
        px, py, st = deciles(c)
        ax.plot([0, 0.8], [0, 0.8], color=F.AXIS, lw=1, zorder=1)
        ax.plot(gx, gy, color=F.MUTED, lw=1.5, alpha=0.6, ls=(0, (4, 2)), zorder=2)
        col = colour(c, dark)
        ax.plot(px, py, color=col, lw=2, zorder=3)
        ax.plot(px, py, "o", ms=5.5, color=col, mec=F.SURFACE, mew=1.2, zorder=4)
        ax.set_title(name, loc="left", fontsize=11.5, pad=16)
        ax.text(0, 1.03, f"{how}  ·  AUC {st['auc']:.3f}  ·  ECE {st['ece']:.3f}",
                transform=ax.transAxes, fontsize=8, color=F.INK_2)  # fmt: skip
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(labelsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Promised default rate", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual default rate", fontsize=9.5)
    F.header(fig, "After learning from the same 20,000 customers",
             "Promised vs actual default rate in ten risk groups, raw output before the shared "
             "correction. Diagonal: perfectly honest.\nDashed grey line in every panel: gradient "
             "boosting, trained on the same 20,000.")  # fmt: skip
    F.footer(fig, TEST_LINE + " " + HOW_LINES[0] + "\n" + HOW_LINES[1])
    return F.save(fig, "m6_reliability.png")


@cache
def gap(contestant: str, ref: str) -> tuple[float, float, float]:
    """Contestant minus reference AUC with its paired bootstrap interval, widened for comparing
    the eight models with each reference (Bonferroni). Same resamples as analysis/fair_fight.py,
    so it agrees with the post's Holm wording: only Kev-4B clears the scorecard."""
    y, p, _ = load(contestant, "trained")
    _, p_ref, _ = load(ref, "trained")
    d = paired_auc(y, p_ref, p, alpha=0.05 / len(MODELS))
    return -d["difference"], -d["hi"], -d["lo"]


def how_close(dark: bool = False) -> Path:
    """Paired AUC gaps to gradient boosting and to the scorecard, one panel each."""
    F.set_theme(dark)
    plt = F.plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.6), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.76, bottom=0.2, wspace=0.1)
    refs = (("gbm", "Against gradient boosting", "-"),
            ("scorecard", "Against the scorecard", (0, (4, 3))))  # fmt: skip
    for ax, (ref, title, ls) in zip(axes, refs, strict=True):
        ax.axvline(0, color=F.COLOR[ref], lw=1.4, ls=ls, zorder=1)
        for y, (c, _zs, _label, _how) in enumerate(MODELS):
            est, lo, hi = gap(c, ref)
            col = colour(c, dark)
            ax.plot([lo, hi], [y, y], color=col, lw=3, alpha=0.45, solid_capstyle="round",
                    zorder=2)  # fmt: skip
            ax.plot(est, y, "o", ms=9, color=col, mec=F.SURFACE, mew=1.5, zorder=3)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlim(-0.085, 0.045)
        ax.set_xticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Difference in ranking (AUC)")
        for x, ha, text in ((-0.003, "right", "ranks worse"), (0.003, "left", "ranks better")):
            ax.text(x, len(MODELS) - 0.55, text, ha=ha, va="center", fontsize=8.5,
                    color=F.INK_2)  # fmt: skip
    axes[0].set_yticks(range(len(MODELS)), [lab for _, _, lab, _ in MODELS])
    axes[0].set_ylim(len(MODELS) - 0.3, -0.6)
    axes[0].tick_params(axis="y", length=0, labelsize=10.5, labelcolor=F.INK)
    axes[1].tick_params(axis="y", length=0)
    F.header(fig, "How close is close?",
             "Each model's ranking minus the reference's, on the same 2,000 customers. Bars: the\n"
             "range a resampling test can't rule out, widened for comparing eight models at once.\n"
             "A bar that crosses the line can't be told apart from the reference.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE,
                             "Paired bootstrap, 2,000 resamples; intervals at 99.4% "
                             "(95% with a Bonferroni correction for eight models).")))  # fmt: skip
    return F.save(fig, "m6_how_close.png")


def kev_calibration(dark: bool = False) -> Path:
    """Kev straight out of training (open) and after the shared correction (filled)."""
    F.set_theme(dark)
    plt = F.plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.7, bottom=0.18, wspace=0.12)
    for ax, (c, name) in zip(axes, (("kev-0.8b-ft", "Kev-0.8B"), ("kev-4b-ft", "Kev-4B")),
                             strict=True):  # fmt: skip
        col = colour(c, dark)
        rx, ry, rs = deciles(c, "trained")
        cx, cy, cs = deciles(c, "trained_calibrated")
        ax.plot([0, 0.8], [0, 0.8], color=F.AXIS, lw=1, zorder=1)
        ax.text(0.62, 0.6, "Perfectly honest", rotation=45, rotation_mode="anchor",
                fontsize=8.5, color=F.INK_2, ha="center", va="top")  # fmt: skip
        ax.plot(rx, ry, color=col, lw=1.5, alpha=0.5, ls=(0, (3, 2)), zorder=2)
        ax.plot(rx, ry, "o", ms=7, color=F.SURFACE, mec=col, mew=1.8, zorder=3)
        ax.plot(cx, cy, color=col, lw=2, zorder=4)
        ax.plot(cx, cy, "o", ms=7, color=col, mec=F.SURFACE, mew=1.2, zorder=5)
        ax.set_title(name, loc="left", fontsize=12, pad=18)
        ax.text(0, 1.03, f"{rs['ece'] * 100:.1f} points off straight out of training, "
                f"{cs['ece'] * 100:.1f} after the correction", transform=ax.transAxes,
                fontsize=8.5, color=F.INK_2)  # fmt: skip
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.xaxis.set_major_formatter(F.PCT)
        ax.yaxis.set_major_formatter(F.PCT)
        ax.set_xlabel("Promised default rate")
    # The one point the text quotes: Kev-0.8B's sixth risk group.
    rx, ry, _ = deciles("kev-0.8b-ft", "trained")
    axes[0].annotate(f"Promised {rx[5]:.0%},\n{ry[5]:.0%} defaulted", (rx[5], ry[5]),
                     xytext=(0.2, 0.42), fontsize=9, color=F.INK, ha="left",
                     arrowprops={"arrowstyle": "-", "color": F.INK_2, "lw": 0.8})  # fmt: skip
    axes[0].set_ylabel("Actual default rate")
    axes[0].legend(handles=dot_handles(plt, "Straight out of training",
                                       "After the same correction"),
                   loc="lower left", bbox_to_anchor=(0, 1.11), ncol=2, fontsize=9.5,
                   handletextpad=0.3, columnspacing=1.4)  # fmt: skip
    F.header(fig, "Training is not calibration",
             "Promised vs actual default rate in ten risk groups. Straight out of training, "
             "both Kev\n"
             "models promise too little in the middle groups. The same correction fixes Kev-0.8B,\n"
             "but not Kev-4B.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE,
                             "The correction (Platt or isotonic, chosen by cross-validation) uses "
                             "the 2,000 tune customers only.")))  # fmt: skip
    return F.save(fig, "m6_kev_calibration.png")


def cost_per_1000(contestant: str, condition: str = "trained") -> float:
    costs = [r["cost_usd"] for r in read_records(contestant)
             if r["split"] == "test" and r["condition"] == condition]  # fmt: skip
    return 1000 * float(np.mean(costs))


COST_LABEL = {  # (dx, dy) label offsets in points, placed by eye to avoid collisions
    "kumo-tabular": (0, 11), "tabicl-v2": (0, 11), "kev-4b-ft": (0, 11),
    "kev-0.8b-ft": (0, -14), "qwen-ft": (9, -11), "laya-ft": (0, -14),
    "sol-examples": (-10, 0), "jev-examples": (10, 0),
    "gbm": (10, 0), "scorecard": (10, 0), "neighbours-20": (10, 0),
}  # fmt: skip


def cost(dark: bool = False) -> Path:
    """Ranking against running cost per 1,000 decisions, after learning from the 20,000."""
    F.set_theme(dark)
    plt = F.plt
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.83, bottom=0.2)
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.5)
    points = [(c, name, colour(c, dark)) for c, _zs, name, _how in MODELS] + [
        ("gbm", "Gradient boosting", F.COLOR["gbm"]),
        ("scorecard", "Scorecard", F.COLOR["scorecard"]),
        ("neighbours-20", "20 neighbours, averaged", F.MUTED),
    ]
    for c, name, col in points:
        x, y = cost_per_1000(c), metric(c, "trained")[0]
        ax.plot(x, y, "o", ms=10, color=col, mec=F.SURFACE, mew=1.8, zorder=3)
        dx, dy = COST_LABEL[c]
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=9.5, color=F.INK)  # fmt: skip
    ax.set_xlim(-0.00003, 40)
    ax.set_ylim(0.735, 0.805)
    ax.set_xticks([0, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "$0" if v == 0 else f"${v:g}"))
    ax.set_xlabel("Running cost per 1,000 decisions (US dollars, log scale)")
    ax.set_ylabel("Ranking (AUC)")
    F.header(fig, "Paying more didn't buy better ranking",
             "Ranking against running cost, after learning from the 20,000 customers.\n"
             "Fine-tuning also cost about a dollar of GPU time at most, "
             "once per model.")  # fmt: skip
    F.footer(fig, "\n".join((
        TEST_LINE,
        "Open models priced at rented L4 GPU rates; GPT-6.2 Sol and Jev at their API prices.",
        "The baselines and the neighbours average cost next to nothing to run once built.",
    )))  # fmt: skip
    return F.save(fig, "m6_cost.png")


def sol_neighbours(dark: bool = False) -> Path:
    """Each answer against how many of the customer's 20 look-alikes defaulted."""
    F.set_theme(dark)
    plt = F.plt
    _, p_nb, ids = load("neighbours-20", "trained")
    k = np.rint(p_nb * 20).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.7, bottom=0.2, wspace=0.12)
    rng = np.random.default_rng(0)  # horizontal jitter only, so the dots don't stack
    for ax, (c, name) in zip(axes, (("sol-examples", "GPT-6.2 Sol"), ("jev-examples", "Jev")),
                             strict=True):  # fmt: skip
        _, p, ids_c = load(c, "trained")
        assert ids_c == ids
        col = colour(c, dark)
        ax.scatter(k + rng.uniform(-0.3, 0.3, len(k)), p, s=8, color=col, alpha=0.22, lw=0,
                   zorder=2)  # fmt: skip
        ks = [v for v in range(21) if (k == v).sum() >= 10]
        med = [float(np.median(p[k == v])) for v in ks]
        ax.plot(ks, med, color=col, lw=2, zorder=4)
        ax.plot(ks, med, "o", ms=6, color=col, mec=F.SURFACE, mew=1.2, zorder=5)
        ax.plot([0, 20], [0, 1], color=F.INK_2, lw=1.3, ls=(0, (4, 3)), zorder=3)
        r = np.corrcoef(p, p_nb)[0, 1]
        ax.set_title(name, loc="left", fontsize=12, pad=18)
        ax.text(0, 1.03, f"Ranking (AUC) {metric(c, 'trained')[0]:.3f}  ·  correlation with "
                f"the count {r:.2f}", transform=ax.transAxes, fontsize=8.5,
                color=F.INK_2)  # fmt: skip
        ax.set_xlim(-0.7, 20.7)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.yaxis.set_major_formatter(F.PCT)
        ax.set_xlabel("How many of its 20 look-alikes defaulted")
    axes[0].set_ylabel("The model's answer: chance of default")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=5, color=F.INK_2, alpha=0.5,
                   label="One test customer"),
        plt.Line2D([], [], marker="o", ms=6, lw=2, color=F.INK_2, mec=F.SURFACE,
                   label="Middle answer (median)"),
        plt.Line2D([], [], color=F.INK_2, lw=1.3, ls=(0, (4, 3)),
                   label="Just counting (3 of 20 defaulted: 15%)"),
    ]  # fmt: skip
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.11), ncol=3,
                   fontsize=9.5, handletextpad=0.4, columnspacing=1.4)  # fmt: skip
    F.header(fig, "Shown 20 look-alikes, Sol mostly counts them",
             "Each model saw its customer's 20 most similar training customers and what "
             "happened\nto them. Counting how many defaulted ranks at 0.764 on its own. "
             "Sol's answers follow\nthe count closely; Jev's start higher, around 20% when "
             "only 1 of the 20 defaulted.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE,
                             "Medians shown where at least 10 customers share a count. "
                             "Raw answers, before any correction.")))  # fmt: skip
    return F.save(fig, "m6_sol_neighbours.png")


CEILING = ("kumo-tabular", "tabicl-v2", "kev-4b-ft", "kev-0.8b-ft", "gbm", "qwen-ft", "laya-ft",
           "scorecard")  # fmt: skip
CEILING_NAME = {c: lab for c, _zs, lab, _how in MODELS} | {"gbm": "Gradient boosting",
                                                          "scorecard": "Scorecard"}  # fmt: skip


def ceiling(dark: bool = False) -> Path:
    """Every model that learned from the 20,000, on the full scale from coin flip to perfect, with
    the scorecard trained on 2,000 (results/robustness.json) for the value of ten times the data."""
    import json

    F.set_theme(dark)
    plt = F.plt
    small = json.loads((ROOT / "results" / "robustness.json").read_text("utf-8"))
    small = small["scorecard_on_2000"]["auc"]
    rows = sorted(CEILING, key=lambda c: -metric(c, "trained")[0])
    aucs = {c: metric(c, "trained")[0] for c in rows}
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.75, bottom=0.17)
    lo, hi = min(aucs.values()), max(aucs.values())
    ax.axvspan(lo, hi, color=F.GRID, alpha=0.8, lw=0, zorder=0)
    ax.text((lo + hi) / 2, -0.85, f"all eight: {lo:.3f} to {hi:.3f}", ha="center", va="bottom",
            fontsize=9, color=F.INK_2)  # fmt: skip
    for y, c in enumerate(rows):
        col = F.COLOR[c] if c in F.BASELINES else colour(c, dark)
        ax.plot(aucs[c], y, "o", ms=10, color=col, mec=F.SURFACE, mew=1.5, zorder=3)
        ax.text(hi + 0.012, y, f"{aucs[c]:.3f}", va="center", fontsize=9.5, color=F.INK_2)
    ys = len(rows) + 0.5
    ax.plot(small, ys, "o", ms=10, color=F.SURFACE, mec=F.COLOR["scorecard"], mew=2, zorder=3)
    ax.text(small - 0.012, ys, f"{small:.3f}", ha="right", va="center", fontsize=9.5,
            color=F.INK_2)  # fmt: skip
    ax.annotate("", (aucs["scorecard"], len(rows) - 1 + 0.25), (small, ys - 0.25),
                arrowprops={"arrowstyle": "->", "color": F.INK_2, "lw": 1})  # fmt: skip
    ax.text(hi + 0.012, ys - 0.5, f"10 times the data:\n+{aucs['scorecard'] - small:.3f}",
            va="center", fontsize=9, color=F.INK_2)  # fmt: skip
    ax.text(0.99, len(rows) / 2 - 0.5, "None of the eight\ngets past 0.80", ha="right",
            va="center", fontsize=10, color=F.INK_2)  # fmt: skip
    labels = [CEILING_NAME[c] for c in rows] + ["Scorecard, 2,000 customers"]
    ax.set_yticks([*range(len(rows)), ys], labels)
    ax.set_ylim(ys + 0.6, -1.1)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=F.INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  ["0.5\ncoin flip", "0.6", "0.7", "0.8", "0.9", "1.0\nperfect"])  # fmt: skip
    ax.set_xlabel("Ranking (AUC)")
    F.header(fig, "Eight very different models hit the same ceiling",
             "Ranking on the full scale, from a coin flip to perfect. Every model that learned\n"
             "from the 20,000 customers lands within 0.02 of the others. A scorecard that saw\n"
             f"a tenth of the data, 2,000 customers, is only {aucs['scorecard'] - small:.3f} "
             "behind.")  # fmt: skip
    F.footer(fig, "\n".join((TEST_LINE,
                             "The 2,000-customer scorecard is the first post's check, trained on "
                             "the 2,000 tune customers.")))  # fmt: skip
    return F.save(fig, "m6_ceiling.png")


FIGURES = (catch_up, honesty, reliability, how_close, kev_calibration, cost, sol_neighbours,
           ceiling)  # fmt: skip


def main() -> None:
    names = set(sys.argv[1:])  # optional: only these, e.g. `cost how_close`
    for dark in (False, True):
        for fig in FIGURES:
            if not names or fig.__name__ in names:
                print(f"Saved -> {fig(dark)}")


if __name__ == "__main__":
    main()
