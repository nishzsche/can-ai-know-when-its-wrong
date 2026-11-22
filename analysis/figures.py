"""The five figures (PLAN.md section 5), from results/summary.csv and summary_detail.json.

Run analysis/summarize.py first. Contestants without results (e.g. Sol before its run
finishes) are drawn as "pending" panels or left out, so this can run again later.

    uv run python analysis/figures.py

Design: reference palette (dataviz skill, light mode), first six categorical slots in their
documented order for the AI contestants; baselines in neutral grays as references. One colour
per contestant across every figure. Direct labels everywhere (three of the slots sit below 3:1
contrast on the light surface). Sized for phones (4:5 portrait or 1:1).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import ROOT  # noqa: E402

SUMMARY_CSV = ROOT / "results" / "summary.csv"
DETAIL_JSON = ROOT / "results" / "summary_detail.json"
FIG_DIR = ROOT / "results" / "figures"

# ---- tokens (reference palette, light) ----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

AI = ("jev", "kev-4b", "kev-0.8b", "laya", "qwen-logprobs", "frontier-verbal")
BASELINES = ("scorecard", "gbm")
COLOR = {
    "jev": "#2a78d6",  # slot 1 blue
    "kev-4b": "#eb6834",  # slot 2 orange
    "kev-0.8b": "#1baf7a",  # slot 3 aqua
    "laya": "#eda100",  # slot 4 yellow
    "qwen-logprobs": "#e87ba4",  # slot 5 magenta
    # Slot 7 violet at its earlier, darker step. The slot 6 green sat next to Kev-0.8B's aqua
    # (normal-vision dE 15.6) and was nearly identical to Kev-4B's orange for red-green colour
    # blindness (dE 3.2); this violet's nearest pairs are Jev (16.3) and the grey scorecard (CVD
    # 7.0, which every chart's name labels cover).
    "frontier-verbal": "#4a3aa7",
    "scorecard": "#52514e",  # neutral reference
    "gbm": "#898781",  # neutral reference
}
NAME = {
    "jev": "Jev", "kev-4b": "Kev-4B", "kev-0.8b": "Kev-0.8B", "laya": "Laya",
    "qwen-logprobs": "Qwen3.5-4B", "frontier-verbal": "GPT-6.2 Sol",
    "scorecard": "Scorecard", "gbm": "Gradient boosting",
}  # fmt: skip
BASE_CONDITION = {"scorecard": "trained", "gbm": "trained"}
FOOTNOTE = (
    "Test set: 2,000 real credit card customers (UCI, Taiwan 2005), 22.1% defaulted.\n"
    "AI contestants are zero-shot; the scorecard and gradient boosting learned from 20,000 "
    "labelled customers."
)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 11, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })  # fmt: skip


def load() -> tuple[dict, dict]:
    rows = {}
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows[(r["contestant"], r["condition"])] = r
    return rows, json.loads(DETAIL_JSON.read_text("utf-8"))


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.975, title, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.938, subtitle, fontsize=11, color=INK_2, va="top")


def footer(fig, text: str = FOOTNOTE) -> None:
    fig.text(0.04, 0.012, text, fontsize=8.5, color=MUTED, va="bottom")


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


PCT = PercentFormatter(1.0, decimals=0)


# ---- 1. hero: promised vs actual by decile ------------------------------------------------------


def fig_reliability(rows, detail, condition: str = "out_of_box") -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.12, hspace=0.32, wspace=0.12)
    ref = detail.get("scorecard", {}).get("trained", {}).get("deciles")
    top = max(max(x["promised"], x["actual"]) for c in (*AI, "scorecard") for d in
              detail.get(c, {}).values() for x in d["deciles"])  # fmt: skip
    lim = 0.8 if top < 0.76 else (0.9 if top < 0.86 else 1.0)
    ticks = [0, 0.2, 0.4, 0.6, 0.8]  # no 100% tick: it would collide with the next panel's 0%
    for ax, c in zip(axes.flat, AI, strict=True):
        ax.plot([0, lim], [0, lim], color=AXIS, lw=1.2, zorder=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if ref:
            ax.plot([d["promised"] for d in ref], [d["actual"] for d in ref],
                    color=COLOR["scorecard"], lw=1.2, alpha=0.55, zorder=2)  # fmt: skip
        d = detail.get(c, {}).get(condition)
        if not d:
            ax.set_title(NAME[c], loc="left", color=MUTED)
            ax.text(lim / 2, lim / 2, "results pending", ha="center", va="center", color=MUTED)
            continue
        dec = d["deciles"]
        ax.plot([x["promised"] for x in dec], [x["actual"] for x in dec], color=COLOR[c], lw=2,
                marker="o", ms=5.5, mec=SURFACE, mew=1.2, zorder=3)  # fmt: skip
        r = rows[(c, condition)]
        ax.set_title(NAME[c], loc="left")
        ax.text(0.03, 0.97, f"says {r['mean_predicted']:.0%} on average\nECE {r['ece']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK_2)  # fmt: skip
    for ax in axes[-1]:
        ax.set_xlabel("Promised default rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual default rate")
    header(fig, "Promised vs actual default rate",
           "Each dot is a tenth of the customers, ranked by predicted risk.\n"
           "On the diagonal = honest. Out of the box; gray line: the scorecard.")  # fmt: skip
    footer(fig)
    return save(fig, "1_promised_vs_actual.png")


# ---- 2. the honesty arc -------------------------------------------------------------------------

STAGES = (("out_of_box", "Out of the box", "o", "none"),
          ("base_rate", "Told the base rate", "^", None),
          ("recalibrated", "Recalibrated on tune", "o", None))  # fmt: skip


def fig_honesty_arc(rows) -> Path:
    present = [c for c in AI if (c, "out_of_box") in rows]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.17)
    order = sorted(present, key=lambda c: rows[(c, "out_of_box")]["ece"])
    for y, c in enumerate(order):
        xs = [rows[(c, s)]["ece"] for s, *_ in STAGES if (c, s) in rows]
        ax.plot([min(xs), max(xs)], [y, y], color=COLOR[c], lw=2, alpha=0.35, zorder=2)
        for s, _, marker, face in STAGES:
            if (c, s) in rows:
                dy = 0.17 if s == "base_rate" else 0.0  # keep coincident markers visible
                ax.plot(rows[(c, s)]["ece"], y + dy, marker=marker, ms=10 if marker == "o" else 9,
                        color=COLOR[c], mfc=face or COLOR[c], mew=2, mec=COLOR[c], ls="none",
                        zorder=3)  # fmt: skip
    ref = rows.get(("scorecard", "trained"))
    if ref:
        ax.axvline(ref["ece"], color=COLOR["scorecard"], lw=1.2, zorder=1)
        ax.text(ref["ece"] + 0.004, len(order) - 0.45, f"Scorecard {ref['ece']:.3f}",
                color=INK_2, fontsize=9, va="center")  # fmt: skip
    ax.set_yticks(range(len(order)), [NAME[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.01, max(0.3, max(rows[(c, s)]["ece"] for c in order for s, *_ in STAGES
                                if (c, s) in rows) * 1.08))  # fmt: skip
    ax.set_xlabel("Calibration error (ECE): lower is more honest")
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=9, color=INK_2,
                          mfc=f or INK_2, mew=2, mec=INK_2, label=lab)
               for _, lab, mk, f in STAGES]  # fmt: skip
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.09), ncol=3,
              fontsize=9.5, handletextpad=0.3, columnspacing=1.2)  # fmt: skip
    header(fig, "The honesty arc",
           "How far each AI's probabilities are from the truth, and what fixes it.\n"
           "Recalibration needs labelled data (2,000 tune customers).")  # fmt: skip
    footer(fig)
    return save(fig, "2_honesty_arc.png")


# ---- 3. ranking: AUC and KS ---------------------------------------------------------------------


def _ranking_rows(rows) -> list[tuple[str, str]]:
    keys = [(c, BASE_CONDITION[c]) for c in BASELINES] + [(c, "out_of_box") for c in AI]
    keys = [k for k in keys if k in rows]
    return sorted(keys, key=lambda k: rows[k]["auc"])


def fig_ranking(rows) -> Path:
    keys = _ranking_rows(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 7.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.35, 1]})  # fmt: skip
    fig.subplots_adjust(left=0.24, right=0.96, top=0.8, bottom=0.17, wspace=0.12)
    ys = range(len(keys))
    for y, k in zip(ys, keys, strict=True):
        r, c = rows[k], k[0]
        ax1.barh(y, r["auc"] - 0.5, left=0.5, height=0.5, color=COLOR[c], zorder=2)
        ax1.plot([r["auc_lo"], r["auc_hi"]], [y, y], color=INK, lw=1.2, zorder=3)
        ax1.text(r["auc_hi"] + 0.006, y, f"{r['auc']:.3f}", va="center", fontsize=9.5, color=INK)
        ax2.barh(y, r["ks"], height=0.5, color=COLOR[c], zorder=2)
        ax2.text(r["ks"] + 0.008, y, f"{r['ks']:.2f}", va="center", fontsize=9.5, color=INK)
    ax1.set_yticks(list(ys), [NAME[k[0]] + ("*" if k[0] in BASELINES else "") for k in keys])
    ax1.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ax1.set_xlim(0.5, 0.86)
    ax1.set_xlabel("AUC (0.5 = coin flip)")
    ax1.set_title("AUC with 95% interval", loc="left", fontsize=11)
    ax2.set_xlim(0, 0.55)
    ax2.set_xlabel("KS statistic")
    ax2.set_title("KS", loc="left", fontsize=11)
    header(fig, "Can it tell risky from safe customers?",
           "Ranking ability, out of the box. *Trained on 20,000 labelled customers.\n"
           "Recalibration doesn't change ranking.")  # fmt: skip
    footer(fig)
    return save(fig, "3_ranking_auc_ks.png")


# ---- 4. risk bands table ------------------------------------------------------------------------


def fig_risk_bands(rows, detail) -> Path:
    keys = [(c, BASE_CONDITION[c]) for c in BASELINES] + [(c, "out_of_box") for c in AI]
    keys = [k for k in keys if k[0] in detail and k[1] in detail[k[0]]]
    bands = detail[keys[0][0]][keys[0][1]]["risk_bands"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.17)
    ax.axis("off")
    ax.grid(False)
    ncol = len(bands)
    x0, colw = 0.27, (1 - 0.27) / ncol
    rowh = 0.8 / (len(keys) + 1)
    ytop = 0.95
    ax.text(0.0, ytop, "Predicted risk →", fontsize=10, color=MUTED, va="center")
    for j, b in enumerate(bands):
        ax.text(x0 + (j + 0.5) * colw, ytop, b["band"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)  # fmt: skip
    for i, (c, cond) in enumerate(keys):
        y = ytop - (i + 1) * rowh
        ax.plot([0, 1], [y + rowh / 2, y + rowh / 2], color=GRID, lw=0.8)
        ax.plot([0.0, 0.0], [y - 0.022, y + 0.022], color=COLOR[c], lw=5, solid_capstyle="butt")
        ax.text(0.02, y, NAME[c] + ("*" if c in BASELINES else ""), va="center", fontsize=11,
                color=INK)  # fmt: skip
        for j, b in enumerate(detail[c][cond]["risk_bands"]):
            cx = x0 + (j + 0.5) * colw
            if not b["n"]:
                ax.text(cx, y, "—", ha="center", va="center", color=MUTED)
                continue
            share = "<1%" if b["share"] < 0.005 else f"{b['share']:.0%}"
            ax.text(cx, y + 0.016, f"{share} of customers", ha="center", va="center",
                    fontsize=8.5, color=INK_2)  # fmt: skip
            ax.text(cx, y - 0.017, f"{b['promised']:.0%} → {b['actual']:.0%}", ha="center",
                    va="center", fontsize=10.5, color=INK, fontweight="bold")  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(ytop - (len(keys) + 0.6) * rowh, 1.0)
    header(fig, "What a bank would see",
           "Customers grouped by predicted risk, out of the box: promised → actual default "
           "rate.\n*Trained on 20,000 labelled customers.")  # fmt: skip
    footer(fig)
    return save(fig, "4_risk_bands.png")


# ---- 5. cost vs accuracy ------------------------------------------------------------------------


LABEL_LEFT = {"kev-0.8b"}  # its right-hand label would run into Qwen's dot


def fig_cost(rows) -> Path:
    keys = [(c, BASE_CONDITION[c]) for c in BASELINES] + [(c, "out_of_box") for c in AI]
    keys = [k for k in keys if k in rows]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.8, bottom=0.17)
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.6)
    for c, cond in keys:
        r = rows[(c, cond)]
        x = r["cost_per_1000_usd"]
        ax.plot(x, r["auc"], "o", ms=11, color=COLOR[c], mec=SURFACE, mew=2, zorder=3)
        left = c in LABEL_LEFT
        ax.annotate(NAME[c], (x, r["auc"]), xytext=(-9 if left else 9, 0),
                    textcoords="offset points", ha="right" if left else "left",
                    va="center", fontsize=10, color=INK)  # fmt: skip
    ax.set_xlim(-0.0002, 10)
    ax.set_ylim(0.55, 0.82)
    ax.set_xticks([0, 0.001, 0.01, 0.1, 1, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "$0" if v == 0 else f"${v:g}"))
    ax.set_xlabel("Cost per 1,000 decisions (log scale; local models at L4 GPU rates)")
    ax.set_ylabel("AUC (ranking ability)")
    header(fig, "Cost vs ranking ability",
           "Out of the box. The baselines cost ~$0 to run once trained.")  # fmt: skip
    footer(fig)
    return save(fig, "5_cost_vs_auc.png")


def main() -> None:
    style()
    rows, detail = load()
    for path in (fig_reliability(rows, detail), fig_honesty_arc(rows), fig_ranking(rows),
                 fig_risk_bands(rows, detail), fig_cost(rows)):  # fmt: skip
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
