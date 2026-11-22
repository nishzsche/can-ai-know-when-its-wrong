"""Disguise check: how much of GPT-6.2 Sol's calibration comes from recognizing the dataset?

Sol names the UCI "Default of Credit Card Clients" dataset for every account summary and knows
its default rate (data/contamination.md, recognition probe). It cites the NT$ amounts, the
April-September 2005 months and the repayment codes. This check removes those cues and asks Sol
again, out of the box, on the same test customers:

- Currency: "NT$" is dropped and one line says "Amounts are in local currency units."
- Months: "Sep 2005" ... "Apr 2005" become "Month 6" ... "Month 1" (the header still says
  "most recent first").
- Repayment codes: the raw numbers go. Codes -2 and 0, both "not late", stay distinct as
  "type A" and "type B"; "paid on time" and "N months late" keep their words.

Everything else (the question, the answer line, model settings, pacing) is the contestant's.
The disguise also removes real context (Taiwan, 2005), so a change measures recognition and
context together. It's a diagnostic, not a contestant condition: output goes to
results/checks/, and the headline results are unchanged.

    # does Sol still recognize the disguised summaries? (same 50 customers as the original probe)
    uv run --group runners python runners/checks/sol_disguise.py --probe 50
    # 2,000 test customers, out of the box, resumable
    uv run --group runners python runners/checks/sol_disguise.py --workers 4
    # original vs disguised on the same customers
    uv run --group runners python runners/checks/sol_disguise.py --report
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "data"))
sys.path.insert(0, str(HERE.parents[2] / "analysis"))

from frontier_verbal import DEFAULT_RPM, FrontierVerbalPredictor  # noqa: E402
from harness import ROOT, build_parser, load_requests, resolve_args, run  # noqa: E402
from render import MONTHS, STATUS_TEXT  # noqa: E402

DISGUISE_VERSION = "d1"
CHECK_DIR = ROOT / "results" / "checks" / "disguise"
PROBE_JSON = ROOT / "results" / "checks" / "sol_disguise_recognition.json"
SUMMARY_JSON = ROOT / "results" / "checks" / "sol_disguise_summary.json"
HEADER = "Credit card account summary"
UNITS_LINE = "Amounts are in local currency units."
MONTH_LABELS = {m: f"Month {len(MONTHS) - i}" for i, m in enumerate(MONTHS)}


def _plain_status(code: int, label: str) -> str:
    if code == -2:
        return "not late (type A)"
    if code == 0:
        return "not late (type B)"
    return re.sub(r" \(code -?\d+\)$", "", label)


STATUS_MAP = {label: _plain_status(code, label) for code, label in STATUS_TEXT.items()}


def disguise(state: str) -> str:
    """The same summary without the currency, the dates or the raw repayment codes."""
    out = []
    for line in state.split("\n"):
        line = line.replace("NT$", "")
        for month, label in MONTH_LABELS.items():
            if line.startswith(f"{month} |"):
                line = label + line[len(month) :]
        head, sep, status = line.rpartition(" | ")
        if sep and status in STATUS_MAP:
            line = f"{head} | {STATUS_MAP[status]}"
        out.append(line)
        if line == HEADER:
            out.append(UNITS_LINE)
    text = "\n".join(out)
    for cue in ("NT$", "2005", "(code"):
        if cue in text:
            raise ValueError(f"disguise left {cue!r} in the summary")
    return text


class DisguisedSol(FrontierVerbalPredictor):
    contestant = "frontier-verbal-disguised"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_version = f"{self.model_version}; disguise {DISGUISE_VERSION}"
        self.meta = {**self.meta, "disguise": DISGUISE_VERSION, "units_line": UNITS_LINE,
                     "month_labels": MONTH_LABELS, "status_map": STATUS_MAP}  # fmt: skip

    def predict(self, request: dict):
        return super().predict({**request, "state": disguise(request["state"])})


def probe(n: int) -> None:
    """The recognition question on disguised summaries (same customers as the original probe)."""
    import recognition as rg
    from env import load_env

    load_env()
    ask = rg.Asker(rg_model(), "https://openrouter.ai/api/v1", dry_run=False)
    rendered = load_requests("test")
    customers = rg.sample_customers(rendered, n, [])
    rows = []
    for r in customers:
        state = disguise(r["requests"]["out_of_box"]["state"])
        got = ask(f"{state}\n\n{rg.RECOGNITION_QUESTION}", rg.RECOGNITION_SCHEMA, "source_guess")
        rows.append({"id": r["id"], "group": "random", **got})
        a = got["answer"] or {}
        print(f"{r['id']}: {a.get('named_dataset')!r} conf {a.get('confidence_percent')} "
              f"rate {a.get('dataset_default_rate_percent')}")  # fmt: skip
    result = {
        "model": rg_model(),
        "disguise": DISGUISE_VERSION,
        "example_state": disguise(customers[0]["requests"]["out_of_box"]["state"]),
        "recognition_question": rg.RECOGNITION_QUESTION,
        "summary": rg.summarize(rows),
        "cost_usd": round(sum(float(x["usage"].get("cost") or 0) for x in rows), 4),
        "recognition": rows,
    }
    PROBE_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    print(f"cost ${result['cost_usd']}; saved -> {PROBE_JSON}")


def rg_model() -> str:
    from frontier_verbal import MODEL

    return MODEL


def report(n_boot: int = 1000, seed: int = 20261122) -> dict:
    """Original vs disguised, out of the box, on the same test customers; paired bootstrap."""
    import metrics as m
    from records import read_records

    orig = {r["id"]: r for r in read_records("frontier-verbal")
            if r["split"] == "test" and r["condition"] == "out_of_box"}  # fmt: skip
    disg = {r["id"]: r for r in read_records(DisguisedSol.contestant, CHECK_DIR)}
    ids = sorted(set(orig) & set(disg))
    y = np.array([orig[i]["label"] for i in ids], dtype=float)
    p0 = np.array([orig[i]["p_default"] for i in ids])
    p1 = np.array([disg[i]["p_default"] for i in ids])
    funcs = {"auc": m.auc, "ece": m.ece, "mean_predicted": m.mean_predicted,
             "citl": m.calibration_in_the_large}  # fmt: skip
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in funcs}
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        if y[idx].min() == y[idx].max():
            continue
        for k, f in funcs.items():
            boots[k].append(f(y[idx], p1[idx]) - f(y[idx], p0[idx]))
    result = {
        "n": len(ids),
        "observed_rate": float(y.mean()),
        "original": m.summarize(y, p0, n_boot=n_boot),
        "disguised": m.summarize(y, p1, n_boot=n_boot),
        "difference_disguised_minus_original": {
            k: {
                "estimate": float(f(y, p1) - f(y, p0)),
                "lo": float(np.percentile(boots[k], 2.5)),
                "hi": float(np.percentile(boots[k], 97.5)),
            }
            for k, f in funcs.items()
        },  # fmt: skip
        "spearman": float(
            np.corrcoef(np.argsort(np.argsort(p0)), np.argsort(np.argsort(p1)))[0, 1]
        ),  # fmt: skip
        "mean_abs_change": float(np.mean(np.abs(p1 - p0))),
        "cost_usd": float(sum(disg[i]["cost_usd"] for i in ids)),
    }
    SUMMARY_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    d = result["difference_disguised_minus_original"]
    print(f"n={result['n']}  actual {result['observed_rate']:.3f}")
    for k in funcs:
        print(f"{k:<15} original {result['original'][k]:.3f}  disguised "
              f"{result['disguised'][k]:.3f}  diff {d[k]['estimate']:+.3f} "
              f"[{d[k]['lo']:+.3f}, {d[k]['hi']:+.3f}]")  # fmt: skip
    print(
        f"rank correlation {result['spearman']:.3f}; mean |change| {result['mean_abs_change']:.3f}"
    )
    print(f"Saved -> {SUMMARY_JSON}")
    return result


def main() -> None:
    ap = build_parser(__doc__)
    ap.set_defaults(split="test", condition="out_of_box")
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM, help="requests-per-minute cap")
    ap.add_argument("--probe", type=int, metavar="N", help="recognition question on N summaries")
    ap.add_argument("--report", action="store_true", help="compare with the original run")
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe)
    if args.report:
        report()
        return
    if args.condition != "out_of_box":
        ap.error("the disguise check is defined for out_of_box only")
    if args.out_dir is None and not args.smoke:
        args.out_dir = CHECK_DIR
    args = resolve_args(args)
    from env import load_env

    load_env()
    run(DisguisedSol(rpm=args.rpm), args)


if __name__ == "__main__":
    main()
