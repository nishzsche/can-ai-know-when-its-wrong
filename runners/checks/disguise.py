"""Disguise check: do the summaries' identifying cues (NT$, 2005 dates, raw codes) change the
AI contestants' answers?

The summaries name the period ("Sep 2005" ... "Apr 2005") and the currency (NT$), and show the
raw repayment codes. GPT-6.2 Sol recognises the dataset from them, and any model could read the
year and country as context (Taiwan's 2005-06 card-debt crisis). This check removes those cues
and asks the same model again, out of the box, on the same test customers:

- Currency: "NT$" is dropped and one line says "Amounts are in local currency units."
- Months: "Sep 2005" ... "Apr 2005" become "Month 6" ... "Month 1" (the header still says
  "most recent first").
- Repayment codes: the raw numbers go. Codes -2 and 0, both "not late", stay distinct as
  "type A" and "type B"; "paid on time" and "N months late" keep their words.

Everything else (the question, model settings, pacing) is the contestant's own. A control run
asks the original summaries again for a fixed sample, which measures run-to-run noise (and, for
Laya, the switch from the L4 GPU to this laptop's CPU), so a disguise effect can be told apart
from noise. A diagnostic, not a contestant condition: output goes to results/checks/disguise/,
and the headline results are unchanged.

    # does Sol still recognise the disguised summaries? (same 50 customers as the original probe)
    uv run --group runners python runners/checks/disguise.py --probe 50
    # 2,000 test customers, disguised (resumable); then the 200-customer control
    uv run --group runners python runners/checks/disguise.py --contestant jev --workers 8
    uv run --group runners python runners/checks/disguise.py --contestant jev --control --workers 8
    uv run --group runners --group laya python runners/checks/disguise.py --contestant laya
    uv run --group runners python runners/checks/disguise.py --workers 4   # Sol (the default)
    # layout check: each bill beside the following month's payment (template v2 pairs each payment
    # with the same month's bill). Tune first, then test, reported next to v2.
    uv run --group runners python runners/checks/disguise.py --contestant jev --variant aligned \
        --split tune   # then the same with --split test
    # post 5, formats (PLAN.md §12): the same values as sentences, a markdown table, CSV or JSON
    # (data/formats.py), 2,000 test customers each, into results/checks/formats/
    uv run --group runners python runners/checks/disguise.py --contestant jev --variant json \
        --workers 8
    # compare with the original runs (disguise, layout and formats)
    uv run --group runners python runners/checks/disguise.py --report
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "data"))
sys.path.insert(0, str(HERE.parents[2] / "analysis"))

from formats import FORMAT_VERSION, FORMATS, render  # noqa: E402
from harness import L4_USD_PER_HOUR, ROOT, build_parser, load_requests, resolve_args, run  # noqa: E402
from render import MONTHS, STATUS_TEXT  # noqa: E402

DISGUISE_VERSION = "d1"
CHECK_DIR = ROOT / "results" / "checks" / "disguise"
PROBE_JSON = ROOT / "results" / "checks" / "sol_disguise_recognition.json"
SUMMARY_JSON = ROOT / "results" / "checks" / "disguise_summary.json"
FORMAT_DIR = ROOT / "results" / "checks" / "formats"  # post 5
FORMAT_SUMMARY_JSON = ROOT / "results" / "checks" / "formats_summary.json"
CONTESTANTS = ("jev", "laya", "frontier-verbal", "kev-4b", "kev-0.8b", "qwen-logprobs")
CONTROL_N, CONTROL_SEED = 200, 20261124
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


LAYOUT_VERSION = "a1"
ALIGNED_HEADER = (
    "Month (most recent first) | Bill amount | Paid in the following month | Repayment status"
)


def align(state: str) -> str:
    """The same summary with each bill shown next to the payment made the following month.

    In this data a month's payment usually settles the previous month's bill (README, data
    quirk 3), but template v2 shows each payment beside the same month's bill, which can look
    like underpayment. Here each bill row shows the payment made the next month. The most recent
    bill's payment isn't in the data yet, and the earliest payment gets its own line, so every
    number in the original summary appears exactly once.
    """
    lines = state.split("\n")
    head = next(k for k, line in enumerate(lines) if line.startswith("Month (most recent first)"))
    rows = [line.split(" | ") for line in lines[head + 1 :] if " | " in line]
    out = [*lines[:head], ALIGNED_HEADER]
    for k, (month, bill, _paid, status) in enumerate(rows):
        following = f"{rows[k - 1][2]} (paid in {rows[k - 1][0]})" if k else "not in the data yet"
        out.append(f"{month} | {bill} | {following} | {status}")
    out.append(f"Earliest payment in the data: {rows[-1][2]} paid in {rows[-1][0]}")
    return "\n".join(out)


def _formatter(fmt: str):
    def transform(state: str) -> str:
        return render(state, fmt)

    transform.__name__ = f"to_{fmt}"
    return transform


VARIANTS = {  # suffix -> (transform, model_version tag)
    "disguised": (disguise, f"disguise {DISGUISE_VERSION}"),
    "aligned": (align, f"layout aligned {LAYOUT_VERSION}"),
    "control": (None, "control: original summaries"),
    **{fmt: (_formatter(fmt), f"format {fmt} {FORMAT_VERSION}") for fmt in FORMATS},  # post 5
}


def wrap(base: type, control: bool = False, variant: str = "disguised") -> type:
    """A contestant's predictor on transformed summaries (control: on the original ones)."""
    suffix = "control" if control else variant
    transform, tag = VARIANTS[suffix]

    class Wrapped(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Kev sets its name per size in __init__; the others name it on the class.
            name = self.__dict__.get("contestant") or base.contestant
            self.contestant = f"{name}-{suffix}"
            self.model_version = f"{self.model_version}; {tag}"
            self.meta = {**getattr(self, "meta", {}), "check": suffix}

        def predict(self, request: dict):
            if transform:
                request = {**request, "state": transform(request["state"])}
            return super().predict(request)

    Wrapped.__name__ = f"{base.__name__}{suffix.title()}"
    return Wrapped


def make_predictor(contestant: str, args, control: bool, variant: str = "disguised"):
    if contestant == "frontier-verbal":
        from frontier_verbal import FrontierVerbalPredictor

        return wrap(FrontierVerbalPredictor, control, variant)(rpm=args.rpm)
    if contestant == "jev":
        from jev import JevPredictor

        return wrap(JevPredictor, control, variant)()
    if contestant == "laya":
        from laya_classic import LayaPredictor

        cost = L4_USD_PER_HOUR if args.device == "cuda" else 0.0
        return wrap(LayaPredictor, control, variant)(args.device, cost)
    if contestant in ("kev-4b", "kev-0.8b"):
        from kev import KevPredictor

        url = args.base_url or "http://127.0.0.1:8009"
        return wrap(KevPredictor, control, variant)(contestant.split("-")[1], url, L4_USD_PER_HOUR)
    if contestant == "qwen-logprobs":
        from qwen_logprobs import SERVED_NAME, QwenPredictor

        url = args.base_url or "http://127.0.0.1:8000/v1"
        return wrap(QwenPredictor, control, variant)(url, SERVED_NAME, L4_USD_PER_HOUR)
    raise SystemExit(f"unknown contestant {contestant!r}; choose from {CONTESTANTS}")


def control_sample(rows: list[dict]) -> list[dict]:
    return sorted(random.Random(CONTROL_SEED).sample(rows, CONTROL_N), key=lambda r: r["id"])


def probe(n: int) -> None:
    """The recognition question on disguised summaries (same customers as the original probe)."""
    import recognition as rg
    from env import load_env
    from frontier_verbal import MODEL

    load_env()
    ask = rg.Asker(MODEL, "https://openrouter.ai/api/v1", dry_run=False)
    customers = rg.sample_customers(load_requests("test"), n, [])
    rows = []
    for r in customers:
        state = disguise(r["requests"]["out_of_box"]["state"])
        got = ask(f"{state}\n\n{rg.RECOGNITION_QUESTION}", rg.RECOGNITION_SCHEMA, "source_guess")
        rows.append({"id": r["id"], "group": "random", **got})
    result = {
        "model": MODEL,
        "disguise": DISGUISE_VERSION,
        "example_state": disguise(customers[0]["requests"]["out_of_box"]["state"]),
        "recognition_question": rg.RECOGNITION_QUESTION,
        "summary": rg.summarize(rows),
        "cost_usd": round(sum(float(x["usage"].get("cost") or 0) for x in rows), 4),
        "recognition": rows,
    }
    PROBE_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


def _records(name: str, directory: Path | None = None, split: str = "test") -> dict[str, dict]:
    from records import read_records

    path = (directory or ROOT / "results" / "predictions") / f"{name}.jsonl"
    if not path.exists():
        return {}
    recs = read_records(name, directory) if directory else read_records(name)
    return {r["id"]: r for r in recs if r["split"] == split and r["condition"] == "out_of_box"}


def report_one(contestant: str, n_boot: int = 1000, seed: int = 20261122,
               variant: str = "disguised", directory: Path = CHECK_DIR,
               control_prefix: str = "disguise") -> dict | None:  # fmt: skip
    """Original vs a variant (disguised, or a post 5 format) on the same customers, with the
    disguise check's control as the noise floor."""
    import metrics as m

    orig = _records(contestant)
    disg = _records(f"{contestant}-{variant}", directory)
    ctrl = _records(f"{contestant}-control", CHECK_DIR)
    ids = sorted(set(orig) & set(disg))
    if not ids:
        return None
    y = np.array([orig[i]["label"] for i in ids], dtype=float)
    p0 = np.array([orig[i]["p_default"] for i in ids])
    p1 = np.array([disg[i]["p_default"] for i in ids])
    rng = np.random.default_rng(seed)
    shifts = [float((p1[idx] - p0[idx]).mean())
              for idx in (rng.integers(0, len(ids), len(ids)) for _ in range(n_boot))]  # fmt: skip
    result = {
        "n": len(ids),
        "observed_rate": float(y.mean()),
        "mean_predicted": {"original": float(p0.mean()), variant: float(p1.mean())},
        "mean_shift": {
            "estimate": float((p1 - p0).mean()),
            "lo": float(np.percentile(shifts, 2.5)),
            "hi": float(np.percentile(shifts, 97.5)),
        },  # fmt: skip
        "mean_abs_change": float(np.abs(p1 - p0).mean()),
        "ece": {"original": m.ece(y, p0), variant: m.ece(y, p1)},
        "auc": {"original": m.auc(y, p0), variant: m.auc(y, p1)},
        "spearman": float(
            np.corrcoef(np.argsort(np.argsort(p0)), np.argsort(np.argsort(p1)))[0, 1]
        ),  # fmt: skip
        "cost_usd": float(sum(disg[i]["cost_usd"] for i in ids)),
    }
    cids = sorted(set(orig) & set(ctrl) & set(disg))
    if cids:
        c0 = np.array([orig[i]["p_default"] for i in cids])
        cc = np.array([ctrl[i]["p_default"] for i in cids])
        cd = np.array([disg[i]["p_default"] for i in cids])
        result["control"] = {
            "n": len(cids),
            "noise_mean_shift": float((cc - c0).mean()),
            "noise_mean_abs_change": float(np.abs(cc - c0).mean()),
            f"{control_prefix}_mean_shift_same_customers": float((cd - c0).mean()),
            f"{control_prefix}_mean_abs_change_same_customers": float(np.abs(cd - c0).mean()),
        }
    return result


def report_aligned(
    contestant: str, split: str = "tune", n_boot: int = 1000, seed: int = 20261122
) -> dict | None:
    """Original vs aligned layout on one split, overall and by latest repayment status."""
    import metrics as m
    import pandas as pd

    orig = _records(contestant, split=split)
    alig = _records(f"{contestant}-aligned", CHECK_DIR, split=split)
    ids = sorted(set(orig) & set(alig))
    if not ids:
        return None
    y = np.array([orig[i]["label"] for i in ids], dtype=float)
    p0 = np.array([orig[i]["p_default"] for i in ids])
    p1 = np.array([alig[i]["p_default"] for i in ids])
    rng = np.random.default_rng(seed)
    shifts = [float((p1[idx] - p0[idx]).mean())
              for idx in (rng.integers(0, len(ids), len(ids)) for _ in range(n_boot))]  # fmt: skip
    status = pd.read_parquet(ROOT / "data" / "splits" / f"{split}.parquet").set_index("id")["PAY_0"]
    codes = np.array([status[i] for i in ids])
    groups = {}
    for code, label in ((-2, "not late (code -2)"), (-1, "paid on time (code -1)"),
                        (0, "not late (code 0)")):  # fmt: skip
        g = codes == code
        groups[label] = {
            "n": int(g.sum()),
            "actual": float(y[g].mean()),
            "original": float(p0[g].mean()),
            "aligned": float(p1[g].mean()),
        }
    return {
        "split": split,
        "n": len(ids),
        "observed_rate": float(y.mean()),
        "mean_predicted": {"original": float(p0.mean()), "aligned": float(p1.mean())},
        "mean_shift": {
            "estimate": float((p1 - p0).mean()),
            "lo": float(np.percentile(shifts, 2.5)),
            "hi": float(np.percentile(shifts, 97.5)),
        },  # fmt: skip
        "mean_abs_change": float(np.abs(p1 - p0).mean()),
        "ece": {"original": m.ece(y, p0), "aligned": m.ece(y, p1)},
        "auc": {"original": m.auc(y, p0), "aligned": m.auc(y, p1)},
        "by_status": groups,
        "cost_usd": float(sum(alig[i]["cost_usd"] for i in ids)),
    }


def report() -> dict:
    results = {c: r for c in CONTESTANTS if (r := report_one(c))}
    aligned = {split: {c: r for c in CONTESTANTS if (r := report_aligned(c, split))}
               for split in ("tune", "test")}  # fmt: skip
    SUMMARY_JSON.write_text(json.dumps({"disguise": results, "aligned_layout": aligned}, indent=2)
                            + "\n", encoding="utf-8")  # fmt: skip
    for split, by_contestant in aligned.items():
        for c, r in by_contestant.items():
            s, st = r["mean_shift"], r["by_status"]["not late (code 0)"]
            print(f"aligned {c} ({split}, n={r['n']}): mean p "
                  f"{r['mean_predicted']['original']:.3f} -> {r['mean_predicted']['aligned']:.3f}"
                  f"  shift {s['estimate']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]  ECE "
                  f"{r['ece']['original']:.3f} -> {r['ece']['aligned']:.3f}  AUC "
                  f"{r['auc']['original']:.3f} -> {r['auc']['aligned']:.3f}  | code 0: actual "
                  f"{st['actual']:.2f}, {st['original']:.2f} -> {st['aligned']:.2f}")  # fmt: skip
    for c, r in results.items():
        s = r["mean_shift"]
        print(f"{c}: n={r['n']}  mean p {r['mean_predicted']['original']:.3f} -> "
              f"{r['mean_predicted']['disguised']:.3f}  shift {s['estimate']:+.3f} "
              f"[{s['lo']:+.3f}, {s['hi']:+.3f}]  ECE {r['ece']['original']:.3f} -> "
              f"{r['ece']['disguised']:.3f}  AUC {r['auc']['original']:.3f} -> "
              f"{r['auc']['disguised']:.3f}  rank corr {r['spearman']:.2f}")  # fmt: skip
        if "control" in r:
            k = r["control"]
            print(f"   control (n={k['n']}): noise shift {k['noise_mean_shift']:+.3f}, "
                  f"|change| {k['noise_mean_abs_change']:.3f} vs disguise shift "
                  f"{k['disguise_mean_shift_same_customers']:+.3f}, |change| "
                  f"{k['disguise_mean_abs_change_same_customers']:.3f}")  # fmt: skip
    print(f"Saved -> {SUMMARY_JSON}")
    report_formats()
    return results


def report_formats() -> dict:
    """Post 5: each format against v2 on the same test customers (results/checks/formats/).
    A run is scored only once it covers every test customer; shorter ones are listed as pending."""
    n_test = len(load_requests("test"))
    out, pending = {fmt: {} for fmt in FORMATS}, []
    for fmt in FORMATS:
        for c in CONTESTANTS:
            got = len(_records(f"{c}-{fmt}", FORMAT_DIR))
            if 0 < got < n_test:
                pending.append(f"{c}-{fmt} ({got} of {n_test})")
            elif got and (r := report_one(c, variant=fmt, directory=FORMAT_DIR,
                                          control_prefix=fmt)):  # fmt: skip
                out[fmt][c] = r
    if pending:
        print("formats still running: " + ", ".join(pending))
    if not any(out.values()):
        return out
    payload = {"format_version": FORMAT_VERSION, "formats": out, "pending": pending}
    FORMAT_SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for fmt, by_contestant in out.items():
        for c, r in by_contestant.items():
            s = r["mean_shift"]
            print(f"format {fmt} {c}: n={r['n']}  mean p {r['mean_predicted']['original']:.3f} -> "
                  f"{r['mean_predicted'][fmt]:.3f}  shift {s['estimate']:+.3f} "
                  f"[{s['lo']:+.3f}, {s['hi']:+.3f}]  |change| {r['mean_abs_change']:.3f}  ECE "
                  f"{r['ece']['original']:.3f} -> {r['ece'][fmt]:.3f}  AUC "
                  f"{r['auc']['original']:.3f} -> {r['auc'][fmt]:.3f}")  # fmt: skip
            if "control" in r:
                print(f"   noise floor (n={r['control']['n']}): |change| "
                      f"{r['control']['noise_mean_abs_change']:.3f}")  # fmt: skip
    print(f"Saved -> {FORMAT_SUMMARY_JSON}")
    return out


def main() -> None:
    ap = build_parser(__doc__)
    ap.set_defaults(split="test", condition="out_of_box")
    ap.add_argument("--contestant", choices=CONTESTANTS, default="frontier-verbal")
    ap.add_argument("--variant", choices=("disguised", "aligned", *FORMATS), default="disguised",
                    help="disguised: no NT$/dates/codes (test); aligned: each bill next to "
                         "the next month's payment (tune, then test); prose, markdown, csv, "
                         "json: post 5's formats (test)")  # fmt: skip
    ap.add_argument("--control", action="store_true",
                    help=f"original summaries, {CONTROL_N} customers")  # fmt: skip
    ap.add_argument("--rpm", type=float, default=18.0, help="Sol's requests-per-minute cap")
    ap.add_argument("--device", default="cpu", help="Laya's device (default: cpu)")
    ap.add_argument("--base-url", help="Kev or Qwen server (defaults: :8009 for Kev, :8000/v1)")
    ap.add_argument(
        "--probe", type=int, metavar="N", help="Sol recognition question on N summaries"
    )
    ap.add_argument("--report", action="store_true", help="compare with the original runs")
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe)
    if args.report:
        report()
        return
    if args.condition != "out_of_box":
        ap.error("these checks run on out_of_box requests only")
    if args.variant in ("disguised", *FORMATS) and args.split != "test" and not args.smoke:
        ap.error(f"the {args.variant} check runs on test (use --variant aligned for tune)")
    if args.variant in FORMATS and args.control:
        ap.error("the formats use the disguise check's control as their noise floor")
    if args.variant == "aligned" and args.control:
        ap.error(
            "--control is for the disguise check; the aligned layout uses its test noise floor"
        )
    if args.out_dir is None and not args.smoke:
        args.out_dir = FORMAT_DIR if args.variant in FORMATS else CHECK_DIR
    args = resolve_args(args)
    from env import load_env

    load_env()
    rows = load_requests(args.split)
    requests = control_sample(rows) if args.control else rows
    run(make_predictor(args.contestant, args, args.control, args.variant), args,
        requests=requests[: args.limit or None])  # fmt: skip


if __name__ == "__main__":
    main()
