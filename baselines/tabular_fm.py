"""M6 Part B: tabular foundation models, learning in context from the 20,000 `train` customers.

    T='--group tabular'
    uv run $T python baselines/tabular_fm.py --model kumo-tabular --smoke   # L4, ~1 min
    uv run $T python baselines/tabular_fm.py --model kumo-tabular           # tune + test
    uv run $T python baselines/tabular_fm.py --model tabicl-v2

Post 5 addendum, part A (PLAN.md §12): the same model on the table laid out another way
(data/arrange.py), into results/checks/arrangements/<model>-<arrangement>[-s<k>][-e1].jsonl:

    uv run $T python baselines/tabular_fm.py --model kumo-tabular --arrangement original
    uv run $T python baselines/tabular_fm.py --model kumo-tabular --arrangement original -s 2
    uv run $T python baselines/tabular_fm.py --model tabicl-v2 --arrangement by-month --estimators 1

`original` (at any seed or estimators) answers tune and test; the others answer test only.

These are AI models built for tables (PLAN.md §11, Part B). They don't train with gradients: each
reads the labelled `train` rows as its context and predicts new rows. They read the same 19
financial fields as the scorecard and LightGBM (raw values, no text, no demographics), with all
20,000 `train` customers in context, and answer every `tune` and `test` customer. Condition
`trained`; `analysis/recalibrate.py` adds `trained_calibrated` from tune, as for every M6 model.

Each maker's defaults (choice 1):
- kumo-tabular: NVIDIA's `nvidia/Kumo-Tabular`, "large", checkpoint v1.0.1 (pinned inside the
  `structured-data-models` package), 8 estimators as in its model card, fp16 autocast on CUDA as
  in NVIDIA's quickstart. Pretrained on tables up to 60,000 rows and 100 columns. Every feature
  is passed as numerical (the model takes only numerical features; LightGBM also reads the
  repayment codes as numbers).
- tabicl-v2: Inria's `tabicl` package, `TabICLClassifier()` defaults. Pretrained on 300-48K rows.
TabPFN-3.5 is out (the author's decision, 2026-11-24): its weights need a Prior Labs login, and
its licence is non-commercial for the weights and their outputs.

Latency is one batched call per split divided by the customers in it, as for LightGBM; cost is
that GPU time at the L4 rate. Writes results/predictions/<model>.jsonl and
results/baselines/<model>.json. `--smoke`: 500 context rows and 100 tune customers, printed only.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))  # arrange (post 5 addendum)

from arrange import ARRANGE_DIR, ARRANGEMENTS, arrange, context_order, describe, run_name  # noqa: E402
from common import (  # noqa: E402
    BASELINE_DIR,
    SEED,
    build_records,
    load_split,
    quick_report,
    train_fingerprint,
)
from records import write_records  # noqa: E402

L4_USD_PER_HOUR = 0.44  # = runners/harness.py
MODELS = ("kumo-tabular", "tabicl-v2")
KUMO_ESTIMATORS = 8  # the nvidia/Kumo-Tabular model card's example
TARGET_COL = "target"


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _seeded(cls, seed: int = SEED, **kwargs):
    """Construct with the maker's defaults, adding a seed only where the class takes one."""
    params = inspect.signature(cls).parameters
    for name in ("random_state", "seed"):
        if name in params:
            kwargs[name] = seed
            break
    return cls(**kwargs)


def unshuffled_kumo_recipe():
    """Kumo-Tabular's default recipe without its `ShuffleColumns(method="latin")` step.

    The default ensemble shuffles the column order for each estimator and averages, and so does
    TabICL's (`feat_shuffle_method="latin"`). With one estimator it would still apply one shuffle
    of its own, so the diagnostic switches the shuffle off: the model sees the columns in the
    arrangement's order. Every other step is the maker's.
    """
    import sdm.processing as sp
    from sdm.models.kumo.tabular.recipe import default_recipe

    rec = default_recipe()
    steps = list(rec._features._modules.values())
    kept = [s for s in steps if not isinstance(s, sp.ShuffleColumns)]
    assert len(kept) == len(steps) - 1, "expected exactly one ShuffleColumns step"
    return sp.Recipe(features=kept, target=rec._target, output=rec._output)


class KumoTabular:
    def __init__(self, device: str, size: str = "large", seed: int = SEED,
                 estimators: int | None = None):  # fmt: skip
        import sdm
        import torch

        self.sdm, self.torch, self.device = sdm, torch, torch.device(device)
        self.seed, self.estimators = seed, estimators or KUMO_ESTIMATORS
        self.model = sdm.models.KumoTabular(task="classification", size=size, device=self.device)
        sdm_version = _version("structured-data-models")
        self.version = (f"nvidia/Kumo-Tabular {size} v1.0.1 via structured-data-models "
                        f"{sdm_version}; {self.estimators} estimators")  # fmt: skip
        self.recipe = None  # the maker's default recipe, as in M6
        if estimators == 1:  # the addendum's diagnostic: the columns in the order given
            self.recipe = unshuffled_kumo_recipe()
            self.version += ", column shuffle off"

    def _table(self, X, y=None):
        df = X.reset_index(drop=True).astype(float)
        overrides = {c: "numerical" for c in df.columns}
        if y is not None:
            df[TARGET_COL] = np.asarray(y).astype(int).astype(str)
            overrides[TARGET_COL] = "categorical"
        stypes = self.sdm.infer_stypes(df, overrides=overrides)
        return self.sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=self.device)

    def fit(self, X, y):
        self.context = self._table(X, y)
        return self

    def predict_proba1(self, X, batch: int = 500) -> np.ndarray:
        torch = self.torch
        x_ctx = self.context.drop_columns(TARGET_COL)
        y_ctx = self.context[:, TARGET_COL]
        out = []
        for i in range(0, len(X), batch):
            query = self._table(X.iloc[i : i + batch])
            # Same seed for every batch: every customer gets the same 8-estimator ensemble.
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            cuda = self.device.type == "cuda"
            with (
                torch.inference_mode(),
                torch.amp.autocast(self.device.type, torch.float16, enabled=cuda),
            ):
                extra = {"recipe": self.recipe} if self.recipe is not None else {}
                pred = self.model(x_context=x_ctx, y_context=y_ctx, x_query=query,
                                  num_estimators=self.estimators, generator=gen,
                                  **extra)  # fmt: skip
            scores = pred.numerical.float()
            if scores.dim() == 3:  # [estimators, rows, classes] if the recipe didn't average
                scores = scores.mean(0)
            cols = [str(c).strip() for c in pred.numerical_columns]
            one = next(j for j, c in enumerate(cols) if c in ("1", "1.0"))
            scores = scores.clamp_min(0)
            out.append((scores[:, one] / scores.sum(-1)).cpu().numpy())
        return np.concatenate(out)


class SklearnStyle:
    """TabICLClassifier: fit(X, y), predict_proba(X)."""

    def __init__(self, name: str, device: str, seed: int = SEED, estimators: int | None = None):
        from tabicl import TabICLClassifier as cls

        pkg = "tabicl"
        params = inspect.signature(cls).parameters
        kwargs = {"device": device} if "device" in params else {}
        setting = "defaults"
        if estimators is not None:  # the diagnostic: one estimator, columns in the order given
            if not {"n_estimators", "feat_shuffle_method"} <= set(params):
                raise SystemExit(f"{cls.__name__} can't run the diagnostic: record the failure")
            kwargs.update(n_estimators=estimators, feat_shuffle_method="none")
            setting = f"n_estimators={estimators}, feat_shuffle_method='none'"
        self.clf = _seeded(cls, seed, **kwargs)
        self.version = f"{pkg} {_version(pkg)} {type(self.clf).__name__}() {setting}"

    def fit(self, X, y):
        self.clf.fit(X.to_numpy(dtype=float), np.asarray(y).astype(int))
        ckpt = getattr(self.clf, "checkpoint_version", None)
        if ckpt:
            self.version += f"; checkpoint {ckpt}"
        return self

    def predict_proba1(self, X) -> np.ndarray:
        proba = self.clf.predict_proba(X.to_numpy(dtype=float))
        return proba[:, list(self.clf.classes_).index(1)]


def build(name: str, device: str, kumo_size: str = "large", seed: int = SEED,
          estimators: int | None = None):  # fmt: skip
    if name == "kumo-tabular":
        return KumoTabular(device, kumo_size, seed, estimators)
    return SklearnStyle(name, device, seed, estimators)


def timed(fn, X) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    p = np.asarray(fn(X), dtype=float)
    secs = time.perf_counter() - start
    return p, secs * 1000 / len(X), secs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="500 context rows, 100 tune customers")
    ap.add_argument("--kumo-size", default="large", choices=("small", "medium", "large"),
                    help="large is the contestant; small only for a quick CPU smoke")  # fmt: skip
    ap.add_argument("--arrangement", choices=ARRANGEMENTS,
                    help="post 5 addendum: the table laid out another way (data/arrange.py), "
                         "into results/checks/arrangements/")  # fmt: skip
    ap.add_argument("-s", "--seed-offset", type=int, default=0,
                    help="addendum: seed = M6's + K, the noise floor (original only)")  # fmt: skip
    ap.add_argument("--estimators", type=int, choices=(1,),
                    help="addendum: the single-estimator diagnostic")  # fmt: skip
    args = ap.parse_args()
    addendum = args.arrangement is not None
    if (args.seed_offset or args.estimators) and not addendum:
        ap.error("--seed-offset and --estimators are for the addendum: add --arrangement")
    if args.seed_offset and args.arrangement != "original":
        ap.error("other seeds are the noise floor: run them on --arrangement original")

    _, X_train, y_train = load_split("train")
    if args.smoke:
        X_train, y_train = X_train.iloc[:500], y_train[:500]
    if args.kumo_size != "large" and not args.smoke:
        sys.exit("The contestant is Kumo-Tabular large; other sizes are for --smoke only.")
    name, tag = args.model, ""
    if addendum:
        name = run_name(args.model, args.arrangement, args.seed_offset, args.estimators)
        tag = f"; {describe(args.arrangement)}; seed {SEED + args.seed_offset}"
        order = context_order(len(X_train), args.arrangement)
        if order is not None:
            X_train, y_train = X_train.iloc[order], y_train[order]
        X_train = arrange(X_train, args.arrangement)
    model = build(args.model, args.device, args.kumo_size, SEED + args.seed_offset,
                  args.estimators)  # fmt: skip
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - start
    version = (f"{model.version}; context = train {train_fingerprint()} ({len(X_train):,} rows)"
               f"{tag}")  # fmt: skip
    print(f"{name}: {version} (fit {fit_s:.1f}s)")

    # The addendum's reference (original, any seed or estimators) answers tune too, for the
    # cut-off and the reproduction of M6; every other arrangement answers test only.
    splits = ("tune", "test") if not addendum or args.arrangement == "original" else ("test",)
    records, report = [], {"model": name, "model_version": version, "fit_seconds": fit_s}
    for split in ("tune",) if args.smoke else splits:
        df, X, y = load_split(split)
        if args.smoke:
            df, X, y = df.iloc[:100], X.iloc[:100], y[:100]
        if addendum:
            X = arrange(X, args.arrangement)
        p, ms, secs = timed(model.predict_proba1, X)
        print(quick_report(name, split, p, y))
        report[split] = {"seconds": secs, "ms_per_customer": ms, "mean_p": float(p.mean()),
                         "exactly_0_or_1": int(((p == 0) | (p == 1)).sum())}  # fmt: skip
        if not args.smoke:
            for rec in build_records(df, p, split=split, contestant=name, condition="trained",
                                     model_version=version, latency_ms=ms):  # fmt: skip
                rec["cost_usd"] = ms / 3_600_000 * L4_USD_PER_HOUR
                records.append(rec)
    if args.smoke:
        print("smoke run: nothing written")
        return
    out_dir, meta_dir = (ARRANGE_DIR, ARRANGE_DIR / "meta") if addendum else (None, BASELINE_DIR)
    path = write_records(name, records, out_dir) if out_dir else write_records(name, records)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"wrote {len(records)} records -> {path}")


if __name__ == "__main__":
    main()
