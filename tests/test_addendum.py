"""Post 5 addendum (PLAN.md §12): the record names, the trained-model layout wrapper, and both
scorers end to end on small synthetic runs, so the L4 session can't be the first time they meet
real files."""

import csv
import json
from functools import partial
from pathlib import Path

import arrange as A
import arrangements as AR
import formats
import layouts as L
import numpy as np
import pytest
from records import make_record, utc_now, write_records

ROOT = Path(__file__).resolve().parents[1]
N = 60  # synthetic customers (the real runs have 2,000)
IDS = [f"cust_{i:05d}" for i in range(N)]


def _write(directory, name, p, labels, splits=("test",)):
    recs = [make_record(id=cid, split=split, contestant=name, condition="trained",
                        model_version=f"{name} v1", p_default=float(np.clip(pi, 0.001, 0.999)),
                        label=int(yi), latency_ms=1.0, cost_usd=0.0, ts=utc_now())
            for split in splits for cid, pi, yi in zip(IDS, p, labels, strict=True)]  # fmt: skip
    write_records(name, recs, directory)


@pytest.fixture
def rng(monkeypatch):
    """Also fewer bootstrap draws (2,000 in the real runs), so the end-to-end tests take seconds."""
    import quantization
    import robustness

    monkeypatch.setattr(L, "paired_auc", partial(robustness.paired_auc, n_boot=40))
    monkeypatch.setattr(L, "paired_ece", partial(quantization.paired_ece, n_boot=40))
    monkeypatch.setattr(L, "ece_noise_floor", partial(robustness.ece_noise_floor, sims=20))
    return np.random.default_rng(7)


def test_run_names():
    assert A.run_name("kumo-tabular", "original") == "kumo-tabular-original"
    assert A.run_name("kumo-tabular", "original", 2) == "kumo-tabular-original-s2"
    assert A.run_name("tabicl-v2", "by-month", 0, 1) == "tabicl-v2-by-month-e1"


def test_layout_wrapper_rewrites_only_the_summary_and_names_the_run():
    import layout

    class Inner:
        contestant, model_version, meta = "laya-ft", "laya-ft vX", {"device": "cpu"}

        def __init__(self):
            self.seen = []

        def predict(self, request):
            self.seen.append(request)
            return request["state"]

    row = json.loads((ROOT / "data" / "rendered" / "test.jsonl").open(encoding="utf-8").readline())
    state = row["requests"]["out_of_box"]["state"]
    inner = Inner()
    w = layout.with_layout(inner, "json")
    out = w.predict({"state": state, "question": "q"})
    assert w.contestant == "laya-ft-json" and "format json" in w.model_version
    assert out == formats.render(state, "json") != state and inner.seen[0]["question"] == "q"
    assert layout.with_layout(Inner(), "aligned").predict({"state": state}) != state
    again = layout.with_layout(Inner(), "again")
    assert again.contestant == "laya-ft-again" and again.predict({"state": state}) == state
    assert layout.with_layout(inner, None) is inner


def test_layout_all_runs_the_five_then_again_and_none_runs_the_original():
    import layout

    assert layout.layouts_to_run("all") == (*layout.LAYOUTS, "again")
    assert layout.layouts_to_run("csv") == ("csv",) and layout.layouts_to_run(None) == (None,)


def test_layout_runs_answer_test_only_and_never_write_to_predictions():
    import argparse

    import harness
    import layout

    ap = argparse.ArgumentParser()
    args = argparse.Namespace(layout="json", condition=harness.TRAINED, split="test", smoke=False,
                              out_dir=None, reproduce_check=None)  # fmt: skip
    layout.check_layout_args(ap, args)
    assert args.out_dir == layout.LAYOUT_DIR
    for bad in ({"split": "tune"}, {"condition": "out_of_box"}, {"reproduce_check": 50}):
        with pytest.raises(SystemExit):
            layout.check_layout_args(ap, argparse.Namespace(**{**vars(args), **bad}))


def test_trained_scorer_end_to_end(tmp_path, monkeypatch, rng):
    preds, trained = tmp_path / "predictions", tmp_path / "layouts_trained"
    y = rng.random(N) < 0.25
    published = []
    for ref, cold_name in L.TRAINED_REFS.items():
        base = np.clip(0.15 + 0.4 * y + rng.normal(0, 0.1, N), 0.01, 0.99)
        _write(preds, ref, base, y, ("tune", "test"))
        _write(trained, f"{ref}-again", base + rng.normal(0, 0.005, N), y)
        for k, layout in enumerate(L.LAYOUTS):
            _write(trained, f"{ref}-{layout}", base + rng.normal(0, 0.02 * (k + 1), N), y)
            published.append({"model": cold_name, "layout": layout, "mean_abs_change": 0.1,
                              "decisions_changed": 0.2, "auc_vs_v2": -0.01})  # fmt: skip
    cold_csv = tmp_path / "layouts.csv"
    with cold_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(published[0]))
        w.writeheader()
        w.writerows(published)
    for name, value in (("PREDICTIONS_DIR", preds), ("TRAINED_DIR", trained),
                        ("OUT_CSV", cold_csv), ("N_TEST", N)):  # fmt: skip
        monkeypatch.setattr(L, name, value)
    rows = L.build_trained()
    assert len(rows) == 20 and {r["model"] for r in rows} == set(L.TRAINED_REFS)
    assert all(0 <= r["auc_vs_v2_p_holm"] <= 1 and r["cold_mean_abs_change"] == 0.1 for r in rows)
    first = next(r for r in rows if r["layout"] == "markdown")
    last = next(r for r in rows if r["layout"] == "aligned" and r["model"] == first["model"])
    assert first["noise_n"] == N and last["mean_abs_change"] > first["mean_abs_change"]


def test_arrangements_scorer_end_to_end(tmp_path, monkeypatch, rng):
    adir, preds = tmp_path / "arrangements", tmp_path / "predictions"
    y = rng.random(N) < 0.25
    for model in AR.MODELS:
        base = np.clip(0.15 + 0.4 * y + rng.normal(0, 0.1, N), 0.01, 0.99)
        _write(preds, model, base, y, ("tune", "test"))  # M6
        for est in AR.SETTINGS.values():
            _write(adir, A.run_name(model, "original", 0, est), base, y, ("tune", "test"))
            for k in AR.SEED_OFFSETS:
                seed_run = base + rng.normal(0, 0.01, N)
                _write(adir, A.run_name(model, "original", k, est), seed_run, y)
            for arr in AR.OTHERS:
                shift = 0.15 if arr == "shuffled-1" else 0.0  # one arrangement that moves a lot
                _write(adir, A.run_name(model, arr, 0, est), base + shift, y)
    for mod, name, value in ((AR, "ARRANGE_DIR", adir), (AR, "PREDICTIONS_DIR", preds),
                             (AR, "N_TEST", N), (L, "N_TEST", N)):  # fmt: skip
        monkeypatch.setattr(mod, name, value)
    rows = AR.build()
    assert len(rows) == 2 * 2 * 14
    by = {(r["setting"], r["model"], r["arrangement"]): r for r in rows}
    moved = by["defaults", "kumo-tabular", "shuffled-1"]
    still = by["defaults", "kumo-tabular", "thousands"]
    assert moved["beyond_seed_noise"] and not still["beyond_seed_noise"]
    assert still["mean_abs_change"] == 0 and moved["mean_shift"] == pytest.approx(0.15, abs=0.02)
    assert moved["repro_test_mean_abs"] == 0  # the session's original equals M6 here
    assert "repro_test_mean_abs" not in by["e1", "kumo-tabular", "thousands"]
    assert all("auc_vs_original_p_holm" in r for r in rows)
