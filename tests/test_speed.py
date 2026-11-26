"""Post 4 speed addendum (PLAN.md §12): the speed tags, the Kev launcher's settings for them, the
Kev runner against a fake server, the throughput sweep and the analysis, on synthetic records.
The fast kernels themselves (int4 matmul, GemLite, FP8) need the L4's GPU; L4_RUNBOOK.md §8 checks
them there with a reproduction or smoke run first."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import harness
import numpy as np
import pytest
import quantize as q
import records as rc
import speed as sp

# ---- names ----------------------------------------------


def test_speed_names_are_valid_distinct_and_never_post_4_names():
    names = sp.all_speed_contestants()
    assert len(names) == len(set(names)) == 15
    for name in names:
        assert rc._CONTESTANT.match(name)
        assert sp.speed_name(*sp.split_speed_name(name)) == name
        with pytest.raises(ValueError):
            q.split_name(name)  # never mistaken for a post 4 precision (e.g. hqq-4)
    assert set(names).isdisjoint(q.all_contestants())
    assert sp.all_speed_contestants(sp.FULL_RUN) == tuple(
        n for n in names if n.endswith(("aoint4", "gemlite", "fp8-dynamic")))  # fmt: skip
    with pytest.raises(ValueError):
        sp.speed_name("kev-4b-ft", "fp4")


def test_every_tag_has_a_base_and_a_check():
    for tag in sp.SPEED_TAGS:
        assert sp.BASE[tag] in q.PRECISIONS
        # each tag is either re-run in full (new arithmetic) or reproduced against post 4's run
        assert (tag in sp.FULL_RUN) != (tag in sp.REPRODUCE)
    assert sp.REPRODUCE == {"bf16-tuned": "bf16", "bnb-nf4-tuned": "bnb-nf4"}


def test_descriptions_name_the_kernels_and_the_installed_versions():
    d = sp.describe("fp8-dynamic", {"torchao": "0.13.0", "torch": "2.8.0"})
    assert d.startswith("fp8-dynamic (") and "no data used" in d and "torchao 0.13.0" in d
    assert "_weight_int4pack_mm" in sp.describe("hqq-4-aoint4", {"hqq": "0.2.8.post1"})
    assert "not installed" in sp.describe("hqq-4-gemlite", {"gemlite": None})
    assert set(sp.versions("hqq-4-gemlite")) >= {"hqq", "gemlite", "torch", *sp.FAST_LAYERS}


def test_apply_bf16_tuned_changes_no_weights_and_refuses_unknown_tags():
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(8, 8)).to(torch.bfloat16)
    before = model[0].weight.clone()
    info = sp.apply(model, "bf16-tuned", device="cpu")
    assert info["tag"] == "bf16-tuned" and info["quantized_layers"] == 0
    assert torch.equal(model[0].weight, before) and isinstance(model[0], nn.Linear)
    with pytest.raises(ValueError):
        sp.apply(model, "int4", device="cpu")


def test_apply_refuses_a_tag_whose_library_is_missing(monkeypatch):
    import torch.nn as nn

    monkeypatch.setattr(sp, "version", lambda lib: None if lib == "torchao" else "1.0")
    with pytest.raises(SystemExit, match="torchao"):
        sp.apply(nn.Linear(4, 4), "fp8-dynamic", device="cpu")


# ---- the Kev launcher's settings ----------------------------------------------


def test_launcher_turns_graphs_on_for_speed_tags_and_fused_off_for_shrunk_ones():
    import kev_quant_serve as ks

    base = {"KEV_TEMPERATURE": "1.0"}
    env = ks.check_env(base, "bf16-tuned")
    assert env["KEV_CUDA_GRAPHS"] == "1" and "KEV_FUSED" not in env  # Kev decides (its FLA pin)
    for tag in ("bnb-nf4-tuned", "hqq-4-aoint4", "hqq-4-gemlite", "fp8-dynamic"):
        env = ks.check_env(base, tag)
        assert env["KEV_CUDA_GRAPHS"] == "1" and env["KEV_FUSED"] == "0"
    assert ks.check_env(base, "hqq-4")["KEV_CUDA_GRAPHS"] == "0"  # post 4's runs are unchanged
    for env, tag in (({**base, "KEV_CUDA_GRAPHS": "0"}, "fp8-dynamic"),
                     ({**base, "KEV_FUSED": "1"}, "hqq-4-aoint4"),
                     ({**base, "KEV_CUDA_GRAPHS": "1"}, "bnb-nf4"),
                     (base, "int4")):  # fmt: skip
        with pytest.raises(SystemExit):
            ks.check_env(env, tag)


def test_launcher_reports_what_the_model_actually_runs_with():
    import kev_quant_serve as ks

    class Graphs:
        def stats(self):
            return {"captured": 4, "kept": 4, "pending": 0, "failed": 1}

    class Model:
        graphs = Graphs()

    class Opts:
        fused = True

    assert ks.features(Model(), Opts()) == {"cuda_graphs": Graphs().stats(), "fused": True}
    assert ks.features(object(), None) == {"cuda_graphs": False, "fused": False}


# ---- the Kev runner against a fake fast server ----------------------------------------------


def _fake_server(precision="fp8-dynamic", graphs=True, p=0.3):
    quant = {"precision": sp.BASE[precision], "tag": precision,  # as speed.apply reports
             "description": sp.describe(precision, {"torchao": "0.13.0"}),
             "cuda_graphs": {"captured": 2, "kept": 2, "pending": 0, "failed": 0} if graphs
             else False, "fused": False, "fast_layers_missing": [],
             "memory_now": {"max_allocated": 5 * 1024**3}}  # fmt: skip
    models = {"models": [{"name": "kev-latest", "run": "/home/runs/kev-0.8b-ft",
                          "temperature": 1.0}]}  # fmt: skip

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(models if self.path == "/v1/models" else quant)

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self._send({"answers": {"will_default": {"type": "noul", "noul": p}}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_dir(tmp_path):
    d = tmp_path / "kev-0.8b-ft"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"adapter")
    (d / "head.pt").write_bytes(b"head")
    return d


def test_kev_runner_names_a_speed_tag_and_records_the_setup(tmp_path):
    import kev

    server = _fake_server()
    try:
        pred = kev.KevPredictor("0.8b", f"http://127.0.0.1:{server.server_port}", 0.44,
                                finetuned=_run_dir(tmp_path), precision="fp8-dynamic")  # fmt: skip
        assert pred.contestant == "kev-0.8b-ft-fp8-dynamic"
        assert "CUDA graphs on, fused kernels off" in pred.model_version
        assert "torchao 0.13.0" in pred.model_version
        out = pred.predict(harness.load_requests("tune", 1)[0]["requests"]["out_of_box"])
        assert out.raw["cuda_graphs"]["captured"] == 2
    finally:
        server.shutdown()


def test_kev_runner_refuses_a_speed_tag_without_graphs(tmp_path):
    import kev

    server = _fake_server(graphs=False)
    try:
        with pytest.raises(SystemExit, match="CUDA graphs"):
            kev.KevPredictor("0.8b", f"http://127.0.0.1:{server.server_port}", 0.44,
                             finetuned=_run_dir(tmp_path), precision="fp8-dynamic")  # fmt: skip
    finally:
        server.shutdown()


def test_a_failed_speed_tag_can_be_recorded(tmp_path):
    import quantization as qa

    path = tmp_path / "failures.json"
    out = qa.record_failure("kev-4b-ft-fp8-dynamic", "load", "graph capture failed", path)
    assert (out["model"], out["precision"]) == ("kev-4b-ft", "fp8-dynamic")
    with pytest.raises(ValueError):
        qa.record_failure("kev-4b-ft-int4", "load", "x", path)


def test_reproduction_references():
    import kev

    assert kev.reproduce_reference("kev-4b-ft", "bf16") == "kev-4b-ft"
    assert kev.reproduce_reference("kev-4b-ft", "bf16-tuned") == "kev-4b-ft-bf16"
    assert kev.reproduce_reference("kev-4b-ft", "bnb-nf4-tuned") == "kev-4b-ft-bnb-nf4"
    assert kev.reproduce_reference("kev-4b-ft", "fp8-dynamic") is None
    assert kev.reproduce_reference("kev-4b-ft", None) is None


# ---- the throughput sweep ----------------------------------------------


def test_sweep_times_every_batch_size_on_the_same_customers():
    import throughput as tp

    rows = [{"id": f"c{i}"} for i in range(40)]
    calls, settled = [], []

    def score(group):  # 10 ms per call, whatever the size: bigger batches go faster
        calls.append(len(group))
        return {r["id"]: 0.25 for r in group}, 0.01, None

    out = tp.sweep(score, rows, batches=(1, 8), usd_per_hour=0.36, warmup=8,
                   settle=lambda: settled.append(1), log=lambda s: None)  # fmt: skip
    assert out["1"]["customers"] == out["8"]["customers"] == 40
    assert out["1"]["seconds"] == pytest.approx(0.40) and out["8"]["seconds"] == pytest.approx(0.05)
    assert out["8"]["customers_per_second"] == pytest.approx(800)
    assert out["8"]["cost_per_1000_usd"] == pytest.approx(1000 / 800 / 3600 * 0.36)
    assert out["8"]["max_abs_diff_vs_batch_1"] == 0.0 and len(settled) == 2
    assert calls[:8] == [1] * 8  # the untimed warm-up comes first


def test_sweep_reports_answers_that_batching_changed():
    import throughput as tp

    rows = [{"id": f"c{i}"} for i in range(4)]

    def score(group):
        p = 0.2 if len(group) == 1 else 0.23
        return {r["id"]: p for r in group}, 0.01, [5.0] * len(group)

    out = tp.sweep(score, rows, batches=(1, 4), warmup=0, log=lambda s: None)
    assert out["4"]["max_abs_diff_vs_batch_1"] == pytest.approx(0.03)
    assert out["1"]["latency_ms_median"] == 5.0


# ---- the analysis on synthetic records ----------------------------------------------


def _write(path, name, p_test, p_tune, y_test, y_tune, ids_test, ids_tune):
    def recs(split, cond, ids, ys, ps):
        return [rc.make_record(id=i, split=split, contestant=name, condition=cond,
                               model_version=f"{name} v", p_default=float(p), label=int(y),
                               latency_ms=50.0, cost_usd=1e-5, ts=rc.utc_now())
                for i, y, p in zip(ids, ys, ps, strict=True)]  # fmt: skip

    rc.write_records(name, recs("tune", "trained", ids_tune, y_tune, p_tune)
                     + recs("test", "trained", ids_test, y_test, p_test), path)  # fmt: skip


def test_report_scores_full_runs_against_bf16_and_reads_throughput(tmp_path):
    import quantization as qa
    import speed_report as sr

    rng = np.random.default_rng(1)
    n = qa.N_TEST
    ids_te, ids_tu = [f"cust_{i:05d}" for i in range(n)], [f"cust_{i:05d}" for i in range(n, 2 * n)]
    p_te, p_tu = rng.uniform(0.02, 0.9, n), rng.uniform(0.02, 0.9, n)
    y_te, y_tu = rng.uniform(size=n) < p_te, rng.uniform(size=n) < p_tu
    preds = tmp_path / "predictions"
    _write(preds, "kev-0.8b-ft-bf16", p_te, p_tu, y_te, y_tu, ids_te, ids_tu)
    _write(preds, "kev-0.8b-ft-fp8-dynamic", np.clip(p_te + 0.01, 0, 1), np.clip(p_tu + 0.01, 0, 1),
           y_te, y_tu, ids_te, ids_tu)  # fmt: skip
    speed_dir = tmp_path / "speed"
    speed_dir.mkdir()
    (speed_dir / "kev-0.8b-ft-fp8-dynamic.json").write_text(json.dumps({
        "contestant": "kev-0.8b-ft-fp8-dynamic", "by_batch": {"1": {
            "customers": 500, "seconds": 10.0, "customers_per_second": 50.0,
            "cost_per_1000_usd": 0.0024, "max_abs_diff_vs_batch_1": 0.0}}}), "utf-8")  # fmt: skip
    # The runbook saves the launcher's stats alongside; the report must skip them.
    (speed_dir / "kev-0.8b-ft-fp8-dynamic.server.json").write_text(
        json.dumps({"tag": "fp8-dynamic", "cuda_graphs": {"captured": 1}}), "utf-8")
    out = sr.build(preds, speed_dir, tmp_path / "repro")
    (row,) = out["honesty"]
    assert row["contestant"] == "kev-0.8b-ft-fp8-dynamic"
    assert row["mean_shift_vs_bf16"] == pytest.approx(0.01, abs=1e-3)
    assert "ece_vs_bf16_p_holm" in row and "auc_vs_bf16_p_holm" in row
    assert len(out["throughput"]) == 1
    assert out["throughput"][0]["customers_per_second"] == 50.0
    assert "qwen-ft-hqq-4-aoint4" in out["not_scored"]
