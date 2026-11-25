"""Post 4 (PLAN.md §12): the quantization helpers, the Kev launcher's settings, the runners'
precision checks (against a fake Kev server), the reproduction check, the manifest group and the
analysis, on synthetic records."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import harness
import numpy as np
import pytest
import quantize as q
import records as rc

# ---- names and settings ----------------------------------------------------------------------


def test_eighteen_record_names_all_valid_and_invertible():
    names = q.all_contestants()
    assert len(names) == len(set(names)) == 18
    for name in names:
        assert rc._CONTESTANT.match(name)
        model, precision = q.split_name(name)
        assert q.contestant_name(model, precision) == name
    assert "kev-0.8b-ft-bnb-nf4" in names and "qwen-ft-hqq-2" in names
    with pytest.raises(ValueError):
        q.contestant_name("kev-0.8b-ft", "int4")
    with pytest.raises(ValueError):
        q.contestant_name("laya-ft", "bf16")  # Laya isn't in post 4


def test_descriptions_name_the_settings_that_differ_from_bare_library_defaults():
    assert q.describe("bf16") == "bf16 (not quantized)"
    assert "LLM.int8, outlier threshold 6.0" in q.describe("bnb-int8", "0.50.2")
    nf4 = q.describe("bnb-nf4", "0.50.2")
    assert "bitsandbytes 0.50.2" in nf4 and "nf4" in nf4 and "no double quantization" in nf4
    assert "2-bit, group size 64, axis 1" in q.describe("hqq-2", "0.2.8.post1")
    assert q.SETTINGS["bnb-int8"]["threshold"] == 6.0  # transformers' load_in_8bit, not 0.0
    assert q.SETTINGS["bnb-nf4"]["compress_statistics"] is False  # transformers' default


@pytest.mark.parametrize("name, kept", [
    ("lm_head", True),
    ("model.visual.blocks.0.attn.qkv", True),
    ("model.language_model.layers.0.mlp.up_proj", False),
    ("model.language_model.layers.3.linear_attn.in_proj_qkv", False),
    ("visualizer.proj", False),  # only whole name parts count
])  # fmt: skip
def test_skip_keeps_the_readout_and_the_vision_tower(name, kept):
    assert q.skipped(name) is kept


def _toy():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 128)
            self.layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128, bias=False))
                 for _ in range(2)])  # fmt: skip
            self.lm_head = nn.Linear(128, 10, bias=False)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = h + layer(h)
            return self.lm_head(h)

    torch.manual_seed(0)
    return torch, Toy().to(torch.bfloat16).eval()


def test_bf16_changes_nothing():
    torch, model = _toy()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    info = q.quantize_linears(model, "bf16", device="cpu")
    assert info["quantized_layers"] == 0 and info["linear_layers"] == 5
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_every_precision_replaces_the_backbone_and_keeps_the_readout():
    torch, _ = _toy()
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("hqq")
    x = torch.randint(0, 10, (1, 7))
    errors = {}
    for precision in q.QUANTIZED:
        torch.manual_seed(0)
        _, model = _toy()
        with torch.inference_mode():
            ref = model(x).float()
        info = q.quantize_linears(model, precision, device="cpu")
        assert info["quantized_layers"] == 4 and "lm_head 1" in info["kept_bf16"]
        assert type(model.lm_head) is torch.nn.Linear
        assert all(type(s[0]) is not torch.nn.Linear for s in model.layers)
        # Kev takes its masks' dtype from the first parameter (the embeddings)
        assert next(model.parameters()).dtype == torch.bfloat16
        with torch.inference_mode():
            out = model(x).float()
        assert torch.isfinite(out).all()
        errors[precision] = float((out - ref).abs().max())
    assert errors["hqq-4"] < errors["hqq-3"] < errors["hqq-2"]  # fewer bits, further off


def test_a_different_library_version_is_refused(monkeypatch):
    _, model = _toy()
    monkeypatch.setattr(q, "library_version", lambda precision: "0.0.1")
    with pytest.raises(SystemExit, match="pins"):
        q.quantize_linears(model, "hqq-4", device="cpu")


# ---- the Kev launcher's environment ----------------------------------------------------------


def test_launcher_requires_temperature_one_and_turns_graphs_and_fused_off():
    import kev_quant_serve as ks

    env = ks.check_env({"KEV_TEMPERATURE": "1.0"})
    assert env["KEV_CUDA_GRAPHS"] == "0" and env["KEV_FUSED"] == "0"
    assert ks.check_env({"KEV_TEMPERATURE": "1.0", "KEV_DTYPE": "bf16"})
    for bad in ({}, {"KEV_TEMPERATURE": "0.8"},
                {"KEV_TEMPERATURE": "1.0", "KEV_CUDA_GRAPHS": "1"},
                {"KEV_TEMPERATURE": "1.0", "KEV_FUSED": "1"},
                {"KEV_TEMPERATURE": "1.0", "KEV_DTYPE": "fp32"},
                {"KEV_TEMPERATURE": "1.0", "KEV_MERGE": "0"}):  # fmt: skip
        with pytest.raises(SystemExit):
            ks.check_env(bad)


# ---- the Kev runner against a fake launcher --------------------------------------------------


def _fake_launcher(run_name, precision="bnb-nf4", quantization=True):
    quant = {"precision": precision, "description": q.describe(precision, "0.50.2"),
             "memory_after_load": {"allocated": 3 * 1024**3}, "cuda_graphs": False,
             "fused": False, "memory_now": {"max_allocated": 4 * 1024**3}}  # fmt: skip
    card = {"name": "kev-latest", "run": f"/home/runs/{run_name}", "temperature": 1.0,
            "backend": f"torch; precision {precision}"}  # fmt: skip
    models = {"models": [card]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(models)
            elif self.path == "/v1/quantization" and quantization:
                self._send(quant)
            else:
                self._send({"detail": "Not Found"}, 404)

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self._send({"answers": {"will_default": {"type": "noul", "noul": 0.3}},
                        "usage": {}, "latency_ms": 5})  # fmt: skip

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_dir(tmp_path):
    d = tmp_path / "kev-0.8b-ft"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"adapter")
    (d / "head.pt").write_bytes(b"head")
    return d


def test_kev_runner_names_and_versions_the_precision(tmp_path):
    import kev

    server = _fake_launcher("kev-0.8b-ft")
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        pred = kev.KevPredictor("0.8b", url, 0.44, finetuned=_run_dir(tmp_path),
                                precision="bnb-nf4")  # fmt: skip
        assert pred.contestant == "kev-0.8b-ft-bnb-nf4"
        assert "kev_quant_serve.py" in pred.model_version and "nf4" in pred.model_version
        assert pred.meta["quantization"]["precision"] == "bnb-nf4"
        out = pred.predict(harness.load_requests("tune", 1)[0]["requests"]["out_of_box"])
        assert out.p_default == 0.3 and out.raw["gpu_max_allocated"] == 4 * 1024**3
    finally:
        server.shutdown()


def test_kev_runner_refuses_another_precision_or_a_plain_server(tmp_path):
    import kev

    run_dir = _run_dir(tmp_path)
    for server, match in ((_fake_launcher("kev-0.8b-ft", "hqq-4"), "hqq-4"),
                          (_fake_launcher("kev-0.8b-ft", quantization=False),
                           "kev_quant_serve")):  # fmt: skip
        try:
            with pytest.raises(SystemExit, match=match):
                kev.KevPredictor("0.8b", f"http://127.0.0.1:{server.server_port}", 0.44,
                                 finetuned=run_dir, precision="bnb-nf4")  # fmt: skip
        finally:
            server.shutdown()


# ---- the reproduction check ------------------------------------------------------------------


class ConstantPredictor:
    contestant = "kev-0.8b-ft-bf16"
    model_version = "fake"
    meta: dict = {}

    def __init__(self, p):
        self.p = p

    def predict(self, request):
        return harness.Prediction(p_default=self.p, latency_ms=1.0, cost_usd=0.0)


def _reference(tmp_path, n=5, p=0.30):
    recs = [rc.make_record(id=row["id"], split="tune", contestant="kev-0.8b-ft",
                           condition="trained", model_version="m6", p_default=p,
                           label=row["label"], latency_ms=1.0, cost_usd=0.0, ts=rc.utc_now())
            for row in harness.load_requests("tune", n)]  # fmt: skip
    rc.write_records("kev-0.8b-ft", recs, tmp_path)


def test_reproduction_check_passes_close_and_stops_far(tmp_path):
    _reference(tmp_path)
    out = tmp_path / "checks"
    ok = harness.reproduction_check(ConstantPredictor(0.31), "kev-0.8b-ft", 5,
                                    directory=tmp_path, out_dir=out)  # fmt: skip
    assert ok["passed"] and ok["mean_abs_diff"] == pytest.approx(0.01)
    assert (out / "repro_kev-0.8b-ft-bf16.json").exists()
    with pytest.raises(SystemExit, match="doesn't reproduce"):
        harness.reproduction_check(ConstantPredictor(0.40), "kev-0.8b-ft", 5,
                                   directory=tmp_path, out_dir=out)  # fmt: skip
    assert not json.loads((out / "repro_kev-0.8b-ft-bf16.json").read_text("utf-8"))["passed"]


# ---- manifest and recalibration --------------------------------------------------------------


def test_manifest_counts_recorded_failures_as_accounted_never_as_done(tmp_path):
    import manifest

    assert set(manifest.QUANT_GROUPS) == set(q.all_contestants())
    names = q.all_contestants()
    some = manifest.build_manifest(tmp_path, manifest.QUANT_GROUPS, {names[0]: {"error": "x"}})
    assert not some["complete"] and some["recorded_failures"] == [names[0]]
    assert not some["contestants"][names[0]]["complete"]
    every = manifest.build_manifest(tmp_path, manifest.QUANT_GROUPS,
                                    {n: {"error": "x"} for n in names})  # fmt: skip
    assert every["complete"]
    assert "recorded_failures" not in manifest.build_manifest(tmp_path, manifest.M6_GROUPS)


def test_recalibration_covers_every_precision():
    import recalibrate

    assert set(q.all_contestants()) <= set(recalibrate.QUANT_CONTESTANTS)


# ---- the analysis, on synthetic records ------------------------------------------------------


def _write(tmp_path, name, p_test, p_tune, y_test, y_tune, ids_test, ids_tune, peak=None):
    def recs(split, cond, ids, ys, ps):
        return [rc.make_record(id=i, split=split, contestant=name, condition=cond,
                               model_version=f"{name} v", p_default=float(p), label=int(y),
                               latency_ms=50.0, cost_usd=1e-5, ts=rc.utc_now())
                for i, y, p in zip(ids, ys, ps, strict=True)]  # fmt: skip

    all_recs = (recs("tune", "trained", ids_tune, y_tune, p_tune)
                + recs("test", "trained", ids_test, y_test, p_test)
                + recs("test", "trained_calibrated", ids_test, y_test, p_test))  # fmt: skip
    rc.write_records(name, all_recs, tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{name}.meta.json").write_text(json.dumps(
        [{"quantization": {"memory_after_load": {"allocated": 2 * 1024**3}}}]), "utf-8")
    if peak:
        (raw / f"{name}.jsonl").write_text("\n".join(
            json.dumps({"id": i, "split": "test", "raw": {"gpu_max_allocated": peak}})
            for i in ids_test[:3]) + "\n", "utf-8")  # fmt: skip


def test_analysis_scores_against_the_models_own_bf16(tmp_path, monkeypatch):
    import quantization as qa

    monkeypatch.setattr(qa, "REPRO_DIR", tmp_path / "none")
    rng = np.random.default_rng(0)
    ids_te = [f"cust_{i:05d}" for i in range(qa.N_TEST)]
    ids_tu = [f"cust_{i:05d}" for i in range(qa.N_TEST, 2 * qa.N_TEST)]
    p_te, p_tu = rng.uniform(0.02, 0.9, qa.N_TEST), rng.uniform(0.02, 0.9, qa.N_TEST)
    y_te, y_tu = (rng.uniform(size=qa.N_TEST) < p_te), (rng.uniform(size=qa.N_TEST) < p_tu)
    _write(tmp_path, "kev-0.8b-ft-bf16", p_te, p_tu, y_te, y_tu, ids_te, ids_tu)
    _write(tmp_path, "kev-0.8b-ft-hqq-2", np.clip(p_te + 0.05, 0, 1), np.clip(p_tu + 0.05, 0, 1),
           y_te, y_tu, ids_te, ids_tu, peak=3 * 1024**3)  # fmt: skip
    out = qa.build(tmp_path, failures={"kev-4b-ft-hqq-2": {"error": "load failed"}})
    rows = {r["contestant"]: r for r in out["rows"]}
    assert set(rows) == {"kev-0.8b-ft-bf16", "kev-0.8b-ft-hqq-2"}
    shrunk = rows["kev-0.8b-ft-hqq-2"]
    assert shrunk["mean_shift_vs_bf16"] == pytest.approx(0.05, abs=1e-3)
    assert shrunk["spearman_vs_bf16"] > 0.99 and abs(shrunk["auc_vs_bf16"]) < 0.01
    assert "auc_vs_bf16_p_holm" in shrunk and "auc_vs_bf16" not in rows["kev-0.8b-ft-bf16"]
    assert shrunk["gpu_weights_gb"] == 2.0 and shrunk["gpu_peak_gb"] == 3.0
    assert shrunk["latency_ms_median"] == 50.0
    assert shrunk["cost_per_1000_usd"] == pytest.approx(0.01)
    assert out["not_scored"]["kev-4b-ft-hqq-2"] == {"recorded_failure": {"error": "load failed"}}
    assert "incomplete" in out["not_scored"]["qwen-ft-bf16"]


def test_record_failure_writes_one_entry_per_pair(tmp_path):
    import quantization as qa

    path = tmp_path / "failures.json"
    qa.record_failure("kev-4b-ft-hqq-2", "load", "boom", path)
    qa.record_failure("qwen-ft-bnb-int8", "smoke", "nan", path)
    got = json.loads(path.read_text("utf-8"))
    assert got["kev-4b-ft-hqq-2"]["precision"] == "hqq-2"
    assert got["qwen-ft-bnb-int8"]["stage"] == "smoke"
    with pytest.raises(ValueError):
        qa.record_failure("laya-ft-hqq-2", "load", "x", path)
