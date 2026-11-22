"""Tests for the runner harness, the Kev client (against a fake server), .env loading, and the
M2 smoke outputs (each AI runner writes valid records on 20 tune customers)."""

import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import harness
import pytest
import records as rc
from env import load_env

# ---- .env loading ----------------------------------------------------------------------------


def test_load_env_sets_missing_names_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport A_KEY='abc'\nB_KEY=\"def\"\nC_KEY=keep-me\nnot a line\n", "utf-8"
    )
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "already-set")
    assert load_env(env) == ["A_KEY", "B_KEY"]
    import os

    assert os.environ["A_KEY"] == "abc" and os.environ["B_KEY"] == "def"
    assert os.environ["C_KEY"] == "already-set"


def test_load_env_without_file_is_a_no_op(tmp_path):
    assert load_env(tmp_path / "missing.env") == []


def test_env_file_is_git_ignored():
    ignored = (harness.ROOT / ".gitignore").read_text("utf-8").splitlines()
    assert ".env" in ignored


# ---- harness ---------------------------------------------------------------------------------


class FakePredictor:
    contestant = "fake-model"
    model_version = "fake@1"
    meta = {"kind": "test"}

    def __init__(self):
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        assert set(request) == {"state", "questions"}
        p = 0.3 if "portfolio" in request["questions"]["will_default"]["instructions"] else 0.2
        return harness.Prediction(p_default=p, latency_ms=5.0, cost_usd=0.0, raw={"p": p})


def _args(tmp_path, **kw):
    ns = Namespace(split="tune", condition="both", limit=5, smoke=False, out_dir=tmp_path)
    for k, v in kw.items():
        setattr(ns, k, v)
    return harness.resolve_args(ns)


def test_run_writes_valid_records_for_both_conditions(tmp_path):
    pred = FakePredictor()
    path = harness.run(pred, _args(tmp_path), log=lambda _: None)
    recs = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert len(recs) == 10 and pred.calls == 10
    for rec in recs:
        rc.validate_record(rec)
    ids = [r["id"] for r in harness.load_requests("tune", 5)]
    assert [r["id"] for r in recs if r["condition"] == "out_of_box"] == ids
    # the base-rate request carries the extra sentence, the out-of-box one doesn't
    assert {r["p_default"] for r in recs if r["condition"] == "base_rate"} == {0.3}
    raw = (tmp_path / "raw" / "fake-model.jsonl").read_text("utf-8").splitlines()
    assert len(raw) == 10
    meta = json.loads((tmp_path / "raw" / "fake-model.meta.json").read_text("utf-8"))
    assert meta["model_version"] == "fake@1" and meta["kind"] == "test"


def test_run_resumes_without_repeating_work(tmp_path):
    harness.run(FakePredictor(), _args(tmp_path, limit=3), log=lambda _: None)
    pred = FakePredictor()
    path = harness.run(pred, _args(tmp_path, limit=5), log=lambda _: None)
    assert pred.calls == 4  # only customers 4 and 5, for each condition
    keys = [
        (r["condition"], r["id"]) for r in map(json.loads, path.read_text("utf-8").splitlines())
    ]
    assert len(keys) == len(set(keys)) == 10


def test_smoke_flag_means_20_tune_customers_in_results_smoke():
    ns = Namespace(split="test", condition="both", limit=None, smoke=True, out_dir=None)
    args = harness.resolve_args(ns)
    assert (args.split, args.limit, args.out_dir) == ("tune", 20, harness.SMOKE_DIR)
    assert args.conditions == ("out_of_box", "base_rate")


def test_gpu_cost_is_time_times_price():
    assert harness.gpu_cost(3_600_000, 0.44) == pytest.approx(0.44)
    assert harness.gpu_cost(100, 0.44) == pytest.approx(0.44 / 36_000)


# ---- Kev client against a fake Kev server ----------------------------------------------------


def _fake_kev_server(models_payload):
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
            self._send(models_payload)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            assert req["model"] == "kev-latest" and "state" in req
            q = req["questions"]["will_default"]
            assert q["type"] == "noul"
            self._send(
                {
                    "model": "kev-latest",
                    "answers": {"will_default": {"type": "noul", "noul": 0.42}},
                    "usage": {"input_tokens": 230, "output_tokens": 9},
                    "latency_ms": 17,
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_kev_predictor_reads_noul_from_the_system_one_response():
    import kev

    server = _fake_kev_server(
        {"data": [{"id": "kev-latest", "run": "jaredpalmer/kev-4b@139fdd94"}]}
    )
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        pred = kev.KevPredictor("4b", url, usd_per_hour=0.44)
        assert pred.contestant == "kev-4b" and pred.model_version == "jaredpalmer/kev-4b@139fdd94"
        request = harness.load_requests("tune", 1)[0]["requests"]["out_of_box"]
        out = pred.predict(request)
        assert out.p_default == 0.42 and out.raw["server_latency_ms"] == 17
        assert out.cost_usd == pytest.approx(harness.gpu_cost(out.latency_ms, 0.44))
    finally:
        server.shutdown()


def test_kev_predictor_refuses_the_wrong_checkpoint():
    import kev

    server = _fake_kev_server(
        {"data": [{"id": "kev-latest", "run": "jaredpalmer/kev-4b@deadbeef"}]}
    )
    try:
        with pytest.raises(SystemExit, match="139fdd94"):
            kev.KevPredictor("4b", f"http://127.0.0.1:{server.server_port}", usd_per_hour=0.44)
    finally:
        server.shutdown()


# ---- M2 check: each AI runner's smoke output -------------------------------------------------

AI_CONTESTANTS = ["jev", "kev-0.8b", "kev-4b", "laya", "qwen-logprobs", "frontier-verbal"]


@pytest.mark.parametrize("contestant", AI_CONTESTANTS)
def test_smoke_output_is_20_valid_tune_records_per_condition(contestant):
    path = harness.SMOKE_DIR / f"{contestant}.jsonl"
    if not path.exists():
        pytest.skip(f"no smoke run yet for {contestant}")
    recs = rc.read_records(contestant, directory=harness.SMOKE_DIR)
    first20 = harness.load_requests("tune", 20)
    for condition in harness.RUN_CONDITIONS:
        part = [r for r in recs if r["condition"] == condition]
        assert [r["id"] for r in part] == [row["id"] for row in first20]
        assert [r["label"] for r in part] == [row["label"] for row in first20]
        assert all(r["split"] == "tune" and r["contestant"] == contestant for r in part)
        assert len({r["model_version"] for r in part}) == 1
