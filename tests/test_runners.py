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
    runs = json.loads((tmp_path / "raw" / "fake-model.meta.json").read_text("utf-8"))
    assert len(runs) == 1 and runs[0]["model_version"] == "fake@1" and runs[0]["kind"] == "test"
    assert runs[0]["split"] == "tune" and "python" in runs[0]["environment"]


def test_each_run_adds_a_metadata_entry(tmp_path):
    harness.run(FakePredictor(), _args(tmp_path, limit=2), log=lambda _: None)
    harness.run(FakePredictor(), _args(tmp_path, limit=2, split="test"), log=lambda _: None)
    runs = json.loads((tmp_path / "raw" / "fake-model.meta.json").read_text("utf-8"))
    assert [r["split"] for r in runs] == ["tune", "test"]


def test_a_file_never_mixes_model_versions(tmp_path):
    harness.run(FakePredictor(), _args(tmp_path, limit=2), log=lambda _: None)
    newer = FakePredictor()
    newer.model_version = "fake@2"
    with pytest.raises(SystemExit, match="mix model versions"):
        harness.run(newer, _args(tmp_path, limit=3), log=lambda _: None)
    assert newer.calls == 0


def test_recalibrated_records_dont_block_a_resume(tmp_path):
    path = harness.run(FakePredictor(), _args(tmp_path, limit=2), log=lambda _: None)
    first = json.loads(path.read_text("utf-8").splitlines()[0])
    recal = {**first, "split": "test", "condition": "recalibrated",
             "model_version": "fake@1; recalibrated: platt fit on tune"}  # fmt: skip
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(recal) + "\n")
    pred = FakePredictor()
    harness.run(pred, _args(tmp_path, limit=3), log=lambda _: None)
    assert pred.calls == 2  # customer 3, both conditions


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


@pytest.mark.parametrize(
    "served",
    [
        "jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c",  # v1.0 tag commit
        "jaredpalmer/kev-4b@139fdd94",  # weights revision from the README table
        "jaredpalmer/kev-4b@v1.0",  # the tag name itself
    ],
)
def test_kev_predictor_accepts_every_name_for_v1_0(served):
    import kev

    server = _fake_kev_server({"data": [{"id": "kev-latest", "run": served}]})
    try:
        pred = kev.KevPredictor("4b", f"http://127.0.0.1:{server.server_port}", usd_per_hour=0.44)
        assert pred.model_version == "jaredpalmer/kev-4b@v1.0 (591dcb5b; weights 139fdd94)"
    finally:
        server.shutdown()


def test_kev_predictor_reads_noul_from_the_system_one_response():
    import kev

    server = _fake_kev_server(
        {"data": [{"id": "kev-latest", "run": "jaredpalmer/kev-4b@139fdd94"}]}
    )
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        pred = kev.KevPredictor("4b", url, usd_per_hour=0.44)
        assert pred.contestant == "kev-4b"
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


# ---- frontier-verbal client against a fake OpenRouter --------------------------------------


def _fake_openrouter(contents):
    """Chat Completions server that answers with the queued `contents`, one per request."""
    queue, prompts = list(contents), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            assert req["model"] == "openai/gpt-6.2-sol"
            assert req["reasoning"] == {"effort": "medium"} and req["seed"] == 0
            assert req["response_format"]["json_schema"]["strict"] is True
            prompts.append(req["messages"][0]["content"])
            if queue and queue[0] == "RATE_LIMIT":
                queue.pop(0)
                body = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})
                self.send_response(429)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())
                return
            body = json.dumps({
                "id": "gen-1", "object": "chat.completion", "created": 0,
                "model": "openai/gpt-6.2-sol", "provider": "OpenAI",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": queue.pop(0)}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 300, "total_tokens": 700,
                          "cost": 0.0038},
            }).encode()  # fmt: skip
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, prompts


def _verbal(server, retries=2):
    import frontier_verbal as fv

    url = f"http://127.0.0.1:{server.server_port}"
    pred = fv.FrontierVerbalPredictor(
        base_url=url, api_key="test-key", retries=retries, rpm=None, backoff_s=0.05
    )
    return fv, pred


def test_frontier_waits_out_a_rate_limit_without_counting_it_as_latency():
    server, prompts = _fake_openrouter(["RATE_LIMIT", "RATE_LIMIT", '{"probability": 12}'])
    try:
        _, pred = _verbal(server)
        pred.backoff_s = 0.5  # 0.5 s, then 1.0 s of waiting before the call that succeeds
        out = pred.predict(harness.load_requests("tune", 1)[0]["requests"]["out_of_box"])
        assert out.p_default == pytest.approx(0.12) and len(prompts) == 3
        assert out.raw["attempts"][-1]["waited_for_rate_limit_s"] == pytest.approx(1.5)
        assert out.latency_ms < 1000  # the 1.5 s of waiting isn't Sol's latency
    finally:
        server.shutdown()


def test_rate_limiter_spaces_request_starts():
    import time as _time

    limiter = harness.RateLimiter(per_minute=600)  # one start every 0.1 s
    t0 = _time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert _time.monotonic() - t0 >= 0.29
    assert harness.RateLimiter(None).interval == 0


def test_frontier_prompt_is_the_shared_request_plus_one_answer_line():
    server, prompts = _fake_openrouter(['{"probability": 23}'])
    try:
        fv, pred = _verbal(server)
        request = harness.load_requests("tune", 1)[0]["requests"]["base_rate"]
        out = pred.predict(request)
        assert out.p_default == pytest.approx(0.23) and out.cost_usd == pytest.approx(0.0038)
        q = request["questions"]["will_default"]["instructions"]
        assert prompts[0] == f"{request['state']}\n\n{q}\n\n{fv.ANSWER_LINE}"
        assert "22.1%" in prompts[0]  # the base-rate sentence reaches the model
    finally:
        server.shutdown()


def test_frontier_retries_a_malformed_answer_and_counts_both_costs():
    server, _ = _fake_openrouter(["not json", '{"probability": 140}', '{"probability": 7.5}'])
    try:
        _, pred = _verbal(server, retries=2)
        out = pred.predict(harness.load_requests("tune", 1)[0]["requests"]["out_of_box"])
        assert out.p_default == pytest.approx(0.075)
        assert out.cost_usd == pytest.approx(3 * 0.0038)
        assert len(out.raw["attempts"]) == 3
    finally:
        server.shutdown()


def test_frontier_gives_up_without_inventing_a_probability(tmp_path):
    server, _ = _fake_openrouter(["I can't help with that."] * 3 + ['{"probability": 40}'] * 3)
    try:
        _, pred = _verbal(server, retries=2)
        logs = []
        args = _args(tmp_path, limit=2, condition="out_of_box")
        path = harness.run(pred, args, log=logs.append)
        recs = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        first, second = (r["id"] for r in harness.load_requests("tune", 2))
        assert [r["id"] for r in recs] == [second]  # the refused customer is skipped
        assert any(first in line and "will retry" in line for line in logs)
        raw = [json.loads(line) for line in (tmp_path / "raw" / "frontier-verbal.jsonl")
               .read_text("utf-8").splitlines()]  # fmt: skip
        assert raw[0]["id"] == first and "failed" in raw[0]
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("content", "expected"),
    [('{"probability": 0}', 0.0), ('{"probability": 100}', 1.0), ('{"probability": 22.5}', 0.225)],
)
def test_parse_probability_accepts_the_full_range(content, expected):
    import frontier_verbal as fv

    assert fv.parse_probability(content) == pytest.approx(expected)


@pytest.mark.parametrize(
    "content", ["", "{}", '{"probability": "high"}', '{"probability": -1}', '{"probability": true}']
)
def test_parse_probability_rejects_bad_answers(content):
    import frontier_verbal as fv

    with pytest.raises((ValueError, KeyError)):
        fv.parse_probability(content)


# ---- Qwen logprob readout (bypasses jevper 0.7.12's last-variant-wins bug) -----------------------

import math  # noqa: E402

import qwen_logprobs as ql  # noqa: E402

# First-token top logprobs recorded for cust_00005 in the 2026-11-22 L4 smoke run (trimmed).
# jevper reported P(yes) = 4.0e-5 because " A" (-10.54) overwrote "A" (-1.16).
CUST_00005_TOP = [
    {"token": "B", "logprob": -0.41}, {"token": "A", "logprob": -1.16},
    {"token": "No", "logprob": -3.91}, {"token": "Yes", "logprob": -6.66},
    {"token": " B", "logprob": -9.60}, {"token": " A", "logprob": -10.54},
    {"token": "b", "logprob": -11.60},
]  # fmt: skip


def _lse(*xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def test_readout_combines_variants_instead_of_last_one_winning():
    p, missing = ql.p_yes_from_top_logprobs(CUST_00005_TOP, "A", "B")
    a, b = _lse(-1.16, -10.54), _lse(-0.41, -9.60, -11.60)
    assert missing == []
    assert p == pytest.approx(math.exp(a) / (math.exp(a) + math.exp(b)))
    assert 0.3 < p < 0.33  # not jevper's 4.0e-5


def test_readout_is_symmetric_in_label_order():
    p, _ = ql.p_yes_from_top_logprobs(CUST_00005_TOP, "A", "B")
    q, _ = ql.p_yes_from_top_logprobs(list(reversed(CUST_00005_TOP)), "A", "B")
    r, _ = ql.p_yes_from_top_logprobs(CUST_00005_TOP, "B", "A")
    assert p == pytest.approx(q) and p == pytest.approx(1 - r)


def test_readout_missing_label_and_bad_values():
    p, missing = ql.p_yes_from_top_logprobs([{"token": "A", "logprob": -0.1}], "A", "B")
    assert p == 1.0 and missing == ["B"]
    with pytest.raises(ValueError):
        ql.p_yes_from_top_logprobs([{"token": "No", "logprob": -0.1}], "A", "B")
    with pytest.raises(ValueError):
        ql.p_yes_from_top_logprobs([{"token": "A", "logprob": 0.5}], "A", "B")


def _raw(options: str, first_token: str, top: list[dict]) -> dict:
    request = {"input": [
        {"role": "system", "content": "You are a precise classification engine."},
        {"role": "user", "content": f"Question:\nWill they default?\n\nOptions:\n{options}"},
        {"role": "user", "content": "<document>\nA: Yes\n</document>"},
    ]}  # fmt: skip
    response = {"output": [{"content": [{"logprobs": [
        {"token": first_token, "logprob": -0.4, "top_logprobs": top},
        {"token": "<|im_end|>", "logprob": -0.04, "top_logprobs": []},
    ]}]}]}  # fmt: skip
    return {"debug": {"llm_attempts": [
        {"question_id": "will_default", "request": request, "response": response}
    ]}}  # fmt: skip


def test_readout_from_raw_reads_letters_from_the_request():
    p, _ = ql.p_yes_from_raw(_raw("A: Yes\nB: No", "B", CUST_00005_TOP), "will_default")
    assert 0.3 < p < 0.33
    # Swapped options: the same tokens now mean the opposite answer.
    q, _ = ql.p_yes_from_raw(_raw("A: No\nB: Yes", "B", CUST_00005_TOP), "will_default")
    assert q == pytest.approx(1 - p)


def test_readout_from_raw_rejects_non_label_answer():
    with pytest.raises(ValueError):
        ql.p_yes_from_raw(_raw("A: Yes\nB: No", "The", CUST_00005_TOP), "will_default")


# ---- Jev client against a fake OpenRouter System One endpoint ------------------------------

RESOLVED = "typesafe/jev-1.14-20261105"


def _fake_systemone(accept_dated=True, drift_after=None):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/systemone"
            assert self.headers["authorization"] == "Bearer test-key"
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append(req["model"])
            if req["model"] == RESOLVED and not accept_dated:
                return self._send(400, {"error": {"code": 400, "message": "unknown model"}})
            model = RESOLVED
            if drift_after is not None and len(seen) > drift_after:
                model = "typesafe/jev-1.15-20261201"
            qid = next(iter(req["questions"]))
            self._send(200, {"id": "gen-dec-1", "model": model, "provider": "TypeSafe",
                             "answers": {qid: {"type": "noul", "noul": 0.37}},
                             "usage": {"cost": 0.00002, "input_tokens": 300,
                                       "output_tokens": 5}})  # fmt: skip

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_jev_resolves_latest_once_then_pins_that_exact_version():
    import jev

    server, seen = _fake_systemone()
    try:
        pred = jev.JevPredictor(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key"
        )
        assert pred.model_version == RESOLVED and pred.request_model == RESOLVED
        out = pred.predict(harness.load_requests("tune", 1)[0]["requests"]["out_of_box"])
        assert out.p_default == 0.37 and out.cost_usd == pytest.approx(0.00002)
        assert seen == ["jev-latest", RESOLVED, RESOLVED]
    finally:
        server.shutdown()


def test_jev_falls_back_to_the_version_family_if_dated_ids_are_refused():
    import jev

    server, seen = _fake_systemone(accept_dated=False)
    try:
        pred = jev.JevPredictor(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key"
        )
        assert pred.request_model == "typesafe/jev-1.14" and pred.model_version == RESOLVED
    finally:
        server.shutdown()


def test_jev_stops_if_the_version_changes_mid_run():
    import jev

    server, _ = _fake_systemone(drift_after=3)
    try:
        pred = jev.JevPredictor(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key"
        )
        request = harness.load_requests("tune", 1)[0]["requests"]["out_of_box"]
        pred.predict(request)  # third call: still 1.14
        with pytest.raises(SystemExit, match="changed version"):
            pred.predict(request)
    finally:
        server.shutdown()


# ---- run manifest ----------------------------------------------------------------------------


def _write_fake(directory, contestant, groups, ids, version="fake@1"):
    recs = [
        rc.make_record(
            id=i,
            split=split,
            contestant=contestant,
            condition=cond,
            model_version=version,
            p_default=0.2,
            label=0,
            latency_ms=10.0,
            cost_usd=0.001,
            ts="2026-11-22T10:00:00Z",
        )  # fmt: skip
        for cond, split in groups
        for i in ids[split]
    ]
    rc.write_records(contestant, recs, directory=directory)


def test_manifest_marks_complete_and_incomplete_contestants(tmp_path):
    import manifest

    ids = {"tune": ["cust_00001", "cust_00002"], "test": ["cust_00003"]}
    groups = manifest.GROUPS["jev"]
    _write_fake(tmp_path, "jev", groups, ids)
    done = manifest.summarize("jev", groups, ids, tmp_path)
    assert done["complete"] and done["records"] == 6 and done["cost_usd_total"] == 0.006
    short = {"tune": ids["tune"][:1], "test": ids["test"]}
    _write_fake(tmp_path, "laya", groups, short)
    partial = manifest.summarize("laya", groups, ids, tmp_path)
    assert not partial["complete"]
    assert partial["groups"]["out_of_box/tune"] == {"records": 1, "missing": 1, "unexpected": 0,
                                                    "complete": False}  # fmt: skip
    assert manifest.summarize("kev-4b", groups, ids, tmp_path) == {
        "present": False,
        "complete": False,
        "groups": {},
    }


def test_manifest_covers_every_contestant_and_its_conditions():
    import manifest

    assert set(manifest.GROUPS) == {*manifest.AI_CONTESTANTS, "scorecard", "gbm"}
    assert len(manifest.GROUPS["frontier-verbal"]) == 4  # 2 conditions x 2 splits
    assert ("trained_calibrated", "tune") not in manifest.GROUPS["gbm"]


def test_manifest_rejects_two_versions_within_one_condition(tmp_path):
    import manifest

    ids = {"tune": ["cust_00001"], "test": ["cust_00003"]}
    groups = manifest.GROUPS["jev"]
    recs = [
        rc.make_record(
            id=ids[s][0],
            split=s,
            contestant="jev",
            condition=c,
            model_version="v1" if s == "tune" else "v2",
            p_default=0.2,
            label=0,
            latency_ms=10.0,
            cost_usd=0.0,
            ts="2026-11-22T10:00:00Z",
        )  # fmt: skip
        for c, s in groups
    ]
    rc.write_records("jev", recs, directory=tmp_path)
    assert not manifest.summarize("jev", groups, ids, tmp_path)["complete"]


# ---- parallel requests (API runners) ---------------------------------------------------------


def test_parallel_run_writes_the_same_records_as_a_sequential_one(tmp_path):
    seq, par = tmp_path / "seq", tmp_path / "par"
    harness.run(FakePredictor(), _args(seq, limit=12), log=lambda _: None)
    pred = FakePredictor()
    harness.run(pred, _args(par, limit=12, workers=4), log=lambda _: None)
    assert pred.calls == 24

    def keyed(d):
        recs = map(json.loads, (d / "fake-model.jsonl").read_text("utf-8").splitlines())
        return {(r["condition"], r["id"]): r["p_default"] for r in recs}

    assert keyed(seq) == keyed(par)
    runs = json.loads((par / "raw" / "fake-model.meta.json").read_text("utf-8"))
    assert runs[-1]["workers"] == 4


def test_parallel_run_stops_and_cancels_on_a_hard_failure(tmp_path):
    class Exploding(FakePredictor):
        def predict(self, request):
            self.calls += 1
            raise SystemExit("version changed mid-run")

    pred = Exploding()
    with pytest.raises(SystemExit, match="version changed"):
        harness.run(pred, _args(tmp_path, limit=50, workers=2), log=lambda _: None)
    assert pred.calls < 100  # queued requests were cancelled, not all sent


def test_run_accepts_an_explicit_customer_list(tmp_path):
    rows = harness.load_requests("tune")
    chosen = [rows[5], rows[2]]
    pred = FakePredictor()
    path = harness.run(pred, _args(tmp_path), log=lambda _: None, requests=chosen)
    ids = {json.loads(line)["id"] for line in path.read_text("utf-8").splitlines()}
    assert ids == {rows[5]["id"], rows[2]["id"]}
