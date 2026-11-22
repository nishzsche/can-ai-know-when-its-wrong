"""Tests for the memorization follow-ups: the depth walk's scoring (data/memdepth.py), the
recognition probe's helpers (data/recognition.py) and the with/without comparison in
analysis/summarize.py. No model calls."""

import memdepth as md
import numpy as np
import pytest
import recognition as rg
import records as rc
import summarize as sm

# ---- depth walk scoring ----------------------------------------------------------------------


def fake_csv(n: int = 40) -> str:
    """A file shaped like the UCI CSV: header, then n customers with 25 fields each."""
    rng = np.random.default_rng(0)
    lines = ["ID," + ",".join(f"X{i}" for i in range(1, 24)) + ",Y"]
    for cid in range(1, n + 1):
        limit = int(rng.integers(1, 50)) * 10000
        demo = [int(rng.integers(1, 3)), int(rng.integers(1, 4)), int(rng.integers(1, 3)),
                int(rng.integers(21, 70))]  # fmt: skip
        pay = [int(x) for x in rng.integers(-2, 3, 6)]
        money = [int(x) for x in rng.integers(1, 90000, 12)]
        lines.append(",".join(map(str, [cid, limit, *demo, *pay, *money, cid % 2])))
    return "\n".join(lines) + "\n"


def test_row_starts_and_cut_offset():
    data = fake_csv()
    starts, lines = md.row_starts(data), data.split("\n")
    assert data[starts[7] :].startswith("7,")
    off = md.cut_offset(data, starts, 7)
    assert starts[7] + len(lines[7]) // 3 <= off < starts[7] + 2 * len(lines[7]) // 3
    assert off == md.cut_offset(data, starts, 7)  # reproducible


def score_answer(data: str, k: int, answer: str) -> dict:
    starts = md.row_starts(data)
    off = md.cut_offset(data, starts, k)
    return md.score(data, md.file_rows(data), k, off, starts[k], answer, 300)


def continuation(data: str, k: int, chars: int = 300) -> str:
    off = md.cut_offset(data, md.row_starts(data), k)
    return data[off : off + chars]


def test_perfect_continuation_counts_whole_rows_and_the_cut_row():
    data = fake_csv()
    s = score_answer(data, 5, continuation(data, 5))
    rows = md.file_rows(data)
    expected_full = [c for c in range(6, 41) if f"\n{rows[c]}\n" in "\n" + continuation(data, 5)]
    assert s["exact_rows"] == expected_full and expected_full
    assert s["cut_row_tail_exact"] and s["cut_row_tail_informative"]
    assert s["memorized"] == [5, *expected_full]
    assert s["exact_prefix_chars"] == 300


def test_altered_field_drops_only_that_customer():
    data = fake_csv()
    rows = md.file_rows(data)
    answer = continuation(data, 5).replace(rows[7], rows[7].replace(",", ",9", 1))
    s = score_answer(data, 5, answer)
    assert 7 not in s["memorized"] and 6 in s["memorized"]
    assert any(r["id"] == 7 for r in s["inexact_rows"])


def test_code_fence_is_ignored():
    data = fake_csv()
    plain = score_answer(data, 5, continuation(data, 5))
    fenced = score_answer(data, 5, "```csv\n" + continuation(data, 5) + "\n```")
    assert fenced["memorized"] == plain["memorized"]


def test_rows_from_the_prompt_dont_count():
    data = fake_csv()
    rows = md.file_rows(data)
    answer = continuation(data, 10).split("\n")[0] + "\n" + rows[3] + "\n" + rows[4]
    s = score_answer(data, 10, answer)
    assert s["exact_rows"] == []


def test_all_zero_tail_is_not_informative():
    tail = "0,0,0,0,0,0,0,0,0,0,0,0,0".split(",")
    assert not md.has_nonzero_money(tail, max(0, len(tail) - 13))
    tail = "0,0,0,0,0,0,0,0,0,0,689,0,1".split(",")
    assert md.has_nonzero_money(tail, max(0, len(tail) - 13))
    assert not md.has_nonzero_money(["0", "1"], 0)  # the label alone isn't money


def test_wrong_answer_memorizes_nothing():
    data = fake_csv()
    s = score_answer(data, 5, "I can't reproduce rows from this dataset.")
    assert s["memorized"] == [] and s["exact_prefix_chars"] == 0


# ---- recognition probe -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "named"),
    [
        ({"named_dataset": "UCI Default of Credit Card Clients", "source": ""}, True),
        (
            {"named_dataset": "", "source": "Likely the Taiwan credit card default data (2005)."},
            True,
        ),  # fmt: skip
        ({"named_dataset": "Yeh & Lien (2009)", "source": ""}, True),
        ({"named_dataset": "", "source": "A synthetic example of a bank statement."}, False),
        ({"named_dataset": "German Credit", "source": "A public credit dataset."}, False),
    ],
)
def test_names_this_dataset(answer, named):
    assert rg.names_this_dataset(answer) is named


def test_sample_is_fixed_and_adds_memorized_customers_once():
    rendered = [{"id": f"cust_{i:05d}"} for i in range(1, 501)]
    a = rg.sample_customers(rendered, 20, ["cust_00004", "cust_99999"])
    b = rg.sample_customers(rendered, 20, ["cust_00004", "cust_99999"])
    assert [r["id"] for r in a] == [r["id"] for r in b]
    ids = [r["id"] for r in a]
    assert "cust_00004" in ids and "cust_99999" not in ids and len(ids) == len(set(ids))


def test_recognition_schema_is_strict():
    for schema in (rg.RECOGNITION_SCHEMA, rg.KNOWLEDGE_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


# ---- with/without memorized customers --------------------------------------------------------


def test_build_excludes_the_same_customers_for_everyone(tmp_path, monkeypatch):
    n = 60
    monkeypatch.setattr(sm, "N_TEST", n)
    monkeypatch.setattr(sm, "CONTESTANTS", ("jev", "laya"))
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, n)
    for contestant in ("jev", "laya"):
        recs = [
            rc.make_record(
                id=f"cust_{i + 1:05d}", split="test", contestant=contestant,
                condition="out_of_box", model_version="v1",
                p_default=float(rng.uniform(0.01, 0.99)),
                label=int(labels[i]), latency_ms=1.0, cost_usd=0.0, ts="2026-11-22T00:00:00Z",
            )
            for i in range(n)
        ]  # fmt: skip
        rc.write_records(contestant, recs, tmp_path)
    full, _ = sm.build(tmp_path, n_boot=20)
    without, _ = sm.build(tmp_path, n_boot=20, exclude=frozenset({"cust_00004", "cust_00005"}))
    assert [r["n"] for r in full] == [n, n]
    assert [r["n"] for r in without] == [n - 2, n - 2]


def test_memorized_ids_reads_the_depth_result(tmp_path):
    assert sm.memorized_ids(tmp_path / "missing.json") == frozenset()
    path = tmp_path / "depth.json"
    path.write_text('{"memorized_ids": ["cust_00004", "cust_00005"]}', "utf-8")
    assert sm.memorized_ids(path) == {"cust_00004", "cust_00005"}
