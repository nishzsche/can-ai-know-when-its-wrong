# L4 runbook: M2 smoke tests, then M3 GPU runs

Everything in this project that needs the GPU, in order. Read `STATUS.md` first (setup, `/home` persistence, money). The L4 bills per second, so don't leave it idle.

- **API contestants** (Jev, GPT-6.2 Sol) run on the laptop, not here.
- **No API key needed:** this runbook uses no API key, so the L4 needs no `.env`.

## 0. Every session

```bash
source /home/env.sh
cd /home/can-ai-know-when-its-wrong && git pull
uv sync --group runners --group laya
nvidia-smi                      # check nothing else holds GPU memory
mkdir -p /home/logs
tmux new -s l4 || tmux attach -t l4   # runs survive a dropped SSH connection
```

**One model on the GPU at a time.** vLLM takes ~85% of the GPU, so stop it before starting a Kev server, and vice versa. Run `pkill -f "vllm serve"` / `pkill -f "kev.serve"` on their own lines; inside a compound command they also kill the calling shell.

## 1. Kev server environment (first time only)

```bash
git clone https://github.com/jaredpalmer/kev.git /home/kev
cd /home/kev && git checkout 65b1e763943ace53fd3a4aae00d8d08dbd849772   # main on 2026-11-20, the README we followed
uv sync --extra serve
```

- **Python:** Kev's `.python-version` asks for Python 3.13. uv downloads it into `/home/uv-python`, which persists.
- **flash-linear-attention:** if `kev.serve` warns that it's missing, run `cd /home/kev && uv pip install flash-linear-attention`. It affects speed only.

**Kev v1.0 weights:** start the servers at these exact commits, which the `v1.0` tags point at. The adapter, head and tokenizer files are byte-identical to the README's "weights revision" (checked 2026-11-22). `runners/kev.py` refuses a server that isn't serving v1.0.

| Model | `--run` |
|---|---|
| Kev-4B | `jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c` |
| Kev-0.8B | `jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d` |

## 2. M2: smoke tests (20 tune customers × 2 conditions)

Laya and Sol already passed theirs. Kev-0.8B, Kev-4B and Qwen need the GPU.

**Kev-4B**
```bash
# window 1: server (first start downloads Qwen3.5-4B-Base, ~8 GB, into /home/hf-cache)
cd /home/kev && uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --port 8009 2>&1 | tee /home/logs/kev-4b.log
# window 2: wait until this answers, then run the smoke test
curl -s 127.0.0.1:8009/v1/models
cd /home/can-ai-know-when-its-wrong && uv run python runners/kev.py --size 4b --smoke
```
Then stop the server (Ctrl-C in window 1).

**Kev-0.8B:** the same, with `--run jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d` and `runners/kev.py --size 0.8b --smoke`.

**Qwen3.5-4B:** start vLLM with the exact command in `STATUS.md` ("Serving Qwen3.5-4B"). Once `curl -s 127.0.0.1:8000/v1/models` answers, run:
```bash
uv run --group runners python runners/qwen_logprobs.py --smoke
```
Then `pkill -f "vllm serve"` on its own line.

**Check, then commit:**
```bash
uv run --group runners --group laya pytest -q tests/test_runners.py
git add results/smoke && git commit -m "M2: Kev and Qwen smoke tests on the L4" && git push
```

The pytest smoke checks for `kev-0.8b`, `kev-4b` and `qwen-logprobs` now pass instead of skipping. In each runner's summary line, look at:
- **mean p:** should be plausible against the actual rate (0.20 on these 20 customers).
- **"exactly 0 or 1":** should be ~0. For Qwen, many exact 0s and 1s mean vLLM is returning processed rather than raw logprobs; check `vllm serve --help | grep -i logprobs` and ask for raw logprobs.
- **latency.**

## Gate: before any M3 run

Continue only if both hold; otherwise stop here and **pause the instance**.
1. **M2 is complete:** after `git pull`, `results/smoke/` has all six AI contestants, `jev.jsonl` included (that one comes from the laptop), and `uv run --group runners --group laya pytest -q` passes.
2. **`PREDICTIONS.md` is filled in and committed.** The plan requires the user's guesses to be written before any full run. Check with `grep -c "NOT FILLED IN" PREDICTIONS.md`, which must print 0.

## 3. M3: GPU runs (tune and test, both conditions)

Each contestant writes 8,000 records: 2,000 customers × 2 splits × 2 conditions. Every command resumes where it stopped, so if anything is interrupted, run the same command again.

```bash
# Laya: in-process, no server (~5 min)
uv run --group laya python runners/laya_classic.py --split tune --device cuda
uv run --group laya python runners/laya_classic.py --split test --device cuda

# Qwen: start vLLM as in step 2 first
uv run --group runners python runners/qwen_logprobs.py --split tune
uv run --group runners python runners/qwen_logprobs.py --split test
#   then: pkill -f "vllm serve"

# Kev-4B: start its server as in step 2 first
uv run python runners/kev.py --size 4b --split tune
uv run python runners/kev.py --size 4b --split test
#   then stop the server

# Kev-0.8B: start its server as in step 2 first
uv run python runners/kev.py --size 0.8b --split tune
uv run python runners/kev.py --size 0.8b --split test
```

After each contestant, `wc -l results/predictions/<contestant>.jsonl` should print 8000.

## 4. Wrap up

```bash
uv run --group runners --group laya python runners/manifest.py --allow-incomplete
git add results/predictions results/run_manifest.json
git commit -m "M3: GPU contestants' tune and test predictions" && git push
```

- **The manifest script:** `runners/manifest.py` checks coverage and writes `results/run_manifest.json`. `--allow-incomplete` is needed here because Jev and Sol run on the laptop; the final manifest without that flag is written once every contestant is in.
- **Pause the instance.** Keep it paused, not deleted, until M4's metrics confirm nothing needs re-running. Then delete it, since storage bills while paused.
