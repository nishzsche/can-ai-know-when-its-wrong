# Status and handoff

Last updated 2026-11-22. Read this together with `PLAN.md` and `CLAUDE.md` before starting work.

## Where things stand

**M1 (data) is done, except the two LLM memorization runs.**

- `data/load.py`: fetches UCI 350 through `ucimlrepo`, drops SEX/MARRIAGE/AGE/EDUCATION before saving, and records the version, a content hash and quirk counts in `data/raw/source.json`.
- `data/splits.py`: stratified 20,000 / 2,000 / 2,000 / 6,000 split, seed 20261122. The default rate is 22.10–22.13% in every split and 22.12% for train.
- `data/render.py`: template **v2**. `data/rendered/{tune,test}.jsonl` hold ready-to-send System One requests for the `out_of_box` and `base_rate` conditions.
- `README.md` has the licence, citation and data quirks; `data/contamination.md` has the per-contestant contamination analysis.
- Tests: `uv run pytest`, 41 passing. The parked insurance work has its own 67 tests: `uv run pytest parked/insurance`.

- **Qwen memorization test: done (2026-11-22).** Qwen3.5-4B, chat and raw-completion modes, both file versions: 0/98 rows reproduced and no non-trivial header continuation. Details and the results table are in `data/contamination.md`.

**Still open for M1:**
1. **Frontier-model memorization test.** Needs `OPENROUTER_API_KEY` and a choice of frontier model (the user hasn't picked one yet). Run `uv run --group memcheck python data/memcheck.py --model <openrouter-id>`, then add its rows to the table in `data/contamination.md`. It needs no GPU.

## Decisions already made (don't revisit without the user)

- **Insurance set parked** in `parked/insurance/` for a later post.
- **Never disadvantage the AI contestants.** Template v2 dropped the "common interpretation" of repayment codes because the data contradicts it (e.g. 40.6% of "−2 = no balance" months carry a positive bill). Labels now say only what's documented and always show the raw code, for example "not late (code -2)" and "paid on time (code -1)". Use train data only for checks like this.
- **Base-rate condition:** the sentence "In this portfolio, about 22.1% of customers miss next month's payment." goes in front of the question instructions. The `state` stays identical across conditions.
- **Recalibrated condition** is computed from the saved `out_of_box` probabilities, so it needs no new API calls.
- **`qwen-logprobs` contestant = `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`** (user chose it over Qwen3-4B-Instruct-2507, 2026-11-22). It's the newest ~4B Qwen, Apache-2.0. Run it text-only with thinking off (`enable_thinking=false`) so P(yes) comes from the first answer token's logprobs.

## Things to know for M2 (runners)

- System One `noul` answers already give P(yes); use that as `p_default`. The rescaled `confidence` formula only applies to `choice` questions.
- English Laya (`convaiinnovations/laya` @ `fd4f9dbc`) has a 512-token context, with about 320 tokens left for the state. Summaries are 192 tokens at most. The card says base Laya is near chance zero-shot on its own benchmark.
- Kev: model cards at github.com/jaredpalmer/kev (checked at commit `65b1e763`). Kev-4B in bf16 is about 12.5 GB; the README only lists L40S/H100 as tested, so fall back to OpenRouter if the L4 runs out of memory.
- Jev on OpenRouter: on 2026-11-22 the model list showed `typesafe/jev-router`, not a versioned `jev-1.x` ID. Check which ID to use and record the resolved version from each response.
- GPU time on the L4 is billed per minute. Do CPU work (baselines, runner code, API runs) wherever it's cheapest and keep the GPU for serving open models.

## The L4 instance (JarvisLabs, launched 2026-11-22)

1× NVIDIA L4 (24 GB), PyTorch template (a container), region IN2, 28 cores and 124 GB RAM. Compute bills per second; storage keeps billing while the instance is paused.

**Only `/home` survives a pause.** Anything installed elsewhere is gone after a resume: system packages, `~/.cache`, a `gh` login. So keep the repo, uv, its Python builds and every model cache under `/home`.

### First-time setup

```bash
cat > /home/env.sh <<'EOF'
export HF_HOME=/home/hf-cache                  # model weights (~25 GB), survive pauses
export UV_CACHE_DIR=/home/uv-cache
export UV_PYTHON_INSTALL_DIR=/home/uv-python   # venvs link to this Python, so it must persist
export PATH=/home/bin:$PATH
EOF
source /home/env.sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/home/bin sh

# The repo is private: authenticate first (`gh auth login`, installing gh if missing,
# or use a GitHub personal access token), then clone it under /home.
cd /home && git clone https://github.com/nishzsche/can-ai-know-when-its-wrong.git
cd can-ai-know-when-its-wrong
uv sync --group memcheck
uv run pytest                     # 41 passing; the token test downloads two tokenizers
export OPENROUTER_API_KEY=...     # never commit keys; repeat in each new shell
```

### After every resume or new shell

Run `source /home/env.sh`, then redo anything that lived outside `/home`, such as `OPENROUTER_API_KEY`.

**GitHub:** `gh` 2.105.0 is installed at `/home/bin/gh`. `env.sh` sets `GH_CONFIG_DIR=/home/gh-config`, so the `gh` login (and its token) survives pauses, and this repo's local git config uses `gh` as the credential helper for github.com. `git push` works only in a shell that has sourced `env.sh`. Log in again with `gh auth login -h github.com -p https -w` if you delete `/home/gh-config`.

### Running models on the GPU

- **One model at a time.** vLLM takes ~90% of GPU memory by default, so stop it (or start it with `--gpu-memory-utilization`) before starting a Kev server.
- **Memory fits:**

  | Model | GPU memory |
  |---|---|
  | Kev-4B | 14.3 GB resident, ~16 GB peak while its adapter loads |
  | Kev-0.8B | 3.8 GB |
  | Laya | well under 2 GB |
  | Qwen ~4B chat model on vLLM | ~8 GB of weights plus cache |

- **Kev-4B speed:** Kev's README calls the L4 "too slow for Kev-4B" when serving production traffic. Our job is ~8,000 short requests (4,000 customers × 2 conditions), which is fine. Fall back to OpenRouter only if it actually fails.
- **vLLM lives in its own venv, `/home/vllm-env`** (vLLM 0.34.0, torch 2.14.0+cu130). It's kept out of the project lockfile so CPU development stays light. To recreate it: `uv venv /home/vllm-env --python 3.12 && VIRTUAL_ENV=/home/vllm-env uv pip install vllm`. Qwen3.5-4B weights are in `/home/hf-cache` (8.8 GB).
- **Serving Qwen3.5-4B** (~18.5 GB used at 0.85 utilization, up in ~1–2 min). This is the exact command used for the memcheck; use the same one for predictions:
  ```bash
  /home/vllm-env/bin/vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --served-model-name qwen3.5-4b --host 127.0.0.1 --port 8000 --max-model-len 16384 \
    --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization 0.85 --seed 0 > /home/logs/vllm-qwen.log 2>&1
  ```
  Stop it with `pkill -f "vllm serve"`; run that on its own, because inside a compound command it also kills the calling shell.
- **The Qwen memcheck (the first GPU job) is done.** No GPU work is pending until the M2 runners are written and tested on CPU/APIs.

### Money

- **Wallet:** if the JarvisLabs wallet runs out of funds, all resources are released, **storage included** (per JarvisLabs' FAQ). Keep it topped up.
- **Autopause:** it works on a timer or a balance threshold, not on idle detection. Set it, and pause whenever you step away.
