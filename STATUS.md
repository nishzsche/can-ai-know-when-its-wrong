# Status and handoff

Last updated 2026-11-22. Read this together with `PLAN.md` and `CLAUDE.md` before starting work.

## Where things stand

**M1 (data) is done, except the two LLM memorization runs.**

- `data/load.py`: fetches UCI 350 through `ucimlrepo`, drops SEX/MARRIAGE/AGE/EDUCATION before saving, and records the version, a content hash and quirk counts in `data/raw/source.json`.
- `data/splits.py`: stratified 20,000 / 2,000 / 2,000 / 6,000 split, seed 20261122. The default rate is 22.10–22.13% in every split and 22.12% for train.
- `data/render.py`: template **v2**. `data/rendered/{tune,test}.jsonl` hold ready-to-send System One requests for the `out_of_box` and `base_rate` conditions.
- `README.md` has the licence, citation and data quirks; `data/contamination.md` has the per-contestant contamination analysis.
- Tests: `uv run pytest`, 41 passing. The parked insurance work has its own 67 tests: `uv run pytest parked/insurance`.

**Still open for M1:**
1. **Frontier-model memorization test.** Needs `OPENROUTER_API_KEY` and a choice of frontier model (the user hasn't picked one yet). Run `uv run --group memcheck python data/memcheck.py --model <openrouter-id>`.
2. **Qwen memorization test.** OpenRouter has no ~4B Qwen, so this runs on the L4 against vLLM, as the first GPU job and before any Qwen predictions. No API key is needed on localhost: `--base-url http://127.0.0.1:8000/v1 --model <served-name>`.

When the results come in, fill the results table in `data/contamination.md`.

## Decisions already made (don't revisit without the user)

- **Insurance set parked** in `parked/insurance/` for a later post.
- **Never disadvantage the AI contestants.** Template v2 dropped the "common interpretation" of repayment codes because the data contradicts it (e.g. 40.6% of "−2 = no balance" months carry a positive bill). Labels now say only what's documented and always show the raw code, for example "not late (code -2)" and "paid on time (code -1)". Use train data only for checks like this.
- **Base-rate condition:** the sentence "In this portfolio, about 22.1% of customers miss next month's payment." goes in front of the question instructions. The `state` stays identical across conditions.
- **Recalibrated condition** is computed from the saved `out_of_box` probabilities, so it needs no new API calls.

## Things to know for M2 (runners)

- System One `noul` answers already give P(yes); use that as `p_default`. The rescaled `confidence` formula only applies to `choice` questions.
- English Laya (`convaiinnovations/laya` @ `fd4f9dbc`) has a 512-token context, with about 320 tokens left for the state. Summaries are 192 tokens at most. The card says base Laya is near chance zero-shot on its own benchmark.
- Kev: model cards at github.com/jaredpalmer/kev (checked at commit `65b1e763`). Kev-4B in bf16 is about 12.5 GB; the README only lists L40S/H100 as tested, so fall back to OpenRouter if the L4 runs out of memory.
- Jev on OpenRouter: on 2026-11-22 the model list showed `typesafe/jev-router`, not a versioned `jev-1.x` ID. Check which ID to use and record the resolved version from each response.
- GPU time on the L4 is billed per minute. Do CPU work (baselines, runner code, API runs) wherever it's cheapest and keep the GPU for serving open models.

## Setting up on the L4

```bash
git clone https://github.com/nishzsche/can-ai-know-when-its-wrong.git
cd can-ai-know-when-its-wrong
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't installed
uv sync --group memcheck
uv run pytest                                       # 41 passing; the token test downloads two tokenizers
export OPENROUTER_API_KEY=...                       # never commit keys
```
