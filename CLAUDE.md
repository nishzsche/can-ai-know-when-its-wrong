# can-ai-know-when-its-wrong

An experiment and write-up: give AI decision models and classical credit-risk models the same 2,000 real credit card customers, ask each for the probability of default, and measure whether those probabilities can be trusted (calibration), the way banks are expected to.

Full spec: see `PLAN.md`. Work milestone by milestone; don't start a milestone before the previous one meets its "done when" checks.

## Hard rules
- **One dataset:** UCI "Default of Credit Card Clients" (Yeh & Lien, 2009), loaded via `ucimlrepo` (id 350). Keep its licence and citation in the README. No other data, and nothing from any employer.
- **No demographic fields in any model input:** `SEX`, `MARRIAGE`, `AGE`, `EDUCATION` are dropped before rendering or training.
- **Same test set for every contestant.** Never tune anything on `test`. Recalibration and threshold choices use `tune` only.
- **One rendering template** for every AI contestant; it's unit-tested and versioned.
- **Record everything:** every prediction is saved with its probability, model version, condition, latency and cost (schema below). Analysis reads only from `results/predictions/`.
- **Be fair in the write-up:** the scorecard and GBM learn from 20,000 labelled customers; the AI contestants are zero-shot. Say so wherever results are compared.
- **Never disadvantage the AI contestants:** their inputs must not be misleading or carry less information than the baselines get (e.g. the template shows raw repayment codes rather than an unverified interpretation). Flag any choice that could tilt the comparison before making it.

Current status, open items and decisions already made: `STATUS.md`.

## Stack
- Python 3.12, `uv` for environments, `ruff` for lint, `pytest` for tests. scikit-learn, LightGBM or XGBoost, pandas, matplotlib.
- Open models served on a rented NVIDIA L4 (24 GB). GPU time is billed per minute: develop and test on CPU/APIs first, use the GPU only for the open-model runs, then shut it down.
- APIs: Jev via OpenRouter or TypeSafe; one frontier chat model. Keys come from environment variables (`OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, etc.); never commit them.

## Layout
```
data/        load.py, render.py, splits.py, splits/{train,tune,test}.parquet, rendered/{tune,test}.jsonl, contamination.md
baselines/   scorecard.py, gbm.py
runners/     one script per AI contestant -> results/predictions/<contestant>.jsonl
analysis/    metrics.py, figures.py, report.ipynb
results/     predictions/, figures/, summary.csv, run_manifest.json
writeup/     blog.md, carousel.md, x_post.md
PREDICTIONS.md   my guesses, written BEFORE any runs
```

## Prediction record (one JSON object per line)
```json
{"id": "cust_01234", "split": "test", "contestant": "kev-4b", "condition": "out_of_box",
 "model_version": "jaredpalmer/kev-4b@<rev>", "p_default": 0.31, "label": 1,
 "latency_ms": 184, "cost_usd": 0.0000, "ts": "2026-11-23T10:00:00Z"}
```
`condition` is one of `out_of_box`, `base_rate`, `recalibrated` (baselines use `trained` / `trained_calibrated`).

## Commands (fill in as they exist)
- `uv run python data/load.py && uv run python data/splits.py` — fetch data and build splits
- `uv run python data/render.py --split tune` — render account summaries
- `uv run python data/render.py --show 10` / `--token-report` — hand-check summaries / token counts per contestant tokenizer
- `uv run python baselines/scorecard.py` / `uv run python baselines/gbm.py` — train the baselines on `train` and write their `tune`/`test` predictions (CPU, seconds)
- `uv run pytest` — render, split, record, baseline and metric tests
- `uv run --group memcheck python data/memcheck.py --model <id>` — LLM memorization tests (needs `OPENROUTER_API_KEY`; `--dry-run` works offline)
- `uv run python runners/<name>.py --split test --condition out_of_box` — run one contestant (`--smoke` = 20 tune customers into `results/smoke/`; Laya needs `--group laya`, Qwen/API runners `--group runners`; keys come from a git-ignored `.env`)
- `uv run python analysis/figures.py` — regenerate all charts
