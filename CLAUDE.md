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

## Working with the author
- **Times in IST.** Whenever you tell the author a time (ETAs, timestamps, "finished at"), give it in IST (UTC+5:30). Logs and records stay in UTC; convert when reporting. The L4 box's clock is UTC.

## Stack
- Python 3.12, `uv` for environments, `ruff` for lint, `pytest` for tests. scikit-learn, LightGBM or XGBoost, pandas, matplotlib.
- Open models served on a rented NVIDIA L4 (24 GB). GPU time is billed per minute: develop and test on CPU/APIs first, use the GPU only for the open-model runs, then shut it down.
- APIs: Jev via OpenRouter or TypeSafe; one frontier chat model. Keys come from environment variables (`OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, etc.); never commit them.

## Layout
```
data/        load.py, render.py, splits.py, splits/{train,tune,test}.parquet, rendered/{tune,test}.jsonl, contamination.md
baselines/   scorecard.py, gbm.py
runners/     one script per AI contestant -> results/predictions/<contestant>.jsonl
training/    M6: export.py, qwen_lora.py, laya_ft.py (fine-tuning on train; data/ is git-ignored)
analysis/    metrics.py, figures.py, report.ipynb
results/     predictions/, figures/, summary.csv, run_manifest.json
writeup/     blog.md, deep_dive.md, carousel.md, x_post.md, accuracy_post.md (post 2 draft)
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
- `uv run --group memcheck --group runners python data/memdepth.py --model openai/gpt-6.2-sol` — how deep Sol's memory of the file goes; writes the memorized-customer list `summarize.py` excludes for the with/without table
- `uv run --group runners python data/recognition.py --model openai/gpt-6.2-sol` — does Sol recognize a summary as this dataset, and does it know the default rate
- `uv run --group runners python runners/checks/disguise.py --contestant <jev|laya|frontier-verbal>` — the contestant on 2,000 test summaries without NT$, dates or raw codes, into `results/checks/disguise/` (`--control` = the original summaries for 200 customers, the noise floor; Laya needs `--group laya` and runs on CPU; `--probe 50` = Sol's recognition question on disguised summaries; `--report` compares)
- `uv run --group runners python runners/checks/sol_false_rate.py --workers 4` — Sol on 500 test customers told the base rate is 10.0% instead of 22.1% (`--report` gives the pass-through)
- `uv run python runners/<name>.py --split test --condition out_of_box` — run one contestant (`--smoke` = 20 tune customers into `results/smoke/`; Laya needs `--group laya`, Qwen/API runners `--group runners`; keys come from a git-ignored `.env`)
- `uv run --group runners python runners/manifest.py` — check every contestant's coverage and write `results/run_manifest.json` (`--allow-incomplete` to write a partial one)
- `uv run python analysis/recalibrate.py` — the `recalibrated` condition: Platt vs isotonic chosen per AI contestant by 5-fold CV on tune, applied to test (re-run when a contestant completes)
- `uv run python analysis/summarize.py` — score every contestant x condition on test -> `results/summary.csv`, `results/summary_detail.json`, plus `results/summary_excluding_memorized.csv` once `memdepth.py` has run
- `uv run python analysis/explain.py` — why the results look as they do: by repayment status, prediction spread, and calibration at other default rates -> `results/explain.json`
- `uv run python analysis/robustness.py` — the review panel's statistics: equal-mass ECE, perfect-calibration noise floor, Hosmer–Lemeshow, paired AUC vs scorecard with Holm, approval at 80%, label-free shift to 22.1%, scorecard on 2,000 tune labels -> `results/robustness.json`
- `uv run --group runners python runners/checks/disguise.py --contestant <c> --variant aligned --split tune` — layout check: each bill beside the following month's payment (tune only)
- `uv run python analysis/figures.py` — regenerate all five charts into `results/figures/`, each as a light and a `-dark.png` version for the blog's dark mode (run the two above first)
- **M6, the fair fight (PLAN.md §11):**
  - `uv run python data/render.py --split train && uv run --group runners python training/export.py` builds the training files (git-ignored).
  - `uv run python data/neighbours.py` finds each customer's 20 nearest `train` customers and writes the `neighbours-20` reference.
  - `uv run --group runners python runners/examples.py --contestant <jev|sol> --split <tune|test>` runs Jev or Sol with 20 examples in the prompt (`--show 1` prints a request without calling the API).
  - `training/qwen_lora.py`, `training/laya_ft.py` and Kev's `kev.train` fine-tune the models on the L4. `runners/qwen_ft.py`, `runners/kev.py --finetuned` and `runners/laya_classic.py --checkpoint` score them. Exact commands are in L4_RUNBOOK.md §6.
  - `uv run --group runners python runners/manifest.py --m6` checks M6 coverage.
  - `uv run --group tabular python baselines/tabular_fm.py --model <kumo-tabular|tabicl-v2>` runs a tabular foundation model, with `train` as its context (GPU; `--smoke` for a quick check).
  - `uv run python analysis/fair_fight.py` builds the M6 scoreboard: zero-shot → trained per model against the baselines, with paired AUC tests -> `results/fair_fight.csv`.
  - `uv run python analysis/fair_fight_figures.py [name ...]` draws post 3's seven charts (`results/figures/m6_*.png`, light and dark) and its social card (`m6_social_card.png`); names such as `cost how_close` or `social_card` redraw only those.
- `uv run python analysis/accuracy.py` — post 2's numbers and chart (accuracy vs a lending decision, from the M3 predictions; cut-offs chosen on tune) -> `results/accuracy.json`, `results/figures/post2_profit.png`
- `uv run python writeup/build_carousel.py` — the LinkedIn carousel PDF (`writeup/carousel.pdf`) from `writeup/carousel.md` and the results (`--png DIR` for slide previews)
- `uv run python writeup/build_accuracy_carousel.py` — post 2's carousel (`writeup/accuracy_carousel.pdf`) from the `## Slides` section of `writeup/accuracy_linkedin.md` (`--png DIR` for previews). Every post gets one; see the carousel rules in `writeup/STYLE.md`.
- `uv run python writeup/build_fair_fight_carousel.py` — post 3's carousel (`writeup/fair_fight_carousel.pdf`) from the `## Slides` section of `writeup/fair_fight_linkedin.md`, drawn from the M6 predictions through `analysis/fair_fight_figures.py` (`--png DIR` for previews; the paired test on slide 3 takes a minute or two)
- `uv run python writeup/build_quantization_carousel.py` — post 4's carousel (`writeup/quantization_carousel.pdf`) from the `## Slides` section of `writeup/quantization_linkedin.md`, drawn from `results/quantization.csv`, the speed results and the predictions through `analysis/quantization_figures.py` (`--png DIR` for previews)
- **Posts 4 and 5 (PLAN.md §12; the GPU commands are in L4_RUNBOOK.md §7):**
  - **Precision tags:** `bf16`, `bnb-int8`, `bnb-nf4`, `hqq-4`, `hqq-3`, `hqq-2` (`runners/quantize.py`; the pinned libraries are in `--group quant`).
  - **Serving and scoring:**
    - `runners/kev_quant_serve.py --precision <tag> --run DIR` serves a fine-tuned Kev with its backbone quantized; it runs in Kev's environment.
    - `runners/kev.py --finetuned DIR --precision <tag>` and `runners/qwen_ft.py --adapter DIR --precision <tag> [--both-splits]` score the shrunk models. Run `--precision bf16 --reproduce-check 50` first.
  - `uv run --group runners python runners/manifest.py --quant` checks post 4's coverage.
  - `uv run python analysis/quantization.py` scores post 4 -> `results/quantization.csv`. `--record-failure NAME --stage S --error TEXT` records a precision that couldn't run.
  - `uv run python analysis/quantization_figures.py [name ...]` draws post 4's four charts (`results/figures/post4_*.png`, light and dark); names such as `drift cost` redraw only those.
  - **Post 4 speed addendum (L4_RUNBOOK.md §8):**
    - **Speed tags** (`runners/speed.py`): `bf16-tuned`, `bnb-nf4-tuned`, `hqq-4-aoint4`, `hqq-4-gemlite`, `fp8-dynamic`. The launcher, `runners/kev.py` and `runners/qwen_ft.py` take them as `--precision`, and `--reproduce-check 50` works for the two tuned tags.
    - `runners/throughput.py --kev <size> --finetuned DIR --precision <tag>` (or `--qwen ADAPTER`) measures 500 tune customers at 1, 8 and 32 at a time -> `results/checks/speed/`.
    - `uv run python analysis/speed_report.py` -> `results/speed.csv`, `results/speed_throughput.csv`, `results/speed.json`.
  - `uv run python data/formats.py --show ID` / `--token-report` show post 5's four formats for one customer / count tokens per format.
  - `uv run --group runners python runners/checks/disguise.py --contestant <c> --variant <prose|markdown|csv|json>` runs one format on 2,000 test customers, into `results/checks/formats/` (`--report` compares each with v2).
  - `uv run python analysis/layouts.py` scores post 5 -> `results/layouts.csv`, `.json`: the four formats plus the realigned table against v2, with paired tests (Holm over 30), the share moved 5+ points and decisions at v2's tune cut-off (about 4 minutes).
  - `uv run python analysis/layouts_figures.py [name ...]` draws post 5's five charts (`results/figures/post5_*.png`, light and dark); names such as `moved row_pieces` redraw only those.
