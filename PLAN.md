# Plan: "Can AI actually know when it's wrong?"

First post in a series on trusting AI with decisions that matter. Format: curiosity question → real experiment → technical insight.

**The question:** banks are audited on whether their risk scores are honest: if a model says "30% chance of default", roughly 30% of those customers should default. Do the new AI decision models meet that bar on real credit card customers? And how do they compare with a 20-line logistic-regression scorecard?

---

## 1. Data

**Source:** UCI "Default of Credit Card Clients" (Yeh & Lien, 2009). 30,000 real credit card customers in Taiwan, April–September 2005. Target: `default.payment.next.month` (about 22% default; compute the exact rate). Licence: CC BY 4.0 as listed on UCI/OpenML (confirm on the UCI page and cite Yeh & Lien in the README). Load with `ucimlrepo` (`fetch_ucirepo(id=350)`) and record the version.

**Fields used as model input (financial behaviour only):**
- `LIMIT_BAL`: credit limit (NT$)
- `PAY_0, PAY_2..PAY_6`: repayment status for the last 6 months
- `BILL_AMT1..6`: bill amounts for the last 6 months
- `PAY_AMT1..6`: amounts paid for the last 6 months

**Fields excluded from every model input:** `SEX`, `MARRIAGE`, `AGE`, `EDUCATION`. Using protected characteristics in credit decisions is off-limits in practice, and leaving them out keeps the post about honesty, not demographics. Note the exclusion in the write-up.

**Known data quirks (document them, don't hide them):**
- Repayment status codes −2 and 0 aren't defined in the original documentation (only −1 = "pay duly" and 1–9 = months of delay are). The "common interpretation" (−2 = no balance/no use, −1 = paid in full, 0 = minimum paid) is contradicted by the data, so the template states only what's documented and shows the raw code (−2/0 → "not late (code X)", −1 → "paid on time (code −1)"). This avoids putting the AI models at a disadvantage against baselines that see the raw codes. Decided 2026-11-22 (template v2); README has the numbers.
- Amounts are 2005 NT$; keep the currency as is.

### Rendering each customer as text (`data/render.py`)
Decision models read text, so each customer becomes a short, neutral account summary (~200–300 tokens so every contestant fits). Example shape:

```
Credit card account summary
Credit limit: NT$200,000
Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$45,210 | NT$2,000 | paid minimum (revolving)
Aug 2005 | ...
```

- One fixed template, unit-tested. Same text for every contestant.
- Optional robustness check later: a second wording (prose "loan officer note") on the `tune` split only, to see whether wording moves the probabilities.

### Splits (`data/splits.py`, fixed seed, stratified on the target)
- `train` 20,000: only for the classical baselines
- `tune` 2,000: recalibration and any threshold choices
- `test` 2,000: every contestant is scored here. Never tune on it.
- The remaining 6,000 stay unused as a reserve.

**Done when:** default rate printed per split; 10 rendered summaries read correctly against the raw rows; render tests pass; licence and citation in the README.

### Memorization check (`data/contamination.md`, before any runs)
This is a famous dataset, and LLMs are known to memorize popular tabular datasets. Run the `tabmemcheck` tests (Bordt et al., "Elephants Never Forget") against the general LLM and the frontier model: header test and row-completion test. Decision models (Jev, Kev, Laya) can't generate text, so they can't be probed this way; say so. Report results honestly in the write-up.

## 2. The question asked
- **will_default** (yes/no, "noul" in System One terms): "Based on this account summary, will this customer miss next month's payment?" The model's probability of "yes" is its predicted default risk.
- Use the System One request shape (`state` = account summary, `questions` = the typed question) wherever possible, so every decision model gets identical input.

### Three conditions for the AI contestants
1. **Out of the box:** the summary and the question, nothing else.
2. **Told the base rate:** one added sentence: "In this portfolio, about 22% of customers miss next month's payment." (Use the exact `train` rate.) Tests whether a single line of context fixes the honesty problem.
3. **Recalibrated:** condition 1's probabilities, recalibrated on `tune` (Platt scaling or isotonic regression; temperature scaling where logits are available).

## 3. Contestants

| id | What | Where it runs | Notes |
|---|---|---|---|
| `jev` | TypeSafe Jev (closed) | API: OpenRouter (`typesafe/jev-...`) or `api.typesafe.ai/v1/systemone` | Use `jev-latest`; record the resolved version (1.13 / 1.14) |
| `kev-0.8b` | Kev 0.8B (open, Apache-2.0) | L4 | github.com/jaredpalmer/kev; ships with a fitted temperature |
| `kev-4b` | Kev 4B (open) | L4 (~12.5 GB bf16) or OpenRouter (`jaredpalmer/kev-4b`) | README lists L40S/H100 as tested; fall back to OpenRouter if L4 memory is tight |
| `laya` | Laya (open) | L4 (`pip install laya`) | Small context (~1k tokens): check the summary fits |
| `qwen-logprobs` | Small general LLM (e.g. Qwen 4B instruct) | L4 via vLLM OpenAI server + `jevper` (`method="logprobs"`) | P(yes) from token logprobs |
| `frontier-verbal` | A frontier chat model asked "What is the probability (0–100) that this customer misses next month's payment? Reply as JSON." | API | The relatable "ask ChatGPT" baseline; gives a probability directly |
| `scorecard` | Logistic regression on the same fields (optionally weight-of-evidence binning, the classic credit-risk scorecard) | CPU | Trained on `train`. The old-school reference. |
| `gbm` | Gradient boosting (LightGBM or XGBoost), raw and isotonic-calibrated | CPU | Trained on `train`; calibrated on `tune`. The "modern ML" reference |

Be explicit in every comparison: the scorecard and GBM learn from 20,000 labelled customers; the AI contestants are zero-shot. That *is* the practical question: do you need the new AI, or 20 lines of logistic regression?

**Done when:** each runner works on 20 `tune` customers and writes valid records (see CLAUDE.md schema).

## 4. Metrics (`analysis/metrics.py`, unit-tested)
**Ranking ("can it tell risky from safe customers?")**
- AUC and Gini; KS statistic.

**Honesty ("do its probabilities mean what they say?")**
- Reliability by risk decile: average predicted default rate vs actual default rate in each decile. (Hero chart: "promised vs actual".)
- Calibration-in-the-large: mean predicted vs observed default rate.
- Calibration slope and intercept (logistic recalibration fit).
- ECE (10 equal-width bins) and Brier score.

**Decision view ("what happens if a bank acts on it?")**
- Risk bands (e.g. <10%, 10–20%, 20–40%, >40%): share of customers in each band, promised default rate vs actual.
- "Approve if predicted risk < X": approval rate and actual default rate among approved, for X = 10%, 15%, 20%.

**Practical**
- Cost per 1,000 decisions and median latency per contestant.
- Bootstrap 95% intervals for AUC, ECE and calibration-in-the-large.

## 5. Figures (`analysis/figures.py`)
1. **Promised vs actual default rate by decile**, one line per contestant, diagonal = perfectly honest. (Hero image.)
2. **The honesty arc:** for each AI contestant, calibration error out of the box → told the base rate → recalibrated, with the scorecard as a reference line.
3. AUC / KS bar chart (ranking ability).
4. Risk-band table rendered as an image (promised vs actual).
5. Cost vs accuracy scatter.
Readable at phone size; one consistent colour per contestant.

## 6. Compute plan
- Build and test everything on CPU and APIs first. Then one focused L4 session (JarvisLabs, ~$0.44/hr, per-minute billing): serve the open models, run `tune` + `test`, save predictions, shut down. Expected 2–3 GPU hours.
- API runs: 2,000 test + 2,000 tune customers × 3 conditions. Estimate token cost before running and log actual cost.
- Pin model revisions and library versions in `results/run_manifest.json`.

## 7. Write-up deliverables (`writeup/`)
- **Before running:** `PREDICTIONS.md` — which contestant I expect to be most and least honest, and whether telling the base rate will help.
- **Blog post** (canonical): the question, the setup, the results, the honesty arc, what a bank should take from it, limitations, how to reproduce.
- **README**: results table, hero chart, cost, reproduce steps, dataset citation.
- **LinkedIn carousel** (6–8 slides): hook ("Banks get audited on whether their risk scores are honest. I held AI to the same standard.") → setup (30,000 real customers, 8 contestants) → what "honest" looks like → promised vs actual → the honesty arc → the scorecard punchline → what's next ("Why 99% accuracy can still be a disaster").
- **X long-form post**: hero chart up front, method in replies.

## 8. Milestones
| # | Milestone | Done when |
|---|---|---|
| M1 | Load data, render summaries, splits, memorization check | Section 1 checks pass; `contamination.md` written |
| M2 | Baselines (scorecard, GBM) + runners for all AI contestants | Baselines scored on `test`; each AI runner writes valid records on 20 `tune` customers |
| M3 | Full runs (L4 session + APIs), all three conditions | All `tune` + `test` predictions saved; manifest written |
| M4 | Metrics, figures, recalibration | Tests pass; all five figures generated |
| M5 | Write-up | Blog, README, carousel and X drafts ready for review |
| M6 | The fair fight: AI models learn from the same 20,000 (§11) | §11 "done when" |

## 9. Limitations to state openly
2005 data from one Taiwanese bank; famous dataset that may have been seen in training; demographic fields deliberately excluded; undocumented repayment-status codes; zero-shot AI vs trained baselines; vendor models change over time (record versions and dates).

## 10. Parked for later posts
- Synthetic insurance appetite set (insurance triage) and open document sets (ContractNLI, CUAD insurance clauses) → the document-pipeline series. Now planned as the documents arc, §12 posts 7–9.
- English vs Tamil honesty test (IndicSentiment, IndicXNLI, DravidianCodeMix) → a later post.
- HMDA mortgage decisions (2025 data) → possible follow-up on lending.

---

## 11. M6: the fair fight (added 2026-11-23)

**The question:** the published post compared zero-shot AI models with baselines that learned from 20,000 labelled customers. Here every AI model gets those same 20,000 (`train`) and is scored on the same `test`. The author asked for this after launch; the closed-model treatment below is the author's choice (2026-11-23).

**Who can learn from `train`, and how (checked 2026-11-23):**

| id | How it learns from `train` | Settings | Where |
|---|---|---|---|
| `kev-4b-ft` | LoRA fine-tune starting from released Kev-4B v1.0: the maker's delta recipe, `kev.train --data … --init_from jaredpalmer/kev-4b@591dcb5b` | lr 2e-5, LoRA rank 16, batch 1 × accum 8, bf16, checkpointing | L4 |
| `kev-0.8b-ft` | The same, from Kev-0.8B v1.0 (`@788ddbdd`) | the same | L4 |
| `qwen-ft` | LoRA on `Qwen/Qwen3.5-4B@851bf6e8` with the exact zero-shot prompt, loss on the yes/no answer token only; P(yes) read from logprobs as before | Common LoRA defaults (no maker recipe exists): rank 16, alpha 32, all linear layers, lr 1e-4, effective batch 16 | L4 |
| `laya-ft` | `laya.train.finetune` from `convaiinnovations/laya@fd4f9dbc`, the maker's defaults: 4 epochs, its proper-scoring "rlcd" objective, its own temperature fit on 400 held-out `train` rows | maker defaults | L4 (minutes) |
| `jev-examples` | Can't be fine-tuned (TypeSafe docs: "Jev is not fine-tuned or LoRA-adapted with customer data… the same weights serve every account"). Each customer is shown with its 20 most similar `train` customers and their outcomes | k = 20 | API, under $1 |
| `sol-examples` | Can't be fine-tuned (OpenAI's fine-tuning is closed to new users and never covered GPT-6.2). The same 20 examples in the prompt | k = 20, effort medium as before | API, ~$40 (needs a top-up) |

**Conditions:** `trained` (the model's own output after learning from `train`) and `trained_calibrated` (Platt vs isotonic chosen by 5-fold CV on `tune`, applied to `test`), the same names and the same treatment as the baselines. The zero-shot rows stay as they are, so each model shows zero-shot → trained.

**Rules carried over:** template v2 for every summary and every example; no demographic fields; training data and examples come from `train` only; nothing is tuned on `test`; every prediction goes in the usual record schema.

**Choices that could tilt the comparison (flagged before running):**
1. **Training budget:** each maker's defaults, no hyperparameter search, all 20,000 `train` customers. One epoch for Kev and Qwen (Kev's default and its own worked example). A second epoch is tried only if the first takes under an hour, and kept only if it lowers `tune` log loss. The baselines got about the same: LightGBM has fixed settings with its round count chosen by CV inside `train`; the scorecard is a fixed recipe.
2. **Layout:** training uses the template v2 summaries, the same as zero-shot, so zero-shot → trained is like for like. Training teaches the model what each column means, so v2's payment-month pairing should matter much less than it did zero-shot.
3. **Kev's temperature:** the fine-tuned Kev models are served at temperature 1.0 (raw output of a model trained on our data). The released checkpoints' temperatures were fitted for other data. Laya keeps its own fit on `train` rows (its default); Qwen gives raw logprobs.
4. **Neighbours for Jev and Sol:** the 20 nearest `train` customers on the 19 financial fields (amounts as signed log, then standardized; repayment codes as numbers), Euclidean, most similar first. That retrieval is itself a small model fitted on `train`, so the neighbours' own default rate is reported as a reference, to show whether the AI adds anything beyond counting.
5. **Class balance:** the natural 22% default rate, no reweighting, so probabilities can stay honest.
6. **Seeds:** one run per model (seed 0). If GPU time allows, Kev-0.8B is re-run with two more seeds as a variance check.

**Compute:** build and test on CPU first: export the training files, run each trainer for a few steps on CPU (Kev-0.8B and Laya; Qwen with a small Qwen as stand-in), and smoke-test the example runners on 20 `tune` customers. Then one L4 session: Kev-0.8B → Laya → Kev-4B → Qwen, each trained and then scored on `tune` and `test`. Measure throughput in the first minutes; stop and check with the author if Kev-4B's epoch projects past 3 hours. Expected 4–6 GPU hours (~$2–3). Keep the trained adapters (a GitHub release asset) so the runs can be reproduced.

**Before running:** the author adds M6 guesses to `PREDICTIONS.md`.

**Write-up:** decided once the results are in (probably a follow-up post). Laya's training objective rewards honest probabilities, the idea behind the sequel the post promised, so the two may merge.

### Part B: tabular foundation models (added 2026-11-23, the author's suggestion)

AI built for tables: these models read the training rows themselves (in context, no gradient training) and predict new rows. A third-party index ([AIMultiple's AIM Tabular Index](https://aimultiple.com/tabular), updated 2026-10-12, 19 datasets, default settings) ranks eight of them above CatBoost, LightGBM and XGBoost. Its ratings are accuracy-type, and it reports no calibration at all. Whether they're honest is exactly this project's question, so they join the fair fight. They read the same 19 financial fields as the scorecard and LightGBM (no text, no demographics), learn from the same `train`, and are scored on `test`.

| Candidate | Maker | Licence (per the index) |
|---|---|---|
| Kumo-Tabular | NVIDIA | OpenMDW-1.1 (top of the index) |
| TabPFN-3.5 | Prior Labs | non-commercial: **out** (the author's decision, 2026-11-24; the licence also covers outputs) |
| TabICLv2 | Inria SODA | BSD-3-Clause |
| Mitra-v2 | Amazon | Apache-2.0 |
| LimiX-2 | StableAI | non-commercial |

**Check before any run** (the index doesn't say):
- how to get each model (weights, pip package or API);
- how many training rows it takes in context. If it's fewer than 20,000, that's a disadvantage: flag it, and use the maker's own subsampling or ensembling if there is one;
- GPU memory on a 24 GB L4;
- whether a non-commercial licence suits a personal blog.

**Chosen (2026-11-24): Kumo-Tabular and TabICLv2.** TabPFN-3.5 is out on licence; Mitra is out because no v2 exists and v1 caps at 10,000 rows; LimiX-2 was never pursued (non-commercial). Conditions `trained` / `trained_calibrated`. They aren't text models, so they're compared with the baselines and the trained AI models, not with the zero-shot runs. They run in the same L4 session if the checks are done in time, otherwise in a short second one.

**Not in M6: the newer open decision models (Jev alternatives).** The strongest are 27–31B and don't fit an L4, and none publishes a fine-tuning recipe, so they can't join a trained comparison. The ones that should fit one L4 suit a cheap zero-shot "round 2" (§12, smaller items):
- Karpo Decision 4B / 2B (Apache-2.0, System One API);
- Elephant-1 9B (MIT, about 19 GB by estimate);
- GLiDE 9B (Fastino community licence, attribution required);
- pplx-decider Flash 4B (Apache-2.0, its own API, free on OpenRouter).

Source: a third-party round-up ([Pinggy, 2026-11-18](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)) built from leaderboards and vendor cards; verify each model before use.

**Done when:** all six new contestants (plus the chosen tabular foundation models and the `neighbours-20` reference) have `tune` and `test` records for `trained` and `test` records for `trained_calibrated`; `summary.csv` includes them, with paired AUC tests against the scorecard and LightGBM; one figure shows zero-shot → trained per model against the baselines.

---

## 12. Next posts (revised 2026-11-25; first proposed 2026-11-23)

Numbers are the series' post numbers. Titles lead with the question, per `writeup/STYLE.md`. A Notion number is the topic's rank on the author's page "Content Strategy: Curiosity First, Expertise Second".

**The author's decisions (2026-11-25):**
- **Order:** 4 → 5 → 6.
- **Kev in post 4:** we quantize it ourselves, since its maker's code can't (flagged in the post 4 spec).
- **Stress levels:** 3-bit and 2-bit are in.

| # | Working title | What it answers | New compute | Status |
|---|---|---|---|---|
| 2 | **Why 99% accuracy can still be a disaster** (Notion #2) | Accuracy vs ranking vs honesty vs what each mistake costs a bank | None | Published 2026-11-24 |
| 3 | **I gave AI the same data as a bank's model. It caught up.** (Notion #3) | M6, the fair fight (§11) | Done | Draft in review |
| 4 | **Does shrinking an AI model make it less honest?** (Notion #7, "Why cheaper AI doesn't always mean cheaper bills") | Post 3's closing promise. The fine-tuned Kev-0.8B, Kev-4B and Qwen at 8, 4, 3 and 2 bits. Does ranking hold while the percentages drift, and does the smaller model actually cost less? | Done (session A) | Draft in review (`writeup/quantization_post.md`) |
| 5 | **Why AI can write code but struggle to read a table** (Notion #5) | The series already has evidence. Moving each bill beside the next month's payment took zero-shot Kev-4B's AUC from 0.594 to 0.720, and Laya's from 0.666 down to 0.569. The table models read the raw numbers and reach 0.794 with no training run. One new check: the same customers as sentences, a markdown table, CSV and JSON | L4 session A, plus about $8 of API | Spec below |
| 6 | **Teaching a small model to say "I don't know"** (Notion #9) | Promised in posts 1 and 3. Posts 1 and 3 already asked whether its 30% is honest. This one asks whether it flags the customers unlike anyone it learned from | L4 session B | Design to settle (spec below) |
| 7 | **Does retrieval make AI more honest, or just more confident?** (documents arc, 1 of 3) | RAG on one public document set: no retrieval, good retrieval, noisy retrieval, conflicting passages, and the whole document in context. Accuracy, honesty, "I don't know" rate and cost | API runs on public documents; a small L4 session for the open models | Choose the dataset now: it has the longest lead time |
| 8 | **Can a PDF trick an AI into making a bad decision?** (Notion #4; documents arc, 2 of 3) | Hidden instructions in the same pipeline's documents (indirect prompt injection): how often they change the answer or the confidence, and which defences hold | Same set-up as post 7, plus crafted documents | |
| 9 | **Can you trust an AI to grade an AI?** (documents arc, 3 of 3) | Post 7's answers graded by AI judges and by people: agreement beyond chance, position bias, run-to-run consistency, and whether a judge's confidence is itself honest | Judge API calls; a few hundred human labels by the author | |
| 10 | **Fresh data no model has seen** | Post 1's biggest limit: Sol knows the Taiwan file. HMDA 2025 mortgage decisions (§10) can't be memorized, and they're closer to a real lending decision | A bigger project: new data, new splits, re-run the best few | |

**Smaller items:**
- **Honesty leaderboard, round 2** (the newer open decision models, zero-shot, §11 Part B list): a README table, not a post.
- **Notion #8, "Can a tiny AI beat a giant one at one job?":** post 3 already answers it. Fine-tuned Kev-0.8B ranked at 0.790; Sol with 20 examples ranked at 0.762. It could be a LinkedIn-only piece.
- **After post 10:** English vs Tamil (§10).

**Two GPU sessions.** Session A runs post 4 and post 5's open models with one setup; a fresh L4 takes 20–30 billed minutes to set up. Session B is post 6's retraining, once its design is settled. Each session needs the author's guesses for its posts in `PREDICTIONS.md` first.

**The reserve.** The 6,000 `reserve` customers are still untouched, while `test` has been reused across three posts and many checks. Keep them sealed for one replication of the headline numbers when the credit-card posts end.

### Post 4 spec: does shrinking an AI model make it less honest? (revised 2026-11-25)

**The question:** quantizing a model (storing its weights in fewer than 16 bits) cuts its memory, and the usual check is that accuracy holds. A model can keep its ranking while its percentages drift, which is the failure this series measures. The cost angle (Notion #7): fewer bits don't automatically mean faster or cheaper answers, so measure it rather than assume it.

**Models:** the fine-tuned Kev-0.8B, Kev-4B and Qwen from M6 (the `m6-weights` release). Each is compared only with its own bf16 run.
- **Laya is out.** At 421M parameters it already runs on a laptop CPU, so shrinking isn't its deployment question.
- **The zero-shot models are out.**

**Precisions.** Every method needs no calibration data, so no data is used to quantize and nothing touches `tune` or `test`.

| Tag | Method | Why |
|---|---|---|
| `bf16` | None, re-run through the same serving path as the others | The reference |
| `bnb-int8` | bitsandbytes 0.50.2, LLM.int8 with outlier threshold 6.0 | The off-the-shelf 8-bit most people use |
| `bnb-nf4` | bitsandbytes 0.50.2, NF4, blocksize 64, no double quantization, bf16 compute | The off-the-shelf 4-bit most people use |
| `hqq-4`, `hqq-3`, `hqq-2` | HQQ 0.2.8.post1 at 4, 3 and 2 bits, its defaults (group size 64, axis 1), bf16 compute | One method all the way down, so the curve shows the bit width rather than the method. `hqq-4` against `bnb-nf4` checks how much the method matters at the same width |

- **"Off the shelf" means what transformers sets.** The bitsandbytes settings are the ones transformers uses for `load_in_8bit` and for `load_in_4bit` with NF4. The bare bitsandbytes layers default to something else: plain int8 with no outlier threshold, and FP4 with double quantization (checked in the 0.50.2 source, 2026-11-25).
- **Versions:** both libraries are pinned (pyproject's `quant` group; `runners/quantize.py` refuses any other version), and the version goes into `model_version`.
- **GPTQ and AWQ are out:** they need calibration data, which adds a data choice this comparison doesn't need.
- **A method that fails** on a model's architecture is recorded and reported, never silently replaced by another.

**Choices that could tilt the comparison (flagged before building):**
1. **We quantize Kev ourselves** (the author's decision, 2026-11-25). Kev's loader only offers bf16, fp16 and fp32 (`LoadOptions` in `kev/checkpoint.py`, checked 2026-11-25).
   - **The launcher** (`runners/kev_quant_serve.py`): it loads the fine-tuned run through Kev's own `Checkpoint.load`, quantizes the backbone's linear layers, then starts Kev's own server. Request handling, the pointer head and the readout stay the maker's.
   - **Guard:** with quantization off (`bf16`), the new path must reproduce M6's predictions on 50 `tune` customers (mean |difference| at most 0.02, as the Qwen in-process check used) before any quantized run. It catches a broken path, which would move predictions by tenths. The comparisons themselves all use the fresh bf16 run.
2. **Merge, then quantize, for every model.** The LoRA delta goes into the base weights first, so the result is one shrunk model, which is what a deployer would ship. Both models merge the same way, in fp32 with one rounding to bf16:
   - **Kev:** its loader's default (`KEV_MERGE`).
   - **Qwen:** the base is loaded in fp32 on the CPU, `merge_and_unload` adds the adapter, then the model is cast to bf16.
3. **What stays in bf16:** the readout (Kev's pointer head, which sits outside the backbone; Qwen's `lm_head`), the embeddings and the norms. Qwen's vision tower, unused here, isn't quantized either.
4. **The same serving for every precision, bf16 included.** CUDA graphs and Kev's fused kernels are off, because neither works with quantized layers. M6 served bf16 with CUDA graphs on, so post 4 re-runs bf16 (`<model>-bf16`) rather than borrowing M6's latency: every precision's speed is then measured the same way.
5. **Everything else as in M6:** template v2, each model's own readout, Kev served at temperature 1.0, one request at a time.

**Data and conditions:**
- `tune` and `test` at every precision.
- Raw predictions are `trained`. M6's correction (Platt vs isotonic, chosen by 5-fold CV on `tune`) gives `trained_calibrated`.
- Quantization is a deployment choice, not a new contestant. Each precision gets its own record name, `<model>-<tag>` (e.g. `kev-0.8b-ft-bnb-nf4`, `qwen-ft-hqq-2`), in `results/predictions/`. The published post 1 and post 3 tables list their models by name, so they don't change.

**Measures (on `test`):**
- **Drift, without labels:** mean prediction against bf16's, mean change per customer, and rank correlation with bf16.
- **Ranking:** AUC, with a paired bootstrap against the model's own bf16 run, Holm-corrected over the 15 quantized rows.
- **Honesty:** ECE against the noise floor, plus calibration-in-the-large and slope, raw and after the correction.
- **Lending:** post 2's profit at the model's own percentages, with the cut-off chosen on `tune`.
- **Cost:** GPU memory (weights and peak), median latency, and cost per 1,000 decisions at the L4 rate ($0.44/h).

**Compute:** about 72,000 requests (3 models × 6 precisions, bf16 included, × 4,000 customers).
- bitsandbytes 8-bit can be several times slower than bf16 per request. So allow 4–5 hours for post 4, plus about 1 hour for post 5's open models: 5–6 GPU hours in all, about $2.50.
- Stop and ask if any one run's ETA is over 2 hours.

**Done when:**
- Every model has `tune` and `test` predictions at each precision, with the precision and library versions in `model_version`, or a recorded failure (`analysis/quantization.py --record-failure`).
- `runners/manifest.py --quant` reports every pair complete or recorded as failed.
- `analysis/quantization.py` writes `results/quantization.csv`.
- The charts follow the `writeup/STYLE.md` budget. Candidates: honesty by bits with the noise band, ranking by bits, drift in the mean prediction, and memory/latency/cost.

**Build list: done on the laptop (2026-11-25), 197 tests passing.**
1. **`runners/quantize.py`:** the shared quantizer. All six precisions were checked on a toy bf16 model on the CPU: the right layer types, the readout kept, and the error growing as the bits shrink.
2. **`runners/kev_quant_serve.py`:** the Kev launcher, run in Kev's environment on the L4. It reports its precision, settings and GPU memory at `/v1/quantization`. `runners/kev.py --precision <tag>` refuses a server that reports a different precision, as it already does for temperature, and reads the peak memory after each request.
3. **`runners/qwen_ft.py --precision <tag>`:** merge in fp32, round once to bf16, quantize. `--both-splits` scores tune and test with one load.
4. **`--reproduce-check N`** on both runners (`harness.reproduction_check`).
5. **`analysis/recalibrate.py` and `runners/manifest.py --quant`:** the 18 record names.
6. **`analysis/quantization.py`:** the measures above. The figures come with the draft, once the results are known, as post 3's did.
7. **L4_RUNBOOK.md §7:** session A as copy-paste commands, with the prompt for Claude Code on the L4.

**Not testable on the laptop:** the launcher inside Kev's server, and the bitsandbytes CUDA kernels. The L4 checks both with the reproduction check and a 20-customer smoke run per precision before any full run.

### Post 5 spec: why AI can write code but struggle to read a table (added 2026-11-25)

**The question:** if the same numbers are laid out differently, does an AI model reach a different conclusion? The series has already seen it twice: in the layout check (`--variant aligned`) and the disguise check (`data/contamination.md` §4). This post adds formats.

**Formats** (`data/formats.py`, run as variants of `runners/checks/disguise.py`, each versioned as `format <name> f1`). Every format keeps v2's first line, "Credit card account summary":
- **`prose`:** one sentence per month, after the credit limit line.
- **`markdown`:** v2's pipe table with the header separator and the outer pipes added.
- **`csv`:** the same table as CSV, with the amounts that contain thousands separators quoted, as CSV requires.
- **`json`:** one compact JSON object holding the credit limit and the six months.

The reference is template v2 (a pipe-separated table without the separator row), using the M3 `out_of_box` runs.

**Fairness:**
- **Same values, byte for byte:** every value string is copied from v2: the credit limit, each month's name, the amounts as written ("NT$8,617") and the repayment labels with their raw codes, with no interpretation. Only the layout changes. A unit test reads every tune and test customer back from each format and checks the values against v2's (`tests/test_formats.py`).
- **Laya's limit:** checked (`uv run python data/formats.py --token-report`, `results/checks/formats_tokens.json`).
  - On the 2,000 test customers, the longest summary in Laya's tokens is 183 for v2, 262 for prose, 203 for markdown, 179 for CSV and 320 for JSON. All are within the model card's 320-token budget.
  - Laya's real room is larger. It's 512 minus the question header minus 1 (`laya.common`), about 470 for our question.
  - `runners/laya_classic.py` now stops if Laya ever reports truncating a summary, so no truncated input is scored.

**Runs:**
- **Who and what:** the six zero-shot AI contestants (Jev, Sol, Kev-4B, Kev-0.8B, Qwen, Laya), `out_of_box`, on 2,000 `test` customers per format, into `results/checks/formats/`.
- **A diagnostic, not a contestant condition:** nothing is tuned, and the headline results stay on v2.
- **Noise floor:** the disguise check's 200-customer control.
- **Sol's caveat applies:** it recognizes the dataset.
- **Cost:** about $2 per format for Sol and cents for Jev. The open models run in session A.

**Measures, per format against v2:** mean shift, AUC, ECE, and the change per customer. The tabular foundation models (0.794 with no training run) are the reference for reading the numbers directly.

**Done when:** every contestant has every format that fits its context, `disguise.py --report` writes `results/checks/formats_summary.json`, and the charts follow the budget.

**Built on the laptop (2026-11-25):** `data/formats.py`, the four variants and the formats report in `runners/checks/disguise.py`, and the tests. The disguise check's published summary is unchanged by the refactor.

### Post 6 spec: teaching a small model to say "I don't know" (redesigned 2026-11-25; settle before session B)

**Why the redesign:**
- **The first version** compared a proper-scoring reward with cross-entropy.
- **The flaw:** cross-entropy is already a proper scoring rule, and Laya-ft was trained with one ("rlcd"). So that contrast tests whether the percentages are honest, which posts 1 and 3 already covered.
- **What it misses:** post 1's promise, "rewarding it for knowing when it doesn't know".

**Two kinds of unsure:**
- **The coin-flip customer:** an honest 50%. No model can do better on them.
- **The unfamiliar customer:** nobody like them in the training data. Flagging these is what "knowing when it doesn't know" means. It also answers the post 1 reviewer's point that calibration isn't self-awareness.

**Proposal:**
1. **Free signals first** (laptop, existing predictions):
   - how much the three seeds of Kev-0.8B and Laya disagree (`results/checks/m6_seeds/`);
   - distance to the 20 nearest `train` customers (`data/neighbours/`);
   - closeness to the cut-off.
2. **The real test (session B):** retrain Kev-0.8B and Laya on `train` with one slice of customers held out, chosen from `train` statistics only (for example, the highest credit limits). On that slice in `test`, do the signals rise, and do the models' errors?
3. **Optional, training to defer:** a "refer" answer that has a cost, against the M6 model plus a referral band.
4. **The bank frame:** send the unsure cases to an underwriter. Report the approval rate, the default rate among approvals and the underwriter workload with 5%, 10% and 20% of customers referred. Every choice is made on `tune`.

**To settle before session B:** the held-out slice, whether step 3 is in, and the referral rates.

### Documents arc spec: posts 7–9 (added 2026-11-24; renumbered 2026-11-25)

**Why these three:** 2026 interview guides put RAG first, then evaluation (including AI judges), agents and production judgment: cost, latency, security. These guides are career and vendor blogs, informed opinion rather than hiring data. One document pipeline, examined three ways, covers the top three areas and builds one story (Notion principle 5). It sits next to the author's insurance work, using **public data only**: nothing from an employer.

**Data:** one public document set with questions whose answers are known. Candidates are CUAD (contract clauses), ContractNLI, or public insurance policy wordings with author-written questions. Check each licence before use. A frozen test split, as in post 1. Nothing is tuned on test.

**Post 7, retrieval and honesty.**
- **Conditions:** no retrieval (the model's own knowledge); good retrieval; noisy retrieval (irrelevant passages mixed in); conflicting passages; the whole document in context.
- **Measures:**
  - accuracy;
  - honesty (ECE, as in post 1);
  - how often the model says "I don't know", and whether it says it at the right times;
  - retrieval quality (did the right passage come back);
  - cost and latency per question.
- **Prior work:**
  - [NOVA, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.705/): noisy or conflicting retrieval makes models more overconfident.
  - [LaRA](https://arxiv.org/pdf/2605.10235): stronger models do better with the whole document, and retrieval handles refusals better.
  - [Token Tax, June 2026](https://arxiv.org/html/2606.20898v1): the whole document is most accurate but about ten times the cost per question.

**Post 8, the PDF trick.** Same pipeline. Plant hidden instructions in some documents (white text, metadata, alt text), as in [OpenRAG-Soc, WWW '26](https://arxiv.org/pdf/2601.10923) and [Chang et al., USENIX Security '26](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_chang.pdf). Measure how often the answer or the confidence changes, and test simple defences: strip hidden text, quote the document as data, check against the source. Plant only harmless payloads, such as "answer 'yes'" or "say you're certain"; this is a measurement of the risk, not a toolkit.

**Post 9, AI grading AI.** Post 7's free-text answers graded by two or three AI judges and by the author (a few hundred labels).
- **Measures:**
  - agreement beyond chance (Cohen's kappa, not raw agreement);
  - position bias (swap the order);
  - run-to-run consistency;
  - whether the judge's own confidence is honest.
- **Prior work:** [Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544): raw agreement overstates judge quality, and a judge that gives the same score every time can still be heavily biased by which answer it sees first.

**Done when** (per post): the experiment's predictions are recorded in the usual schema, with model versions; one table, one chart, and a draft following `writeup/STYLE.md`.
