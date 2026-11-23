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
- Synthetic insurance appetite set (insurance triage) and open document sets (ContractNLI, CUAD insurance clauses) → the document-pipeline series. Now planned as the documents arc, §12 posts 5–7.
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
| TabPFN-3.5 | Prior Labs | non-commercial |
| TabICLv2 | Inria SODA | BSD-3-Clause |
| Mitra-v2 | Amazon | Apache-2.0 |
| LimiX-2 | StableAI | non-commercial |

**Check before any run** (the index doesn't say):
- how to get each model (weights, pip package or API);
- how many training rows it takes in context. If it's fewer than 20,000, that's a disadvantage: flag it, and use the maker's own subsampling or ensembling if there is one;
- GPU memory on a 24 GB L4;
- whether a non-commercial licence suits a personal blog.

Then pick two or three. Recommended: Kumo-Tabular, TabPFN-3.5, and TabICLv2 or Mitra-v2. Conditions `trained` / `trained_calibrated`. They aren't text models, so they're compared with the baselines and the trained AI models, not with the zero-shot runs. They run in the same L4 session if the checks are done in time, otherwise in a short second one.

**Not in M6: the newer open decision models (Jev alternatives).** The strongest are 27–31B and don't fit an L4, and none publishes a fine-tuning recipe, so they can't join a trained comparison. The ones that should fit one L4 suit a cheap zero-shot "round 2" (§12, post 4):
- Karpo Decision 4B / 2B (Apache-2.0, System One API);
- Elephant-1 9B (MIT, about 19 GB by estimate);
- GLiDE 9B (Fastino community licence, attribution required);
- pplx-decider Flash 4B (Apache-2.0, its own API, free on OpenRouter).

Source: a third-party round-up ([Pinggy, 2026-11-18](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)) built from leaderboards and vendor cards; verify each model before use.

**Done when:** all six new contestants (plus the chosen tabular foundation models and the `neighbours-20` reference) have `tune` and `test` records for `trained` and `test` records for `trained_calibrated`; `summary.csv` includes them, with paired AUC tests against the scorecard and LightGBM; one figure shows zero-shot → trained per model against the baselines.

---

## 12. Next posts (proposed 2026-11-23, for the author to choose; quantization and the documents arc added 2026-11-24)

Ordered so each one keeps a promise or answers a limit of the one before. Titles lead with the question, per `writeup/STYLE.md`. Where a post matches a topic on the author's Notion list ("Content Strategy: Curiosity First, Expertise Second"), its number is given.

| # | Working title | What it answers | New compute |
|---|---|---|---|
| 1 | **Why 99% accuracy can still be a disaster** (Notion #2) | Promised at the end of post 1. A model that says "nobody defaults" is 78% accurate here; accuracy vs ranking (AUC) vs honesty (calibration) vs what each mistake costs a bank. Built from results already in `results/` | None. It can go out while M6 runs |
| 2 | **The fair fight: what happens when AI gets the same 20,000 customers** (Notion #3) | M6. Fine-tuned Kev, Qwen and Laya; Jev and Sol shown 20 look-alike customers; tabular foundation models; the scorecard and LightGBM. Twenty look-alikes alone already rank at AUC 0.764 and are nearly honest (ECE 0.023), so "does the AI add anything?" is the hook | One L4 session (~$3) plus ~$44 of API |
| 3 | **Does shrinking an AI model make it less honest?** (Notion #7) | Quantization (spec below). The fair fight's trained models, re-scored at full precision, 8-bit and 4-bit on the same `test`. Does ranking hold while the percentages drift? Pairs with post 1's point that accuracy alone misses what matters | ~1 L4 hour, no API |
| 4 | **Teaching a small model to say "I don't know"** (Notion #9) | The sequel post 1 promised. Laya and Kev-0.8B trained with a proper-scoring reward vs plain cross-entropy, plus abstention: refer the unsure cases to a person, and see what happens to approval and default rates. Builds on M6's setup; merge into post 2 if it's short | A short L4 session |
| 5 | **Does retrieval make AI more honest, or just more confident?** (documents arc, 1 of 3) | RAG on one public document set: no retrieval, good retrieval, noisy retrieval, conflicting passages, and the whole document in context. Accuracy, honesty, "I don't know" rate and cost (spec below) | API runs on public documents; a small L4 session for the open models |
| 6 | **Can a PDF trick an AI into making a bad decision?** (Notion #4; documents arc, 2 of 3) | Hidden instructions in the same pipeline's documents (indirect prompt injection): how often they change the answer or the confidence, and which defences hold | Same set-up as post 5, plus crafted documents |
| 7 | **Can you trust an AI to grade an AI?** (new topic; documents arc, 3 of 3) | Post 5's answers graded by AI judges and by people: agreement beyond chance, position bias, run-to-run consistency, and whether a judge's confidence is itself honest | Judge API calls; a few hundred human labels by the author |
| 8 | **Honesty leaderboard, round 2** | The newer open decision models that fit one L4, zero-shot, on the same `test` (§11 Part B list). A short update post, or just a new README table | ~1 L4 hour; pplx Flash is free on OpenRouter |
| 9 | **Fresh data no model has seen** | Post 1's biggest limit: Sol knows the Taiwan file. HMDA 2025 mortgage decisions (§10) can't be memorized, and they're closer to a real lending decision | A bigger project: new data, new splits, re-run the best few |

After that, English vs Tamil (§10). The document-pipeline series parked in §10 is now the documents arc (posts 5–7).

### Post 3 spec: does shrinking an AI model make it less honest? (added 2026-11-24)

**The question:** quantizing a model (storing its weights in 8 or 4 bits instead of 16) cuts its memory and serving cost, and the usual check is that accuracy holds. A model can keep its ranking while its percentages drift, which is the failure this series measures.

**Setup:**
- **Models:** the M6 trained models that run on a GPU (Kev-4B, Kev-0.8B, Qwen fine-tuned; weights from the `m6-weights` GitHub release). Optionally their zero-shot versions too, to see whether training makes a model more or less robust to shrinking.
- **Precisions:** full (bf16, the M6 run itself), 8-bit and 4-bit, with standard off-the-shelf methods (e.g. bitsandbytes 8-bit and NF4). Pin each method's version, and use no calibration data from `test`.
- **Data:** the same 2,000 `test` customers, the same template v2 summaries and the same readout as M6. Each model is compared only with itself, so nothing else changes between rows.
- **Measures:**
  - ranking (AUC, with a paired bootstrap against the full-precision model);
  - honesty (ECE, with the noise floor);
  - the post 2 lending profit, at the model's own percentages;
  - GPU memory, latency and cost per 1,000 decisions.
- **Fairness:** quantization is a deployment choice, not a new contestant. A quantized model that drifts can be recalibrated on `tune` like any other, so report both raw and recalibrated.

**Done when:** every model has `test` predictions at each precision, recorded with the precision in `model_version`; one table and one chart (honesty and ranking by precision, per model).

### Documents arc spec: posts 5–7 (added 2026-11-24)

**Why these three:** 2026 interview guides put RAG first, then evaluation (including AI judges), agents and production judgment: cost, latency, security. These guides are career and vendor blogs, informed opinion rather than hiring data. One document pipeline, examined three ways, covers the top three areas and builds one story (Notion principle 5). It sits next to the author's insurance work, using **public data only**: nothing from an employer.

**Data:** one public document set with questions whose answers are known. Candidates are CUAD (contract clauses), ContractNLI, or public insurance policy wordings with author-written questions. Check each licence before use. A frozen test split, as in post 1. Nothing is tuned on test.

**Post 5, retrieval and honesty.**
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

**Post 6, the PDF trick.** Same pipeline. Plant hidden instructions in some documents (white text, metadata, alt text), as in [OpenRAG-Soc, WWW '26](https://arxiv.org/pdf/2601.10923) and [Chang et al., USENIX Security '26](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_chang.pdf). Measure how often the answer or the confidence changes, and test simple defences: strip hidden text, quote the document as data, check against the source. Plant only harmless payloads, such as "answer 'yes'" or "say you're certain"; this is a measurement of the risk, not a toolkit.

**Post 7, AI grading AI.** Post 5's free-text answers graded by two or three AI judges and by the author (a few hundred labels).
- **Measures:**
  - agreement beyond chance (Cohen's kappa, not raw agreement);
  - position bias (swap the order);
  - run-to-run consistency;
  - whether the judge's own confidence is honest.
- **Prior work:** [Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544): raw agreement overstates judge quality, and a judge that gives the same score every time can still be heavily biased by which answer it sees first.

**Done when** (per post): the experiment's predictions are recorded in the usual schema, with model versions; one table, one chart, and a draft following `writeup/STYLE.md`.
