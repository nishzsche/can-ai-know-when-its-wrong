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

## 9. Limitations to state openly
2005 data from one Taiwanese bank; famous dataset that may have been seen in training; demographic fields deliberately excluded; undocumented repayment-status codes; zero-shot AI vs trained baselines; vendor models change over time (record versions and dates).

## 10. Parked for later posts
- Synthetic insurance appetite set (insurance triage) and open document sets (ContractNLI, CUAD insurance clauses) → the document-pipeline series.
- English vs Tamil honesty test (IndicSentiment, IndicXNLI, DravidianCodeMix) → a later post.
- HMDA mortgage decisions (2025 data) → possible follow-up on lending.
