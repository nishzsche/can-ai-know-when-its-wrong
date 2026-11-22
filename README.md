# Can AI actually know when it's wrong?

Banks are audited on whether their risk scores are honest: if a model says "30% chance of default", about 30% of those customers should default. This project holds AI decision models to the same standard on 2,000 real credit card customers, and compares them with a logistic-regression scorecard and gradient boosting.

Status: **M1 (data) in progress.** Results, charts and the full method will be added here when the runs are done. See [PLAN.md](PLAN.md).

## Data

**Default of Credit Card Clients**: 30,000 credit card customers of a bank in Taiwan, April–September 2005, and whether each one defaulted on the next month's payment.

- Source: UCI Machine Learning Repository, dataset 350, loaded with `ucimlrepo` 0.0.7. UCI last updated it on 29 March 2024; `data/raw/source.json` records the content hash we used.
- **Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**, as stated on the [UCI dataset page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- **Please cite:**
  - Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
  - Yeh, I-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Changes we made: we dropped four columns (below), renamed the columns from UCI's `X1`–`X23` to their documented names, and rendered each customer as a short text summary.

### What the models see

Every model sees only financial behaviour:

- `LIMIT_BAL`: credit limit (NT$)
- `PAY_0`, `PAY_2`–`PAY_6`: repayment status, September back to April 2005
- `BILL_AMT1`–`BILL_AMT6`: bill amounts, September back to April 2005
- `PAY_AMT1`–`PAY_AMT6`: amounts paid, September back to April 2005

**Excluded from every model input: `SEX`, `MARRIAGE`, `AGE`, `EDUCATION`.** Using protected characteristics in credit decisions is off-limits in practice. Leaving them out also keeps this about whether the probabilities are honest, not about demographics. `data/load.py` drops them before anything is saved.

The target is `default_next_month` (UCI's `Y`, "default payment next month"). The overall default rate is **22.12%** (6,636 of 30,000).

### How each customer is shown to the AI models

The AI models read text, so each customer becomes one fixed summary (`data/render.py`, template `v2`). Every AI contestant sees the same text:

```
Credit card account summary
Credit limit: NT$100,000

Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$4,829 | NT$0 | paid on time (code -1)
Aug 2005 | NT$0 | NT$0 | paid on time (code -1)
Jul 2005 | NT$0 | NT$0 | not late (code -2)
...
```

The question is "Based on this account summary, will this customer miss next month's payment?" and the model's probability of "yes" is taken as its predicted default risk. In the "told the base rate" condition, one sentence goes before the question: "In this portfolio, about 22.1% of customers miss next month's payment." The summary text doesn't change.

Summaries are 164 tokens on average and 192 at most with Laya's tokenizer, which leaves about 320 tokens for the summary. With Kev's (Qwen3.5) tokenizer they're 218 on average and 250 at most.

### Known data quirks

We document these rather than hide them.

1. **Repayment-status codes −2 and 0 aren't documented.** Yeh & Lien define −1 as "pay duly" and 1–9 as months of payment delay; −2 and 0 fall on the not-late side of that scale. The often-quoted "common interpretation" (−2 = no balance / card not used, −1 = paid in full, 0 = paid the minimum) is contradicted by the data:

   | Code | Common interpretation | What the train split shows |
   |---|---|---|
   | −2 | "no balance / card not used" | 40.6% of these months still carry a positive bill (48.5% non-zero across all 30,000 customers) |
   | −1 | "paid in full" | 24% of payments fall short of the previous bill (documented only as "pay duly") |
   | 0 | "paid the minimum" | 95% are partial payments, but "minimum" can't be verified |

   So the summaries (template v2) say only what's documented or follows from the scale, and always show the raw code: −2 → "not late (code −2)", −1 → "paid on time (code −1)", 0 → "not late (code 0)", 1–9 → "N months late (code N)". This gives the AI models the same information the trained baselines see, without wording that would mislead them. (Template v1 used the common interpretation and was replaced before any runs.) Code 9 never occurs.
2. **Code 1 ("one month late") appears almost only in September**: 3,688 times there, but 28, 4, 2, 0 and 0 times in August–April. The coding seems to differ between months.
3. **Payments usually settle the previous month's bill.** "Amount paid" in a month matches the *previous* month's bill 21,213 times, against 3,720 for the same month. We keep UCI's month alignment, so each row shows that month's bill and that month's payment.
4. **Negative bills (credit balances):** 3,932 entries across 1,930 customers, down to −NT$339,603. The summary shows them as `-NT$1,234 (credit)`.
5. **Bills above the credit limit:** 8,274 entries.
6. **Identical inputs:** 817 customers have exactly the same financial fields as another customer (795 have no bills or payments at all), and some of them have different outcomes.
7. Amounts are 2005 NT$ and are kept as they are.

### Splits

The splits are stratified on the target with a fixed seed (`data/splits.py`):

| split | customers | defaults | default rate | used for |
|---|---:|---:|---:|---|
| train | 20,000 | 4,424 | 22.12% | the scorecard and GBM only |
| tune | 2,000 | 442 | 22.10% | recalibration and threshold choices |
| test | 2,000 | 442 | 22.10% | scoring every contestant; nothing is tuned on it |
| reserve | 6,000 | 1,328 | 22.13% | unused |

### Memorization

This is a famous dataset, so `data/contamination.md` records what each contestant could have seen in training, along with the LLM memorization tests.

## Reproduce

```bash
uv sync
uv run python data/load.py && uv run python data/splits.py   # fetch data, build splits
uv run python data/render.py --split tune --split test       # render account summaries
uv run python data/render.py --token-report                  # token counts per contestant tokenizer
uv run pytest                                                 # render and split tests
uv run --group memcheck python data/memcheck.py --model <id>  # LLM memorization tests (needs an API key)
```
