# LinkedIn carousel: "Can AI actually know when it's wrong?"

Eight square slides (1080 × 1080), exported as one PDF. One idea per slide, big type, a chart where there's evidence to show. The numbers come from `results/summary.csv` and `data/contamination.md`, and the charts from `results/figures/`.

## Post text

> Banks get audited on whether their risk scores are honest. I held AI to the same standard.
>
> When a model says a customer is "30% likely to default", about 30 in every 100 such customers should default. That's calibration, and banks are expected to prove it.
>
> So I gave eight models the same 2,000 real credit-card customers and one question: will this customer miss next month's payment?
>
> What I found:
> → Most AI models overstated the risk, two of them by more than double
> → One sentence of context fixed most of it, and broke one model
> → The most "honest" model had most likely seen the answers
> → A decades-old scorecard still ranked customers best
>
> I also wrote down my predictions before running anything. I got three of four wrong.
>
> To be fair to the AI: the classic models learned from 20,000 labelled customers, and the AI models answered cold. That's the real trade-off.
>
> Full write-up, code and every prediction in the first comment.

## First comment

> The full write-up, including how I checked whether the models had memorized the data: [blog link]
> Code and all predictions: [github.com/nishzsche/can-ai-know-when-its-wrong](https://github.com/nishzsche/can-ai-know-when-its-wrong)
> Data: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009), CC BY 4.0.

## Slides

### 1. Hook

**Headline:** Banks get audited on whether their risk scores are honest.
**Body:** I held AI to the same standard.
**Footer:** 8 models · 2,000 real customers · 1 question
**Visual:** text only. Large headline, the body line in a second colour.

### 2. The setup

**Headline:** Same customers. Same question.
**Body:**
- 2,000 real credit-card customers (Taiwan, 2005). 22% missed a payment.
- "Will this customer miss next month's payment?"
- 6 AI models answering cold
- 2 classic models trained on 20,000 customers
- No age, sex, marital status or education
**Visual:** two columns of model names. AI: Jev, Kev-4B, Kev-0.8B, Laya, Qwen3.5-4B, GPT-6.2 Sol. Classic: Scorecard, Gradient boosting.

### 3. What "honest" looks like

**Headline:** When it says 30%, do 30% default?
**Body:** A weather app that says "70% chance of rain" should be right on about 7 of every 10 such days. A risk model is the same. On the diagonal, the model is honest.
**Visual:** a simple diagonal line labelled "honest", with "too pessimistic" below it and "too optimistic" above it. Or the scorecard panel from figure 1.

### 4. Promised vs actual

**Headline:** Most AI models think customers are riskier than they are.
**Body:** Jev said 48% would default. Kev-4B said 49%. The real figure: 22%.
**Visual:** `results/figures/1_promised_vs_actual.png`, or crop it to the Jev and Kev-4B panels for phone size.

### 5. The honesty arc

**Headline:** One sentence fixes most of it.
**Body:** "In this portfolio, about 22% of customers miss a payment." Calibration improved for 5 of 6 models. Kev-4B went from 27 points off to 3. It made Kev-0.8B worse.
**Visual:** `results/figures/2_honesty_arc.png`.

### 6. The twist

**Headline:** The most "honest" model had most likely seen the answers.
**Body:** GPT-6.2 Sol named the dataset 53 times out of 53 and knew its exact default rate: 22.12%. Told a false rate (10%), it followed 70% of the way. It already knew the real one.
**Visual:** three big numbers in a row: **53/53** recognised · **22.12%** recalled · **70%** followed a false hint.

### 7. The punchline

**Headline:** A decades-old scorecard still ranks customers best.
**Body:** Scorecard AUC 0.777. The best AI models: 0.757 and 0.754, close but not ahead. The scorecard learned from 20,000 customers; the AI models from none.
**Visual:** `results/figures/3_ranking_auc_ks.png`.

### 8. What to take away

**Headline:** Ask how honest it is, not just how accurate.
**Body:**
- Check calibration, not just accuracy
- Recalibrate zero-shot AI on your own labelled data (2,000 customers was enough)
- Test on data the model can't have seen
**Footer:** Next: why 99% accuracy can still be a disaster.
**Visual:** text only, with the next-post teaser at the bottom.
