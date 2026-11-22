# LinkedIn carousel: "Can AI actually know when it's wrong?"

Ten square slides (1080 × 1080), exported as one PDF by `writeup/build_carousel.py`. One idea per slide, big type, a chart where there's evidence to show. The numbers come from `results/summary.csv` and `data/contamination.md`, and the charts from `results/figures/`.

## Post text

Written for anyone, tech or not: every term is explained the first time it appears.

> Banks get audited on whether their risk scores are honest. I held AI to the same standard.
>
> "Honest" has a precise meaning here: when a model says a customer is 30% likely to miss a payment, about 30 in every 100 such customers should actually miss it. Statisticians call this calibration, and banks have to prove it.
>
> So I gave eight models the same 2,000 real credit-card customers and one question: will this customer miss next month's payment? Six were AI models, given no examples to learn from. Two were traditional bank models that learned from 20,000 past customers.
>
> What I found:
> → Most AI models said customers were riskier than they were. Two of them more than doubled it.
> → They read the safest customers, people paying part of their bill on time, as the riskiest.
> → One extra sentence, "about 22% of customers miss a payment", fixed most of it, and confused one model.
> → The most "honest" AI had most likely seen this data before: it's a famous public dataset.
> → A decades-old bank scorecard still sorted risky customers from safe ones best.
> → Who looks honest depends on the bank: if half the customers missed payments, the order flips.
>
> I also wrote down my predictions before running anything. I got three of four wrong.
>
> To be fair to the AI: the traditional models learned from thousands of past outcomes, and the AI models got none. That's the real trade-off.
>
> Full write-up, code and every prediction in the first comment.

## First comment

> The full write-up, including how I checked whether the models had memorized the data: [blog link]
> Code and all predictions: [github.com/nishzsche/can-ai-know-when-its-wrong](https://github.com/nishzsche/can-ai-know-when-its-wrong)
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI "Default of Credit Card Clients", Yeh & Lien, 2009; CC BY 4.0).

## Slides

### 1. Hook

**Headline:** Banks get audited on whether their risk scores are honest.
**Body:** I held AI to the same standard.
**Footer:** 8 models · 2,000 real customers · 1 question
**Visual:** text only. Large headline, the body line in a second colour.

### 2. The setup

**Headline:** Same customers. Same question.
**Body:**
- 2,000 real credit-card customers (Taiwan, 2005): their bills, payments and lateness. 22% missed a payment.
- One question: "Will this customer miss next month's payment?"
- 6 AI models, given no examples to learn from
- 2 traditional bank models that learned from 20,000 past customers
- No age, sex, marital status or education
**Visual:** two columns of model names, each with a few words on what it is. AI: Jev (built for yes/no decisions), Kev-4B and Kev-0.8B (free, open-source), Laya (open-source, tiny), Qwen3.5-4B (free, general-purpose), GPT-6.2 Sol (ChatGPT's model). Traditional: Scorecard (banks' classic), Gradient boosting (today's standard).

### 3. What "honest" looks like

**Headline:** When it says 30%, do 30% default?
**Body:** A weather app that says "70% chance of rain" should see rain on about 7 of every 10 such days. A risk model is the same: when it says 30% will miss a payment ("default"), about 30% should. On the diagonal, the model is honest.
**Visual:** a simple diagonal line labelled "honest", with "too pessimistic" below it and "too optimistic" above it. Or the scorecard panel from figure 1.

### 4. Promised vs actual

**Headline:** Most AI models think customers are riskier than they are.
**Body:** On average, Jev said 48% of these customers would miss a payment, and Kev-4B 49%. The real figure: 22%.
**Visual:** `results/figures/1_promised_vs_actual.png`, or crop it to the Jev and Kev-4B panels for phone size.

### 5. Where it goes wrong

**Headline:** They read the safest customers as the riskiest.
**Body:** Half the customers pay part of their balance on time. Only 12% of them defaulted, the lowest of any group. Jev predicted 51% for them, Kev-4B 54%.
**Visual:** bars: each model's average prediction for these customers, against the 12% who actually defaulted (`results/explain.json`).

### 6. The honesty arc

**Headline:** One sentence fixes most of it.
**Body:** Adding "About 22% of customers miss a payment" made 5 of 6 models more honest. Kev-4B went from 27 percentage points off, on average, to 3. It confused Kev-0.8B, which got worse.
**Visual:** `results/figures/2_honesty_arc.png`.

### 7. The twist

**Headline:** The most "honest" model had most likely seen the answers.
**Body:** These customers come from a famous public dataset. GPT-6.2 Sol recognised it 53 times out of 53 and knew its exact rate of missed payments: 22.12%. Told a false rate (10%), it moved 70% of the way there. It already knew the real one.
**Visual:** three big numbers in a row: **53/53** recognised · **22.12%** recalled · **70%** followed a false hint.

### 8. The punchline

**Headline:** A decades-old scorecard still ranks customers best.
**Body:** At sorting risky customers from safe ones, on a score where 0.5 is a coin flip and 1 is perfect: the scorecard 0.777, the best AI models 0.757 and 0.754. Close, but not ahead, and the scorecard learned from 20,000 past customers.
**Visual:** `results/figures/3_ranking_auc_ks.png`.

### 9. It depends on the portfolio

**Headline:** Who looks honest depends on your portfolio.
**Body:** Take the same customers, mixed to look like a bank where 5% or 50% miss payments. Where half do, Kev-4B becomes the most honest. Where 5% do, every model overstates the risk, the scorecard too.
**Visual:** table of how far off each model is, in percentage points, at 5%, 22% and 50%, with the most honest model in each column highlighted (`results/explain.json`).

### 10. What to take away

**Headline:** Ask how honest it is, not just how accurate.
**Body:**
- Check that its percentages come true, not just that it ranks well
- Check on your own customers: honesty changes with how many miss payments
- Correct its percentages with your own past outcomes (2,000 customers was enough)
- Test on data it can't have seen before
**Footer:** Next: why 99% accuracy can still be a disaster.
**Visual:** text only, with the next-post teaser at the bottom.
