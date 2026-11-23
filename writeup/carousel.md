# LinkedIn carousel: "Can AI know when it's wrong? I tested Jev and 7 other models on 2,000 real credit-card customers"

Ten square slides (1080 × 1080), exported as one PDF by `writeup/build_carousel.py`. One idea per slide, big type, a chart where there's evidence to show. The numbers come from `results/summary.csv` and `data/contamination.md`, and the charts from `results/figures/`.

## Post text

Written for anyone, tech or not: every term is explained the first time it appears.

> Everyone's talking about Jev. I gave it, and 7 other models, 2,000 real credit-card customers and checked whether their percentages come true: the standard banks get audited against.
>
> "Honest" has a precise meaning here: when a model says a customer is 30% likely to miss a payment, about 30 in every 100 such customers should actually miss it. Statisticians call this calibration, and banks have to prove it.
>
> Every model got the same customers and one question: will this customer miss next month's payment? Six were AI models, Jev among them, given no examples to learn from. Two were traditional bank models that learned from 20,000 past customers.
>
> What I found:
> → Most AI models said customers were riskier than they were. Two of them more than doubled it.
> → Two of them treated some of the safest customers, people paying part of their bill on time, as coin flips.
> → One extra sentence, "about 22% of customers miss a payment", helped four models, made no clear difference to one, and broke one.
> → Simply shifting each AI's numbers to match that 22%, with no past outcomes needed, fixed most of the error for four of six.
> → The closest AI already knew this famous public dataset and its default rate, so its score isn't a clean test.
> → The trained models still sorted risky customers from safe ones best. For a simple approve-or-decline call, the best AI models did about as well.
> → Who looks honest depends on the bank: if half the customers missed payments, the order changes.
> → One model was partly misled by my table's layout: shown each payment beside the bill it pays, Kev-4B's average fell from 49% to 36%.
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
**Body:** I held Jev and 7 other models to the same standard.
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
**Visual:** two columns of model names, each with a few words on what it is. AI: Jev (built for yes/no decisions), Kev-4B and Kev-0.8B (free, open-source), Laya (open-source, tiny), Qwen3.5-4B (free, general-purpose), GPT-6.2 Sol (OpenAI's chat model). Traditional: Scorecard (banks' classic), Gradient boosting (today's standard).

### 3. What "honest" looks like

**Headline:** When it says 30%, do 30% default?
**Body:** A weather app that says "70% chance of rain" should see rain on about 7 of every 10 such days. A risk model is the same: when it says 30% will miss a payment ("default"), about 30% should. On the diagonal, the model is honest.
**Visual:** a simple diagonal line labelled "honest", with "too pessimistic" below it and "too optimistic" above it. Or the scorecard panel from figure 1.

### 4. Promised vs actual

**Headline:** Most AI models think customers are riskier than they are.
**Body:** On average, Jev said 48% of these customers would miss a payment, and Kev-4B 49%. The real figure: 22%.
**Visual:** `results/figures/1_promised_vs_actual.png`, or crop it to the Jev and Kev-4B panels for phone size.

### 5. Where it goes wrong

**Headline:** Two models treated some of the safest customers as coin flips.
**Body:** Half the customers pay part of their balance on time. Only 12% of them defaulted, about as few as those who pay in full. Jev predicted 51% for them, Kev-4B 54%. (Kev-4B said 36% when my table showed each payment beside the bill it pays.)
**Visual:** bars: each model's average prediction for these customers, against the 12% who actually defaulted (`results/explain.json`).

### 6. The honesty arc

**Headline:** One sentence helps, but unpredictably.
**Body:** Adding "About 22% of customers miss a payment" clearly helped four models: Kev-4B went from 27 percentage points off, on average, to 3. It made no clear difference to Laya and broke Kev-0.8B, from 9 points off to 27.
**Visual:** `results/figures/2_honesty_arc.png`.

### 7. The twist

**Headline:** The closest model already knew this dataset.
**Body:** These customers come from a famous public dataset. Asked which dataset a summary came from, GPT-6.2 Sol named it every time and knew its exact rate of missed payments. So its score isn't a clean test of judging customers from scratch.
**Visual:** three big numbers: **53/53** named the dataset · **22.12%** default rate recalled · **26→18%** its average when told a false 10%. Note: the question invited a dataset name, and Sol ranks the "not late" groups in the opposite order to this data, so it knows the dataset, not its patterns.

### 8. The punchline

**Headline:** The trained models still rank customers best.
**Body:** On a score where 0.5 is a coin flip and 1 is perfect: gradient boosting 0.789, the scorecard 0.777, the best AI out of the box 0.754. For a simple approve-or-decline call, the best AI models did about as well.
**Visual:** `results/figures/3_ranking_auc_ks.png`.

### 9. It depends on the portfolio

**Headline:** Who looks honest depends on your portfolio.
**Body:** Mix the same customers to look like a bank where 5% or 50% miss payments. Where half do, Kev-4B is the most honest; where 5% do, everyone overstates. It's mostly each model's average.
**Visual:** table of how far off each model is, in percentage points, at 5%, 22% and 50%, with the most honest model in each column highlighted (`results/explain.json`).

### 10. What to take away

**Headline:** Ask how honest it is, not just how accurate.
**Body:**
- Check that its percentages come true, not just that it ranks well
- Shift its numbers to your own default rate: no past outcomes needed
- Then correct them with your own past outcomes (2,000 customers was enough)
- Test on data it can't have seen, laid out more than one way
**Footer:** Next: why 99% accuracy can still be a disaster.
**Visual:** text only, with the next-post teaser at the bottom.
