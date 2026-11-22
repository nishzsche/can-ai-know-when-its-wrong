# X long-form post: "Can AI actually know when it's wrong?"

One long-form post with the main chart attached, and no links in it. The method and links go in the first reply. The numbers come from `results/summary.csv` and `data/contamination.md`.

**Attach:** `results/figures/1_promised_vs_actual.png`

## Post

> When an AI says "30% likely", does it mean 30%?
>
> I tested 8 models on 2,000 real credit-card customers, the way a bank checks its risk models: does the promised default rate match what actually happens?
>
> The contestants:
> • 6 AI models, answering cold: Jev, Kev-4B, Kev-0.8B, Laya, Qwen3.5-4B (probability read from its logprobs), GPT-6.2 Sol (asked for a %)
> • 2 classic models trained on 20,000 labelled customers: a scorecard and gradient boosting
>
> Same inputs for all of them: credit limit plus 6 months of bills, payments and repayment status. No demographics.
>
> What happened:
>
> 1. Most AI models overstate risk. Jev said 48% would default and Kev-4B 49%. Actual: 22%.
>
> 2. One sentence ("about 22% of customers miss a payment") cut calibration error for 5 of 6. Kev-4B went from 0.27 to 0.03. Kev-0.8B went the other way: 0.09 → 0.27.
>
> 3. GPT-6.2 Sol tied with Laya as the most honest out of the box (0.05). Then I checked:
> – it recites 31 customers from the original data file verbatim
> – it names the dataset for 53 of 53 summaries and states its exact default rate, 22.12%
> – stripping the currency, dates and codes doesn't stop it: 50 of 50
> – told a false rate (10%), it moves 70% of the way there
> It does listen to stated rates, and it most likely already knew this one.
>
> 4. No AI model beat the scorecard at ranking (AUC 0.777 vs best 0.757). The scorecard was trained; the AI models weren't.
>
> 5. Recalibrating on 2,000 labelled customers brings every AI model to 0.003–0.041 calibration error. Honesty is fixable, but it costs labels.
>
> 6. Cost per 1,000 decisions: Sol $0.74, the open models under $0.02 on one L4.
>
> Bonus: a bug in jevper's logprob readout turned Qwen's 0.32 into 0.00004. I worked around it and reported it upstream.
>
> I wrote my predictions down before running anything. Three of four were wrong.

## First reply

> Full write-up: [blog link]
> Code, all predictions and every check: github.com/nishzsche/can-ai-know-when-its-wrong
>
> Method in short: the same test set for every model; nothing tuned on test; recalibration (Platt or isotonic, picked by CV) fitted on a separate 2,000; bootstrap intervals; one unit-tested template for every AI model. Data: UCI Default of Credit Card Clients (CC BY 4.0).
