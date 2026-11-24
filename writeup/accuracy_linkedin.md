# LinkedIn post: "Why 99% accuracy can still be a disaster"

A single-image post (no carousel), with the post's chart attached. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from `results/accuracy.json` (`analysis/accuracy.py`), the same ones the published post uses.

**Attach:** `results/figures/post2_profit.png` (portrait, reads well on a phone). Alternative: the link-preview card `results/figures/post2_social_card.png`.

## Post text

> A fraud detector that never flags anything is 99.9% accurate. It has never caught a fraudster.
>
> Accuracy counts how often a model agrees with what happened. When one answer is far more common, agreeing with the majority is easy and worthless.
>
> I tested this on the same 2,000 real credit-card customers from my last experiment. 22% missed a payment, so a "model" that says nobody will default is right 77.9% of the time. That's the floor, not zero.
>
> What happened when I judged eight models the way most people do, by accuracy:
>
> → The best model beat doing nothing by less than 5 points (82.5%).
> → Three of the six AI models were less accurate than doing nothing.
> → A small AI model (Qwen) tied a traditional bank scorecard: 81.9% vs 81.8%.
>
> Then I made them lend. Say each customer who pays earns a bank 1 unit and each default costs 4.
>
> → The 78%-accurate "nobody defaults" model approves everyone and loses 105 units per 1,000 customers.
> → Qwen, as accurate as the scorecard, earns 17 when you lend on its own percentages. The scorecard earns 284. At the lending cut-off, Qwen catches 428 of 442 defaulters, but turns away 1,468 of the 1,558 customers who would have paid.
> → Another AI model ranks customers well enough to earn 306, the best in the test, once you pick its cut-off from past outcomes. On its own percentages, it earns 99.
> → The models whose percentages come true earn about the same either way. That's what honest numbers buy you: you can use them as they come.
>
> Raise the cost of a default to 6 units and approving everyone loses 547 per 1,000. The costlier a mistake, the more the decision rests on honest numbers.
>
> So, instead of accuracy:
> → Compare it with the do-nothing score. 82% sounds good until you learn 78% is free.
> → Ask whether the model can tell risky from safe at all.
> → Ask whether its percentages come true.
> → Set the cut-off from your own costs. Here the right one was 20%, not the 50% accuracy assumes.
>
> To be fair to the AI: the two traditional models learned from 20,000 past customers, and the AI models answered cold. Next post: what happens when they all get the same 20,000.
>
> Write-up, code and every number in the first comment.

## First comment

> The full write-up: https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/
> The first experiment, on whether AI's percentages come true: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/
> Code and every number (analysis/accuracy.py): https://github.com/nishzsche/can-ai-know-when-its-wrong
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI "Default of Credit Card Clients", Yeh & Lien, 2009; CC BY 4.0). The lending costs are illustrative.
