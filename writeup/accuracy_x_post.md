# X long-form post: "Why 99% accuracy can still be a disaster"

One long-form post with the chart attached, and no links in it. Links go in the first reply. Written for anyone, tech or not. Numbers come from `results/accuracy.json` (`analysis/accuracy.py`).

**Attach:** `results/figures/post2_profit.png`

## Post

> A fraud detector that never flags anything is 99.9% accurate. It has never caught a fraudster.
>
> Accuracy rewards agreeing with the majority. The rarer the thing you care about, the better a model that ignores it scores.
>
> I checked how much this matters on 2,000 real credit-card customers (Taiwan, 2005), using the predictions from my last experiment: 6 AI models answering cold, plus a scorecard and gradient boosting trained on 20,000 past customers.
>
> 1. The floor isn't zero. 22% missed a payment, so "nobody defaults" is 77.9% accurate. The best model scored 82.5%: less than 5 points better than doing nothing.
>
> 2. Three AI models scored worse than doing nothing: Laya 76.5%, Jev 60.9%, Kev-4B 55.2%. Meanwhile Qwen3.5-4B tied the scorecard: 81.9% vs 81.8%.
>
> 3. Accuracy hides which mistakes a model makes. At the usual 50% cut-off, the scorecard missed 291 of 442 defaulters but wrongly turned away only 74 payers. Jev caught 316 defaulters by turning away 656 people who would have paid. Accuracy only adds up the right answers.
>
> 4. So I made them lend. Each paying customer earns 1 unit, each default costs 4 (illustrative), so approve anyone under 20% risk.
> – Approve everyone (the 77.9%-accurate model): loses 105 per 1,000 customers
> – Scorecard, on its own percentages: +284
> – Gradient boosting: +303
> – GPT-6.2 Sol: +294
> – Qwen, as accurate as the scorecard: +17
>
> 5. Why Qwen earns nothing: its percentages run about 1.6 times too high. At the 20% cut-off it catches 428 of 442 defaulters, and turns away 1,468 of 1,558 good customers.
>
> 6. Ranking and honesty are different things. Ignore Kev-0.8B's percentages, use only its ranking, and pick the cut-off from 2,000 other customers with known outcomes: it earns 306, the best of all. On its own percentages: 99. Jev: 226 vs 14.
>
> 7. The models with honest percentages earn about the same either way. That's what calibration buys: you can use the number as it comes.
>
> 8. Make defaults costlier (6 units) and approving everyone loses 547. The trained models still earn about 190 on their own numbers; the best AI on its own numbers, Sol, 110. At 10 units, only the trained models make real money.
>
> What to look at instead of accuracy: the do-nothing score next to it, whether the model can rank risky above safe, whether its percentages come true, and a cut-off set from your own costs (20% here, not 50%).
>
> Caveats: one public dataset, illustrative costs, and the AI models answered cold while the traditional models learned from 20,000 customers. Sol also recognizes this dataset. Next: what happens when every AI model gets the same 20,000.

## First reply

> Full write-up: https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/
> The first experiment: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/
> Code and every number (analysis/accuracy.py): github.com/nishzsche/can-ai-know-when-its-wrong
>
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI, CC BY 4.0). Cut-offs were chosen on 2,000 separate customers; nothing was tuned on the test set.
