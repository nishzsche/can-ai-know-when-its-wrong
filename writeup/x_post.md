# X long-form post: "Can AI know when it's wrong? I tested Jev and 7 other models on 2,000 real credit-card customers"

One long-form post with the main chart attached, and no links in it. Links go in the first reply. Written for anyone, tech or not: every term is explained in passing, and the blog carries the technical detail. The numbers come from `results/summary.csv`, `results/explain.json` and `data/contamination.md`.

**Attach:** `results/figures/1_promised_vs_actual.png`

## Post

> Everyone's talking about Jev. When it says "30% likely", does it mean 30%?
>
> Banks have to prove their risk numbers are honest: if a model says 30% of a group will miss a payment, about 30% should. (The technical word is calibration.) I held Jev and 7 other models to that standard, on 2,000 real credit-card customers.
>
> The contestants:
> • 6 AI models, given no examples to learn from: Jev (a commercial AI built for yes/no decisions), Kev-4B, Kev-0.8B and Laya (free open-source ones), Qwen3.5-4B (a free general-purpose AI; I read its probability from inside the model) and GPT-6.2 Sol (OpenAI's chat model, asked for a percentage)
> • 2 traditional bank models that learned from 20,000 past customers: a scorecard and gradient boosting
>
> All saw the same thing: credit limit and six months of bills, payments and lateness. No age, sex, marital status or education.
>
> What happened:
>
> 1. Most AI models overstated the risk. Jev said 48% would miss a payment, Kev-4B 49%. Actual: 22%.
>
> 2. Where they went wrong: half the customers pay part of their bill, on time. Only 12% of them missed a payment, about as few as those who pay in full. Jev predicted 51% for them, Kev-4B 54%. (Part of Kev-4B's error was my table's fault: shown each payment beside the bill it actually pays, it said 36%, and ranked customers much better.)
>
> 3. One extra sentence ("about 22% of customers miss a payment") helped four. Kev-4B went from 27 percentage points off, on average, to 3. It did nothing clear for Laya, and one small model got worse: 9 to 27.
>
> Simpler still: shift each AI's numbers until they average 22%. No past outcomes needed, and it beat the sentence for four of six.
>
> 4. GPT-6.2 Sol looked the most honest, tied with Laya at 5 points off. Then I checked:
> – it can recite 31 customers from the original data file, word for word
> – asked which public dataset a customer came from, it named this one (53 of 53) and knew its exact rate of missed payments, 22.12%
> – hiding the currency, dates and codes didn't stop it: 50 of 50
> – told a false rate (10%), its average fell from 26% to 18%
> It knows this dataset, so its score isn't a clean test. (My question did invite a dataset name, and it doesn't seem to know the file's patterns.)
>
> 5. No AI beat the trained models at sorting risky customers from safe ones. On a ranking score where 0.5 is a coin flip and 1 is perfect: gradient boosting 0.789, scorecard 0.777, best AI 0.757. For a simple approve-or-decline call, though, the best AI did about as well.
>
> 6. Correcting each AI with 2,000 past customers whose outcomes were known brought it to within 2 to 4 points. But that needs exactly the kind of data the AI was meant to replace, and with those 2,000 a scorecard trained from scratch ranks as well as the best AI.
>
> 7. Who looks honest depends on the bank. Mix the same customers to look like a bank where half miss payments, and Kev-4B becomes the most honest. Where 5% do, every model overstates the risk, the scorecard too. That's mostly down to each model's average.
>
> 8. Cost per 1,000 decisions: Sol about 74 cents, Jev about 2 cents, the open models under 2 cents on one rented GPU.
>
> Bonus: a bug in an open-source tool (jevper) turned one model's 32% into 0.004%. I worked around it and reported it.
>
> I wrote my predictions down before running anything. Three of four were wrong.
>
> Caveats: the AI models got no examples while the traditional models learned from 20,000, and this is one public dataset from one bank in 2005.

## First reply

> Full write-up: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/
> The deep dive, with the method and every check: https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/writeup/deep_dive.md
> Code and every prediction: github.com/nishzsche/can-ai-know-when-its-wrong
>
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI, CC BY 4.0). Every model was scored on the same 2,000 customers, and nothing was adjusted using them.
