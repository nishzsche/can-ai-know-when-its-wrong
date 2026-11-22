# X long-form post: "Can AI actually know when it's wrong?"

One long-form post with the main chart attached, and no links in it. Links go in the first reply. Written for anyone, tech or not: every term is explained in passing, and the blog carries the technical detail. The numbers come from `results/summary.csv`, `results/explain.json` and `data/contamination.md`.

**Attach:** `results/figures/1_promised_vs_actual.png`

## Post

> When an AI says "30% likely", does it mean 30%?
>
> Banks have to prove their risk numbers are honest: if a model says 30% of a group will miss a payment, about 30% should. (The technical word is calibration.) I held 8 models to that standard, on 2,000 real credit-card customers.
>
> The contestants:
> • 6 AI models, given no examples to learn from: Jev (a commercial AI built for yes/no decisions), Kev-4B, Kev-0.8B and Laya (free open-source ones), Qwen3.5-4B (a free general-purpose AI; I read its probability from inside the model) and GPT-6.2 Sol (ChatGPT's model, asked for a percentage)
> • 2 traditional bank models that learned from 20,000 past customers: a scorecard and gradient boosting
>
> All saw the same thing: credit limit and six months of bills, payments and lateness. No age, sex, marital status or education.
>
> What happened:
>
> 1. Most AI models overstated the risk. Jev said 48% would miss a payment, Kev-4B 49%. Actual: 22%.
>
> 2. Where they went wrong: half the customers pay part of their bill, on time. Only 12% of them missed a payment, the safest group of all. Jev predicted 51% for them, Kev-4B 54%.
>
> 3. One extra sentence ("about 22% of customers miss a payment") made 5 of 6 more honest. Kev-4B went from 27 percentage points off, on average, to 3. One small model got worse: 9 to 27.
>
> 4. GPT-6.2 Sol looked the most honest, tied with Laya at 5 points off. Then I checked:
> – it can recite 31 customers from the original data file, word for word
> – shown any customer, it named the public dataset they come from (53 of 53) and knew its exact rate of missed payments, 22.12%
> – hiding the currency, dates and codes didn't stop it: 50 of 50
> – told a false rate (10%), it moved 70% of the way there
> It listens to what it's told, and it most likely already knew the answer.
>
> 5. No AI beat the traditional scorecard at sorting risky customers from safe ones. On a ranking score where 0.5 is a coin flip and 1 is perfect: scorecard 0.777, best AI 0.757.
>
> 6. Correcting each AI with 2,000 past customers whose outcomes were known fixed its honesty: 0.3 to 4 points off. But that needs exactly the kind of data the AI was meant to replace.
>
> 7. Who looks honest depends on the bank. Mix the same customers to look like a bank where half miss payments, and Kev-4B becomes the most honest. Where 5% do, every model overstates the risk, the scorecard too.
>
> 8. Cost per 1,000 decisions: Sol about 74 cents, the open models under 2 cents on one rented GPU.
>
> Bonus: a bug in an open-source tool (jevper) turned one model's 32% into 0.004%. I worked around it and reported it.
>
> I wrote my predictions down before running anything. Three of four were wrong.

## First reply

> Full write-up, with the method, every chart and links to every technique: [blog link]
> Code and every prediction: github.com/nishzsche/can-ai-know-when-its-wrong
>
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI, CC BY 4.0). Every model was scored on the same 2,000 customers, and nothing was adjusted using them.
