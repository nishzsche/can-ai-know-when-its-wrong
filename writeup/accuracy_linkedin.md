# LinkedIn post: "Why 99% accuracy can still be a disaster"

A document post (a swipeable PDF carousel, like post 1's): nine square slides built by `writeup/build_accuracy_carousel.py` from the slide text at the end of this file. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from `results/accuracy.json` (`analysis/accuracy.py`), the same ones the published post uses.

**Attach:** `writeup/accuracy_carousel.pdf`, as a document ("Add a document" in the post composer). Document title (LinkedIn allows 58 characters): "Why 99% accuracy can still be a disaster: 8 models tested". A post takes images or a document, not both.

**Replacing the first version:** the post first went up boosted, with the profit chart as a single image. LinkedIn can't change a published post's media, so it is replaced: delete the boost (Manage, then Delete ad), delete the post, and repost with the carousel.

**If posting the charts as images instead** (`results/figures/post2_accuracy.png`, `post2_confusion.png`, `post2_profit.png`, `post2_cost_ratio.png`, in that order), the alt text for each, under 300 characters (LinkedIn's limit may be 120 or 300, depending on the source):

- `post2_accuracy.png`: Two dot charts. Left, accuracy at the 50% cut-off: saying "won't default" to everyone scores 77.9%, the best model 82.5%, and Laya, Jev and Kev-4B fall below doing nothing. Right, share of the 442 defaulters caught: Jev (71%) and Kev-4B (64%) catch the most.
- `post2_confusion.png`: Bars showing where 2,000 customers landed at the 50% cut-off. Of the 442 who defaulted, doing nothing misses all 442, the scorecard 291, Jev 126. Of the 1,558 who paid, doing nothing turns away none, the scorecard 74, Jev 656. Accuracy counts only the right calls.
- `post2_profit.png`: Profit per 1,000 customers, earning 1 per good customer and losing 4 per default. Gradient boosting, the scorecard and GPT-6.2 Sol earn about 300 either way. On their own percentages Qwen earns 17, Jev 14, Kev-0.8B 99; with a learned cut-off, 252, 226, 306. Approving everyone loses 105.
- `post2_cost_ratio.png`: Profit per 1,000 customers as a default costs 1 to 10 times what a good customer earns. Approving everyone loses 547 at 6. Gradient boosting and the scorecard stay profitable, about 190 at 6. Sol and Laya lose money by 10; Jev, Qwen and both Kevs approve almost nobody.

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
> → Accuracy can't see which mistakes a model makes. One AI model caught twice as many defaulters as the scorecard, by turning away 656 people who would have paid.
>
> Then I made them lend. Say each customer who pays earns a bank 1 unit and each default costs 4.
>
> → The 78%-accurate "nobody defaults" model approves everyone and loses 105 units per 1,000 customers.
> → Qwen, as accurate as the scorecard, earns 17 when you lend on its own percentages. The scorecard earns 284. At the lending cut-off, Qwen catches 428 of 442 defaulters, but turns away 1,468 of the 1,558 customers who would have paid.
> → Another AI model ranks customers well enough to earn 306, the best in the test, once you pick its cut-off from past outcomes. On its own percentages, it earns 99.
> → The models whose percentages come true earn about the same either way. That's what honest numbers buy you: you can use them as they come.
>
> Raise the cost of a default to 6 units and approving everyone loses 547 per 1,000. The two traditional models, taken at their word, still earn about 190. At 10 to 1, every AI model taken at its word either loses money or approves almost nobody. The costlier a mistake, the more the decision rests on honest numbers.
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

## Slides

Nine square slides (1080 × 1080). One idea per slide, big type, a chart where there's evidence to show. Charts are drawn by `writeup/build_accuracy_carousel.py` from `results/accuracy.json`.

### 1. Hook

**Headline:** Why 99% accuracy can still be a disaster

**Body:** A model that says nobody will default is 78% accurate. It loses money.

**Footer:** 8 models. 2,000 real credit-card customers.

### 2. The fraud detector

**Headline:** A fraud detector that never flags anything is 99.9% accurate.

**Body:** It has never caught a fraudster. When one answer is far more common, agreeing with the majority is easy, and worthless.

**Footer:** 1 fraud in every 1,000 payments. Saying "not fraud" every time is right 999 times.

### 3. The floor

**Headline:** On my credit data, doing nothing is 77.9% accurate

**Body:** 22% of 2,000 customers missed a payment, so saying "won't default" to everyone is right 77.9% of the time. The best model beats that by 4.6 points.

### 4. Which mistakes

**Headline:** Accuracy can't see which mistakes a model makes

**Body:** Jev catches twice as many defaulters as the scorecard, by turning away 656 people who would have paid. Accuracy only adds up the right calls.

### 5. Make them lend

**Headline:** So I made them lend

**Body:**
- Each customer who pays earns the bank 1 unit. Each default costs it 4.
- So a customer is worth approving only below a 20% chance of default.
- Doing nothing approves everyone: per 1,000 customers it earns 779 and loses 884.

**Footer:** The 78%-accurate model loses 105.

### 6. More accurate isn't worth more

**Headline:** More accurate isn't worth more

**Body:** Qwen is as accurate as the scorecard. Lending on its own percentages, it earns 17 per 1,000 customers. The scorecard earns 284.

### 7. When mistakes cost more

**Headline:** The costlier a mistake, the more honesty pays

**Body:** At 6 to 1, approving everyone loses 547 per 1,000, and the trained models still earn about 190. At 10 to 1, every AI model taken at its word loses money or approves almost nobody.

### 8. What to look at

**Headline:** So what should you look at instead?

**Body:**
- Accuracy, only next to the do-nothing score. 82% sounds good until you learn 78% is free.
- Ranking: can the model tell risky from safe at all?
- Honesty: do its percentages come true?
- Your own costs, to set the cut-off. Here it was 20%, not 50%.

**Footer:** The 50% cut-off is a convention, not a decision.

### 9. Next

**Headline:** To be fair to the AI

**Body:** The scorecard and gradient boosting learned from 20,000 past customers. The AI models answered cold, with no examples.

**Footer:** Next: what happens when every model gets the same 20,000 customers.
