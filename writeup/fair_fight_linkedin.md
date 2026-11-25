# LinkedIn post: "I gave AI the same data as a bank's model. It caught up."

A document post (a swipeable PDF carousel, like posts 1 and 2): ten square slides built by `writeup/build_fair_fight_carousel.py` from the slide text at the end of this file. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from the M6 predictions through `analysis/fair_fight_figures.py` and `results/robustness.json`, the same ones the blog post uses (`writeup/fair_fight_post.md`).

**Attach:** `writeup/fair_fight_carousel.pdf`, as a document ("Add a document" in the post composer). Document title (LinkedIn allows 58 characters): "I gave AI the same data as a bank's model. It caught up." A post takes images or a document, not both, and its media can't be changed after it's published.

**Blog post:** published 2026-11-25 at https://nishzsche.github.io/2026/11/25/fair-fight/ (from `writeup/fair_fight_post.md`). If it's ever edited in a way that changes a number, rebuild the carousel before reposting.

## Post text

> The biggest objection to my first AI experiment: it wasn't a fair fight. So I made it one.
>
> I'd asked AI models how likely 2,000 real credit-card customers were to miss a payment. None ranked them better than the scorecard banks have used for decades. But the scorecard had studied 20,000 past customers whose outcomes were known. The AI had studied nobody.
>
> So every AI model got the same 20,000 customers, in whatever way it could use them:
> → Four open models were trained on them.
> → Two AI models built for tables, NVIDIA's Kumo-Tabular and TabICLv2, read all 20,000 as examples, with no training.
> → Two closed models, GPT-6.2 Sol and Jev, can't be trained, so each saw the 20 most similar customers in its prompt.
>
> What happened:
>
> → They caught up. The six that were trained or read all 20,000 now rank customers like gradient boosting, today's standard. Kev-4B went from last place to 0.002 behind the top.
> → Caught up, not ahead. None of the six can be told apart from gradient boosting. Against the scorecard, only Kev-4B is clearly better, by 0.015.
> → Honest, after the same correction. One adjustment learned from 2,000 separate customers brings all but one model within chance of perfectly honest.
> → Training is not calibration. Straight out of training, one model promised 11% to a group in which 21% defaulted. Promising too little is the worse direction for a lender.
> → The AI built for tables matched the best with no training run, close to honest out of the box, for a fraction of a cent per 1,000 decisions.
> → The closed models can only look, and that wasn't enough. Shown 20 look-alikes, Sol mostly counted how many had defaulted. Simply counting ranked as well, for next to nothing. Sol cost about 10 dollars per 1,000 decisions.
> → Eight very different models, from a decades-old scorecard to a 4-billion-parameter language model, landed within 0.02 of each other. On this data, there's only so much anyone can tell.
>
> What I'd tell a bank:
> → If everything ties, start with the model you can explain.
> → If you want AI, try the kind built for tables before fine-tuning a language model.
> → Check calibration after training, and keep checking it.
> → A closed model can only look at your data, and every look sends customer records to a vendor.
>
> The fine print: one bank, Taiwan, 2005. With 2,000 test customers, gaps under about 0.02 can't be told apart. And GPT-6.2 Sol recognizes this public dataset and knows its default rate.
>
> Next: does shrinking a model to save money make it less honest?
>
> Write-up, code, trained models and every number in the first comment.

## First comment

> The full write-up: https://nishzsche.github.io/2026/11/25/fair-fight/
> The first two posts: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/ and https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/
> Code, every prediction and the trained models' weights: https://github.com/nishzsche/can-ai-know-when-its-wrong
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI "Default of Credit Card Clients", Yeh & Lien, 2009; CC BY 4.0). No age, sex, marital status or education in any model's input.

## Slides

Ten square slides (1080 × 1080). One idea per slide, big type, a chart where there's evidence to show: one slide for each of the blog post's seven charts, plus the hook, what to do, and the fine print with what's next. Charts are drawn by `writeup/build_fair_fight_carousel.py` from the same predictions as the blog's charts, and every model keeps one row order and one colour throughout.

### 1. Hook

**Headline:** I gave AI the same data as a bank's model. It caught up.

**Body:** The bank's models had studied 20,000 past customers. The AI had studied none. So I gave every model the same 20,000.

**Footer:** 8 AI models. 2,000 real credit-card customers to test on.

### 2. It caught up

**Headline:** Give AI the same 20,000 customers and it catches up

**Body:** Ranking: how often a model rates a customer who defaulted as riskier than one who paid. 0.5 is a coin flip.

### 3. How close

**Headline:** Caught up, not ahead

**Body:** None of the six that trained or read all 20,000 can be told apart from gradient boosting. Against the scorecard, only Kev-4B is clearly better, by 0.015.

### 4. Honest after the same correction

**Headline:** After the same correction, all but one are honest

**Body:** The average gap between the risk a model promised and what happened. Every model gets the same correction, learned from 2,000 separate customers.

### 5. Training is not calibration

**Headline:** Training is not calibration

**Body:** Straight out of training, Kev-0.8B promised 11% to a group in which 21% defaulted. Promising too little is the worse direction for a lender: prices and provisions come out too low.

### 6. Paying more

**Headline:** Paying more didn't buy better ranking

**Body:** The AI built for tables matched the best with no training run, for a fraction of a cent per 1,000 decisions. GPT-6.2 Sol, shown 20 examples, cost about 10 dollars.

### 7. The closed models can only look

**Headline:** Shown 20 look-alikes, Sol mostly counts them

**Body:** Just counting how many of the 20 defaulted ranks as well as Sol does (0.764 against 0.762). Jev ranks worse, and starts too high.

### 8. The ceiling

**Headline:** Eight very different models hit the same ceiling

**Body:** From a decades-old scorecard to a 4-billion-parameter language model, all within 0.02. A scorecard trained on a tenth of the data is only 0.018 behind.

### 9. What I'd tell a bank

**Headline:** What I'd tell a bank

**Body:**
- If everything ties, start with the model you can explain.
- If you want AI, try the kind built for tables before fine-tuning a language model.
- Check calibration after training, and keep checking it.
- A closed model can only look at your data, at many times the cost, and every look sends customer records to a vendor.

**Footer:** The AI caught up. It didn't pull ahead.

### 10. Next

**Headline:** What this can't tell you

**Body:**
- One bank, Taiwan, 2005, and the test customers come from the same months as the training ones.
- With 2,000 test customers, ranking gaps under about 0.02 can't be told apart.
- Every model got the same 20,000 customers, but Jev and Sol could only see 20 at a time.
- GPT-6.2 Sol recognizes this public dataset and knows its default rate.

**Footer:** Next: does shrinking a model to save money make it less honest?
