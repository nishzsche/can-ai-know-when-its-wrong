# LinkedIn post: "I shrank AI models. It saved memory, not time."

A document post (a swipeable PDF carousel, like posts 1–3): nine square slides built by `writeup/build_quantization_carousel.py` from the slide text at the end of this file. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from `results/quantization.csv`, `results/speed_throughput.csv` and the prediction records, the same ones the blog post uses (`writeup/quantization_post.md`).

**Attach:** `writeup/quantization_carousel.pdf`, as a document ("Add a document" in the post composer). Document title (LinkedIn allows 58 characters): "I shrank AI models. It saved memory, not time." A post takes images or a document, not both, and its media can't be changed after it's published.

**Blog post:** https://nishzsche.github.io/2026/11/27/shrinking-ai-models/ (blog PRs #54 and #55; dated 00:55 IST on 27 November, and the review URL redirects to it). If it's ever edited in a way that changes a number, rebuild the carousel before reposting.

## Post text

> Saving a photo at lower quality works for a while. Then it turns into blocks. AI models get shrunk the same way to save money, so I tested how far that goes before a bank should worry.
>
> In my last post I trained three AI models on 20,000 real credit-card customers. This time I squeezed each one from 16 bits per number down to 8, 4, 3 and 2, and asked the same 2,000 test customers the same question: how likely is this customer to miss next month's payment?
>
> What happened:
>
> → Ranking held down to 4 bits. All three stayed within 0.005 of their full size. At 2 bits, two of them ranked customers worse than a coin flip.
> → Honesty held too. Above 2 bits, the only change bigger than chance was an improvement, and nobody chose it: it's a side effect of rounding.
> → But the summary scores hid the customers moving. At 3 bits, 6 in 10 of one model's customers moved by 5 points or more, while its ranking and honesty looked as good as full size.
> → That changes lending decisions. Swap in a 3-bit model and keep the old cut-off, and 10 to 20 in 100 approve-or-decline decisions flip. Even 8 bits changed more than running the same model twice.
> → It saved memory, not time. At 4 bits the models needed less than half the memory. But on a properly tuned GPU, only one of nine shrunk versions answered faster than its full size, by 11%.
> → My first speed test was wrong. It said shrunk models were up to 4 times slower. A review of my draft found my setup held every version back, so I re-ran it properly. Tuning the setup made full size up to 3 times faster, more than any shrinking did.
>
> What I'd tell a bank:
> → Treat shrinking as a model change, not an IT change. Agree how many decisions may change before you test.
> → Compare each customer's number before and after, not just the summary scores. It needs no outcomes, so it can be done on day one.
> → Price the saving on your own setup. It's memory, so it only pays if you can use a smaller or shared GPU.
>
> The fine print: one bank, Taiwan, 2005. Three models, one GPU, and only shrinking methods that need no data. All three models learned from the same 20,000 customers as the scorecard I compare them with.
>
> Next: why AI can write code but struggle to read a table.
>
> Write-up, code and every number in the first comment.

## First comment

> The full write-up: https://nishzsche.github.io/2026/11/27/shrinking-ai-models/
> The first three posts: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/, https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/ and https://nishzsche.github.io/2026/11/25/fair-fight/
> Code and every prediction, with the library versions behind it: https://github.com/nishzsche/can-ai-know-when-its-wrong
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI "Default of Credit Card Clients", Yeh & Lien, 2009; CC BY 4.0). No age, sex, marital status or education in any model's input.

## Slides

Nine square slides (1080 × 1080). One idea per slide, big type, a chart where there's evidence to show: one slide for each of the blog post's four charts, plus the hook, how shrinking works, the lending decisions, what to do, and the fine print with what's next. Charts are drawn by `writeup/build_quantization_carousel.py` from the same results as the blog's charts, and each model keeps its colour throughout.

### 1. Hook

**Headline:** I shrank AI models. It saved memory, not time.

**Body:** Like saving a photo at lower quality: for a while, you can't tell. I squeezed three trained AI models to 8, 4, 3 and 2 bits, and checked what a bank would check.

**Footer:** 3 trained AI models. 2,000 real credit-card customers to test on.

### 2. How shrinking works

**Headline:** Shrinking rounds every number more coarsely

**Body:** A model is billions of numbers, each normally stored in 16 bits. Fewer bits means fewer values each number can take, and less memory to hold them.

### 3. Ranking

**Headline:** Ranking held to 4 bits, and collapsed at 2

**Body:** Ranking: how often a model rates a customer who defaulted as riskier than one who paid. 0.5 is a coin flip.

### 4. Honesty

**Headline:** Honesty held until 2 bits

**Body:** The average gap between the risk a model promised and what happened. Above 2 bits, the only change bigger than chance was an improvement.

### 5. The customers moved

**Headline:** The scores held while the customers moved

**Body:** Each dot is one customer: Kev-0.8B at full size against shrunk. At 3 bits, 6 in 10 moved by 5 points or more, yet its ranking and honesty looked as good as full size.

### 6. Lending decisions

**Headline:** Shrinking quietly changes lending decisions

**Body:** Swap in the shrunk model and keep the cut-off chosen for the full-size one: the share of approve-or-decline decisions that change.

### 7. Speed

**Headline:** It saved memory, not time

**Body:** Time per answer on a tuned GPU, one customer at a time. Only Kev-4B on GemLite beat its full size, by 11%. Tuning the setup mattered more than shrinking.

### 8. What I'd tell a bank

**Headline:** What I'd tell a bank

**Body:**
- Treat shrinking as a model change, not an IT change. Agree how many decisions may change before you test.
- Compare each customer's number before and after, not just the summary scores. It needs no outcomes, so do it on day one.
- 8 bits and the common 4-bit format passed here. That's where to start testing, not a reason to skip it.
- Price the saving on your own setup: it's memory, so it only pays if you can use a smaller or shared GPU.

**Footer:** The scores can hold while the customers move.

### 9. Next

**Headline:** What this can't tell you

**Body:**
- One bank, Taiwan, 2005. Three models, one GPU.
- Only shrinking methods that need no data. Methods that learn from sample data may do better at 3 and 4 bits.
- Speed depends on the whole setup: a bigger GPU or a serving engine could change which version is fastest.
- Fair lending isn't tested: at 3 bits, 10 to 20 in 100 decisions changed, and I haven't checked whose.

**Footer:** Next: why AI can write code but struggle to read a table.
