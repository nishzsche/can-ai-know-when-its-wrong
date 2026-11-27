# LinkedIn post: "I shrank AI models. It saved memory, not time."

A document post (a swipeable PDF carousel, like posts 1–3): nine square slides built by `writeup/build_quantization_carousel.py` from the slide text at the end of this file. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from `results/quantization.csv`, `results/speed_throughput.csv` and the prediction records, the same ones the blog post uses (`writeup/quantization_post.md`).

**Attach:** `writeup/quantization_carousel.pdf`, as a document ("Add a document" in the post composer). Document title (LinkedIn allows 58 characters): "I shrank AI models. It saved memory, not time." A post takes images or a document, not both, and its media can't be changed after it's published.

**Blog post:** https://nishzsche.github.io/2026/11/27/shrinking-ai-models/ (blog PRs #54 and #55; dated 00:55 IST on 27 November, and the review URL redirects to it). If it's ever edited in a way that changes a number, rebuild the carousel before reposting.

## Post text

> Saving a photo at lower quality works for a while. Then it turns into blocks. AI models get shrunk the same way to save money, so I tested how far that goes before a bank should worry.
>
> In my last post I trained three AI models on 20,000 real credit-card customers. This time I shrank each one. Every number inside a model normally takes 16 bits; I cut that to 8, 4, 3 and 2. Then I asked every version about the same 2,000 test customers: how likely is this person to miss next month's payment?
>
> What I found:
>
> → Down to 4 bits, all three models sorted risky customers from safe ones as well as at full size. At 2 bits, two of them did worse than a coin flip.
> → Their risk numbers stayed as honest as at full size down to 3 bits. At 2 bits, that broke too.
> → But those summary scores hid a lot of movement. At 3 bits, one model changed 6 in 10 customers' risk by 5 points or more (say, from 20% to 25%), and its summary scores still looked as good as at full size.
> → That changes lending decisions. If a bank swapped in a 3-bit model and kept its old cut-off, 10 to 20 in every 100 approve-or-decline decisions would flip. Even 8 bits changed more decisions than running the same model twice.
> → It saved memory, not time. At 4 bits, the two larger models used less than half the memory. But even with the GPU set up for speed, only one of nine shrunk versions answered faster than its full size, by 11%.
>
> What I'd tell a bank:
> → Treat shrinking a model as changing the model, not as an IT upgrade. Decide up front how many decisions are allowed to change.
> → Compare each customer's score before and after, not just the summary scores. You don't need to wait for outcomes, so you can do it on day one.
> → Work out the saving on your own hardware. Shrinking saves memory, so it only pays if it lets you use a smaller or shared GPU.
>
> The fine print: one bank's customers, Taiwan, 2005. Three models, one GPU, and only shrinking methods that need no extra data. All three models learned from the same 20,000 customers as the traditional scorecard I compare them with.
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

**Body:** Honesty: the average gap between the risk a model promised and what really happened. Down to 3 bits, none got measurably less honest.

### 5. The customers moved

**Headline:** The scores held while the customers moved

**Body:** Each dot is one customer's risk from Kev-0.8B, full size against shrunk. At 3 bits, 6 in 10 moved by 5 points or more, yet its summary scores held.

### 6. Lending decisions

**Headline:** Shrinking quietly changes lending decisions

**Body:** Swap in the shrunk model and keep the cut-off chosen for the full-size one: the share of approve-or-decline decisions that change.

### 7. Speed

**Headline:** It saved memory, not time

**Body:** At 4 bits, the two larger models used less than half the memory. But even on a GPU set up for speed, only Kev-4B on GemLite beat its full size, by 11%.

### 8. What I'd tell a bank

**Headline:** What I'd tell a bank

**Body:**
- Treat shrinking as changing the model, not as an IT upgrade. Decide up front how many decisions may change.
- Compare each customer's score before and after, not just the summary scores. It needs no outcomes, so do it on day one.
- 8 bits and the common 4-bit format passed here. That's where to start testing, not a reason to skip it.
- Work out the saving on your own hardware: it's memory, so it only pays if you can use a smaller or shared GPU.

**Footer:** The scores can hold while the customers move.

### 9. Next

**Headline:** What this can't tell you

**Body:**
- One bank's customers, Taiwan, 2005. Three models, one GPU.
- Only shrinking methods that need no extra data. Methods that learn from sample data may do better at 3 and 4 bits.
- Speed depends on the hardware and software: a bigger GPU or other serving software could change which version is fastest.
- Fair lending isn't tested: at 3 bits, 10 to 20 in 100 decisions changed, and I haven't checked whose.

**Footer:** Next: why AI can write code but struggle to read a table.
