# LinkedIn post: "Why AI can write code but struggle to read a table"

A document post (a swipeable PDF carousel, like posts 1–4): ten square slides built by `writeup/build_formats_carousel.py` from the slide text at the end of this file. Links go in the first comment, not the post. Written for anyone, tech or not: every term is explained in passing. Numbers come from `results/layouts.csv` and the prediction records, the same ones the blog post uses (`writeup/formats_post.md`).

**Attach:** `writeup/formats_carousel.pdf`, as a document ("Add a document" in the post composer). Document title (LinkedIn allows 58 characters): "Why AI can write code but struggle to read a table" (50). If the blog title changes after review, change `TITLE` in the builder, the hook slide's headline and this file's heading to match, and rebuild. A post takes images or a document, not both, and its media can't be changed after it's published.

**Blog post:** https://nishzsche.github.io/2026/11/27/reading-tables/ (blog PR #63; dated 16:55 IST on 27 November, and the review URL redirects to it). If it's ever edited in a way that changes a number, rebuild the carousel before reposting.

## Post text

> Ask an AI chatbot to write a program that reads a spreadsheet, and it'll do it in seconds. The program reads 46,990 as 46,990, whatever the layout. Give the chatbot the spreadsheet itself, though, and it reads it as text, a few characters at a time. Does the layout change its answer?
>
> I tested it on lending. Six AI models, the same 2,000 real credit-card customers, one question: how likely is this customer to miss next month's payment? Every number stayed the same, character for character. Only the layout changed: a table, sentences, CSV and JSON (two formats programs use to swap data), and two more.
>
> What happened:
>
> → One model's ranking went from its best to worse than a coin flip. Laya ranked customers at 0.716 when they were written as sentences, and at 0.478 as JSON.
> → Another went the other way. Kev-4B, the worst of the six with the original table (0.594), reached 0.729 with JSON.
> → Answers moved even where the ranking held. Laya's moved 26 points per customer with CSV and JSON, four times what shrinking a model to 3 bits did in my last post. Kev-0.8B's ranking never changed, yet with CSV it flipped a quarter of its lending decisions.
> → The percentages moved too. Laya promised 20% risk for these customers as a table and 44% as CSV. 22% actually defaulted.
> → Keep the old approval cut-off and up to 63 in 100 decisions flip. With JSON, Laya's lender would have approved nobody.
> → The frontier chatbot barely moved, hardly more than when it's asked the same table twice. But it knows this dataset, so it may be answering from memory.
>
> What I'd tell a bank:
> → The layout is part of the model. Treat any change to the prompt template as a model change: adding one line of dashes and some bars changed 22 in 100 of one model's decisions.
> → Test a few layouts before trusting one. Score the same customers in two layouts and compare each customer's number. It needs no outcomes.
> → When a decision rests on numbers, let something read them as numbers.
>
> The fine print: one bank, Taiwan, 2005. The AI models answered cold, while the scorecard I compare them with learned from 20,000 customers. 11 of the 30 ranking changes were bigger than chance; the rest weren't.
>
> Write-up, code and every number in the first comment.

## First comment

> The full write-up: https://nishzsche.github.io/2026/11/27/reading-tables/
> The first four posts: https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/, https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/, https://nishzsche.github.io/2026/11/25/fair-fight/ and https://nishzsche.github.io/2026/11/27/shrinking-ai-models/
> Code and every prediction, in every layout: https://github.com/nishzsche/can-ai-know-when-its-wrong
> Data: a public dataset of 30,000 Taiwanese credit-card customers from 2005 (UCI "Default of Credit Card Clients", Yeh & Lien, 2009; CC BY 4.0). No age, sex, marital status or education in any model's input.

## Slides

Ten square slides (1080 × 1080). One idea per slide, big type, a chart where there's evidence to show: one slide for each of the blog post's five charts, plus the hook, the layouts, the lending decisions, what to do, and the fine print. Charts are drawn by `writeup/build_formats_carousel.py` from the same results as the blog's charts. Each model keeps its colour throughout; the original table is an open dot and the other layouts filled ones.

### 1. Hook

**Headline:** Why AI can write code but struggle to read a table

**Body:** I gave six AI models the same 2,000 credit-card customers as a table, as sentences, as CSV and as JSON. Not one number changed. Their answers did.

**Footer:** 6 AI models. 2,000 real credit-card customers. 6 layouts.

### 2. Six layouts

**Headline:** Same customer, six layouts

**Body:** One month of one customer's account, in four of them. Every value is copied character for character: only the layout changes.

### 3. Ranking

**Headline:** Same numbers, new layout, new ranking

**Body:** Ranking: how often a model rates a customer who defaulted as riskier than one who paid. 0.5 is a coin flip.

### 4. Answers moved

**Headline:** Laya's answers moved 26 points. Sol's, 3.

**Body:** How far each model's answer moved per customer, in percentage points, when the same numbers came in another layout.

### 5. Customers

**Headline:** Same customers, same numbers, different answers

**Body:** Each dot is one customer: the answer to the original table against the answer to JSON.

### 6. Promised

**Headline:** The promised risk moved with the layout

**Body:** Each model's average answer for the same 2,000 customers, against the 22.1% who actually defaulted.

### 7. Lending decisions

**Headline:** Keep the old cut-off, and decisions flip

**Body:** The share of approve-or-decline decisions that change when only the layout does, at the cut-off chosen for the original table.

### 8. Why

**Headline:** A program reads four values. A model reads 31 pieces.

**Body:** One row of the original table. Nothing in the pieces says which number is the bill and which the payment: the model has to work it out from where they sit.

### 9. What I'd tell a bank

**Headline:** What I'd tell a bank

**Body:**
- The layout is part of the model. Version the prompt template with it, and treat any change, even a line of dashes, as a model change.
- Test a few layouts before trusting one. Score the same customers in two layouts and compare each customer's number: it needs no outcomes.
- Choose a layout on data set aside for tuning, never on the test set.
- When a decision rests on numbers, let something read them as numbers: a scorecard, a table model, or code that works out the figures first.

**Footer:** The numbers didn't change. The answers did.

### 10. Fine print

**Headline:** What this can't tell you

**Body:**
- One bank, Taiwan, 2005. Six models, answering cold, while the scorecard learned from 20,000 customers.
- One run per layout. 11 of the 30 ranking changes were bigger than chance; the rest weren't.
- GPT-6.2 Sol knows this dataset. That may be why its answers didn't move.
- I didn't test why the layout matters, or whether trained models are as sensitive.

**Footer:** Change the layout and you've changed the model. Test it like one.
