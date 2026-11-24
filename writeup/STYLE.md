# Write-up style: curiosity first, expertise second

How the blog, carousel and X post for this experiment are written. This follows the author's Notion page "Content Strategy: Curiosity First, Expertise Second" (2026-11-19), where this post is topic #1, "Can AI actually know when it's wrong?". Re-read that page if it has changed.

**The rule:** make people curious first, and let them trust the expertise second. Expertise should show through the quality of the question, the experiment and the conclusions, not through how complicated the writing sounds.

**The flow of every piece:** everyday curiosity → real experiment → technical insight.

## Three readers, one post

| Reader | What they should get from it |
|---|---|
| General audience | A question worth thinking about: when an AI says "70% sure", should you believe it? |
| Tech audience | Surprising findings: most AI models overstate risk about 2×, one sentence of context fixes most of it, the most "honest" model has seen this exact data, and a 20-line logistic regression still ranks customers best |
| Hiring managers | Judgement, not just results: a fair setup (same inputs, same test set, no demographic fields), checks on my own results (memorization, recognition, disguise, false base rate), a library bug found and reported upstream |

## The five principles, applied to this post

1. **Lead with curiosity, not jargon.**
   - No acronym in the title, the first line or the first three paragraphs.
   - Introduce each idea before its name: "the average gap between what it promised and what happened" first, then "(calibration error, ECE)".
   - The opening hook is the weather forecast, the bank audit, or the "70% sure" question, never "calibration of decision models".
2. **Show, don't tell.**
   - Use real numbers in plain units: "Jev said 48% of these customers would default; 22% did."
   - Every section earns its place with a chart, a number or a surprise.
   - The "promised vs actual" chart goes near the top.
3. **Simplify without dumbing down.**
   - Keep the technical content exact: ECE with 10 bins, AUC, bootstrap intervals, recalibration fitted on tune only.
   - Explain each in one plain sentence with an everyday analogy.
   - Put the full method at the end and on GitHub, not in the story.
4. **Have a point of view.**
   - Say what a bank, or anyone deploying AI, should do with these results.
   - Show where my own predictions were wrong (from `PREDICTIONS.md`, written before any run). That's honest, and it's the post's own theme.
5. **Build a story, not an isolated post.**
   - This post is part of the decision-models series.
   - End by pointing at the next question, "Why 99% accuracy can still be a disaster", and at the sequel, training my own decision model.

## Voice and language

- **Person:** first person, curious and direct ("I gave eight models the same 2,000 customers").
- **Language:** short sentences and plain words. Each number gets a unit or a comparison.
- **Tone:** no hype and no dunking on vendors. Name the models and report what they did.
- **Fairness, wherever results are compared:** the scorecard and the GBM learned from 20,000 labelled customers; the AI models answered cold.
- **Sol's caveat, wherever Sol's honesty is mentioned:** it recognizes this dataset and knows its default rate.
- **Confidentiality:** a public dataset only. My credit-risk background is described as patterns, never a past employer's internals or numbers.

## Charts (every post in the series)

No study gives a single best number of charts. This budget comes from what the author asked for in post 3: 2 charts in 2,400 words was too few, and 6 in 2,700 was right.

- **How many:** one chart for each finding the reader is asked to believe, which works out to about one every 400–500 words.
  - A 1,500–2,000-word post has 4–5 charts.
  - A 2,500–3,000-word post has 5–7.
  - Fewer than 3 is too few for a data post. More than 7 means the post is doing too much: move the extra charts to the deep dive.
- **Where:**
  - The first chart comes within the first 400 words. It can carry the hook.
  - No stretch of more than about 600 words goes without a chart.
  - Tables don't count toward either rule. A table gives the exact numbers; a chart gives the pattern.
- **Each chart earns its place:**
  - It makes one point, and its title states that point as a sentence ("Paying more didn't buy better ranking", not "Cost vs AUC").
  - It shows something the text and tables around it can't.
  - No two charts make the same point.
  - A chart may repeat a table's numbers only if it adds the shape, as the catch-up chart does in post 3.
- **How it's built:**
  - Draw it with the project's figure code and shared palette (`analysis/figures.py`), in a light version and a `-dark.png` version.
  - Make it readable on a phone.
  - Use the same encoding throughout a post: an open dot for before, a filled dot for after, each model in its own colour.
  - Write plain-language alt text that gives the key numbers.
  - A chart's statistics must match the text's wording. For example, if the text corrects for multiple comparisons, so do the chart's intervals.

## LinkedIn carousel (every post in the series)

Every blog post gets a LinkedIn carousel: a PDF uploaded as a document post, as posts 1 and 2 have. Don't use a single image or several images instead. A LinkedIn post holds either images or a document, and its media can't be changed after it's published, so the carousel has to be ready before the post goes up.

- **How many slides:** 8–10.
  - At least one slide for each blog chart that shows a finding.
  - Plus a hook, a slide on what to do, and a closing slide on fairness and what's next. Post 1 has 10 slides and post 2 has 9.
- **Order:**
  - The hook: the title and the one-line surprise.
  - The setup, or the everyday example.
  - One finding per slide, each with its chart.
  - What to do about it.
  - To be fair to the AI, and what's next.
- **Draw the slides; don't paste the blog charts:**
  - Draw each chart slide at phone size: 1080 × 1080, headlines 27–30 pt, body text 15–20 pt, chart labels at least 12 pt.
  - Use the same results files as the blog. Never paste the blog's PNGs: on a phone their text shrinks to about 6 pt.
  - Use the same colours and encoding as the blog's charts.
  - One idea per slide, with the headline stating the point.
  - Chart slides carry the fairness note (who learned from what), and every slide has the series name in its footer.
- **Where the text lives:**
  - The slide text goes in the post's LinkedIn file under `## Slides`, as a headline, body and footer for each slide.
  - A builder script renders it: `writeup/build_carousel.py` for post 1 and `writeup/build_accuracy_carousel.py` for post 2. New builders reuse the first one's helpers.
- **Check:**
  - Render with `--png` and look at every slide for overlapping or cut-off text.
  - The document title can be at most 58 characters, LinkedIn's limit.
- **The post itself:**
  - The post text tells the story. Document pages can't take alt text, so the text is what screen-reader users get.
  - Links go in the first comment.
- **It ships with the blog post:** a post isn't finished until its carousel is built.

## The blog (`writeup/blog.md`, the canonical piece)

**Audience:** all three readers. About 1,800 words, plus a methods appendix.

1. **Hook (everyday curiosity).** A weather app that says 70% should be right on about 7 of 10 such days. Banks are held to the same standard for the probability of default behind every credit decision. AI models now give confidence percentages too. Are they honest?
2. **The experiment.** 2,000 real credit-card customers from Taiwan, 2005. Eight contestants answer one question: "will this customer miss next month's payment?"
   - **Contestants:** six AI models (a commercial decision model, three open ones, a small general LLM, a frontier chatbot) and two classic models trained the old way.
   - **Same everything:** the same inputs, with no age, sex, marriage or education fields.
3. **What "honest" means.** The "promised vs actual" chart (figure 1). It's one picture, explained once.
4. **Findings, as story beats.**
   - **Most AI models overstate the risk about 2× out of the box.** Jev says 48% and Kev-4B 49%, against 22% actual. Use figure 4: Jev puts 70% of customers in its "40%+" band, where only 26% default.
   - **One sentence of context fixes most of it.** Saying "about 22% of customers default" helps five of six models, and Kev-4B's calibration error drops from 0.27 to 0.03. It backfires for Kev-0.8B (0.09 → 0.27). Use figure 2, the honesty arc.
   - **The most "honest" model has seen the answers.** Sol looks honest out of the box, but it names this dataset from any summary and knows its default rate to two decimals (22.12%), even with the currency and dates removed.
     - **The clincher:** tell it a false rate (10%) and it follows about 70% of the way. So it does use a stated rate, and it already "knew" the real one.
     - **Wording:** "most likely from knowing this dataset", not "proven".
   - **The 20-line model still wins on ranking.** No AI model beats the scorecard's AUC (0.777). The best are 0.757 and 0.754. Use figure 3.
   - **Honesty can be bought back, with labels.** Recalibrating on 2,000 labelled customers brings every AI model's error down to between 0.003 and 0.041. But labelled data is exactly what zero-shot AI was meant to save you.
   - **Cost.** Use figure 5: Sol costs about $0.74 per 1,000 decisions; the open models cost under $0.02 on one GPU.
5. **My predictions vs reality.** Use the table from `PREDICTIONS.md`. Most honest Jev: wrong. Least honest Laya: wrong. The base rate helps most: right. An AI model beats the scorecard: wrong.
6. **What a bank should take from it (point of view).** Check calibration, not just accuracy. Never trust a zero-shot percentage without a recalibration set. Give models the base rate. Test on data the model can't have seen.
7. **Limitations, in plain words.** The data is 2005 from one bank, and the dataset is famous and memorized. Demographics are excluded on purpose. The repayment codes are undocumented. The AI models are zero-shot while the baselines were trained. Vendor models change over time.
8. **What's next.** "Why 99% accuracy can still be a disaster", and training my own decision model.
9. **Appendix: method and reproduce.** The template, splits, metrics, recalibration, every check, costs, and the commands. Link to the repo.

## Post 1's carousel: the original plan (`writeup/carousel.md` has the 10 slides as built)

- **Hook:** a strong first line in the post text: "Banks get audited on whether their risk scores are honest. I held AI to the same standard."
- **Slides:** planned as 6–8 PDF slides, one idea per slide, a chart on most.
  1. Hook.
  2. Setup.
  3. What honest looks like.
  4. Promised vs actual.
  5. The honesty arc.
  6. The memorization twist.
  7. The scorecard punchline.
  8. What's next.
- **Links:** in the first comment, not the post.

## X (`writeup/x_post.md`)

- **Format:** a long-form post, not a thread. The "promised vs actual" chart goes first, then the findings in short lines.
- **Links:** none in the main post. The method and repo go in a reply.
- **Register:** X suits the experimental side: the checks, the jevper bug, and the numbers.

## Before publishing

- [ ] Can someone outside tech follow the first three paragraphs without looking anything up?
- [ ] Does the post meet the chart budget: about one chart per 400–500 words, the first within 400 words, and no 600-word stretch without one?
- [ ] Is its LinkedIn carousel built: 8–10 slides, at least one per chart finding, every slide checked as a PNG, and a document title of 58 characters or fewer?
- [ ] Does every comparison say the baselines were trained and the AI models weren't?
- [ ] Is Sol's recognition caveat next to every mention of Sol's honesty?
- [ ] Is every number traceable to `results/summary.csv` or `data/contamination.md`?
- [ ] Is there nothing from a past employer, and no data beyond the public dataset?
- [ ] Does it end on what's next?
