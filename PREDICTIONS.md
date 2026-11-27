# Predictions (written before any full run)

**Written 2026-11-22, before any M3 run.** The author answered these questions in a Q&A with Claude Code; the answers are recorded verbatim, with no reasons given. The commit that adds them comes before every M3 prediction file.

PLAN.md section 7: which contestant I expect to be most and least honest, and whether telling the base rate will help. Write the reasoning too: the point is to compare it with what actually happens.

## What was already known when this was written

To be fair to readers, this is the information visible before writing:

- **Baselines on test** (from M2):
  - Scorecard: AUC 0.777, ECE 0.014.
  - GBM: AUC 0.789, ECE 0.017.
  - Both promise ~22% and deliver 22%.
- **Smoke tests:** mean predicted risk on the same 20 tune customers (actual default rate 20%). Twenty customers is far too few to judge calibration.

  | Contestant | Out of the box | Told the base rate |
  |---|---|---|
  | Laya | 0.23 | 0.27 |
  | GPT-6.2 Sol | 0.32 | 0.31 |
  | Jev | 0.55 | 0.46 |
  | Kev-4B | 0.55 | 0.30 |
  | Kev-0.8B | 0.36 | 0.52 |
  | Qwen3.5-4B | 0.42 | 0.26 |

- **Memorization:** GPT-6.2 Sol (and Astra) reproduced the first rows of the data file verbatim; Qwen3.5-4B showed no memorization (`data/contamination.md`).

## My guesses

**Most honest AI contestant** (best calibrated out of the box), and why:

> **Jev.**

**Least honest AI contestant**, and why:

> **Laya.**

**Will telling the base rate help?** For which contestants, and by how much:

> **Yes, for most of them.**

**Ranking (AUC):** will any AI contestant beat the 20-line scorecard (AUC 0.777)?

> **Yes, at least one.**

**The "confident and wrong" risk:** which contestant will put the most customers in a confident band that turns out wrong?

> (not given)

**Optional: one number per contestant** (not given). Out-of-the-box ECE guess (lower is better; the baselines get ~0.015):

| Contestant | ECE guess | Base rate helps? (yes/no) |
|---|---|---|
| jev | | |
| kev-0.8b | | |
| kev-4b | | |
| laya | | |
| qwen-logprobs | | |
| frontier-verbal (GPT-6.2 Sol) | | |

---

# M6 predictions: the fair fight (written before any M6 full run)

**Written 2026-11-23, before any M6 full run** (PLAN.md §11). The author answered these questions in a Q&A with Claude Code on the L4; the answers are recorded verbatim, with no reasons given. Only smoke runs (a few training steps, 20 tune customers) had happened. The commit that adds them comes before every M6 prediction file.

## What was already known when this was written

- **The published zero-shot results on test** (`results/summary.csv`):
  - Scorecard AUC 0.776, ECE 0.013; LightGBM AUC 0.788, ECE 0.016.
  - Best zero-shot AI calibration: GPT-6.2 Sol, ECE 0.051 (out of the box). Best zero-shot AI AUC: Kev-4B told the base rate, 0.756.
  - Out-of-the-box ECE: Jev 0.264, Kev-4B 0.266, Kev-0.8B 0.089, Laya 0.052, Qwen 0.140, Sol 0.051.
- **The `neighbours-20` reference** (the 20 nearest `train` customers' default rate): test AUC 0.764, ECE 0.023.
- **Smoke runs:** Jev and Sol with examples on 20 tune customers (actual rate 0.20): mean p 0.40 and 0.27. The trainers ran a few steps each; their losses say nothing about test.

## My guesses

**Most honest AI model after training** (lowest test ECE, raw `trained` output):

> **Qwen3.5-4B.**

**Will any trained AI model beat LightGBM's AUC (0.789)?**

> **Yes, at least one.**

**Will Jev and Sol with 20 examples beat their own neighbours (AUC 0.764, ECE 0.023)?**

> **Both beat them.**

**Least honest AI contestant after training** (highest test ECE, raw `trained` output, Jev and Sol with examples included):

> **Kev-4B.**

---

# Posts 4 and 5 predictions: shrinking the models, and the formats (written before any run)

**Written 2026-11-25, before any post 4 or post 5 run** (PLAN.md §12). The author answered these questions in a Q&A with Claude Code on the laptop. The answers are recorded verbatim, with no reasons given. The commit that adds them comes before every post 4 and post 5 prediction file.

## What was already known when this was written

- **Post 3's published results** for the fine-tuned models at full precision (bf16), on test (`results/fair_fight.csv`):
  - Kev-4B: AUC 0.792, ECE 0.036.
  - Kev-0.8B: AUC 0.790, ECE 0.045.
  - Qwen3.5-4B: AUC 0.780, ECE 0.024.
- **The published layout and disguise checks** (`results/checks/disguise_summary.json`):
  - Placing each bill next to the following month's payment took zero-shot Kev-4B's AUC from 0.594 to 0.720, and Laya's from 0.666 to 0.569.
  - Removing the currency, dates and raw codes moved mean predictions by up to 4.8 points.
- **No post 4 run of any kind,** smoke runs included. The quantization code was tried only on a tiny model with random weights on the laptop, which says nothing about the contestants.
- **Post 5:** token counts per format (`results/checks/formats_tokens.json`). Laya was run on one test customer in each format on the laptop, to check the plumbing. The author wasn't shown those numbers.

## My guesses

### Post 4: shrinking the fine-tuned models (8, 4, 3 and 2 bits)

**At which bit width will calibration error (ECE) first rise clearly above the full-precision score?**

> **Already at 8-bit.**

**Will ranking (AUC) hold at 4-bit, within about 0.01 of each model's full-precision score?**

> **Yes, for all three.**

**Answering one customer at a time on the L4, will 4-bit (NF4) be faster or slower than bf16?**

> **Faster.**

**Which fine-tuned model will hold up best when shrunk (least change in its percentages)?**

> **Qwen3.5-4B.**

### Post 5: the same customers in four formats (zero-shot models)

**Which format will move the zero-shot models' predictions most, compared with today's pipe table?**

> **Prose.**

**Will any format change a model's ranking (AUC) by as much as the layout check did (Kev-4B 0.594 to 0.720)?**

> **No, every change is smaller.**

**Which kind of zero-shot model will be most sensitive to the format?**

> **A small open model** (Kev-0.8B or Laya).

# Post 4 speed addendum predictions (written before any speed run)

**Written 2026-11-26, before any run in PLAN.md §12's "Post 4 speed addendum".** The author answered these in a Q&A with Claude Code on the laptop. The answers are recorded verbatim.

## What was already known when this was written

- **All of post 4's first runs** (`results/quantization.csv`). Those were the untuned setup: every version, full size included, ran two kinds of layers on slow reference code, and the shrunk versions used the libraries' generic paths.
  - **Median time per answer, full size:** 62 ms (Kev-0.8B), 111 ms (Kev-4B), 120 ms (Qwen).
  - **4-bit NF4:** 81, 139 and 152 ms.
  - **4-bit HQQ:** 111, 426 and 286 ms.
  - **8-bit:** 215, 289 and 164 ms.
  - **Honesty:** 8-bit and NF4 stayed within half a point of full size.
- **Post 3's full-size Kev with CUDA graphs on:** 24 ms (Kev-0.8B), 106 ms (Kev-4B).
- **The review panel's reasoning** (STATUS.md), which the author saw: fused kernels or FP8 "could plausibly win", and a ~250-token prompt is "bandwidth-bound in principle" on the L4.
- **No run of the new tags** (`bf16-tuned`, `hqq-4-fast`, `fp8-dynamic`), and no throughput run.

## My guesses

**With fast fused kernels and Kev's shortcut back on, will 4-bit (HQQ) answer one customer faster or slower than a properly tuned full-size model?**

> **Faster.**

**Will FP8 (weights and arithmetic both in 8 bits) answer one customer faster or slower than the tuned full-size model?**

> **Faster.**

**Will FP8 change the models' honesty more or less than bitsandbytes 8-bit did (within half a point of full size)?**

> **Less or the same.**

**Scoring 32 customers at a time, which version will cost least per 1,000 decisions?**

> **4-bit NF4.**

# Post 5 addendum predictions: the table models rearranged, and the trained models in five layouts (written before any run)

**Written 2026-11-27, before any run in PLAN.md §12's "Post 5 addendum".** The author answered these in a Q&A with Claude Code on the laptop. The answers are recorded verbatim.

## What was already known when this was written

- **All of post 5** (`results/layouts.csv`, published 2026-11-27): the six zero-shot AI models in five layouts. Laya moved 26 points per customer (CSV, JSON) and ranked from 0.716 to 0.478; Kev-4B rose from 0.594 to 0.729 with JSON; Kev-0.8B (2.6–5.5 points) and Sol (2.6–2.9) moved least.
- **Post 3's trained models and table models on the original layout** (`results/fair_fight.csv`): Kumo-Tabular and TabICL-v2 at 0.794; fine-tuned Kev-0.8B at 0.790.
- **Post 4:** two full-size runs of the same trained model, on different setups, changed 0.15% to 1.5% of decisions.
- **No run** of any arrangement or of a trained model in a new layout.

## My guesses

**When the same 19 numbers are rearranged (column order, month order, context row order, units), will any arrangement move Kumo-Tabular's or TabICL-v2's answers more than simply changing their random seed does?**

> **No, only seed-level noise.**

**Will post 3's trained AI models be sturdier to layout (answers move less) than the same models were answering cold in post 5?**

> **Much sturdier.**

**Which trained model's answers will move most across the five layouts?**

> **Laya.**
