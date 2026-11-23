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
