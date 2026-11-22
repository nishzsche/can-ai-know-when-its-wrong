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
