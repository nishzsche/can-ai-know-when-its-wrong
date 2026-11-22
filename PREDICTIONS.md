# Predictions (written before any full run)

**Status: NOT FILLED IN YET.** The author writes this before any M3 run. When it's done, replace this line with the date it was written. The L4 runbook's gate checks for that.

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

> _TODO_

**Least honest AI contestant**, and why:

> _TODO_

**Will telling the base rate help?** For which contestants, and by how much:

> _TODO_

**Ranking (AUC):** will any AI contestant beat the 20-line scorecard (AUC 0.777)?

> _TODO_

**The "confident and wrong" risk:** which contestant will put the most customers in a confident band that turns out wrong?

> _TODO_

**Optional: one number per contestant.** Out-of-the-box ECE guess (lower is better; the baselines get ~0.015):

| Contestant | ECE guess | Base rate helps? (yes/no) |
|---|---|---|
| jev | | |
| kev-0.8b | | |
| kev-4b | | |
| laya | | |
| qwen-logprobs | | |
| frontier-verbal (GPT-6.2 Sol) | | |
