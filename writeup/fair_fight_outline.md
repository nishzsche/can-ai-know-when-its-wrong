# Outline: the fair fight (post 3; PLAN.md §11 and §12, Notion topic #3)

Status: outline for the author's review (2026-11-24). Every number below is from `results/fair_fight.csv` (test set, raw `trained` output unless marked calibrated); the AUC comparisons are paired bootstraps, Holm-corrected.

## Title options (question first, no acronym; per `writeup/STYLE.md`)

1. **I gave AI the same data as a 20-line model. It tied.** (recommended: the result is the hook)
2. What happens when AI gets the same 20,000 customers?
3. Train it and AI catches up. Show it examples and it doesn't.

Subtitle: *Post 1's biggest objection, answered: every AI model gets the 20,000 labelled customers the old models learned from.*

## The hook (everyday curiosity → the objection)

- Post 1 ended with a caveat reviewers pushed on: the AI models answered cold, while the scorecard and gradient boosting studied 20,000 labelled customers. A reviewer put it bluntly: "models that are not trained on task-specific data have no fair comparison."
- So: give every AI model the same 20,000 customers, the same test, and see who wins.

## The setup (one table, three ways to learn)

| How a model can use the 20,000 | Models | Why this way |
|---|---|---|
| Fine-tune on them (one epoch, the maker's own defaults) | Kev-4B, Kev-0.8B, Qwen3.5-4B, Laya | Open weights, so they can be trained |
| Read all 20,000 as examples, no training | Kumo-Tabular (NVIDIA), TabICLv2 | AI built for tables: it learns from the rows in one pass |
| See the 20 most similar customers and their outcomes in the prompt | Jev, GPT-6.2 Sol | Closed: TypeSafe and OpenAI don't let you train them on your data |
| Reference: just average those 20 neighbours' outcomes | "20 neighbours" | Shows whether Jev and Sol add anything beyond counting |

**Fairness line, every time results are compared:** same 20,000 `train` customers, same 2,000 `test` customers, no settings tuned on test, maker defaults with no hyperparameter search (the baselines got about the same), calibration fitted on `tune` only.

## Story beats

1. **Train it, and it catches up.** Six AI models land in a tie with gradient boosting: AUC 0.778–0.794 against 0.789, none statistically different. Biggest turnaround: Kev-4B, 0.594 → 0.792. The model that was least honest and close to a coin toss in post 1 now ranks with the best.
   - *Figure (new): the catch-up chart.* Per model, zero-shot AUC → trained AUC, with the scorecard and gradient boosting as reference lines.
2. **Training fixes honesty too, mostly.** Laya (0.017), Kumo (0.020) and Qwen (0.024) are within what a perfectly honest model scores by chance here (about 0.025). Both Kev models under-state risk (18–19% against 22%), Kev-4B even after calibration (0.033). That's an honest wrinkle; say so.
   - *Figure: `results/figures/m6_reliability.png`* (eight panels, promised vs actual).
3. **The AI built for tables wins without training.** Kumo-Tabular and TabICLv2 top the table (0.794), honest out of the box, for almost nothing: TabICLv2's 6,000 predictions cost about a tenth of a cent of GPU time.
4. **The closed models can only look at examples, and that isn't enough.** Sol (0.762) and Jev (0.743) don't beat simply averaging their 20 neighbours (0.764). Sol's run cost about $41; the fine-tuned models cost cents.
5. **A ceiling, not a winner.** Every model that truly learned sits within 0.02 AUC of the others, and the default rate among the 80% each rates safest is 12.9–13.6% for all of them. With these 19 fields, the data sets the limit, not the model. (Word it as "suggests".)
6. **My predictions vs reality** (`PREDICTIONS.md`, M6 section, written before the runs): most honest after training, guessed Qwen, actually Laya (all three leaders within chance of each other); any AI beats gradient boosting on ranking, yes on the numbers, a tie in the statistics; Jev and Sol beat their neighbours, wrong; least honest, guessed Kev-4B, actually Jev with examples.

## What I'd tell a bank (point of view)

- If you have labelled history, the model choice matters less than you'd think: old and new land in the same place.
- Check honesty again after training: Kev learned to rank but not to state the right rate.
- Closed models can't learn from your data except through the prompt, and a prompt full of look-alikes adds nothing over counting them.
- Try the AI built for tables before fine-tuning a language model: same ranking, honest, no training, nearly free.

## What this test can't tell you

- One dataset again (Taiwan, 2005, 22% default).
- One epoch with maker defaults: more tuning might move a model, but the baselines didn't get any either.
- Sol still recognizes this dataset; the tabular models' pretraining data isn't fully documented.
- The neighbour choice for Jev and Sol is itself a design decision (19 fields, Euclidean); a different one could help or hurt.
- Fine-tuning used the same layout as post 1 (template v2); training should make that matter less than it did zero-shot, but it wasn't re-tested.

## What's next

- Post 4: does shrinking a model make it less honest? The trained models here, at 8-bit and 4-bit.
- The promised sequel: teaching a small model to say "I don't know".

## Figures and tables to make

1. The catch-up chart (new): zero-shot → trained AUC per model, with the baselines; a second chart (not a second axis) for ECE.
2. `m6_reliability.png` (done, light and dark).
3. Cost per 1,000 decisions vs AUC for the trained models (table or small chart).
4. The setup table and a results table (AUC, ECE raw and calibrated, cost).

## Deep dive additions (repo only)

The M6 method: training settings and logs (`results/checks/m6_logs/`), the neighbour retrieval, the Qwen readout check, the Kev under-prediction diagnosis, the weights release (`m6-weights`), and the full significance table.
