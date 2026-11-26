# I shrank AI models to save money. They stayed honest, and got slower.

<!-- PENDING (2026-11-26): title, subtitle and the speed section wait for the extra speed runs (STATUS.md, PLAN.md §12 post 4 addendum). The review panel found "stayed honest" overclaims (both Kev models are outside the chance band at full size) and the title is over LinkedIn's 58-character document limit. -->

*Three trained models, squeezed to 8, 4, 3 and 2 bits. Their percentages held up longer than I expected, their speed didn't, and the summary scores hid how far each customer's number moved.*

Save a photo at a lower quality setting and, for a while, you can't tell the difference. Then, past some point, it turns into blocks.

AI models get compressed the same way. A model is billions of numbers, each normally stored in 16 bits, enough for about 65,000 different values. Shrinking a model rounds those numbers more coarsely, like rounding every price in a shop to the nearest ten cents, then the nearest dollar, then the nearest ten dollars. It's called [quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview), and it lets a model fit on a smaller, cheaper GPU, the chip that runs it. The usual check afterwards is whether the model still gets the answers right.

This series checks something else: whether a model's percentages come true. In [my last post](https://nishzsche.github.io/2026/11/25/fair-fight/), I trained four AI models on 20,000 credit-card customers whose outcomes were known, and they ranked customers about as well as the models banks use. This time I shrank three of them, and asked the same 2,000 test customers the same question: how likely is this customer to miss next month's payment? Would the shrunk models still be honest, and would they really be cheaper to run?

| Size | Values each number can take | Memory for the two 4B models' numbers |
|---|---|---|
| Full size, 16 bits | about 65,000 | 8.5 GB |
| 8 bits | 256 | 4.9 GB |
| 4 bits | 16 | 3.3–3.4 GB |
| 3 bits | 8 | 2.9–3.0 GB |
| 2 bits | 4 | 2.4 GB |

Each small block of numbers gets its own scale, so even at 2 bits the four values differ from block to block. The scales take space too, and some parts of each model stay full size, mostly the table that turns words into numbers. That's why 4 bits saves less than three-quarters of the memory.

![Each trained model's ranking (AUC) at each size. Kev-4B, Kev-0.8B and Qwen3.5-4B stay between 0.780 and 0.792 from full size down to 4 bits, with either tool, all within 0.005 of their full-size scores and above the scorecard's 0.777. At 3 bits, Kev-0.8B (0.786) and Qwen (0.779) show no drop the test can detect and Kev-4B slips to 0.776. At 2 bits, Kev-4B falls to 0.60, Kev-0.8B to 0.45 and Qwen to 0.43, the last two worse than a coin flip.](../results/figures/post4_ranking.png)

## Three models, six versions

The three are Kev-4B and Kev-0.8B, two open models built for decisions, and Qwen3.5-4B, a general open model. I left out Laya, the fourth model from last time: at 421 million numbers it already runs on a laptop's processor, so shrinking it doesn't answer a real question.

I used two tools:

- **bitsandbytes,** the most widely used library for this: at 8 bits with its standard method (LLM.int8), and at 4 bits in NF4, the format most guides use.
- **HQQ** at 4, 3 and 2 bits, so that one method goes all the way down, and its 4-bit version shows how much the tool matters at the same size.

Neither tool needs any data to shrink a model, so nothing was tuned on the test customers. Each shrunk model is compared only with its own full-size version, run the same way on the same GPU. And all three models learned from the same 20,000 customers as the scorecard I use as a reference.

## Ranking held to 4 bits, and collapsed at 2

To measure ranking I use the same score as before: pick one customer who defaulted and one who paid, and ask how often the model rates the defaulter as riskier ([AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve); 0.5 is a coin flip).

The chart above tells it:

- **Down to 4 bits, with either tool, nothing moved.** All three models stayed within 0.005 of their full-size scores. Comparing a model with itself on the same customers is precise: the test rules out a drop of more than about 0.004.
- **At 3 bits, Kev-4B slipped,** from 0.792 to 0.776, level with the scorecard (0.777). That was the only drop above 2 bits the test detected, even after allowing for 15 comparisons (three models, five shrunk versions each). Kev-0.8B and Qwen showed no drop the test could detect, though at 3 bits it can't rule out one of up to about 0.015.
- **At 2 bits, all three collapsed.** Kev-4B fell to 0.60, Kev-0.8B to 0.45 and Qwen to 0.43. The last two are worse than a coin flip: they'd do better with their answers reversed.

## Honesty held, with one change bigger than chance

Ranking is half the job. The other half is whether a model's percentages come true. I measure that as before: the average gap, in percentage points, between the risk a model promised and what happened ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)). With 2,000 customers, a perfectly honest model stays below about 2.5 points in 95 samples out of 100. To tell whether shrinking changed a model's honesty, I compared each shrunk version with its full size on the same customers, again allowing for the 15 comparisons.

![Calibration error at each size, in points. At full size Kev-0.8B is 4.4 points off, Kev-4B 3.8 and Qwen3.5-4B 2.4. At 8 bits and 4-bit NF4, each stays within half a point of that. At 4-bit HQQ, Kev-0.8B improves to 2.3, the only change above 2 bits bigger than chance; Kev-4B is 3.1 and Qwen 3.2. At 3 bits, Kev-4B is 4.8, Kev-0.8B 3.1 and Qwen 1.5. At 2 bits all three are off the scale: Kev-0.8B 15, Kev-4B 20 and Qwen 46 points off.](../results/figures/post4_honesty.png)

- **8 bits and 4-bit NF4 changed almost nothing.** Every model stayed within half a point of its full-size honesty.
- **4-bit HQQ made one model more honest, by more than chance.** Kev-0.8B went from 4.4 points off to 2.3. Shrinking nudged its average answer up from 18.1% to 19.9%, toward the 22.1% who actually defaulted (the last post found Kev understates risk). Nobody chose that direction: it's a side effect of rounding. For scale, retraining Kev-0.8B from a different random start in the last post gave an average of 20.0% and 3.0 points off, much the same.
- **It also moved Qwen's average answer, down from 22.1% to 19.2%.** That's measured on the same customers, without outcomes, so it isn't the luck of the sample. Whether it made Qwen less honest can't be told: its error rose from 2.4 to 3.2 points, a change within chance. Kev-4B moved less (3.8 to 3.1).
- **3 bits moved honesty both ways** (Kev-4B 4.8, Kev-0.8B 3.1, Qwen 1.5), none by more than chance.
- **2 bits broke all three:** Kev-0.8B 15, Kev-4B 20 and Qwen 46 points off. Qwen's average answer was 67%, against 22% who defaulted.

One twist at 2 bits. Apply the same correction as in the last post, learned from 2,000 separate customers, and the broken models come out about 1 point off, within chance of perfectly honest. That's because they now give almost everyone a similar answer, near the 22% average. A model that can't tell customers apart can still promise the right average. Honest and useless, which is why this series always measures ranking and honesty separately. (Down to 3 bits, the same correction left every version 1.4 to 3.3 points off, as at full size.)

## The scores held while the customers moved

So far, so reassuring, at least down to 4 bits. But ranking and calibration error are summary scores, and summaries can hide a lot, the way a compressed photo looks fine from across the room. Here is each test customer's answer from Kev-0.8B, full size against shrunk:

![Each dot is one test customer: Kev-0.8B's full-size answer against its shrunk answer, in four panels. At 4-bit NF4 the dots hug the no-change line: the average customer moved 0.9 points and 5 in 1,000 moved 5 points or more. At 4-bit HQQ, 3.0 points, and 8 in 100 moved 5 or more. At 3 bits, 6.6 points, and 62 in 100 moved 5 or more, with low answers pushed up and high answers pulled down, yet ranking 0.786 and 3.1 points off, against 0.790 and 4.4 at full size. At 2 bits the dots form a flat band between about 20% and 55% whatever the full-size answer, 26.6 points on average.](../results/figures/post4_drift.png)

- **At 4-bit NF4, the answers barely moved:** less than a point for the average customer.
- **At 3 bits, 6 in 10 customers' answers moved by 5 points or more.** Yet Kev-0.8B's ranking (0.786) and calibration error (3.1 points off) looked as good as at full size. Its safest customers were pushed up and its riskiest pulled down: the answers were squeezed toward the middle, mostly in the same order, with the average still roughly right.
- **At 2 bits, the answer hardly depends on the customer at all.**

The other two models show the same pattern, less strongly: at 3 bits, 31 in 100 of Kev-4B's customers and 26 in 100 of Qwen's moved 5 points or more.

A dashboard showing only ranking and average calibration error would have passed the 3-bit Kev-0.8B. A standard bank check would not. Its calibration slope, which is 1 when the spread of a model's answers is right, went from 0.86 to 1.31: the mark of answers squeezed toward the middle. And each customer's number feeds a decision.

## What it does to a lending decision

To see what that means for a lender, I used the setup from [post 2](https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/). Each paying customer earns 1 unit and each default costs 4, and the lender approves everyone whose risk is below a cut-off. Suppose the lender swaps in the shrunk model and keeps the cut-off it chose for the full-size one, as an infrastructure team might when it ships a smaller model to save money:

| Version | Approve/decline decisions that differ from full size, at the full-size cut-off |
|---|---|
| The same full-size model, run again through the new code | 0.2% (both Kev models), 1.4% (Qwen) |
| 8 bits | 0.9–3.3% |
| 4 bits, NF4 | 1.5–3.4% |
| 4 bits, HQQ | 2.6–7.1% |
| 3 bits | 10–20% |
| 2 bits | 62–72% |

Even 8 bits changed more decisions than running the same model twice. At 3 bits, the lending policy itself shifted without anyone deciding it: approvals fell from 62 in 100 to 43 for Kev-4B and from 72 to 59 for Kev-0.8B, while Qwen's rose from 63 to 68. Kev-4B's lender would also have earned less, 253 per 1,000 customers instead of 294, though profit gaps this size are hard to pin down with 2,000 customers. Swaps can go the other way too: Kev-4B at 4-bit HQQ earned 308 against 294.

If the lender instead re-chooses each version's cut-off on customers whose outcomes are known, every version down to 3 bits earns about the same: 282 to 304 per 1,000, against 287 to 300 at full size. At 2 bits, little or nothing: −10 to 73.

## Smaller, but slower

<!-- PENDING (2026-11-26): this section is rewritten after the extra speed runs. The review panel found the setup untuned: every version, full size included, ran two kinds of layers (causal_conv1d, chunk_gated_delta_rule) on slow reference code, the shrunk versions used no fused low-bit kernels, and post 3's Kev-4B gained only 5% from CUDA graphs (106 vs 111 ms), so the earlier overhead explanation was wrong and has been removed. The numbers below are from that setup. -->

Now the reason to shrink a model at all: cost.

![GPU memory at peak and median time per answer at each size. Memory falls with every step: the 4B models go from 8.7 to 8.9 GB at full size to 3.5 to 3.9 GB at 4 bits and 2.6 to 2.9 GB at 2 bits, and Kev-0.8B from 1.7 GB to under 1 GB at 4 bits. Time per answer rises at every shrunk size: at full size, 62 ms for Kev-0.8B, 111 ms for Kev-4B and 120 ms for Qwen; at 4-bit NF4, 81, 139 and 152 ms; at 8 bits up to 289 ms; at HQQ 3 bits up to 467 ms for Kev-4B.](../results/figures/post4_cost.png)

- **Memory fell as promised.** At 4 bits the two 4B models needed 3.5 to 3.9 GB at peak instead of 8.7 to 8.9, less than half.
- **On this setup, every shrunk version was slower.**
  - 4-bit NF4 was a quarter to a third slower: Kev-4B went from 111 to 139 milliseconds per answer, Qwen from 120 to 152 and Kev-0.8B from 62 to 81.
  - 8 bits took 1.4 to 3.5 times as long.
  - HQQ took up to about 4 times as long: 426 milliseconds for Kev-4B at 4 bits, 467 at 3.

A rented GPU bills by the minute, so per decision, slower means more expensive. Kev-4B cost 1.3 cents per 1,000 decisions at full size, and 5.2 cents at 4-bit HQQ.

Kev's own server also has a shortcut, called CUDA graphs, that records the GPU's sequence of steps once and replays it for every request. It offers it only for full-size models. With it on, the last post's full-size Kev-0.8B answered in 24 milliseconds instead of 62. Kev-4B gained much less: 106 instead of 111. (To compare sizes fairly, I measured every version here with the shortcut off.)

## I got most of my predictions wrong, again

Before any of these runs, I [wrote down](https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/PREDICTIONS.md) what I expected:

| My guess | What happened |
|---|---|
| Honesty would get clearly worse already at 8 bits | ✗ At 8 bits every model stayed within half a point of full size. Above 2 bits, the only honesty change bigger than chance was an improvement. The first clear break came at 2 bits |
| Ranking would hold at 4 bits for all three | ✓ Within 0.005, with either tool |
| 4-bit NF4 would be faster than full size | ✗ A quarter to a third slower, for all three, on the setup tested <!-- PENDING: revisit after the speed runs --> |
| Qwen would hold up best, its percentages changing least | ✗ Not quite. Kev-4B changed least at 8 and 4 bits (0.3 to 1.1 points per customer). Qwen changed least at 3 bits (4.0 points), then broke hardest at 2 (49) |

## What I'd tell a bank

- **Treat shrinking as a material model change, not an IT change.** Agree tolerances first: how many decisions may change, how far any one customer may move, how far the calibration slope and the spread of scores may shift. Most of this needs no outcomes: score the same customers with both versions on day one. Then revalidate, and re-set the cut-off and any calibration through the usual approval process.
- **8 bits and the common 4-bit format passed here.** The average customer moved less than a point, though a few moved up to 9. That tells you where to start testing, not whether to skip it.
- **The tool matters as much as the size, at least here.** At the same 4 bits, HQQ moved the answers 1.5 to 3.2 times as much as NF4. One likely reason: NF4 spaces its 16 levels to match the bell-shaped spread of a model's numbers, while HQQ spaces them evenly.
- **Record the whole setup with every score.** Each prediction here carries its size and library versions. Record the GPU, the kernels and the drivers too, and treat a library upgrade as a model change: the same full-size Qwen, run a second way, changed 1.4% of decisions.

<!-- PENDING: a speed and cost bullet, after the speed runs. -->

## What this test can't tell you

- **Three models, one dataset, one way of serving:** a bank in Taiwan in 2005, one NVIDIA L4 GPU, one customer at a time. <!-- PENDING: speed limitations after the speed runs. -->
- **Only methods that need no data.** Methods that tune their rounding on sample data, such as GPTQ and AWQ, often do better at 3 and 4 bits. I didn't test them.
- **One training run per model.** Retraining alone moved Kev-0.8B's honesty by 1.2 to 1.5 points in the last post, as much as most of the shrinking did here.
- **Shrunk after training.** Training a model that's already shrunk, or keeping the training update separate and full size, could behave differently.
- **A use its maker doesn't support.** Kev's maker doesn't offer shrinking, so I wrapped its loader myself. Running a vendor's model in a way it doesn't support is a risk of its own.
- **Fair lending isn't tested.** At 3 bits, 10 to 20 in 100 decisions changed. I haven't checked whether those customers differ by group.
- **Small gaps between models are hard to see.** With 2,000 test customers, ranking gaps under about 0.02 between different models can't be told apart; a model compared with itself is measured far more precisely, to about 0.004. These test customers have also now been used in four posts. 6,000 customers that no model has seen are kept aside for one final replication.
- **"Default" is the dataset's label:** missing next month's payment, not the 90-days-overdue definition banks use for capital.

## How I ran it

Each model was merged with its training update first and then shrunk, so the result is one model, as you'd deploy it. The part of each model that turns its reading into a probability stays full size. Both tools' versions are pinned and recorded in every prediction. Before any full run of a shrunk model, the full-size version had to reproduce the last post's predictions through the new code, and it did: on 50 check customers, within 0.002 on average. Scoring took 72,000 predictions and about 4.3 hours of GPU time, about $1.90. The measures were fixed before the runs, except three I added later: the lending-decision counts and the share of customers moving 5 points or more, while writing, and the paired test of honesty changes, after review.

## What's next

Every number, the code and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/), [why 99% accuracy can still be a disaster](https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/) and [giving AI the same data as a bank's model](https://nishzsche.github.io/2026/11/25/fair-fight/).

Next: why AI can write code but struggle to read a table. I gave six AI models the same customers as a table, as sentences, as CSV and as JSON. For one of them, a different layout moved its answers by 26 points per customer on average, about four times what shrinking to 3 bits did to any model here. And the sequel I promised in the first post, teaching a small model to say "I don't know", is still coming.
