# Status and handoff

Last updated 2026-11-27. Read this together with `PLAN.md` and `CLAUDE.md` before starting work.

## L4 session L done (2026-11-27 20:31 → about 23:00 IST): post 5's addendum

L4_RUNBOOK.md §9 on a fresh L4. Gate passed: the author's addendum guesses are in `PREDICTIONS.md` (8d3ec08) before any run. No speed libraries were installed: session A's plain setup, as §9 says.

**Outcome:**
- **Ran:** everything. Part A is 76 runs (`results/checks/arrangements/`). Part B is 4 models × 6 runs (`results/checks/layouts_trained/`).
- **Failures:** none. `runners/manifest.py --addendum` is complete (`results/run_manifest_addendum.json`).
- **Reproduction:**
  - **Part A:** the in-session `original` matches M6's `trained` answers exactly for both table models (0.0 on all 4,000 tune and test answers).
  - **Part B:** reproduction checks passed on 50 tune customers. Mean |diff|: Kev-0.8B 0.0009, Kev-4B 0.0010, Qwen 0.0020 (max 0.031). They first overwrote post 4's records in `results/checks/quantization/`, which post 4's scorer reads. The laptop restored post 4's and moved this session's to `results/checks/addendum_repro/`.
- **Laya's smoke run:** a no-op. The laptop's committed CPU smoke records were already in `results/smoke/`, and they weren't overwritten. Its meta file logs the attempt. The full GPU run checked loading, 0/1 answers and truncation itself.
- **Logs:** `results/checks/addendum_logs/`.

**First read, Part A** (`results/arrangements.csv`; test, against the in-session `original`):
- **Defaults (the headline):**
  - No arrangement moves either model beyond its own seed noise. Kumo's mean |change| per customer is at most 0.36 points (seeds 0.41); TabICL's is at most 0.54 (seeds 0.63).
  - AUC and ECE are unchanged (all Holm p ≈ 1).
  - Row order and units barely register (≤ 0.09 points).
- **One estimator, shuffle off (diagnostic):**
  - Kumo stays within its seed noise (≤ 0.96 vs 1.14).
  - TabICL with one estimator gives identical answers at every seed, so its noise floor is 0. Every arrangement is therefore trivially "beyond seed noise".
  - TabICL's real moves are small: 1.1–1.4 points per customer for column orders, at most 4.3% of customers moved 5+ points, AUC 0.795 → 0.791–0.794. The single Holm p < 0.05 is `rows-3` AUC (0.028, a change of < 0.001).
  - The write-up should say why the noise floor is 0, not just "yes".

**First read, Part B** (`results/layouts_trained.csv`; test, against each model's post 4 `-bf16` run or M6's `laya-ft`):
- **Kev and Qwen:** trained, they move 0.8–4.4 points per customer. Cold, they moved 2.6–15.1. Kev-4B with JSON is 2.0 trained against 15.1 cold.
  - The only AUC change beyond chance (Holm over 20) is Kev-4B with prose: 0.792 → 0.784.
  - Same-session re-run noise is about 0.1 points for Kev and 0.0 for Qwen and Laya (deterministic).
- **Laya:** training didn't make it sturdier.
  - JSON moves 26.4 points per customer (cold 25.8). AUC 0.778 → 0.591, ECE 0.017 → 0.191, rank correlation 0.09, 65% of decisions changed.
  - Prose: AUC −0.034, ECE 0.104. Markdown: AUC −0.012 (Holm p 0.034).
- **Totals:** 4 of 20 AUC changes and 2 of 20 ECE changes are beyond chance (Holm).
- **The author's guesses:**
  - Q1 (no arrangement moves the table models beyond seed noise): ✓ at defaults.
  - Q2 (training makes the AI models much sturdier): ✓ for Kev and Qwen, ✗ for Laya.
  - Q3 (Laya moves most): ✓.

**For the laptop (fixed):** `results/arrangements.json`'s `holm_family` field said 56. The correction itself is per setting (28 each, as specified); `analysis/layouts.write` now takes the family size, and the label says 28.

**Laptop review (2026-11-27, after session L):**
- **Reproduces:** `analysis/arrangements.py`, `analysis/layouts.py --trained` and `runners/manifest.py --addendum`, re-run on the laptop from the raw records, match the committed outputs exactly (only timestamps differ).
- **Laya's JSON isn't cut short:** its JSON summaries are 285–320 of Laya's tokens (median 304; 4 of 2,000 at exactly 320), within its 320-token budget, as PLAN.md's token report found. They're much longer than v2's (at most 183), the layout Laya was trained on.
- **Part B needs more care than "much sturdier":**
  - **Per customer, yes for Kev-4B and Qwen:** 0.8–3.7 and 1.0–2.4 points trained, against 2.8–15.1 and 4.7–9.2 cold, on every layout. Kev-0.8B is sturdier on four layouts, but not the realigned table (4.4 trained, 3.8 cold).
  - **Decisions, not always:** at each model's own tune cut-off, trained Kev and Qwen still flip 1.8–9.6% of decisions. That's more than cold on several layouts: Kev-0.8B on four of five (JSON 7.4% against 0.7%), Qwen with JSON (9.4% against 7.8%), and Kev-4B with prose (9.6% against 9.5%). The trained models' cut-offs are lower (Kev-0.8B 16%, Qwen 20%) with many customers near them, and some layouts shift the answers one way: Qwen with JSON −1.8 points on average (approval 63% → 71%), Kev-0.8B with JSON +1.6 and realigned +1.0.
  - **Laya is mixed, not "no sturdier":** sturdier with CSV (25.7 → 1.3 points per customer), Markdown (6.4 → 4.8) and the realigned table (12.8 → 5.6). Not with prose (10.1 → 11.7) or JSON (25.8 → 26.4). With JSON its average answer jumps from 21% to 41%, and its order of customers is nearly scrambled (rank correlation with v2 0.09).
  - **Noise:** every layout moves the trained Kev models at least 10 times as much as their same-session re-run (0.08–0.10 points); Qwen and Laya re-run identically (0).
- **Part A as session L read it.** At the defaults, column orders move Kumo 0.30–0.36 points and TabICL 0.43–0.54, inside each model's seed noise (0.41, 0.63); rows and units move them at most 0.09. With one estimator, TabICL is deterministic (seed noise 0), so report its sizes (1.1–1.4 points for column orders, AUC within 0.004) rather than "beyond seed noise".
- **Against the cold AI models** (post 5: 2.6–25.8 points per customer), the table models at their defaults move far less: 0.3–0.5. Post 5's line that they "don't have this problem by design" holds; no update note needed.
- **Guesses, revised:** Q1 ✓ (defaults). Q2 ✓ per customer for Kev-4B and Qwen, mostly for Kev-0.8B; ✗ for decisions, and mixed for Laya. Q3 ✓.
- **Still to decide (the author):** where it goes, a short follow-up post or a dated section in post 5.

**GPU time:** the box was up from about 15:00 UTC to about 17:30 UTC, about 2.5 h, about $1.10 at $0.44/h. The JarvisLabs bill is the authority.

**Box note:** the repo's `gh` install and login (§6.0 block 1) hadn't been done. `gh` 2.105.0 was installed mid-session, and the author logged in. Part A's commit waited about an hour to be pushed.

## Post 5: published (2026-11-27 16:55 IST); LinkedIn post and carousel drafted

**Draft:** `writeup/formats_post.md`, "Why AI can write code but struggle to read a table" (the title post 4 promised), about 2,700 words, five charts. No runs were needed: all six zero-shot models already had all four formats (session A and the laptop, below), and the realigned table comes from post 1's layout check.

**New code (laptop, CPU):**
- `analysis/layouts.py` scores the four formats plus the realigned table against v2, per model, on test, and writes `results/layouts.csv` and `.json`. It takes about 4 minutes, mostly bootstraps.
  - **Measures added while writing (the draft says so):** paired AUC and ECE tests (Holm over 30: six models × five layouts), the share of customers moving 5+ points, and decisions changed at v2's tune cut-off (post 2's lender, 4:1).
  - **Noise:** v2 against the disguise check's 200-customer control.
  - `formats_summary.json` is unchanged.
- `analysis/layouts_figures.py` draws five charts, light and dark: `post5_ranking`, `post5_moved`, `post5_customers`, `post5_promised` and `post5_pieces`. The last one tokenizes one row with Kev's and Laya's tokenizers.
- `tests/test_layouts.py`. The full suite passes (217).

**First read (test, zero-shot, against v2):**
- **Laya:** AUC ranges from 0.716 (sentences) to 0.478 (JSON). Answers moved 26 points per customer with CSV and JSON; rank correlation for JSON is 0.08.
- **Kev-4B:** 0.594 → 0.729 with JSON.
- **Kev-0.8B and Sol:** no ranking change beyond chance. Sol's per-customer move (2.6–2.9 points) is about its own re-run noise (2.5). Sol's caveat applies.
- **Totals:** 11 of 30 AUC changes and 17 of 30 ECE changes are beyond chance (Holm).
- **Decisions at v2's cut-off:** up to 63% change (Laya with JSON, which approves nobody). Kev-0.8B with CSV changes 25% although its ranking holds.
- **The author's guesses (`PREDICTIONS.md`):** prose moves most ✗ (JSON did); no ranking change as big as the layout check ✗; a small open model most sensitive: half right (Laya).

**Draft brought into the blog's Experiment format** (the blog's WRITING.md, design phase 4): a link to the methods page beside the ranking measure; the shared "default" caveat replaced by a pointer to the methods page; the promise of post 6 removed (the blog links a next part only once it's published); and a bold last line, "Change the layout and you've changed the model. Test it like one."

**Review copy:** https://nishzsche.github.io/review/reading-tables/ (unlisted, noindex), once blog PR #61 is merged. It's `review/reading-tables.md` on its own branch, `review-post5`, from `main`, so it doesn't touch the blog session's uncommitted design work in the `dev` working copy. The charts are in `public/images/reading-tables/`, embedded with `chart.html`, and their sizes are in `_data/charts.yml`. It was built locally with Jekyll and checked: five charts, both tables, and it isn't in the home page, feed or sitemap. It's generated from the draft (headline removed, links made relative, a one-sentence alt plus "Chart in words" per chart), so draft edits need regenerating into it. The review note asks about the title, the moved-but-ranking-held section, the bank advice, and where readers got lost.

**Published:** https://nishzsche.github.io/2026/11/27/reading-tables/ (blog PR #63, merged 2026-11-27; `_posts/2026-11-27-reading-tables.md`, dated 16:55 IST so Pages built it on merge). The author asked to publish before the review came back, so the title stayed as promised.
- The review URL redirects to the post. The social card is `results/figures/post5_social_card.png` (`layouts_figures.py social_card`), published as `public/images/reading-tables/social-card.png`.
- **Drawing:** #10 Mystical (telescope and constellations, "the same stars, joined up into different pictures"), my pick, flagged in the PR for the author to swap.
- **Methods page:** the shared caveat "changing the wording moved results by up to 13 points" now adds that layout alone moved one model's ranking from 0.716 to 0.478, linking post 5.
- **Rounding:** two cells of the post's decisions table are exact halves shown rounded down: Qwen with Markdown, 197 of 2,000, shows as 9.8%, and Sol with the realigned table, 87 of 2,000, as 4.3%. Post 4 rounded halves up. It was left as published (no correction note for 0.05 points), and the carousel rounds the same way as the post.

**LinkedIn draft (2026-11-27):**
- `writeup/formats_linkedin.md`: post text of 2,238 characters, the first comment (with the published URL) and 10 slides.
- `writeup/build_formats_carousel.py` builds `writeup/formats_carousel.pdf`: hook, the layouts, ranking, answers moved, customers (Laya and Sol with JSON), promised risk, the decisions grid, the pieces, the bank and the limits. Every slide was checked as a PNG.
- Following the blog's new rule, it doesn't promise post 6. The closing line is the post's: "Change the layout and you've changed the model. Test it like one."

**Still open:** the chart rule (the closing sections run about 1,075 words without a chart), and posting on LinkedIn (the author).

**Post 5 addendum (2026-11-27, the author's request):** PLAN.md §12, "Post 5 addendum".
- **Part A:** the table models (Kumo-Tabular, TabICL-v2) rearranged: column orders, context row orders and units, with seed noise and a single-estimator diagnostic.
- **Part B:** post 3's four trained AI models on post 5's five layouts. This is like for like, since both groups learned from the same 20,000 customers.
- **Guesses:** recorded verbatim in `PREDICTIONS.md` from a Q&A, before any code ran. They were not in the file when the author said they were.
  - Q1: no arrangement moves the table models beyond seed noise.
  - Q2: training makes the AI models much sturdier.
  - Q3: Laya moves most.
  - **Commit them before session L.** The laptop checks below printed numbers on tune customers after the guesses were written, but before they were committed; the guesses weren't changed.
- **Built on the laptop** (PLAN.md has the list; 317 tests pass). Findings that changed the plan:
  - Both table models' defaults shuffle the column order across their 8 estimators and average. The diagnostic is therefore one estimator with the shuffle off.
  - Kumo doesn't read column names; TabICL gets a bare array.
  - Laya moves to the L4: 1.9 s per customer on the laptop's CPU, about 20 ms on the L4.
- **Laya's weights** are unpacked at `C:/Users/nisha/runs/laya-ft` on the laptop (outside the repo). They match M6's (hash `c284138fb037`), and a 20-customer smoke run reproduced M6's answers to within 0.0007 on average.
- **Next:** L4 session L (`L4_RUNBOOK.md` §9, about 2.5 h, about $1.10).

## L4 session S done (2026-11-26 20:39 → 2026-11-27 00:00 IST): post 4's speed runs

L4_RUNBOOK.md §8 on a fresh L4. Gate passed: the speed-run guesses are in `PREDICTIONS.md` (9f415ea) before any run.

**Outcome:**
- **Ran:** all three models at `bf16-tuned` and `bnb-nf4-tuned` (reproduction check plus throughput), and at `hqq-4-gemlite` (tune, test, throughput).
  - `hqq-4-aoint4` ran for Kev-0.8B and Kev-4B.
  - `fp8-dynamic` ran for Qwen.
- **Recorded failures (3):** `kev-0.8b-ft-fp8-dynamic` and `kev-4b-ft-fp8-dynamic`, which returned NaN on Kev's server (torchao 0.16.0 Python API, torch 2.8), and `qwen-ft-hqq-4-aoint4`, which failed at load because HQQ 0.2.8's packing is rejected by torch 2.15. These are in `results/checks/quant_failures.json`.
- **Outputs:**
  - `analysis/recalibrate.py`: Platt for all six full-run tags. It rewrote the other prediction files with identical values (max |change| 0; timestamps only).
  - `analysis/speed_report.py` wrote `results/speed.csv`, `results/speed_throughput.csv` and `results/speed.json`.
  - Logs are in `results/checks/speed_logs/`.
- **First read (test, against each model's post 4 bf16):**
  - **Ranking:** every scored tag keeps AUC within 0.003.
  - **Honesty:** ECE changes from −2.2 points (Kev-0.8B, both HQQ backends) to +0.55 (Qwen GemLite). Qwen FP8 is at ECE 1.6 (−0.76).
  - **Speed, one at a time:** GemLite 4-bit beats `bf16-tuned` only for Kev-4B (12.1 vs 10.7 per second). `bf16-tuned` is fastest everywhere else, and at 32 at a time.
  - **Qwen:** batching doesn't speed it up here.
- **GPU time:** the box was up 15:09–18:30 UTC, about 3.35 h, about $1.47 at $0.44/h. That includes ~30 min of setup and the 16:20–16:57 reboot gap. If the gap wasn't billed, it's about 2.7 h, about $1.20. The JarvisLabs bill is the authority.
- **Box note:** after the reboot, `/home/env.sh` had been rewritten without `GH_CONFIG_DIR`. Pushes used `GH_CONFIG_DIR=/home/gh-config` explicitly.

**Fast libraries installed (§8.0), before any run.** Each was chosen to match that environment's torch:

| Environment | torch / Triton | torchao | GemLite | flash-linear-attention | causal-conv1d |
|---|---|---|---|---|---|
| This repo (`uv run --no-sync`) | 2.15.0+cu132 / 3.9.0 | 0.18.0 | 0.6.0.post2 | 0.6.0 (fla-core 0.6.0) | 1.7.0 (built from source, sm_89) |
| Kev (`/home/kev`, 65b1e76) | 2.8.0+cu128 / 3.4.0 | 0.16.0 (not the runbook's 0.13.0; see below) | 0.5.1.post1 | 0.5.2 (fla-core 0.5.2) | 1.7.0 |

- **Post 4's pins are unchanged:** bitsandbytes 0.50.2 and hqq 0.2.8.post1 in both environments.
- **torchao 0.18.0 in this repo:** no torchao release lists torch 2.15. torchao's compatibility table (pytorch/ao#2919) says the CUDA kernels are ABI-stable from 0.17 (0.17 is listed for "torch 2.11.0+"). So the newest release, 0.18.0 (CI pinned to torch 2.13), is the match for 2.15. It imports, and its FP8 dynamic setting loads.
- **GemLite differs between the environments:** GemLite 0.6.x requires a newer Triton, and installing it into Kev's environment would replace Triton 3.4 with 3.9, which breaks torch 2.8. Kev therefore has 0.5.1.post1, the newest that leaves Triton 3.4 in place. This repo (Qwen) has 0.6.0.post2. Each record's `model_version` carries its own version.
- **torchao 0.16.0 in Kev, not 0.13.0 (the author's decision, 2026-11-26):** with 0.13.0 installed, Kev's launcher crashed on load at every tag, `bf16-tuned` included. Kev's PEFT 0.21.0 raises `ImportError` for any torchao below 0.16.0 while it loads the LoRA adapter. torchao's table lists 0.16.0 for torch 2.8 through its Python API only, without torchao's own C++ kernels. FP8 dynamic runs on torch's built-in `_scaled_mm`, and only Kev's `fp8-dynamic` uses torchao. The install left torch 2.8 and Triton 3.4 unchanged.
- **FLA 0.5.2 imports with Kev's Triton**, unlike session A's 0.6.0. Kev's fused path, which pins 0.5.2, is therefore available for `bf16-tuned`.
- **Tests:** 212 passed, 1 skipped (`test_training.py` needs the train split rendered, which §7.0 skips).
- **Runner fix (no design change):** `runners/kev.py` refused every speed-tag server. The launcher reports the base precision under `precision` (e.g. `bf16`) and the tag under `tag` (`speed.apply`), but the runner compared `precision` with the tag. The tests' fake server had reported the tag as `precision`. The runner now checks `tag` and falls back to `precision` for post 4's servers, and the fake server now reports what the real one does.

**Kev-0.8B (done, 21:17–21:35 IST):**
- **Reproduction checks:** `bf16-tuned` against `bf16`, mean |diff| 0.0013 (max 0.0056). `bnb-nf4-tuned` against `bnb-nf4`, 0.0011 (max 0.0064). Both PASS.
- **Fused kernels:** on for `bf16-tuned`. The launcher turns them off for the shrunk tags.
- **CUDA graphs:** none failed. `hqq-4-aoint4` ended with 1 pending.
- **Batching:** every batch-size difference was at most 0.0075.
- **Full runs:** `hqq-4-aoint4` and `hqq-4-gemlite` completed tune and test.
- **`fp8-dynamic` failed (recorded, stage smoke):** the server produced a NaN on the first request (`ValueError: Out of range float values are not JSON compliant: nan`). It ran torchao 0.16.0's Python API on torch 2.8, with CUDA graphs on. Skipped, not replaced.

**Qwen (done, 21:35–23:03 IST, with an interruption):**
- **Reproduction checks:** `bf16-tuned` against `bf16`, mean |diff| 0.0052 (max 0.0310). `bnb-nf4-tuned` against `bnb-nf4`, 0.0058 (max 0.0278). Both PASS on the mean (limit 0.02), although single customers differ by up to 0.03.
- **`hqq-4-aoint4` failed (recorded, stage load):** HQQ 0.2.8.post1's `torchao_int4` backend passes non-uint8 weights to torch 2.15's int4 packing op (`RuntimeError: Expected in.dtype() == at::kByte`). Kev's torch 2.8 accepted them. Skipped, not replaced. No throughput run.
- **Full runs:** `hqq-4-gemlite` and `fp8-dynamic` completed tune and test.
- **Instance reboot:** the instance went down at 16:20 UTC and came back at 16:57, during `hqq-4-gemlite`'s test split. At that point tune was complete and test had 11 of 2,000 records. At 16:58 something outside the session re-synced this repo's `.venv` without the runner groups, which removed torch. The environment was restored to the same versions (the same `uv sync` groups, then the same four fast libraries). The run resumed from its saved records: all 4,000 share one `model_version`, with no duplicates and no NaN. The ~45 idle minutes are billed.
- **Batching moves Qwen's answers by more than 0.01:** up to 0.031 for `bf16-tuned`, `bnb-nf4-tuned` and `hqq-4-gemlite`, and 0.052 (8 at a time) for `fp8-dynamic`. Batches are left-padded, one forward pass each. Kev stayed at 0.0075 or less.
- **Throughput, one at a time:** customers per second, `bf16-tuned` 12.8, `bnb-nf4-tuned` 9.2, `hqq-4-gemlite` 11.8, `fp8-dynamic` 12.9. Batching barely helps, and at 32 at a time it is slower for three of the four tags.

**Kev-4B (done, 23:06–23:58 IST):**
- **Reproduction checks:** `bf16-tuned` against `bf16`, mean |diff| 0.0010 (max 0.0045). `bnb-nf4-tuned` against `bnb-nf4`, 0.0011 (max 0.0038). Both PASS.
- **Fused kernels:** on for `bf16-tuned`.
- **CUDA graphs:** none failed.
- **Batching:** every batch-size difference was at most 0.0073.
- **Full runs:** `hqq-4-aoint4` and `hqq-4-gemlite` completed tune and test.
- **Throughput, customers per second, 1 / 32 at a time:**

  | Tag | 1 at a time | 32 at a time |
  |---|---|---|
  | `bf16-tuned` | 10.7 | 18.2 |
  | `bnb-nf4-tuned` | 7.6 | 13.2 |
  | `hqq-4-aoint4` | 4.1 | 3.8 |
  | `hqq-4-gemlite` | 12.1 | 13.5 |

- **`fp8-dynamic` failed (recorded, stage smoke)** the same way as Kev-0.8B: a NaN on the first request. Skipped.

## Post 4 draft (2026-11-26, version 2 on 2026-11-27): in the author's review

`writeup/quantization_post.md`, "I shrank AI models. It saved memory, not time." (46 characters, under LinkedIn's 58). About 3,500 words with tables and four charts (`analysis/quantization_figures.py` → `results/figures/post4_{ranking,honesty,drift,cost}.png`, light and dark).
- **New measures, added while writing (the post says so):** `analysis/quantization.py` now also reports the lending-decision counts. These are the share of approve/decline decisions that differ from bf16 (each at its own tune cut-off), the shrunk model swapped in at bf16's tune cut-off (approval, profit, decisions changed), and the share of customers moved 5+ points. All existing columns are unchanged. The noise floor quoted in the post is two full-size runs (M6 against the session A bf16 re-run), at the M6 cut-off like the table's other rows: 0.15% of decisions change for Kev-4B, 0.35% for Kev-0.8B and 1.5% for Qwen (corrected 2026-11-27 from 0.2%/1.4%, which were the own-cut-off figures). It isn't a column: `quantization.decisions` on the M6 and bf16 predictions reproduces it, and the carousel builder computes it.
- **Review copy:** https://nishzsche.github.io/review/shrinking-ai-models/ (unlisted, noindex), as post 3's was. It's on the blog's `dev` branch as `review/shrinking-ai-models.md`, with the charts in `public/images/shrinking-ai-models/`. Version 1 went live on 2026-11-26 (blog PR #47). Version 2 (blog PR #48, awaiting the author's merge) adds the speed rewrite and all the panel fixes, which version 1 predates.
- **Review panel (2026-11-26).** Five agents: statistician, fact-checker, bank model-risk validator, editor, and ML engineer with hiring manager. Scores as drafted were 6.5–7 (hiring manager 7.5, "would interview"), and 8.5–9.5 with the fixes. Every number reproduced from the raw predictions. The drift section was called the best idea in the series. Must-fixes, now applied to the draft:
  - **Decisions table:** each row now sits at the full-size model's cut-off (`decisions_changed_bf16_cutoff`). Re-choosing the cut-off on tune flips 3.4% of Kev-0.8B's decisions on its own (post 3 against the bf16 re-run, at their own cut-offs), so the earlier own-cut-off rows were mostly cut-off noise.
  - **Honesty changes:** a paired bootstrap of the ECE change, Holm over 15, is now in `analysis/quantization.py` (`ece_vs_bf16*`). Above 2 bits, only Kev-0.8B at hqq-4 changed by more than chance (−2.2 points, Holm p 0.024). Qwen's hqq-4 rise (+0.8) and every 3-bit change are within chance. Qwen's mean shift (22.1% to 19.2%) is label-free and solid.
  - **Speed explanation:** it was wrong and has been removed. Post 3's Kev-4B gained only 5% from CUDA graphs (106 vs 111 ms). Every version, full size included, ran `causal_conv1d` and Qwen's `chunk_gated_delta_rule` on slow reference code (the logs say so). Each Kev request is two backbone passes (about 245 tokens in, 23 out). The CUDA-graph limit belongs to Kev's server, not to quantization.
  - **Facts:**
    - Post 3 trained four models.
    - Post 4's own GPU time is 4.3 h, about $1.90.
    - NF4 is the common recipe, not transformers' default (FP4, fp32 compute).
    - Kev-4B costs 1.3 cents per 1,000.
    - Memory is now in decimal GB (the CSV columns are GiB).
    - "Within 0.002" was on 50 tune customers.
    - The "moved 5+ points" share was also added after the runs.
  - **Overclaims:**
    - "a dashboard would have passed it" (calibration slope went 0.86 to 1.31);
    - "earns nothing";
    - "perfectly honest";
    - "safe";
    - "0.02 can't be told apart" (between models; a model against itself resolves about 0.004);
    - "ranking held to 3 bits".
  - **Bank advice:** rewritten around governance (material model change, tolerances, day-one score migration, re-set cut-offs through approval, record GPU, kernels and drivers). New limitations: an unsupported vendor use, fair lending, keeping the update unmerged.
- **The author's decision (2026-11-26): hold post 4 for extra speed runs** (PLAN.md §12, post 4 speed addendum). The title, subtitle, speed section, one predictions row and the speed bullets are marked PENDING in the draft. The live review copy is still version 1.
  - **Guesses:** committed before any speed run (9f415ea).
  - **Code:** built on the laptop, 213 tests (PLAN.md §12, the addendum's "Built" note).
  - **Session S done** (see its section above). **Speed rewrite done (2026-11-27):**
    - **New title and subtitle:** "stayed honest" dropped, because both Kev models are outside the chance band at full size.
    - **Speed section, "Smaller, but not faster":** it says a review found the first setup untuned, then lists the tuned setup. That includes switching Kev's CUDA graphs on for the shrunk versions (an unsupported use, now in the limitations) and Kev's fused kernels at full size only.
    - **What it reports:** the setup mattered more than size (62→20, 120→78, 111→93 ms at full size, with a caveat that the timing methods differ). NF4 was 18–40% slower. GemLite beat full size only for Kev-4B (−11%). FP8 matched Qwen's full size. Three failures are recorded. The fast routes give the same answers (0.1 points per customer for Kev, 0.5 for Qwen). At 32 at a time, full size is cheapest for both Kevs. The section gives no mechanism for the slowness ("I didn't measure where the time goes"), after two wrong explanations.
    - **Cost chart redrawn:** `post4_cost.png` now shows memory (the first runs) and time on the tuned setup, with the first runs' times as hollow dots.
    - **Predictions:** four rows added from the speed addendum. Bank advice gains a speed bullet and a batch-size line (batching moved Qwen's answers up to 3 points, 5 with FP8). Limitations gain a speed bullet. "How I ran it" gains the re-runs (reproduction, 500 tune customers, about 3.4 h, about $1.50).
    - **Chart budget:** about 1,800 words follow the last chart (predictions, bank, limitations, method), well over STYLE's 600-word rule.
  - **Review copy version 2 is live** (blog PR #48, merged 2026-11-27).
  - **LinkedIn draft (2026-11-27):**
    - `writeup/quantization_linkedin.md`: post text of 2,270 characters, the first comment (the published URL still to fill in) and 9 slides.
    - `writeup/build_quantization_carousel.py` builds `writeup/quantization_carousel.pdf`: hook, how shrinking works, ranking, honesty, customers moved, lending decisions, speed, bank, limits/next. Every slide was checked as a PNG.
    - The decisions slide computes the same-model-twice floor itself.
    - The carousel picks up whatever `analysis/figures.py` has; it was built with the blog session's uncommitted warm-paper surface (#faf8f3).
    - **Draft fix found while building:** the post's same-model-twice row now reads 0.2% / 0.4% / 1.5% at the full-size cut-off, not 0.2% / 1.4%. Review copy version 2.1 carries it (blog PR #52, on its own branch from main, because another session's PR #51 was open on `dev`).
  - **Publishing (2026-11-27):**
    - Blog PR #54 was merged at 01:00 IST while the post was dated 12:00 IST. GitHub Pages skips future-dated posts and only rebuilds on a push, so the post 404'd, and the review redirect and post 3's "Next" link led there.
    - Blog PR #55 (the author chose to publish now) dates it 00:55 IST and sets the chosen drawing.
    - **URL:** it first built at /2026/11/26/, because the blog had no timezone and Jekyll used UTC. The blog session's PR #56 (`timezone: Asia/Kolkata`) moved it to its intended **https://nishzsche.github.io/2026/11/27/shrinking-ai-models/**. My PR #58, merged just after, had pointed the review redirect and post 3's link at /26/. The blog session is fixing both in its Phase 4 work and adding a /26/ redirect. The LinkedIn first comment uses /27/.
    - `_posts/2026-11-27-shrinking-ai-models.md`.
    - Charts redrawn with the new export and embedded with `chart.html` (alt plus "Chart in words"). The drift chart's numbers moved above each panel: the boxes had hidden the 60–80% customers. The honesty label and the cost footnote were re-laid.
    - Social card `results/figures/post4_social_card.png` (`quantization_figures.py social_card`).
    - Drawing `20261009-163715.jpg` (Hanuman bowing to a stick of RAM), the author's choice, relayed by the blog session. On main it's also post 1's icon until the blog's design Phase 3 reassigns icons and adds `icon_caption:`.
    - The review URL redirects, and post 3's closing line links to post 4.
    - The LinkedIn first comment now carries the URL.
  - **LinkedIn revision (2026-11-27, the author's call):** the post and carousel give the final speed result only. The "my first speed test was wrong" bullet is gone, and slide 7 no longer shows the first runs as hollow dots (the blog keeps that story). The wording is simpler throughout, and the post is now 2,177 characters.
  - **Memory correction (blog PR #64, merged 2026-11-27 22:58 IST):** the bank advice said "Shrinking cut memory by more than half every time". That holds only for the two 4B models at 4 bits: Kev-0.8B went from 1.7 to about 1.0 GB (over 40%), and 8 bits saved 29% to 42%. The post and `writeup/quantization_post.md` now say so, with an `updates:` note. The LinkedIn draft had the same slip ("the models") and now says "the two larger models". PR #64's commit message wrongly says 8 bits saved about 40% for all three (Kev-0.8B saved 29%); the PR description corrects it.
  - **Next:** the author merges PR #55, then posts on LinkedIn around 13:00 IST (as posts 2 and 3) with `writeup/quantization_carousel.pdf`. X post optional (post 3's was never written). Open rule question: exempt closing sections from the 600-word chart limit. The blog session is moving `analysis/figures.py` to the new blog design (warm paper, larger small text). Post 4's charts stay in the current look until that design launches; then redraw them and check the layouts.
- **Still to do after the author's review:** the LinkedIn post and its carousel (`writeup/STYLE.md`: 8–10 slides, a builder script), the X post, the social card, and the README and PLAN status.

## L4 session A done (2026-11-25 19:43 → 2026-11-26 01:30 IST)

L4_RUNBOOK.md §7 ran in full on the resumed M6 instance: about 5.8 GPU hours, about $2.55. Nothing failed, so there are no `--record-failure` entries. Everything is pushed (last commit `84bd30b`); the instance can be deleted and its GitHub token revoked.

**Post 5, the open models on four formats:** Laya, Kev-4B, Kev-0.8B and Qwen, zero-shot, `out_of_box`, 2,000 `test` customers per format, in `results/checks/formats/`. The report against v2 is `results/checks/formats_summary.json`. Jev and Sol ran on the laptop afterwards (Next, item 5), so the summary now covers all six models.

**Post 4, the fine-tuned models at six precisions:** Kev-0.8B, Kev-4B and Qwen at `bf16`, `bnb-int8`, `bnb-nf4`, `hqq-4`, `hqq-3` and `hqq-2`, `tune` and `test`, plus `trained_calibrated` (Platt chosen for all 18 rows).
- **Guards:** all three bf16 reproduction checks passed against M6 (mean |difference| 0.0009 Kev-0.8B, 0.0010 Kev-4B, 0.0020 Qwen; limit 0.02). Every Kev precision passed its 20-customer smoke run first.
- **Outputs:** `runners/manifest.py --quant` reports 18/18 complete (`results/run_manifest_quant.json`). `analysis/quantization.py` wrote `results/quantization.csv` and `.json`. Server reports and logs are in `results/checks/quantization/` and `results/checks/quant_logs/`.
- **First read (test):**
  - Down to 4 bits, ranking holds: AUC stays within 0.005 of each model's own bf16 run, and memory for the 4B models falls to about 40% of bf16.
  - `hqq-3` holds for Kev-0.8B and Qwen. Kev-4B loses 0.017 AUC (Holm p 0.012).
  - `hqq-2` breaks all three: AUC 0.43–0.60, mean prediction 0.36–0.67 against an actual 0.22.
  - Fewer bits weren't faster on this serving path (CUDA graphs and fused kernels off). Per request, bnb-int8 and HQQ were slower than bf16, and bnb-nf4 was about the same.

**Changes and caveats from the session:**
- **Launcher fix (`a0cfa6f`):** `runners/kev_quant_serve.py` imported `runners/kev.py` instead of Kev's package and crashed before loading. It now drops `runners/` from `sys.path` before importing Kev; no design choice changed.
- **Kev's environment:** `flash-linear-attention` is pinned back to 0.5.0, as M6 had. An unpinned install pulled 0.6.0, which fails to import with Kev's Triton. Fused kernels were off in both M6 and session A, because Kev's fused path wants 0.5.2.
- **Logs:** the bnb-int8 logs are trimmed in the repo. bitsandbytes' `MatMul8bitLt` cast warning is kept once, with its count (about 1.5M repeats; one log was 122 MB, over GitHub's limit).
- **Order:** for Kev-4B, the smoke run went before the reproduction check, against the runbook. The check passed before any full run.
- **Rank correlation varies between machines:** re-running `disguise.py --report` moved only the Spearman values in the published `results/checks/disguise_summary.json`, by at most 0.002 for 6 entries. The committed file was kept as published.
  - **Cause (found on the laptop, 2026-11-26):** it is not the recalibration rewrites. `report_one` ranks with `np.argsort(np.argsort(p))`, whose default sort is not stable. Tied predictions get ranks in whatever order the CPU's sort code leaves them, and that order differs between the L4 and the laptop. The formats summary showed the same thing: the L4's four models re-scored on the laptop matched in every field except Spearman, which moved in the fifth decimal.
  - **Size:** this matters for Jev and Sol only, which give about 80 distinct probabilities across 2,000 customers. For them the current figure is up to 0.003 below a standard tie-aware Spearman (`scipy.stats.spearmanr`). At two decimals, only Sol on markdown changes (0.94 → 0.95); nothing in the published disguise check changes.
  - **Decided (author, 2026-11-26): switched to `spearmanr`.** `report_one` now uses tie-aware Spearman, and a test in `tests/test_formats.py` pins it. `disguise_summary.json` and `formats_summary.json` were regenerated: only the 6 + 24 Spearman fields changed, and every other field is identical. The largest change is Sol's (disguise 0.9405 → 0.9437, csv 0.9485 → 0.9515). No write-up quotes the figure.
- **Recalibration** rewrote the M3/M6 prediction files with identical values (timestamps and float noise of at most 2e-16 only).

## After post 3: the plan (2026-11-25)

PLAN.md §12 has the order and the specs.

**The author's decisions (2026-11-25):**
- **Order:**
  - post 4, does shrinking a model make it less honest (quantization);
  - post 5, why AI struggles to read a table (new: formats, plus the existing layout and disguise checks);
  - post 6, "I don't know" (redesigned: flagging unfamiliar customers, not proper scoring);
  - then the documents arc (posts 7–9) and fresh data (post 10).
- **Kev in post 4:** we quantize it ourselves; its loader has no quantization (checked 2026-11-25). The guard is a bf16 reproduction check before any quantized run.
- **Stress levels:** 3-bit and 2-bit are in (HQQ), beside bitsandbytes 8-bit and NF4.

**Still open:**
- the pace between posts;
- post 6's design (the held-out slice, training to defer, referral rates), before session B.

**Built on the laptop (2026-11-25):** posts 4 and 5's code; PLAN §12 lists what was built.
- **Tests:** 197 passing (`uv run --group runners --group laya --group quant pytest`).
- **Post 4:** every precision was checked on a toy bf16 model on the CPU. The launcher inside Kev's server and the bitsandbytes GPU kernels can only be checked on the L4. Their guards there are a reproduction check against M6 and a 20-customer smoke run per precision.
- **Post 5:** the formats read back to v2's values for all 4,000 tune and test customers. All of them fit Laya's context (`results/checks/formats_tokens.json`).
  - Laya's runner now stops if Laya ever reports truncating a summary.
  - The plumbing was tried on one customer on the laptop. Those numbers aren't reported anywhere, so they can't steer the guesses.
- **Session A** is L4_RUNBOOK.md §7. It starts with the prompt for Claude Code on the L4.

**Next:**
1. Post 3's X post (below).
2. **Done (2026-11-25):** the author's post 4 and post 5 guesses are in `PREDICTIONS.md` (commit 44eee55), before any run.
3. **Done:** the OpenRouter balance covered post 5's Jev and Sol runs.
4. **Done (2026-11-26):** L4 session A (L4_RUNBOOK.md §7); results above.
5. **Done (2026-11-26, 04:42 IST):** Jev and Sol on the four formats, 2,000 `test` customers per format, into `results/checks/formats/`.
   - **Versions:** each file has one model version, the same as M3's.
   - **Cost:** Jev $0.18, Sol $7.59. Sol averaged about 16 requests a minute, about 2 h per format.
   - **Report fix:** `report_formats` now scores a run only once it covers every test customer and lists shorter runs as still running. It had crashed on Sol's partial prose run, whose first customers were all non-defaulters, so AUC was undefined.
   - **Summary:** written by calling `disguise.report_formats()` on its own, so the published `disguise_summary.json` stays untouched.
6. **Meanwhile:** choose the documents arc's dataset and check its licence.

## Post 2 published (2026-11-24, 12:55 IST)

- **Live:** [Why 99% accuracy can still be a disaster](https://nishzsche.github.io/2026/11/24/why-99-percent-accuracy/) (blog PR #38), from `writeup/accuracy_post.md`. Numbers, chart (light and dark) and social card come from `analysis/accuracy.py`.
- **Review:** two reviewers. Version 2 added the confusion matrices and named the table models (blog PR #31). The review URL now redirects to the post. Post 1's closing line links to post 2.
- **Post 1:** gained the "Honest isn't self-aware" caveat (blog PR #33), from a reviewer's point that calibration isn't metacognition.
- **Charts (2026-11-24):** post 2 went from 1 chart to 4: `post2_accuracy`, `post2_confusion` and `post2_cost_ratio` were added (blog PR #44). Jev's and Kev-4B's defaulters caught were corrected to 71% and 64%; they had been double-rounded to 72% and 65%.
- **LinkedIn:** the first version went up boosted, with a single image. It is being replaced by a document post, the 9-slide carousel `writeup/accuracy_carousel.pdf` (built by `writeup/build_accuracy_carousel.py` from the `## Slides` section of `writeup/accuracy_linkedin.md`). The document title is "Why 99% accuracy can still be a disaster: 8 models tested", within LinkedIn's 58-character limit.
- **New rules in `writeup/STYLE.md`:** a chart budget (about one chart per 400–500 words, the first within 400 words), and a LinkedIn carousel for every post (8–10 slides, drawn at phone size).

## Post 3 published (2026-11-25, 13:02 IST)

- **Live:** [I gave AI the same data as a bank's model. It caught up.](https://nishzsche.github.io/2026/11/25/fair-fight/) (blog PR #46), from `writeup/fair_fight_post.md`. Charts and the social card (`m6_social_card.png`) come from `analysis/fair_fight_figures.py`. The review URL redirects to the post, and post 2's closing line now links to it.
- **LinkedIn:** the post text and first comment (with the URL) are in `writeup/fair_fight_linkedin.md`. The 10-slide carousel is `writeup/fair_fight_carousel.pdf`, with the document title "I gave AI the same data as a bank's model. It caught up." (56 characters).
- **Still to do:**
  - the X post;
  - the chart-rule decision on the 703 words after the last chart (proposal: exempt the closing sections from the 600-word limit).

## Post 3 (the fair fight): draft and review (2026-11-24)

- **Draft:** `writeup/fair_fight_post.md`. Charts come from `analysis/fair_fight_figures.py`: `m6_catch_up` (ranking), `m6_how_close` (paired gaps to gradient boosting and the scorecard, Bonferroni-widened so the chart matches the Holm wording), `m6_honesty` (raw vs the same correction), `m6_kev_calibration` (Kev raw vs corrected), `m6_cost` (cost vs ranking), `m6_sol_neighbours` (Sol and Jev against their 20 look-alikes), `m6_ceiling` (every model that learned from the 20,000 on the full 0.5–1.0 scale, plus the scorecard trained on 2,000) and `m6_reliability` (deep dive, not in the post). The scoreboard comes from `analysis/fair_fight.py`.
- **Review copy:** https://nishzsche.github.io/review/fair-fight/ (unlisted, noindex). Version 1 went up on 2026-11-24 (blog PR #42); version 1.1 added four charts (PR #43). Version 2 (PR #45) added the seed results, Sol's caveat beside its corrected score, the headline chart moved to the top, and the ceiling chart.
- **Published** on 2026-11-25; see the section above.
- **Review panel** (five agents: statistician, fact-checker, bank model-risk validator, editor, ML engineer and hiring manager). Scores as drafted were 6–7/10, about 8.5 with the fixes. No leakage was found, and every number recomputes from the raw predictions. The must-fixes:
  - **Laya:** 4 passes plus its own temperature fit on 400 train rows, not one pass.
  - **Qwen:** common LoRA settings; it has no maker recipe.
  - **The predictions table:** Kev-0.8B was second least honest, not Kev-4B.
  - **"It tied":** overstated. The first Holm family covered 26 rows, zero-shot and duplicate calibrated rows included.
  - **"Added nothing beyond counting":** never tested.
  - **Wording:** race language, "six" for eight models, "Nobody had checked", and how much tuning gradient boosting got.
- **Kev under-prediction (diagnosed 2026-11-24):**
  - Both fine-tuned Kevs average 18–19% against 22.1%, on tune and test alike, mostly for low- to middle-risk customers.
  - Ruled out: the augmentations (choice-only), the readout (a clean P(yes) over {no, yes}), split differences, truncation.
  - Kev's maker recipe ends with a temperature fit on held-out data, which was **not applied**: Kev was served at T = 1.0 (PLAN §11, choice 3). Laya's equivalent step **was** kept.
  - The models are overconfident at T = 1 (calibration slopes 0.86–0.93). A temperature alone fitted on tune fixes most of it (see `results/fair_fight.json`, `temperature_only_on_tune`).
  - The "overshoot from 49%" story in the first draft was wrong.
- **Author's decisions (2026-11-24):**
  - The post compares honesty **after the same correction on tune** for every model, raw second, and explains the Kev/Laya recipe difference.
  - Title framing: **"It caught up"**.
  - **Extra seeds:** Kev-0.8B and Laya, seeds 1 and 2 (L4_RUNBOOK.md §6.8, `analysis/seeds.py`). Run on 2026-11-24 (commit 0e9f369), with results in `results/seeds.json`:
    - **Ranking:** varied by under 0.01 across runs (Kev-0.8B 0.786–0.790, Laya 0.778–0.786).
    - **Raw calibration error:** varied more (Kev-0.8B 3.0–4.5 points, Laya 1.7–3.5).
    - **After the correction:** steadier (Kev-0.8B 1.3–1.5, Laya 0.9–2.5).
    - **Kev-0.8B's under-prediction:** holds in all three runs (it averages 18–20%).
- **Analysis fix:** `analysis/fair_fight.py` now runs Holm over the eight trained AI models' raw output only, and adds paired tests against each model's own zero-shot score and against the 20 neighbours, an "adds information beyond the neighbours" test for Jev and Sol, and Kev's temperature-only fit.

## M6: the fair fight (started 2026-11-23)

After launch, the author asked to train the AI models on the same 20,000 customers the scorecard and LightGBM learned from. PLAN.md §11 has the design and the choices that could tilt the comparison. §12 proposes the next posts.

**Author's choices (2026-11-23):**
- Jev and Sol take part through 20 look-alike `train` customers in the prompt, since neither can be fine-tuned.
- Tabular foundation models join as Part B (the author's suggestion).
- Write-up venue: decided once the results are in.

**Who can be trained (checked 2026-11-23):**
- **Kev:** yes, with the maker's `kev.train --init_from`, a LoRA delta on the released weights.
- **Laya:** yes, with `laya.train.finetune`.
- **Qwen:** yes, with a standard LoRA.
- **Jev:** no. TypeSafe's docs: "Jev is not fine-tuned or LoRA-adapted with customer data".
- **Sol:** no. OpenAI is winding down fine-tuning, it's closed to new users, and it never covered GPT-6.2.

**Built and tested on the laptop (2026-11-23):**
- **`data/neighbours.py`:** the 20 nearest `train` customers for every `tune` and `test` customer (`data/neighbours/`).
  - Its own answer, the neighbours' default rate, is the reference contestant `neighbours-20`: test AUC 0.764, ECE 0.023; after calibration on tune, 0.762 and 0.015 (isotonic).
  - That already beats every zero-shot AI model's AUC and is close to the scorecard (0.777).
- **`runners/examples.py`:** `jev-examples` and `sol-examples`, condition `trained`. Smoke run on 20 tune customers (actual rate 0.20):
  - Jev: 5,251 input tokens per call, mean p 0.40, $0.0002 per call.
  - Sol: 4,045 tokens per call, mean p 0.27, $0.0107 per call.
  - Full runs: Sol about $43 (tune plus test), Jev about $1.
- **Trainers:** `training/export.py` (training files for Kev, Laya and Qwen from the template v2 summaries), `training/qwen_lora.py`, `training/laya_ft.py`.
- **Scoring:**
  - `runners/qwen_ft.py`: in-process, with a readout check against M3's zero-shot numbers before it scores anything.
  - `runners/kev.py --finetuned`: refuses a server that isn't at temperature 1.0.
  - `runners/laya_classic.py --checkpoint`.
  - The harness's `trained` condition reuses the `out_of_box` request.
- **Analysis plumbing:** `analysis/recalibrate.py` now writes `trained_calibrated` for the M6 contestants, and `runners/manifest.py --m6` writes `results/run_manifest_m6.json`. The published M3 manifest is untouched.
- **Qwen's training prompt:** rebuilt from jevper's own pieces. It matches all 8,000 first-attempt requests recorded in M3 byte for byte (`tests/test_training.py` checks 300).
- **Not smoke-tested yet:** the Kev, Laya and Qwen trainers. The laptop has 1 GB free on C:, so no model weights were downloaded. L4_RUNBOOK.md §6 smoke-tests each one on the L4 for a few steps before its real run.
- **Tests:** `uv run --group runners pytest`, 170 passing.

**Next:**
1. **Author: `PREDICTIONS.md` M6 guesses**, committed before any full run (the gate in L4_RUNBOOK.md §6). For example: which trained AI model is most honest, whether any beats LightGBM's AUC (0.789), and whether Jev and Sol beat their own 20 neighbours (AUC 0.764).
2. **Author: top up OpenRouter by $30.** The balance was $21.51 on 2026-11-23; the M6 API runs need about $44.
3. **L4: L4_RUNBOOK.md §6** (§6.0 sets up a fresh instance from zero), in the order Kev-0.8B → Laya → Kev-4B → Qwen → the tabular models (§6.7). Expected 4–6 GPU hours (~$2–3). Stop and ask if Kev-4B's or Qwen's epoch ETA is over 3 hours. The trained weights go into a GitHub release (`m6-weights`).
4. **Laptop: Jev and Sol with examples: done (2026-11-24, 01:25 IST).** 4,000 records each, no duplicates, no failed customers. Jev $0.86, Sol $40.95 (one Sol request hung for 18 minutes; its recorded latency includes that).
   - **Results so far** (`analysis/fair_fight.py`, test, raw `trained`; AUC gaps are paired bootstrap, Holm-corrected):
     - Kev-0.8B fine-tuned: AUC 0.790, tied with LightGBM (+0.001, p = 1.0); ECE 0.045 (it under-predicts: 18.1% vs 22.1%), 0.014 after calibration on tune.
     - Laya fine-tuned: AUC 0.778, tied with the scorecard; ECE 0.017, as honest as LightGBM, and 0.009 calibrated.
     - Sol with examples: AUC 0.762, ECE 0.035. That's no better than its own 20 neighbours (0.764, 0.023), and it's behind LightGBM (p < 0.001).
     - Jev with examples: AUC 0.743, ECE 0.106. That's below its neighbours, and behind both baselines (p = 0.014 and < 0.001).
   - Original plan for this step: Sol is paced at 18 requests/minute, so each split takes about 2 hours.
   ```bash
   uv run --group runners python runners/examples.py --contestant jev --split tune --workers 4
   uv run --group runners python runners/examples.py --contestant jev --split test --workers 4
   uv run --group runners python runners/examples.py --contestant sol --split tune --workers 4
   uv run --group runners python runners/examples.py --contestant sol --split test --workers 4
   ```
5. **Part B, tabular foundation models: checked (2026-11-23), runner `baselines/tabular_fm.py`, L4_RUNBOOK.md §6.7.**
   - **Kumo-Tabular:** NVIDIA, `nvidia/Kumo-Tabular` (large, checkpoint v1.0.1). Package `structured-data-models` 0.3.0. Weights OpenMDW-1.1, code Apache-2.0. Pretrained on up to 60,000 rows and 100 columns. Runs.
   - **TabICLv2:** Inria, `tabicl` 2.2.0, BSD-3. Checkpoint `tabicl-classifier-v2-20260212`. Pretrained on 300–48K rows. Its default settings soften its own probabilities (temperature 0.9); that's the maker's choice, so it stays. Runs.
   - **TabPFN-3.5: out** (the author's decision, 2026-11-24). Its weights need a Prior Labs login, and its licence is non-commercial for the weights *and their outputs*. It's removed from the runner and the dependencies.
   - **Laptop smoke (CPU, 500 context rows, 100 tune customers):** Kumo-Tabular *small* AUC 0.775, TabICLv2 AUC 0.788. Both code paths work; the contestant is Kumo *large* on the L4.
   - **Mitra:** dropped. No v2 exists in AutoGluon's docs, and v1 caps out at 10,000 training rows, half of what the baselines see.
6. **After the runs:**
   - `uv run python analysis/recalibrate.py` (it also refreshes the zero-shot `recalibrated` records, with the same values).
   - `uv run --group runners python runners/manifest.py --m6`.
   - `uv run python analysis/fair_fight.py` (written 2026-11-23, checked on the existing rows): zero-shot → trained per model against the baselines, with paired AUC tests against the scorecard and LightGBM, Holm-corrected. The figure is still to write.

## Launched (2026-11-23, 16:05 IST)

- **Published:** [the post](https://nishzsche.github.io/2026/11/23/can-ai-know-when-its-wrong/) (`writeup/blog.md`, cut from about 5,400 words to about 2,000). The deep dive (`writeup/deep_dive.md`: method, statistics, Sol's checks, disguise and layout tables, different-data analysis, related work, limits, reproduction) lives in this repo only: it was published as a second post, then removed from the blog at the author's request (2026-11-23), and the post links to the GitHub copy. Blog PRs #22 and #23.
- **Live checks:** the post loads. The post has the contents list, Giscus comments (Discussions in this repo), GoatCounter and the social card, and search engines can index it. The old review URL redirects to the post.
- **Blog housekeeping:** the notebook pipeline is removed (the three workflows republished the two old notebooks as new posts). `notebooks/` stays, unpublished. The About page mentions the credit-risk work and links the post. The `gh-pages` branch is a leftover from that pipeline: Pages deploys from `main`, so it's unused.
- **Carousel and X post** now carry the real links. They're ready for the author to post; `writeup/carousel.pdf` is the LinkedIn document.

**Still to do (author):**
1. Post the LinkedIn carousel (`writeup/carousel.md` post text plus `carousel.pdf`, links in the first comment) and the X thread (`writeup/x_post.md`, chart attached, links in the first reply).
2. Optionally re-check the link preview in LinkedIn Post Inspector (needs a login). The tags themselves are verified.
3. Revoke the GitHub access the L4 used.
4. The Notion review copy still shows the long version 3 draft. It now carries a note that points to the published post.

## Where things stand

**M4 is nearly done (2026-11-23).** All eight contestants are scored on test and all five figures are regenerated with Sol in. Tests pass.
- **Built:**
  - `analysis/metrics.py`: every PLAN §4 metric, 19 tests including cross-checks against scikit-learn.
  - `analysis/recalibrate.py`: the `recalibrated` condition.
  - `analysis/summarize.py`: writes `results/summary.csv`, `results/summary_detail.json` and `results/summary_excluding_memorized.csv`.
  - `analysis/figures.py`: the five figures, in `results/figures/`.
- **Recalibration method (flagged choice):** Platt vs isotonic, picked per AI contestant by 5-fold CV log loss on **tune only**. Platt won for Kev-4B, Kev-0.8B, Laya and Qwen; isotonic for Jev and Sol (Sol narrowly: 0.4524 vs 0.4549). The choice and CV scores are in `results/recalibration.json` and in each record's `model_version`.
- **Calibration intercept** is the slope-fixed-at-1 ("calibration-in-the-large") intercept (Van Calster et al. 2019); the slope comes from the joint logistic fit.
- **Manifest fix:** the cost and latency totals now count model calls only. Recalibrated records copy their source call's cost per decision, which is right per record but was double-counting run spend.
- **Checks done:**
  - Labels agree across every predictions file and match the split parquet.
  - Kev-4B's AUC jump with the base-rate sentence (0.59 → 0.76 on test) reproduces on tune (0.59 → 0.73); its two conditions' rankings correlate at only 0.23. That's real model behaviour, worth a line in the write-up.
- **Sol on test:**
  - Out of the box: AUC 0.754, ECE 0.052, mean p 0.260 (actual 0.221).
  - With the base rate: ECE 0.032; recalibrated: ECE 0.017.
  - Cost: $0.74 per 1,000 decisions (base rate: $1.02).
  - No AI contestant beats the scorecard's AUC (0.777); the best are Kev-4B with the base rate (0.757) and Sol (0.754).
- **Sol's memorization: all three follow-ups done** (user, 2026-11-22). Details are in `data/contamination.md`.
  - **Depth test** (`data/memdepth.py`, $1.14): 31 customers memorized verbatim, dense through customer 33 and sporadic to 92. Of those, test has `cust_00004`, `cust_00016` and `cust_00018`; tune has `cust_00005`–`00007`. tabmemcheck's 1,000-token cap (which includes reasoning) was most likely why the first test stopped early.
  - **With/without:** without the 3 memorized test customers, no contestant's AUC, ECE or calibration-in-the-large moves by more than 0.0006.
  - **Recognition** (`data/recognition.py`, $0.07): Sol named the dataset for 53 of 53 summaries at 99% confidence and gave its exact default rate (22.12%) every time.
  - **For the write-up:** its out-of-the-box calibration must be read as that of a model that knows the population, not zero-shot honesty.
  - **Disguise check** (`runners/checks/disguise.py`, user asked 2026-11-23): removing NT$, the 2005 months and the raw repayment codes doesn't hide the dataset. Sol still named it for 50 of 50 disguised summaries (22.12%, 99% confidence), now citing the six-month bill/payment/status structure itself.
    - **Prediction run, every AI contestant (2026-11-23):** 2,000 disguised test customers each, plus a 200-customer control on the original summaries. Mean shifts: Jev +4.1, Qwen +3.4, Kev-4B +1.5, Sol +0.8, Kev-0.8B −1.3, Laya −4.8 points (noise ≤ 0.3; Kev and Qwen reproduce exactly or nearly). Kev-4B ranks *better* disguised (AUC 0.594 → 0.651). No era effect; wording sensitivity. Details are in `data/contamination.md` §4.
    - **Layout check (`--variant aligned`, tune and test, 2026-11-23):** each bill shown beside the payment made the following month, for the template-v2 tilt the review panel flagged. The author chose to run it on test for every AI contestant and report it next to v2 (headlines stay v2). Test results:
      - **Kev-4B changes a lot:** mean p 0.486 → 0.356, ECE 0.267 → 0.135, AUC 0.594 → 0.720; code 0 customers 0.54 → 0.36 (actual 0.12). v2 put it at a real disadvantage.
      - **Laya goes the other way:** mean p +6.0 points, ECE 0.052 → 0.060, AUC 0.666 → 0.569.
      - **Jev, Qwen, Kev-0.8B:** mean within 0.3 points; AUC 0.712 → 0.692, 0.744 → 0.715, 0.750 → 0.743. v2 suited them slightly better.
      - **Sol** (test only; its tune run was skipped as unneeded): mean p 0.260 → 0.252, ECE 0.052 → 0.052, AUC 0.754 → 0.745. Five copies of the runner ran at once for a while after restarts (Windows kept the old retry scripts alive), so the records were deduplicated, keeping the first answer per customer; the raw log keeps every call.
      - Tune results match test closely. `results/checks/disguise_summary.json` has both.
  - **Review panel (2026-11-23):** four reviewers plus an advisor scored the work 6.5/10 as written and about 8 after fixes. The must-fixes are applied to the blog, carousel, X post and README. `analysis/robustness.py` holds the extra statistics.
  - **False-base-rate check** (`runners/checks/sol_false_rate.py`, user asked 2026-11-23, $0.46): 500 test customers, the base-rate sentence saying 10.0% instead of 22.1%.
    - **Sol follows it:** mean 0.258 → 0.175, a pass-through of 0.69 [0.64, 0.73]. 98% of customers went lower, and AUC was unchanged.
    - **So:** it listens to stated rates, and the true sentence barely moved it because its own estimate was already near 22%. Most likely that's from knowing this dataset.
    - **Status:** a diagnostic only, never a contestant condition. Details are in `data/contamination.md` §5.
- **Still to do for M4:** the author reviews the figures. Possible tweak: Sol's green is close to Kev-0.8B's teal on the honesty arc.

**M3 is done (2026-11-22, 19:16Z).** `runners/manifest.py` passes without `--allow-incomplete`: all eight contestants are complete, and `results/run_manifest.json` says `complete: true`.
- **Jev:** 8,000 records, $0.17 (`ee7afa9`).
- **GPT-6.2 Sol:** 8,000 records, $7.12, median latency 2.0 s. It ran unattended on the laptop (a Windows scheduled task, since deleted), paced at 18 requests/minute.
  - **Answers:** every one ended normally, no customer needed a retry, and no probability was exactly 0 or 1. Sol answers in whole percentages.
  - **Credit stop:** the run paused ~14:55–16:39Z when the OpenRouter credit ran out (402), and resumed after a top-up. Waits are never in the recorded latency.
  - **Rate limit:** OpenRouter caps new accounts at 20 requests/minute for this model. The runner paces at 18/min (`--rpm`), retries 429s with backoff, and keeps all waiting out of the recorded latency (it's in raw as `waited_for_rate_limit_s`).
- **GPU runs (L4): done (2026-11-22, 09:46–10:36Z).** Laya, Qwen3.5-4B, Kev-4B and Kev-0.8B, 8,000 records each, committed with their raw responses.
  - They ran one model at a time with sequential requests, so latencies aren't inflated by batching. Median latency: Laya 16 ms, Kev-0.8B 39 ms, Kev-4B 139 ms, Qwen 142 ms. Amortized GPU cost $0.02–0.16 per contestant.
  - No probabilities were exactly 0 or 1 in any run. Kev ran without flash-linear-attention.
  - The Qwen raw file is 64 MB (full prompt plus top-20 logprobs per request); GitHub warns above 50 MB.
  - **The L4 is deleted** (user, 2026-11-22 ~17:40Z). Everything it produced is in git, and M4's checks of the four GPU contestants found nothing to re-run. It had no `.env`, so no API key was on it. **Pending (user):** revoke the GitHub access it used: a personal access token, or the GitHub CLI OAuth app (that also signs out `gh` on the laptop).
- **Harness fix:** recalibrated records no longer count as a second model version, so a runner can resume a file after `recalibrate.py` has run.

**Next: M5, the write-up** (`writeup/` is empty). Blog, README, carousel and X drafts, per PLAN §7. Say wherever results are compared that the baselines learned from 20,000 labelled customers while the AI contestants are zero-shot, and that Sol recognizes the dataset.

**M2 is done (2026-11-22).** Baselines are scored on test, and all six AI runners passed their 20-customer smoke tests.
- **Built:** the baselines, all six AI runners, and `runners/manifest.py`, which checks coverage and writes `results/run_manifest.json`.
- **Smoke-tested (passed):**
  - Laya (CPU).
  - GPT-6.2 Sol and Jev (OpenRouter).
  - Kev-4B, Kev-0.8B and Qwen (L4).
- **`PREDICTIONS.md`** (the author's guesses, 2026-11-22): most honest Jev, least honest Laya, the base rate helps most contestants, and at least one AI contestant beats the scorecard's AUC.

### API runners (2026-11-22, CPU)

| Contestant | Script | Pinned to | Smoke test |
|---|---|---|---|
| `jev` | `runners/jev.py` (`uv run --group runners`) | asks once for `jev-latest`, then pins what it resolved to: `typesafe/jev-1.14-20261014`. Stops if a response comes from another version | 40 records, median latency 432 ms, $0.00085 total (~$0.17 for 8,000) |
| `frontier-verbal` | `runners/frontier_verbal.py` | `openai/gpt-6.2-sol`, effort medium, prompt v1 | 40 records, $0.04 total (~$8 for 8,000) |

- **IPv4 fix:** on the laptop network, OpenRouter's IPv6 addresses time out (21 s each), which added ~42 s to every fresh connection and would have been recorded as model latency. `harness.http_client()` forces IPv4 and keeps connections alive. It uses `httpx2`, the HTTP client `openai` 3.x is built on. The Sol smoke latencies predate the fix, so a few requests show 44–60 s; they aren't used anywhere.
- **Harness safeguards:** a predictions file can't mix model versions (the run refuses to start), and the run metadata keeps one entry per run, with its software environment and GPU.
- **Sol's version string** no longer contains the run date (each record's `ts` does). Its smoke file still has the old string, which is fine: smoke files aren't analysed.

M1 is done. The frontier-model memorization test found that GPT-6.2 Sol (the contestant) has memorized the start of the data file, as had Astra, tested before the switch. See the M1 section and `data/contamination.md`.

### Open-model runners (written 2026-11-22, on CPU)

All runners share `runners/harness.py`. It:
- feeds each model the identical rendered System One request,
- writes records through `analysis/records.py`,
- keeps raw responses in `<out>/raw/`,
- resumes where it stopped.

`--smoke` means 20 tune customers × both conditions, written to `results/smoke/`; that's the M2 check, and `tests/test_runners.py` validates it. `--split test` writes to `results/predictions/`. `cost_usd` for local models is the amortized GPU time (latency × $0.44/h L4 rate; $0 for Laya on a laptop CPU).

| Contestant | Script | Interface | Pinned to | Status |
|---|---|---|---|---|
| `kev-0.8b`, `kev-4b` | `runners/kev.py --size 0.8b\|4b` | local Kev server, `POST /v1/systemone` | v1.0 (`9a45d25e` / `139fdd94`); refuses to run if `/v1/models` doesn't show it | **smoke passed** on the L4 (2026-11-22; see below) |
| `laya` | `runners/laya_classic.py` (`uv run --group laya`) | in-process `laya.Agent(..., vegaml=False).predict` | `convaiinnovations/laya@fd4f9dbc` | **smoke passed** (CPU, ~1.1 s/request; 40 valid records) |
| `qwen-logprobs` | `runners/qwen_logprobs.py` (`uv run --group runners`) | vLLM + `jevper` `method="logprobs"` | `Qwen/Qwen3.5-4B@851bf6e8`, thinking off; P(yes) computed in the runner, not by jevper | **smoke passed** on the L4 (2026-11-22; see below) |

**Commands on the L4: follow [`L4_RUNBOOK.md`](L4_RUNBOOK.md).** It covers the Kev server setup, the M2 smoke tests, the gate before M3 (Jev smoke done and `PREDICTIONS.md` committed), the M3 GPU runs and the wrap-up, all as copy-paste commands.

**Kev v1.0 commits:** the `v1.0` tags point at `591dcb5b` (4B) and `788ddbdd` (0.8B), not at the README's `139fdd94` / `9a45d25e`. The weight files are byte-identical at both (checked 2026-11-22). `runners/kev.py` accepts any of the names.

**L4 smoke results (2026-11-22, 20 tune customers, actual rate 0.20):**

| Contestant | Mean p, out of box | Mean p, base rate | Range (out of box) | Latency |
|---|---|---|---|---|
| `kev-4b` | 0.55 | 0.30 | 0.34–0.70 | ~0.15 s |
| `kev-0.8b` | 0.36 | 0.52 | 0.19–0.66 | ~0.05 s |
| `qwen-logprobs` | 0.42 | 0.26 | 0.27–0.71 | ~0.15 s |

- No probabilities of exactly 0 or 1 anywhere. Kev-0.8B's base-rate sentence *raised* its predictions; with 20 customers that's only worth rechecking on the full run.
- Kev ran without flash-linear-attention (reference PyTorch kernel). It was fast enough, so it wasn't installed.

**jevper logprob bug (found in the Qwen smoke test, worked around):** jevper 0.7.12's readout strips whitespace from the top-20 tokens and lets the *last* match overwrite the rival label's logprob. A low-ranked `" A"` (−10.5) replaced `"A"` (−1.2), so every Qwen probability came out near 0 or 1 (e.g. 4e-5 where the model's own P(yes) was 0.32), and which extreme depended on the letter sampled at temperature 1.0. `runners/qwen_logprobs.py` now computes P(yes) itself from the recorded first-token top-20: each label's logprob is the logsumexp of all its variants, and the softmax is taken over the two labels, with the Yes letter read from the prompt's options. jevper's value is kept as `jevper_noul` in the raw record, and unit tests cover the exact case. The model version string now ends in `readout: label-variant logsumexp over first-token top-20`. **Reported upstream** (user asked, 2026-11-23): https://github.com/zhulinchng/jevper/issues/1, with a minimal reproduction. jevper 0.7.12 is still the latest release, so the bug is current.

**Qwen option-order check (2026-11-22, `runners/checks/qwen_order_swap.py`):** 200 tune customers, out of box, with jevper's exact request replayed as "A: No / B: Yes". Mean P(yes) shift −0.004, mean |shift| 0.048, correlation 0.91 (Spearman 0.91), no labels missing from the top 20; in the swapped order 2/200 answers were a word, not a letter. No letter bias, so the contestant stays single-order. The check is label-free and writes to `results/checks/`, not `results/predictions/`.

**Alternatives to jevper (researched 2026-11-22):** none is widely used or official for local models.
- **TypeSafe's official `typesafe-sdk`** (MIT, ~316 stars) only calls hosted Jev or gateways that speak TypeSafe's API (OpenRouter, Vercel, Pydantic). It has no local-model or logprob mode, so it's no replacement for jevper. It's an alternative route for Jev only: `runners/jev.py` already goes through OpenRouter and records the resolved version. TypeSafe's docs give the versioned IDs as `jev-1.14.0` / `jev-1.13.0`, and each response's `model` field names the version that answered.
- **Other adapters:** `ryanbeales/decision-proxy` (1 star, a proof of concept) sums variants the way our readout does, but floors missing labels at 1e-4. `ziyu/system-one-sdk` (2 stars) uses the model's self-reported probabilities, not logprobs. jevper (21 stars) is the best documented, and its docs don't describe the overwrite we hit.
- **Decision:** keep jevper for the prompt and transport and our own unit-tested readout for P(yes).

**Checks for the L4 smoke runs:**
- **Kev:** if `/v1/models` names the checkpoint without the revision, look at it and use `--skip-version-check` only after confirming it's the v1.0 weights.
- **Qwen logprobs:** the harness warns when more than 10% of probabilities are exactly 0 or 1. That would mean vLLM is returning processed rather than raw logprobs; check vLLM's `--logprobs-mode` (want raw).

**Laya engine:** since laya 0.5, `laya.Agent()` defaults to a different engine (vegaml, a frozen-LLM reader). The plan's contestant is the 421M classic encoder, so the runner asks for it with `vegaml=False` and refuses anything else.

**OpenRouter key:** it lives in the git-ignored `.env` in the repo root (`OPENROUTER_API_KEY=...`), loaded by `analysis/env.py` in every runner and in `memcheck.py`. On 2026-11-22 OpenRouter accepted it, with a $100 limit. On the L4, create the same `.env` in `/home/can-ai-know-when-its-wrong`.

**Langfuse tracing (on since 2026-11-22 ~19:00Z, confirmed working):** OpenRouter Broadcast sends every request made with keys in the OpenRouter **Default workspace** to the author's Langfuse project. The workspace was renamed after this project in the dashboard the same evening; renaming kept the key, the Broadcast destination and the usage history. The project key belongs to that workspace.
- **Setup lives in OpenRouter's dashboard** (Observability → Langfuse destination). No code, packages or Langfuse keys are in the repo.
- **Only requests since Broadcast was turned on are traced.** All Jev calls and Sol's first ~7,500 records aren't in Langfuse.
- **Recorded latency is unaffected:** OpenRouter sends traces after the response.
- **A viewing copy only.** Analysis still reads only `results/predictions/`.
- **Tagging a request:** add `session_id` and a `trace` object (`trace_name`, `generation_name`, `trace_id` as 32 hex characters, `environment`, any other keys become metadata) to the request body.
- **Don't delete the Default workspace** (it's still the Default one under its new name). OpenRouter says that disables every key not assigned to a workspace, including the project key, and deletes the Broadcast destination. Usage history can't be moved between workspaces; a key's workspace is fixed when the key is created.

**Frontier model: `openai/gpt-6.2-sol` at its default reasoning effort (medium)** (user, 2026-11-22). Switched from GPT-6.2 Astra for cost: Astra at default effort was estimated at $165–350, beyond the $100 key limit. Sol is also what ChatGPT starts new chats on, which matches the plan's "ask ChatGPT" framing.
- **Runner:** `runners/frontier_verbal.py` (`uv run --group runners`).
  - Prompt: the shared summary and question, plus one answer line: probability 0–100 as JSON (prompt v1).
  - Settings: strict JSON schema, `reasoning.effort="medium"` sent explicitly (equal to the default), `seed=0`. Sol accepts no temperature and no logprobs.
  - Records OpenRouter's real per-request cost (`usage.cost`). Retries malformed answers twice, then skips the customer for a later run rather than inventing a value.
- **Versioning:** responses name the model only as `openai/gpt-6.2-sol`, with no dated snapshot, so `model_version` includes the run date.
- **Smoke test (2026-11-22):** 40 valid records, $0.04 in total, no retries.
  - Usage: prompts 262 tokens on average; reasoning only 31 tokens on average (max 210).
  - Projection: about **$0.001 per request, ~$8 for all 8,000**.
  - First impression (20 customers, too few to judge): mean p 0.32 vs actual 0.20; the base-rate line barely moved it (0.31).
- **Prompt caching doesn't apply.** OpenAI caches only prompts of 1,024+ tokens, ours are ~262, and on GPT-5.6+ cache writes cost 1.25× input. The harness already skips finished customers, so re-runs never pay twice.

### M2 so far (2026-11-22, on CPU)

| Contestant | Condition | Test AUC | Test Brier | Test ECE | Mean p vs actual |
|---|---|---|---|---|---|
| `scorecard` (WoE + logistic regression) | `trained` | 0.777 | 0.1365 | 0.014 | 21.8% vs 22.1% |
| `gbm` (LightGBM, 157 rounds) | `trained` | 0.789 | 0.1319 | 0.017 | 21.8% vs 22.1% |
| `gbm` + isotonic fitted on tune | `trained_calibrated` | 0.784 | 0.1328 | 0.020 | 21.4% vs 22.1% |

- **Scripts:** `baselines/scorecard.py` and `baselines/gbm.py` write `results/predictions/{scorecard,gbm}.jsonl`, plus `results/baselines/*.json` with the bin table, coefficients, settings and feature importances. Each takes under 10 seconds, and tests check that a re-run reproduces the saved predictions exactly.
- **Calibration:** isotonic calibration didn't help the GBM. Raw LightGBM is already well calibrated, and 2,000 tune customers add noise. Report this as found.
- **Scorecard sign flips:** 4 of the 19 WoE coefficients are negative (BILL_AMT1 −1.27, BILL_AMT3 −1.48, PAY_2, BILL_AMT5). Checked on train and tune only (test untouched):
  - Default rate across bill-amount bins is flat (19–25%), so bills have no risky direction of their own and act as adjustments given the other fields.
  - The two big negative coefficients are stable: negative in 200/200 bootstrap refits. Only BILL_AMT5 (−0.12) is unstable (134/200).
  - WoE correlation between months is 0.27–0.69 (raw bills: 0.80–0.95).
  - A sign-consistent scorecard drops all six bill amounts plus PAY_2 (12 fields left): 5-fold CV AUC 0.767 vs 0.770, Brier 0.1367 vs 0.1362, ECE 0.013 vs 0.015. Same within noise.

  **Decided: keep all 19 fields** (see "Decisions already made").
- **Shared record format:** every contestant, AI runners included, writes through `analysis/records.py` (`make_record` and `write_records`), which validates the CLAUDE.md schema. `recalibrated` and `trained_calibrated` records may only be written for `test`, since their calibrator was fitted on tune.
- **For M4:** 7 isotonic test probabilities are exactly 0 or 1 (all correct). Clip probabilities before log-loss or logit-based metrics such as the calibration slope.

### M1

- `data/load.py`: fetches UCI 350 through `ucimlrepo`, drops SEX/MARRIAGE/AGE/EDUCATION before saving, and records the version, a content hash and quirk counts in `data/raw/source.json`.
- `data/splits.py`: stratified 20,000 / 2,000 / 2,000 / 6,000 split, seed 20261122. The default rate is 22.10–22.13% in every split and 22.12% for train.
- `data/render.py`: template **v2**. `data/rendered/{tune,test}.jsonl` hold ready-to-send System One requests for the `out_of_box` and `base_rate` conditions.
- `README.md` has the licence, citation and data quirks; `data/contamination.md` has the per-contestant contamination analysis.
- Tests: `uv run pytest`, 71 passing (data, records and baselines). The parked insurance work has its own 67 tests: `uv run pytest parked/insurance`.

- **Qwen memorization test: done (2026-11-22).** Qwen3.5-4B, chat and raw-completion modes, both file versions: 0 of 100 rows reproduced (98 of them non-trivial) and no non-trivial header continuation. Details and the results table are in `data/contamination.md`.

- **Frontier memorization test: done (2026-11-22), memorization found.** Details are in `data/contamination.md`.
  - **GPT-6.2 Sol (the contestant):** chat mode, default effort, about $1. It continued the file verbatim for every character it wrote: customers 8–13 (`uci`) and 2–6 (`kaggle_style`), labels included. Row completion: 0 of 50 exact, with 42 refusals.
  - **GPT-6.2 Astra (tested before the switch; not a contestant):** customers 3–10 verbatim. Row completion: 0 of 50 exact, with 19 refusals; its other answers match no better than guessing each column's most common value.
  - **Reproduced customers we use:** `cust_00004` (test) and `cust_00005`–`cust_00007` (tune). Sol reproduced 4–6, Astra 4–7.

**Sol's memorization follow-ups (user chose all three, 2026-11-22): done.** Results are under M4 above and in `data/contamination.md`.

## Decisions already made (don't revisit without the user)

- **Insurance set parked** in `parked/insurance/` for a later post.
- **Never disadvantage the AI contestants.** Template v2 dropped the "common interpretation" of repayment codes because the data contradicts it (e.g. 40.6% of "−2 = no balance" months carry a positive bill). Labels now say only what's documented and always show the raw code, for example "not late (code -2)" and "paid on time (code -1)". Use train data only for checks like this.
- **Base-rate condition:** the sentence "In this portfolio, about 22.1% of customers miss next month's payment." goes in front of the question instructions. The `state` stays identical across conditions.
- **Recalibrated condition** is computed from the saved `out_of_box` probabilities, so it needs no new API calls.
- **Scorecard keeps all 19 fields** despite 4 negative WoE coefficients (user, 2026-11-22). The textbook sign rule's premises don't hold here, a sign-consistent version performs the same within noise, and all 19 fields keep its inputs identical to every other contestant's. The write-up gets one line: "A production-style scorecard with consistent signs would drop all six bill amounts; it ranks and calibrates the same within noise."
- **`qwen-logprobs` contestant = `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`** (user chose it over Qwen3-4B-Instruct-2507, 2026-11-22). It's the newest ~4B Qwen, Apache-2.0. Run it text-only with thinking off (`enable_thinking=false`) so P(yes) comes from the first answer token's logprobs.
- **Sol's memorization:** run all three follow-ups (longer header test, with/without table, recognition probe), flag Sol's results everywhere, and keep the main results on the full test set (user, 2026-11-22).
- **`frontier-verbal` contestant = `openai/gpt-6.2-sol` at default reasoning effort (medium)** through OpenRouter (user, 2026-11-22; switched from Astra for cost). Astra's memorization result stays in `data/contamination.md` as context about the OpenAI family.

## Things to know for M2 (runners)

- System One `noul` answers already give P(yes); use that as `p_default`. The rescaled `confidence` formula only applies to `choice` questions.
- English Laya (`convaiinnovations/laya` @ `fd4f9dbc`) has a 512-token context, with about 320 tokens left for the state. Summaries are 192 tokens at most. The card says base Laya is near chance zero-shot on its own benchmark.
- Kev: model cards at github.com/jaredpalmer/kev (checked at commit `65b1e763`). Kev-4B in bf16 is about 12.5 GB; the README only lists L40S/H100 as tested, so fall back to OpenRouter if the L4 runs out of memory.
- Jev on OpenRouter: on 2026-11-22 the model list showed only `typesafe/jev-router`, which picks a model and reasoning effort per request. Don't use the router. Pin a version (e.g. `jev-1.14`) through OpenRouter's System One endpoint (`POST /api/v1/systemone`, which maps bare `jev-*` IDs onto `typesafe/`), and record the resolved version from each response.
- GPU time on the L4 is billed per minute. Do CPU work (baselines, runner code, API runs) wherever it's cheapest and keep the GPU for serving open models.

## The L4 instance (JarvisLabs, launched 2026-11-22, deleted 2026-11-22)

**Deleted after the M3 GPU runs.** For M6 or any new GPU work, launch a fresh instance and follow **L4_RUNBOOK.md §6.0**, the current from-zero setup: base tools, `gh`, the project and Kev's environment. The "First-time setup" below is the M1 version, kept for the record: it predates the public repo and the M6 groups.

1× NVIDIA L4 (24 GB), PyTorch template (a container), region IN2, 28 cores and 124 GB RAM. Compute bills per second; storage keeps billing while the instance is paused.

**Only `/home` survives a pause.** Anything installed elsewhere is gone after a resume: system packages, `~/.cache`, a `gh` login. So keep the repo, uv, its Python builds and every model cache under `/home`.

### First-time setup (M1, kept for the record: don't use it)

**Don't run this on a new instance.** It has no dependency groups, so tests that need `torch` or `jevper` fail. Use the setup in the newest L4_RUNBOOK.md section (§9.0 for post 5's addendum), or §6.0.

```bash
cat > /home/env.sh <<'EOF'
export HF_HOME=/home/hf-cache                  # model weights (~25 GB), survive pauses
export UV_CACHE_DIR=/home/uv-cache
export UV_PYTHON_INSTALL_DIR=/home/uv-python   # venvs link to this Python, so it must persist
export PATH=/home/bin:$PATH
EOF
source /home/env.sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/home/bin sh

# The repo is private: authenticate first (`gh auth login`, installing gh if missing,
# or use a GitHub personal access token), then clone it under /home.
cd /home && git clone https://github.com/nishzsche/can-ai-know-when-its-wrong.git
cd can-ai-know-when-its-wrong
uv sync --group memcheck
uv run pytest                     # 41 passing; the token test downloads two tokenizers
export OPENROUTER_API_KEY=...     # never commit keys; repeat in each new shell
```

### After every resume or new shell

Run `source /home/env.sh`, then redo anything that lived outside `/home`, such as `OPENROUTER_API_KEY`.

**GitHub:** `gh` 2.105.0 is installed at `/home/bin/gh`. `env.sh` sets `GH_CONFIG_DIR=/home/gh-config`, so the `gh` login (and its token) survives pauses, and this repo's local git config uses `gh` as the credential helper for github.com. `git push` works only in a shell that has sourced `env.sh`. Log in again with `gh auth login -h github.com -p https -w` if you delete `/home/gh-config`.

### Running models on the GPU

- **One model at a time.** vLLM takes ~90% of GPU memory by default, so stop it (or start it with `--gpu-memory-utilization`) before starting a Kev server.
- **Memory fits:**

  | Model | GPU memory |
  |---|---|
  | Kev-4B | 14.3 GB resident, ~16 GB peak while its adapter loads |
  | Kev-0.8B | 3.8 GB |
  | Laya | well under 2 GB |
  | Qwen ~4B chat model on vLLM | ~8 GB of weights plus cache |

- **Kev-4B speed:** Kev's README calls the L4 "too slow for Kev-4B" when serving production traffic. Our job is ~8,000 short requests (4,000 customers × 2 conditions), which is fine. Fall back to OpenRouter only if it actually fails.
- **vLLM lives in its own venv, `/home/vllm-env`** (vLLM 0.34.0, torch 2.14.0+cu130). It's kept out of the project lockfile so CPU development stays light. To recreate it: `uv venv /home/vllm-env --python 3.12 && VIRTUAL_ENV=/home/vllm-env uv pip install vllm`. Qwen3.5-4B weights are in `/home/hf-cache` (8.8 GB).
- **Serving Qwen3.5-4B** (~18.5 GB used at 0.85 utilization, up in ~1–2 min). This is the exact command used for the memcheck; use the same one for predictions:
  ```bash
  /home/vllm-env/bin/vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --served-model-name qwen3.5-4b --host 127.0.0.1 --port 8000 --max-model-len 16384 \
    --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization 0.85 --seed 0 > /home/logs/vllm-qwen.log 2>&1
  ```
  Stop it with `pkill -f "vllm serve"`; run that on its own, because inside a compound command it also kills the calling shell.
- **The Qwen memcheck (the first GPU job) is done.** No GPU work is pending until the M2 runners are written and tested on CPU/APIs.

### Money

- **Wallet:** if the JarvisLabs wallet runs out of funds, all resources are released, **storage included** (per JarvisLabs' FAQ). Keep it topped up.
- **Autopause:** it works on a timer or a balance threshold, not on idle detection. Set it, and pause whenever you step away.
