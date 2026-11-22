# Contamination and memorization check

Written 2026-11-22, before any contestant runs. Re-check every model card at run time (M3) and update this file if anything changes.

## Why it matters

The UCI credit default data has been public since 2009. It's mirrored on Kaggle (as `UCI_Credit_Card.csv`) and appears in countless tutorials and notebooks, so it's almost certainly in the pretraining data of general-purpose LLMs. LLMs are known to memorize popular tabular datasets (Bordt et al., "Elephants Never Forget", 2024). Each CSV row ends with the default label, so a model that has memorized rows could recall answers instead of estimating risk.

What limits the risk here:
- Contestants never see the CSV. They see our own text rendering (`data/render.py`), with no customer ID, no demographic fields and no label.
- To use a memorized row, a model would have to match our summary back to the original row. A strong LLM could plausibly do that, so it isn't ruled out.

## What each contestant could have seen

| Contestant | Can it be probed with tabmemcheck? | What's known about its training data | Could it have seen this dataset? |
|---|---|---|---|
| `scorecard`, `gbm` | Not needed | Trained only on our `train` split | Only `train`, by design |
| `qwen-logprobs` (Qwen3.5-4B) | Yes: **no memorization detected** (2026-11-22, see below) | Qwen's pretraining data isn't disclosed | **Possibly**: public web data, though the tests found no trace of the file |
| `frontier-verbal` (GPT-6.2 Sol) | Yes: **memorized the start of the file** (2026-11-22, see below) | Not disclosed | **Yes**: it reproduced the start of the published file verbatim (customers 2–6 and 8–13 across the two file versions), labels included. GPT-6.2 Astra, tested before the switch to Sol, did the same (customers 3–10) |
| `jev` | No: decision model, doesn't generate text | Not disclosed. Third-party write-ups say TypeSafe describes its data only as built in-house; no model card | **Unknown** |
| `kev-0.8b`, `kev-4b` | No: decision models | Listed in the model cards (see below). Bases are Qwen3.5-0.8B-Base and Qwen3.5-4B-Base, whose pretraining data isn't disclosed | **Not in Kev's fine-tuning data; possibly in the Qwen3.5 base pretraining** |
| `laya` | No: decision model | The card describes the training method (RLCD) but not the training data. Backbone is ModernBERT-large, pretrained on a large web and code corpus | **Unknown**: possibly via backbone pretraining |

### Kev training data

From `docs/model-cards/kev-0.8b.md` and `kev-4b.md` at github.com/jaredpalmer/kev commit `65b1e763` (2026-11-20). Both cards list the same sets.

- **Base recipe `decision-v7`:** 10,000 records, 1,000 each from banking77, boolq, ag_news, multi_nli, sst5, yelp_review_full, trec, dbpedia_14, amazon_reviews_multi_en and imdb. Plus 896 generated policy examples and 1,680 records from generated rule structures.
- **Follow-up fine-tunes:**
  - `documents-v1`: 5,219 CFPB consumer-finance complaint narratives.
  - `hard-v1`: generated skill records.
  - `devtools-v1`: CodeReviewer, CommitPackFT, FlakeFlagger and Aegis.
- **Evaluation-only suites** (breadth-v1, tasksource-heldout-v1, transfer-v4, longdoc-v1 with CUAD) are never trained on.
- **Takeaway:** no tabular credit data and no UCI credit default data. The CFPB complaints are in the same domain (consumer credit), which counts as familiarity with the domain, not exposure to this dataset.

### Laya

Card at huggingface.co/convaiinnovations/laya, revision `fd4f9dbc` (2026-11-19). The English checkpoint has a 512-token context with about 320 tokens for the state, and our summaries fit (192 tokens at most). The card also says base checkpoints are near chance on its own typed-decisions benchmark zero-shot ("a fast base to specialise, not a zero-shot decision engine"), which matters for predictions but isn't contamination.

## LLM memorization tests (tabmemcheck)

**Status:**
- **Qwen3.5-4B:** done (2026-11-22). No memorization detected.
- **GPT-6.2 Sol** (the frontier contestant since the user switched from Astra for cost, 2026-11-22): done (2026-11-22). Memorized the start of the file; no recall of rows deeper in the file.
- **GPT-6.2 Sol follow-ups** (user chose all three, 2026-11-22): all done. The depth test found 31 customers memorized, 3 of them in test. Without them, no metric moves by more than 0.0006. The recognition probe found that Sol identifies every summary as this dataset and knows its exact default rate (22.12%). See "Follow-ups for GPT-6.2 Sol" below.
- **GPT-6.2 Astra** (the user's first choice, tested before the switch; not a contestant): done (2026-11-22). Same pattern. Kept as context about the OpenAI family.

What runs, for each LLM contestant, on two versions of the file:

1. **Header test:** given the first rows of the file, cut partway through a row, can the model continue it exactly? The reference is the real file's continuation: 500 characters, or as long as the model's answer if that's longer (before a fix on 2026-11-22 the script capped this at 500, which is why the Astra result file records "500/500").
2. **Row-completion test:** given 10 consecutive rows, can it reproduce the next row exactly? 25 queries.

File versions:
- `uci`: the CSV exactly as UCI serves it (`ID,X1,…,X23,Y`; sha256 `45bcf4df…3dd0`).
- `kaggle_style`: the same rows with the header names and float formatting of the widely mirrored Kaggle copy. This is our reconstruction of that format, not a download of the Kaggle file.

The tests send rows of the published file, including its demographic columns, because the question is whether the file itself was memorized. These rows are never a model input for the experiment.

How to read the results:
- **Strong evidence:** an exact header continuation longer than about one row (roughly 60–90 characters), or exact matches on non-trivial rows. "Non-trivial" means at least one non-zero bill or payment; all-zero rows can be guessed without memorization.
- **Absence isn't proof:** zero matches doesn't prove a model never saw the data. Chat models may reformat or refuse, and these query-only tests can miss memorization (Bordt et al. say so too).

How to run:
- Through OpenRouter: `uv run --group memcheck python data/memcheck.py --model <openrouter-id>`
- The Qwen contestant: OpenRouter doesn't list a ~4B Qwen, so test it on the L4 against the vLLM server, as the first job of the GPU session and before any Qwen prediction runs. No API key is needed on localhost: `--base-url http://127.0.0.1:8000/v1 --model <served-name>`.

Results go in `data/memcheck/results/<model>.json` (raw-completion runs: `<model>__completion.json`).

| Model | Mode | File version | Header: exact prefix (chars matched / compared) | Rows: exact matches (non-trivial / non-trivial rows) | Verdict |
|---|---|---|---|---|---|
| Qwen3.5-4B | chat | uci | 6 | 0 (0/24) | No memorization detected |
| Qwen3.5-4B | chat | kaggle_style | 0 | 0 (0/24) | No memorization detected |
| Qwen3.5-4B | raw completion | uci | 2 | 0 (0/25) | No memorization detected |
| Qwen3.5-4B | raw completion | kaggle_style | 15 (a run of `0.0` fields) | 0 (0/25) | No memorization detected |
| GPT-6.2 Sol | chat | uci | **622 / 622**: everything it wrote matches (customers 8–13) | 0 (0/25); 20 refusals | **Memorized the start of the file**, not random rows |
| GPT-6.2 Sol | chat | kaggle_style | **764 / 764**: everything it wrote matches (customers 2–6) | 0 (0/24); 22 refusals | **Memorized the start of the file**, not random rows |
| GPT-6.2 Astra (not a contestant) | chat | uci | **701 / 701**: everything it wrote matches (the rest of customer 3, then customers 4–10) | 0 (0/25); 9 refusals | **Memorized the start of the file**, not random rows |
| GPT-6.2 Astra (not a contestant) | chat | kaggle_style | **669 / 669**: everything it wrote matches (the rest of customer 3, then customers 4–8) | 0 (0/25); 10 refusals | **Memorized the start of the file**, not random rows |

**Qwen3.5-4B run details:** `Qwen/Qwen3.5-4B` at revision `851bf6e8`, served by vLLM 0.34.0 on the L4 with `--language-model-only` and thinking turned off (`enable_thinking=false`, the same setting the contestant uses). tabmemcheck 0.1.6. We ran both modes: chat, which is how the contestant is prompted, and raw `/completions` with no chat template, which is the stronger probe because the model can't reformat or refuse.

**What the responses look like:** in both modes the model writes well-formed CSV rows with the right number of fields and copies the row ID from the prompt. The values are invented: round limits such as 100000 and 150000, bills repeated across months, repayment codes mostly 0. That shows the model knows the format, not the rows. In the raw-completion Kaggle-style header test it repeated customer 1's values (`20000.0,2,2,1,24,2,2,-1,-1,-2,-2,3913.0,3102.0`) under ID 2. Customer 1 was in the prompt (the test cut the file partway through row 1), so this is copying from the prompt, not recall; the real row 2 never appeared.

**Relevance to Kev:** Kev's bases are Qwen3.5-0.8B-Base and Qwen3.5-4B-Base. A clean result on the instruct Qwen3.5-4B is weak, indirect evidence that the shared pretraining didn't fix this file in memory. It doesn't test the base checkpoints or Kev itself.

**GPT-6.2 Sol run details:** `openai/gpt-6.2-sol` through OpenRouter on 2026-11-22T08:51Z, chat mode, default reasoning effort (medium), tabmemcheck 0.1.6, scored with the fixed reference window (no 500-character cap). Cost about $1.
- **Header test:** tabmemcheck cuts the file at a random row and keeps the best of four attempts, so the two file versions started at different places. In both, Sol's continuation matched the real file for every character it wrote: 622 characters (customers 8–13, `uci`) and 764 characters (customers 2–6, `kaggle_style`), default labels included.
- **Row completion:** none of 50 exact. 42 of the 50 answers weren't CSV rows (refusals), so this test says little about deeper rows for Sol.
- **Customers we use:** `cust_00004` (test) and `cust_00005`, `cust_00006` (tune) were reproduced by Sol. Astra also reproduced `cust_00007` (tune). Customers 2, 3, 8 and 10 are in reserve; 9 and 11–13 in train.

**GPT-6.2 Astra run details** (tested before the switch to Sol; not a contestant): `openai/gpt-6.2-astra` through OpenRouter on 2026-11-22. Responses name no dated snapshot. Chat mode only, since there's no raw-completion endpoint. Reasoning can't be turned off, so it ran at the default effort (medium). tabmemcheck 0.1.6 sends `temperature=0`, which Astra doesn't list as supported. A 3-query probe first checked that reasoning wouldn't use up tabmemcheck's 1,000-token answer cap: it used 7–9 reasoning tokens per query and no answer came back empty. Cost about $3 (about 5,400 prompt tokens per query, mostly tabmemcheck's few-shot examples).

**What Astra's responses show:**
- **Header test:** the prompt holds the file header and customers 1–2, cut partway through customer 3. Astra's continuation matches the real file for every character it wrote, default labels included. None of that text is in the prompt, so this is recall, not copying. Bordt et al. treat it as evidence that the model saw the file in training. The first rows are also the most widely copied part of the file (every `df.head()` in a tutorial shows them), so this shows exposure to the file, not knowledge of all 30,000 rows.
- **Row-completion test:** it refused 9 of 25 (`uci`) and 10 of 25 (`kaggle_style`) queries, giving the next ID and saying it "can't determine its remaining values". Its 31 attempted rows match 25–28% of the 24 non-ID fields on average (max 58%). Always guessing each column's most common value matches 32%, so these answers are no better than guessing typical values. There's no sign of recall for rows deeper in the file, but with 38% refusals this test is weaker for Astra than it was for Qwen.

**What it means for the experiment:**
- **Direct leaks look tiny.** Of the customers Sol and Astra reproduced, `cust_00004` is in test and `cust_00005`–`cust_00007` are in tune (2, 3, 8 and 10 are in reserve; 9 and 11–13 in train). One test customer out of 2,000 can't move the metrics.
- **How far the memorized start goes:** measured for Sol by the depth test below. It knows customers 2–20 and most of 21–33 verbatim, which adds test customers `cust_00016` and `cust_00018`. (Its earlier stops at 622 / 764 characters were most likely tabmemcheck's 1,000-token cap, which includes reasoning; with room it wrote 993 and 1,633 matching characters.)
- **The model never sees customer IDs**, only our rendered summaries. Still, rows it knows by heart could be recognizable from their numbers.
- **Recognition is the bigger risk, and it's confirmed for Sol** (recognition probe below). The summaries show NT$ amounts and April–September 2005. Sol named the dataset for 53 of 53 summaries and gave its exact default rate (22.12%) every time. That can flatter its out-of-the-box calibration, since the base-rate condition gives every other contestant the same number.
- **Decided (user, 2026-11-22): all three follow-ups.** See the next section. Per the rule below, Sol's results are flagged wherever they appear either way.

## Follow-ups for GPT-6.2 Sol

### 1. Depth test (`data/memdepth.py`, done 2026-11-22)

**What it does:** a longer, deterministic header test.
- **Cuts:** the file is cut partway through customer k, for k = 2, 12, 22, … (seeded cut points), in both file versions.
- **Prompt:** tabmemcheck's header-test format: its system prompt, one few-shot example from each of its default CSV files cut at the same offset, then our file up to the cut.
- **Length:** few-shot continuations are 1,500 characters, about 18 rows, so the model is shown that a long continuation is wanted.
- **Budget and settings:** 16,000 tokens, so reasoning can't cut the answer short; every answer records why it ended. Settings are the contestant's: effort medium, seed 0.
- **What counts as memorized:** a customer whose whole row (all 25 fields, label included) the model writes exactly without it being in the prompt; or a cut row it completes exactly, when the completed part contains a non-zero bill or payment.
- **Stopping rule:** the walk stops after 3 cuts in a row with nothing memorized.

**Result:** 20 queries, $1.14, every answer ended on its own (none hit the budget). Results are in `data/memcheck/results/openai_gpt-6.2-sol__depth.json`.

| File version | Cuts | Memorized (customer IDs) | Notes |
|---|---|---|---|
| `uci` | 2–62 | 2–20, 22, 23, 25–27, 29, 32 | Whole rows up to customer 29 and the completion of cut row 32. From cut 42 on, only the cut row's completion: wrong at 42, 52 and 62 |
| `kaggle_style` | 2–122 | 2–20, 22, 24–27, 29, 33, 42, 72, 92 | Whole runs to 33. From cut 42 on, Sol completed only the cut row and stopped: exact at 42, 72 and 92, wrong at 52, 62, 82, 102, 112 and 122 |

- **Union** (with the original header test): **31 customers**, 2–20, 22–27, 29, 32, 33, 42, 72 and 92.
- **Customer 92 is the weakest case:** its completed part's only non-zero value (3,968) also appears among its bills in the prompt, so it could have been inferred. Customer 72's completion (an exact bill and five exact payments) and customer 42's (including a payment of 4,042) can't be guessed.
- **How to read the deep end:** past customer ~40 Sol only completes the row in front of it, so the test measures single-row recall there. It got 2 of 9 rows right in the Kaggle-style file and 0 of 3 in the UCI file. Memory is dense for the first ~33 customers (the part shown in tutorials' `df.head()`-style printouts) and sporadic after that. Rows deeper than ~120 weren't reached; tabmemcheck's row-completion test (0 of 50 deeper rows, 42 refusals) is the only probe of them.
- **In our evaluation sets:** test `cust_00004`, `cust_00016`, `cust_00018` (3 of 2,000); tune `cust_00005`–`cust_00007` (3 of 2,000). The rest are in train or reserve.

### 2. Results with and without the memorized customers

`analysis/summarize.py` reads the memorized list from the depth result and writes `results/summary_excluding_memorized.csv`.
- **Same customers for everyone:** every contestant is re-scored on the same 1,997 test customers, so the comparison stays like-for-like.
- **Recalibrated rows:** they keep their calibrators, which were fitted on all of tune, including 3 memorized tune customers out of 2,000.

**Result (2026-11-23):** removing the 3 memorized test customers changes every contestant's AUC, ECE and calibration-in-the-large by at most 0.0006, Sol included. The leaked rows don't move the results.

### 3. Recognition probe (`data/recognition.py`, done 2026-11-22T19:20Z)

**What it asks:**
- **Recognition:** the exact `state` Sol saw as a contestant, followed by: "Where does this account summary most likely come from? If you think it comes from a specific public dataset, name it, and give the share of that dataset's customers who miss next month's payment if you know it." Strict JSON.
  - **Customers:** 50 random test customers (fixed seed) plus the 3 memorized test customers.
  - **Leading by design:** the question mentions public datasets, so it measures whether Sol *can* place a summary when asked.
- **Knowledge:** "In the UCI 'Default of Credit Card Clients' dataset (Yeh & Lien, 2009), what share of customers miss next month's payment?", asked 3 times.
- **Settings:** the contestant's (effort medium, seed 0). Cost $0.07. Results are in `data/memcheck/results/openai_gpt-6.2-sol__recognition.json`.

**Result: Sol recognizes the dataset from any summary and knows its default rate exactly.**
- **Recognition:** 53 of 53 summaries were named as UCI "Default of Credit Card Clients", at 99% confidence, all 50 random customers included. Every answer gave the default rate as 22.12%, the published file's exact rate (6,636 / 30,000).
- **Cues it cited:** the April–September 2005 billing period, the NT$ amounts and the repayment codes. The median answer used no reasoning tokens.
- **Knowledge:** 22.12%, 3 of 3.

**What it means for the comparison:**
- **Not fully zero-shot:** Sol's out-of-the-box condition isn't zero-shot in the sense that matters for calibration. When asked, it identifies the population and knows its default rate, the number the `base_rate` condition gives everyone.
- **Its predictions don't simply recite that rate:**
  - The out-of-the-box mean prediction is 0.260 (test; actual 0.221).
  - The base-rate sentence barely moves it (0.253). By comparison, it moves Jev 0.48 → 0.38, Kev-4B 0.49 → 0.20 and Qwen 0.36 → 0.25.
- **Two readings this probe can't separate:** either Sol already uses the rate it knows, or it ignores the sentence.
- **The disguise check tried to separate them and couldn't** (next section): Sol recognizes the dataset even without the cues.
- **For the write-up:** Sol's out-of-the-box calibration has to be read as "a model that recognizes this dataset and knows its default rate", not as zero-shot honesty. Its ranking (AUC) is less affected: knowing the overall rate doesn't rank individual customers, and the leaked rows don't change its AUC.

### 4. Disguise check (`runners/checks/sol_disguise.py`, recognition step done 2026-11-23)

**Idea:** remove the cues Sol cited, then compare its out-of-the-box predictions with the original run on the same test customers. If recognition drives its calibration, the disguised run should be worse calibrated.

**Disguise d1:** the rest of the summary is unchanged.
- **Currency:** "NT$" is dropped, plus one line: "Amounts are in local currency units."
- **Months:** "Sep 2005"…"Apr 2005" become "Month 6"…"Month 1".
- **Repayment codes:** raw numbers removed. Codes −2 and 0 stay distinct as "not late (type A)" and "(type B)".
- **Real context goes too:** this also removes Taiwan and 2005, so any change would measure recognition and context together.

**First step, does the disguise hide the dataset? No.**
- **Probe:** the recognition question on the same 50 random test customers as the original probe, disguised. Cost $0.07. Results are in `results/checks/sol_disguise_recognition.json`.
- **Result:** Sol still named the dataset for 50 of 50, at 99% confidence, and gave the rate as 22.12% every time.
- **What it cites now:** the structure itself: "six months of bill amounts, payments, and repayment status".

**What follows:**
- **No faithful disguise works:** any faithful rendering of this data has that structure, so any disguise that keeps the information intact won't stop recognition. One that changes the structure (fewer months, aggregated amounts) would also change the information, and the comparison would no longer be clean.
- **The full run wasn't done:** the 2,000-customer disguised run (about $1.50) can't measure what recognition contributes, since Sol still recognizes the data. It would only show whether the surface cues change its numbers. The script can still run it (`--workers 4`, then `--report`).
- **For the write-up:** Sol identifies this dataset from the bare structure of a six-month credit-card history, not just from its currency or dates.

## What the write-up will say

- The scorecard and GBM were trained only on our `train` split.
- Jev, Kev and Laya can't be probed for memorization because they don't generate text. Kev's disclosed fine-tuning data doesn't include this dataset. The pretraining data of every base model is undisclosed, so exposure can't be ruled out for any AI contestant.
- The LLM contestants' tabmemcheck results get reported as measured. If a contestant shows memorization, its results are flagged wherever they appear.
- Qwen3.5-4B showed no memorization in either chat or raw-completion mode: 0 of 100 rows reproduced across both file versions (98 of them non-trivial), and no header continuation beyond trivial runs of zeros. This is absence of evidence, not proof.
- GPT-6.2 Sol, the frontier contestant, has seen this dataset in training. It reproduced the start of the data file verbatim: 31 customers in all, dense through customer 33 with sporadic hits to 92. Three of them are in our 2,000-customer test set; without them, no metric moves by more than 0.0006. More importantly, Sol recognizes every account summary as this dataset and knows its exact default rate (22.12%), so its out-of-the-box calibration is not zero-shot honesty: it is a model that knows the population it's being asked about. It reproduced none of 50 rows from deeper in the file, though it refused 42 of those queries. Its results are flagged as coming from a model that has seen this data.
- GPT-6.2 Astra, OpenAI's flagship, which we tested before switching to Sol for cost, showed the same: customers 3–10 verbatim, none of 50 deeper rows (19 refused).
