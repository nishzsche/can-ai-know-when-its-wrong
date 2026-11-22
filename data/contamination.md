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
| `frontier-verbal` (GPT-6.2 Astra) | Yes: **memorized the start of the file** (2026-11-22, see below) | Not disclosed | **Yes**: it reproduced the start of the published file verbatim (the rest of customer 3, then customers 4–10), labels included |
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
- **GPT-6.2 Astra** (the frontier contestant, chosen by the user 2026-11-22): done (2026-11-22). Memorized the start of the file; no recall of rows deeper in the file.

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
| GPT-6.2 Astra | chat | uci | **701 / 701**: everything it wrote matches (the rest of customer 3, then customers 4–10) | 0 (0/25); 9 refusals | **Memorized the start of the file**, not random rows |
| GPT-6.2 Astra | chat | kaggle_style | **669 / 669**: everything it wrote matches (the rest of customer 3, then customers 4–8) | 0 (0/25); 10 refusals | **Memorized the start of the file**, not random rows |

**Qwen3.5-4B run details:** `Qwen/Qwen3.5-4B` at revision `851bf6e8`, served by vLLM 0.34.0 on the L4 with `--language-model-only` and thinking turned off (`enable_thinking=false`, the same setting the contestant uses). tabmemcheck 0.1.6. We ran both modes: chat, which is how the contestant is prompted, and raw `/completions` with no chat template, which is the stronger probe because the model can't reformat or refuse.

**What the responses look like:** in both modes the model writes well-formed CSV rows with the right number of fields and copies the row ID from the prompt. The values are invented: round limits such as 100000 and 150000, bills repeated across months, repayment codes mostly 0. That shows the model knows the format, not the rows. In the raw-completion Kaggle-style header test it repeated customer 1's values (`20000.0,2,2,1,24,2,2,-1,-1,-2,-2,3913.0,3102.0`) under ID 2. Customer 1 was in the prompt (the test cut the file partway through row 1), so this is copying from the prompt, not recall; the real row 2 never appeared.

**Relevance to Kev:** Kev's bases are Qwen3.5-0.8B-Base and Qwen3.5-4B-Base. A clean result on the instruct Qwen3.5-4B is weak, indirect evidence that the shared pretraining didn't fix this file in memory. It doesn't test the base checkpoints or Kev itself.

**GPT-6.2 Astra run details:** `openai/gpt-6.2-astra` through OpenRouter on 2026-11-22. Responses name no dated snapshot. Chat mode only, since there's no raw-completion endpoint. Reasoning can't be turned off, so it ran at the default effort (medium). tabmemcheck 0.1.6 sends `temperature=0`, which Astra doesn't list as supported. A 3-query probe first checked that reasoning wouldn't use up tabmemcheck's 1,000-token answer cap: it used 7–9 reasoning tokens per query and no answer came back empty. Cost about $3 (about 5,400 prompt tokens per query, mostly tabmemcheck's few-shot examples).

**What Astra's responses show:**
- **Header test:** the prompt holds the file header and customers 1–2, cut partway through customer 3. Astra's continuation matches the real file for every character it wrote, default labels included. None of that text is in the prompt, so this is recall, not copying. Bordt et al. treat it as evidence that the model saw the file in training. The first rows are also the most widely copied part of the file (every `df.head()` in a tutorial shows them), so this shows exposure to the file, not knowledge of all 30,000 rows.
- **Row-completion test:** it refused 9 of 25 (`uci`) and 10 of 25 (`kaggle_style`) queries, giving the next ID and saying it "can't determine its remaining values". Its 31 attempted rows match 25–28% of the 24 non-ID fields on average (max 58%). Always guessing each column's most common value matches 32%, so these answers are no better than guessing typical values. There's no sign of recall for rows deeper in the file, but with 38% refusals this test is weaker for Astra than it was for Qwen.

**What it means for the experiment:**
- **Direct leaks look tiny.** Of the customers it reproduced, `cust_00004` is in test and `cust_00005`–`cust_00007` are in tune (3, 8 and 10 are in reserve, 9 in train). One test customer out of 2,000 can't move the metrics.
- **How far the memorized start goes is untested.** Astra stopped on its own after 701 / 669 characters, still matching. Other low-numbered customers in test (`cust_00016`, `cust_00018`) are beyond what the test reached. A header test with a longer window would measure the depth.
- **The model never sees customer IDs**, only our rendered summaries. Still, rows it knows by heart could be recognizable from their numbers.
- **Recognition is the bigger risk.** The summaries show NT$ amounts and April–September 2005, so a model that knows this dataset could recognize it and recall facts about it, such as its ~22% default rate. That would flatter Astra's out-of-the-box calibration in particular. The base-rate condition gives every contestant that number, so if Astra is well calibrated out of the box while the others need the base-rate sentence, the write-up must point this out. Not tested yet: we haven't asked Astra whether it recognizes a summary.
- **Not decided yet (user):** whether to run a longer header test, report Astra's results with and without the memorized customers, and probe whether it recognizes a summary. Per the rule below, its results are flagged wherever they appear either way.

## What the write-up will say

- The scorecard and GBM were trained only on our `train` split.
- Jev, Kev and Laya can't be probed for memorization because they don't generate text. Kev's disclosed fine-tuning data doesn't include this dataset. The pretraining data of every base model is undisclosed, so exposure can't be ruled out for any AI contestant.
- The LLM contestants' tabmemcheck results get reported as measured. If a contestant shows memorization, its results are flagged wherever they appear.
- Qwen3.5-4B showed no memorization in either chat or raw-completion mode: 0 of 100 rows reproduced across both file versions (98 of them non-trivial), and no header continuation beyond trivial runs of zeros. This is absence of evidence, not proof.
- GPT-6.2 Astra reproduced the start of the data file verbatim (customers 3–10) in the header test, so it has seen this dataset in training. It reproduced none of 50 rows from deeper in the file, though it refused 19 of those queries. Its results are flagged as coming from a model that has seen this data.
