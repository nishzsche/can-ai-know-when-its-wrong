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
| `frontier-verbal` | Yes: **pending** (model not chosen yet) | Not disclosed | **Possibly**: public web data |
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

**Status:** Qwen3.5-4B done (2026-11-22). The frontier model is still pending: it hasn't been chosen yet and needs `OPENROUTER_API_KEY`.

What runs, for each LLM contestant, on two versions of the file:

1. **Header test:** given the first rows of the file, cut partway through a row, can the model continue it exactly? The reference is the next 500 characters of the real file.
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

| Model | Mode | File version | Header: exact prefix (chars of 500) | Rows: exact matches (non-trivial / non-trivial rows) | Verdict |
|---|---|---|---|---|---|
| Qwen3.5-4B | chat | uci | 6 | 0 (0/24) | No memorization detected |
| Qwen3.5-4B | chat | kaggle_style | 0 | 0 (0/24) | No memorization detected |
| Qwen3.5-4B | raw completion | uci | 2 | 0 (0/25) | No memorization detected |
| Qwen3.5-4B | raw completion | kaggle_style | 15 (a run of `0.0` fields) | 0 (0/25) | No memorization detected |
| frontier model | | | *pending* | | |

**Qwen3.5-4B run details:** `Qwen/Qwen3.5-4B` at revision `851bf6e8`, served by vLLM 0.34.0 on the L4 with `--language-model-only` and thinking turned off (`enable_thinking=false`, the same setting the contestant uses). tabmemcheck 0.1.6. We ran both modes: chat, which is how the contestant is prompted, and raw `/completions` with no chat template, which is the stronger probe because the model can't reformat or refuse.

**What the responses look like:** in both modes the model writes well-formed CSV rows with the right number of fields and copies the row ID from the prompt. The values are invented: round limits such as 100000 and 150000, bills repeated across months, repayment codes mostly 0. That shows the model knows the format, not the rows. In the raw-completion Kaggle-style header test it repeated customer 1's values (`20000.0,2,2,1,24,2,2,-1,-1,-2,-2,3913.0,3102.0`) under ID 2. Customer 1 was in the prompt (the test cut the file partway through row 1), so this is copying from the prompt, not recall; the real row 2 never appeared.

**Relevance to Kev:** Kev's bases are Qwen3.5-0.8B-Base and Qwen3.5-4B-Base. A clean result on the instruct Qwen3.5-4B is weak, indirect evidence that the shared pretraining didn't fix this file in memory. It doesn't test the base checkpoints or Kev itself.

## What the write-up will say

- The scorecard and GBM were trained only on our `train` split.
- Jev, Kev and Laya can't be probed for memorization because they don't generate text. Kev's disclosed fine-tuning data doesn't include this dataset. The pretraining data of every base model is undisclosed, so exposure can't be ruled out for any AI contestant.
- The LLM contestants' tabmemcheck results get reported as measured. If a contestant shows memorization, its results are flagged wherever they appear.
- Qwen3.5-4B showed no memorization in either chat or raw-completion mode: 0 of 100 rows reproduced across both file versions (98 of them non-trivial), and no header continuation beyond trivial runs of zeros. This is absence of evidence, not proof.
