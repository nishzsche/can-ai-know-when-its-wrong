# Status and handoff

Last updated 2026-11-23. Read this together with `PLAN.md` and `CLAUDE.md` before starting work.

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
    - **GPU contestants (L4 step 5, 2026-11-22 22:35–23:10Z, about $0.08 of GPU time):** 2,000 disguised test customers each, plus a 200-customer control on the original summaries. Kev and Qwen controls reproduce exactly or nearly (|change| 0.000–0.005), so every shift below is real.
      - **Kev-4B:** mean p 0.486 → 0.502, ECE 0.267 → 0.283, **AUC 0.594 → 0.651**, rank correlation 0.91. Its ranking gets *better* without the cues.
      - **Kev-0.8B:** mean p 0.305 → 0.292, ECE 0.090 → 0.084, AUC 0.750 → 0.747. Slightly better calibrated when disguised.
      - **Qwen:** mean p 0.355 → 0.389, ECE 0.140 → 0.171, AUC 0.744 → 0.738, rank correlation 0.80.
      - **Jev** (laptop, for comparison): mean p +0.041, ECE 0.265 → 0.308.
    - **Layout check on the GPU contestants** (`--variant aligned --split tune`, L4 2026-11-22 23:10–23:31Z, about $0.09): each bill shown next to the payment made the following month, 2,000 tune customers, numbers unchanged. No control on tune, but Kev's and Qwen's test controls reproduced to within 0.005.
      - **Kev-4B changes a lot:** mean p 0.488 → 0.357, **ECE 0.268 → 0.147, AUC 0.594 → 0.711**. For "paid on time (code -1)" customers (actual 0.18) it goes 0.37 → 0.24. Template v2's same-month layout seems to read as underpayment to it.
      - **Laya goes the other way:** mean p 0.206 → 0.265, ECE 0.045 → 0.067, AUC 0.638 → 0.581.
      - **Kev-0.8B and Qwen:** essentially unchanged (shift −0.003 and −0.002; AUC −0.003 and −0.011).
      - **Fairness flag for the author:** under the "never disadvantage the AI contestants" rule, the v2 layout costs Kev-4B a lot and helps Laya. No template change made; that's the author's call, and any change would mean re-running every AI contestant on test.
      - **Laya and Sol's disguised runs are not in git** (the report lists only the four above). If they ran on the laptop, commit them from there.
    - **Full run stopped:** the 2,000-customer disguised run (~$1.50) was stopped before spending, since it can't isolate recognition. It can still be run with `--workers 4`, then `--report`.
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

**Deleted after the M3 GPU runs.** For any new GPU work, launch a fresh instance and follow the first-time setup below; it takes about 20–30 billed minutes.

1× NVIDIA L4 (24 GB), PyTorch template (a container), region IN2, 28 cores and 124 GB RAM. Compute bills per second; storage keeps billing while the instance is paused.

**Only `/home` survives a pause.** Anything installed elsewhere is gone after a resume: system packages, `~/.cache`, a `gh` login. So keep the repo, uv, its Python builds and every model cache under `/home`.

### First-time setup

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
