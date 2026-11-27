# L4 runbook: M2 smoke tests, M3 GPU runs, M6 fine-tuning, posts 4 and 5

Everything in this project that needs the GPU, in order. Read `STATUS.md` first (setup, `/home` persistence, money). The L4 bills per second, so don't leave it idle.

**Done (2026-11-22):** steps 0–4 ran, the results are committed, and the instance was deleted. **Step 5 done (2026-11-22, 22:35–23:31Z)** on a new instance (`jl-584335`): the disguise check for Kev-4B, Kev-0.8B and Qwen, and the layout check (`--variant aligned`, tune) for Kev-4B, Kev-0.8B, Qwen and Laya, at the same weights and setup as M3 (Kev without flash-linear-attention, vLLM 0.34.0). Results are in `results/checks/disguise/` and `results/checks/disguise_summary.json`. **Step 6 (M6, the fair fight) done (2026-11-23/24)**, seed check included; the weights are in the `m6-weights` release. **Step 7 (session A, posts 4 and 5) is next.**

- **API contestants** (Jev, GPT-6.2 Sol) run on the laptop, not here.
- **No API key needed:** this runbook uses no API key, so the L4 needs no `.env`.

## 0. Every session

```bash
source /home/env.sh
cd /home/can-ai-know-when-its-wrong && git pull
uv sync --group runners --group laya
nvidia-smi                      # check nothing else holds GPU memory
mkdir -p /home/logs
tmux new -s l4 || tmux attach -t l4   # runs survive a dropped SSH connection
```

**One model on the GPU at a time.** vLLM takes ~85% of the GPU, so stop it before starting a Kev server, and vice versa. Run `pkill -f "vllm serve"` / `pkill -f "kev.serve"` on their own lines; inside a compound command they also kill the calling shell.

## 1. Kev server environment (first time only)

```bash
git clone https://github.com/jaredpalmer/kev.git /home/kev
cd /home/kev && git checkout 65b1e763943ace53fd3a4aae00d8d08dbd849772   # main on 2026-11-20, the README we followed
uv sync --extra serve
```

- **Python:** Kev's `.python-version` asks for Python 3.13. uv downloads it into `/home/uv-python`, which persists.
- **flash-linear-attention:** if `kev.serve` warns that it's missing, run `cd /home/kev && uv pip install flash-linear-attention`. It affects speed only.

**Kev v1.0 weights:** start the servers at these exact commits, which the `v1.0` tags point at. The adapter, head and tokenizer files are byte-identical to the README's "weights revision" (checked 2026-11-22). `runners/kev.py` refuses a server that isn't serving v1.0.

| Model | `--run` |
|---|---|
| Kev-4B | `jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c` |
| Kev-0.8B | `jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d` |

## 2. M2: smoke tests (20 tune customers × 2 conditions)

Laya and Sol already passed theirs. Kev-0.8B, Kev-4B and Qwen need the GPU.

**Kev-4B**
```bash
# window 1: server (first start downloads Qwen3.5-4B-Base, ~8 GB, into /home/hf-cache)
cd /home/kev && uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --port 8009 2>&1 | tee /home/logs/kev-4b.log
# window 2: wait until this answers, then run the smoke test
curl -s 127.0.0.1:8009/v1/models
cd /home/can-ai-know-when-its-wrong && uv run python runners/kev.py --size 4b --smoke
```
Then stop the server (Ctrl-C in window 1).

**Kev-0.8B:** the same, with `--run jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d` and `runners/kev.py --size 0.8b --smoke`.

**Qwen3.5-4B:** start vLLM with the exact command in `STATUS.md` ("Serving Qwen3.5-4B"). Once `curl -s 127.0.0.1:8000/v1/models` answers, run:
```bash
uv run --group runners python runners/qwen_logprobs.py --smoke
```
Then `pkill -f "vllm serve"` on its own line.

**Check, then commit:**
```bash
uv run --group runners --group laya pytest -q tests/test_runners.py
git add results/smoke && git commit -m "M2: Kev and Qwen smoke tests on the L4" && git push
```

The pytest smoke checks for `kev-0.8b`, `kev-4b` and `qwen-logprobs` now pass instead of skipping. In each runner's summary line, look at:
- **mean p:** should be plausible against the actual rate (0.20 on these 20 customers).
- **"exactly 0 or 1":** should be ~0. For Qwen, many exact 0s and 1s mean vLLM is returning processed rather than raw logprobs; check `vllm serve --help | grep -i logprobs` and ask for raw logprobs.
- **latency.**

## Gate: before any M3 run

Continue only if both hold; otherwise stop here and **pause the instance**.
1. **M2 is complete:** after `git pull`, `results/smoke/` has all six AI contestants, `jev.jsonl` included (that one comes from the laptop), and `uv run --group runners --group laya pytest -q` passes.
2. **`PREDICTIONS.md` is filled in and committed.** The plan requires the user's guesses to be written before any full run. Check with `grep -c "NOT FILLED IN" PREDICTIONS.md`, which must print 0.

## 3. M3: GPU runs (tune and test, both conditions)

Each contestant writes 8,000 records: 2,000 customers × 2 splits × 2 conditions. Every command resumes where it stopped, so if anything is interrupted, run the same command again.

```bash
# Laya: in-process, no server (~5 min)
uv run --group laya python runners/laya_classic.py --split tune --device cuda
uv run --group laya python runners/laya_classic.py --split test --device cuda

# Qwen: start vLLM as in step 2 first
uv run --group runners python runners/qwen_logprobs.py --split tune
uv run --group runners python runners/qwen_logprobs.py --split test
#   then: pkill -f "vllm serve"

# Kev-4B: start its server as in step 2 first
uv run python runners/kev.py --size 4b --split tune
uv run python runners/kev.py --size 4b --split test
#   then stop the server

# Kev-0.8B: start its server as in step 2 first
uv run python runners/kev.py --size 0.8b --split tune
uv run python runners/kev.py --size 0.8b --split test
```

After each contestant, `wc -l results/predictions/<contestant>.jsonl` should print 8000.

## 4. Wrap up

```bash
uv run --group runners --group laya python runners/manifest.py --allow-incomplete
git add results/predictions results/run_manifest.json
git commit -m "M3: GPU contestants' tune and test predictions" && git push
```

- **The manifest script:** `runners/manifest.py` checks coverage and writes `results/run_manifest.json`. `--allow-incomplete` is needed here because Jev and Sol run on the laptop; the final manifest without that flag is written once every contestant is in.
- **Pause the instance.** Keep it paused, not deleted, until M4's metrics confirm nothing needs re-running. Then delete it, since storage bills while paused.

## 5. Disguise and layout checks: Kev-4B, Kev-0.8B, Qwen, Laya (2026-11-23)

**Why:** every summary shows "Sep 2005"…"Apr 2005", NT$ amounts and raw repayment codes.
- **The question:** this step checks whether removing those cues changes each model's answers. For example, does knowing it's Taiwan in 2005, during a card-debt crisis, push its risk up or down?
- **Already done:** Jev, Laya and Sol ran on the laptop. Jev's average went *up* 4 points when disguised, against 0.3 points of run-to-run noise.
- **Details:** `runners/checks/disguise.py`, `data/contamination.md` §4.

**What runs:** for each model, 2,000 disguised test summaries, then a 200-customer control on the *original* summaries, which measures run-to-run noise. Both are out of the box only.

**Layout check (added 2026-11-23, after the review panel):** template v2 shows each payment beside the same month's bill, but payments usually settle the *previous* month's bill (README, data quirk 3). `--variant aligned --split tune` re-asks the 2,000 tune customers with each bill shown next to the payment made the following month. Every number is unchanged and the test set isn't touched. Jev and Sol run it on the laptop; Kev, Qwen and Laya run it here. At ~0.1–0.15 s per request, that's a few minutes per model. Output goes to `results/checks/disguise/`. Every command resumes if interrupted.

**Setup (new instance):**
1. **Base setup:** §6.0 blocks 1 and 2 (for a new session, use the newest section's setup, e.g. §9.0). Then `uv sync --group runners`. No API key is needed.
2. **Kev server environment:** step 1 above. The first Kev start downloads the base models into `/home/hf-cache`.
3. **vLLM venv for Qwen:** `uv venv /home/vllm-env --python 3.12 && VIRTUAL_ENV=/home/vllm-env uv pip install vllm`. Pin vLLM 0.34.0 if a newer one fails.
4. **Logs folder:** `mkdir -p /home/logs`.

```bash
source /home/env.sh && cd /home/can-ai-know-when-its-wrong && git pull

# Kev-4B: start its server in window 1 (the exact command in step 2), wait for
#   curl -s 127.0.0.1:8009/v1/models   then, in window 2:
uv run --group runners python runners/checks/disguise.py --contestant kev-4b
uv run --group runners python runners/checks/disguise.py --contestant kev-4b --control
uv run --group runners python runners/checks/disguise.py --contestant kev-4b --variant aligned --split tune
#   stop the server (Ctrl-C in window 1)

# Kev-0.8B: same, with the 0.8b --run commit from step 1
uv run --group runners python runners/checks/disguise.py --contestant kev-0.8b
uv run --group runners python runners/checks/disguise.py --contestant kev-0.8b --control
uv run --group runners python runners/checks/disguise.py --contestant kev-0.8b --variant aligned --split tune
#   stop the server

# Qwen3.5-4B: start vLLM with the exact command in STATUS.md ("Serving Qwen3.5-4B"), wait for
#   curl -s 127.0.0.1:8000/v1/models   then:
uv run --group runners python runners/checks/disguise.py --contestant qwen-logprobs
uv run --group runners python runners/checks/disguise.py --contestant qwen-logprobs --control
uv run --group runners python runners/checks/disguise.py --contestant qwen-logprobs --variant aligned --split tune
#   then, on its own line:  pkill -f "vllm serve"

# Laya: layout check only (its disguise run was done on the laptop), on the GPU like its M3 run
uv run --group runners --group laya python runners/checks/disguise.py --contestant laya --variant aligned --split tune --device cuda
```

**Check, then push:**
```bash
wc -l results/checks/disguise/{kev-4b,kev-0.8b,qwen-logprobs}-{disguised,control}.jsonl   # 2000 / 200 each
wc -l results/checks/disguise/{kev-4b,kev-0.8b,qwen-logprobs,laya}-aligned.jsonl           # 2000 each
uv run --group runners python runners/checks/disguise.py --report
git add results/checks/disguise && git commit -m "Disguise and layout checks: Kev-4B, Kev-0.8B, Qwen, Laya (L4)" && git push
```

**Then delete the instance.** Nothing else needs the GPU, and storage bills while paused. Also revoke its `gh` token.

**Same weights and revisions as M3:**
- **Kev:** v1.0 commits as in step 1. The runner refuses any other version.
- **Qwen:** `851bf6e8`, with thinking off.

The control run on the original summaries makes any change from the disguise visible against noise.

## 6. M6: the fair fight, fine-tuning on `train` (added 2026-11-23)

PLAN.md §11 has the design and the flagged choices. In short: Kev-0.8B, Laya, Kev-4B and Qwen each learn from the 20,000 `train` customers with the maker's default settings, then answer the usual `tune` and `test` questions as condition `trained`. Jev and Sol run on the laptop (examples in the prompt), not here. Nothing here is tuned on `test`.

**Gate:** `PREDICTIONS.md` has the author's M6 guesses, committed before any full run (smoke runs are fine).

**Order and time** (estimates; measure in the first minutes of each run): Kev-0.8B (~20–30 min) → Laya (~10–20 min) → Kev-4B (~1–2 h) → Qwen (~1–2 h). Scoring is ~0.05–0.2 s per customer, so 4,000 customers take 5–15 min per model. **If Kev-4B's or Qwen's first-epoch ETA is over 3 hours, stop and ask the author** (PLAN.md §11, compute).

### 6.0 A fresh instance, from zero (about 20–30 billed minutes)

JarvisLabs, PyTorch template, one L4. Everything lives under `/home`, which is the only part that survives a pause. The repo is public, so cloning needs no login; pushing results back does.

**1. Base tools**
```bash
cat > /home/env.sh <<'EOF'
export HF_HOME=/home/hf-cache
export UV_CACHE_DIR=/home/uv-cache
export UV_PYTHON_INSTALL_DIR=/home/uv-python
export GH_CONFIG_DIR=/home/gh-config
export PATH=/home/bin:$PATH
EOF
source /home/env.sh
mkdir -p /home/bin /home/runs /home/logs
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/home/bin sh
curl -sL https://github.com/cli/cli/releases/download/v2.105.0/gh_2.105.0_linux_amd64.tar.gz | tar xz -C /tmp   && cp /tmp/gh_2.105.0_linux_amd64/bin/gh /home/bin/
gh auth login -h github.com -p https -w      # only needed to push results back
tmux new -s l4                                # runs survive a dropped connection
```

**2. The project** (inside tmux)
```bash
source /home/env.sh
cd /home && git clone https://github.com/nishzsche/can-ai-know-when-its-wrong.git
cd can-ai-know-when-its-wrong && gh auth setup-git
uv sync --group runners --group laya --group training
uv run --group runners --group laya --group training pytest -q      # 170 passed
uv run python data/render.py --split train                            # 20,000 summaries (git-ignored)
uv run --group runners python training/export.py                      # 3 files, default rate 0.2212
```

**3. Kev's own environment** (its training code comes with it)
```bash
git clone https://github.com/jaredpalmer/kev.git /home/kev
cd /home/kev && git checkout 65b1e763943ace53fd3a4aae00d8d08dbd849772
uv sync --extra serve
uv pip install flash-linear-attention    # optional, speed only; skip it if it won't build
```

**After a resume or in a new shell:** `source /home/env.sh`, then `cd /home/can-ai-know-when-its-wrong && git pull`.

**Running Claude Code here instead:** after block 2, tell it "Read STATUS.md (the M6 section) and follow L4_RUNBOOK.md §6." There's no API key on this box, and none is needed.

### 6.1 Kev-0.8B

```bash
cd /home/kev
REL=jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d
# 1. The released checkpoint's architecture settings; kev.train refuses a warm start that differs.
uv run python -c "from kev.checkpoint import Checkpoint; print(Checkpoint('$REL').meta)"
#    Use its `base` for --base, and pass --head_dim / --option_isolation / --special_embeddings /
#    --lora / --base_revision whenever they differ from kev.train's defaults (256 / 0 / 0 / 16 / unset).
BASE=<meta.base>     # e.g. Qwen/Qwen3.5-0.8B-Base
ARGS="--data /home/can-ai-know-when-its-wrong/training/data/kev_train.jsonl --base $BASE --init_from $REL \
  --epochs 1 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --seed 0"
# 2. Smoke: 5 optimizer steps, then delete it.
uv run python -m kev.train $ARGS --max_steps 5 --out /home/runs/kev-0.8b-smoke && rm -rf /home/runs/kev-0.8b-smoke
# 3. The real run (one epoch over all 20,000), logged.
uv run python -m kev.train $ARGS --out /home/runs/kev-0.8b-ft 2>&1 | tee /home/logs/kev-0.8b-ft.log
# 4. Serve it at temperature 1.0 (window 1) ...
KEV_TEMPERATURE=1.0 uv run --extra serve python -m kev.serve --run /home/runs/kev-0.8b-ft --port 8009
```

```bash
# ... and score it (window 2). The runner refuses a server that isn't at temperature 1.0.
cd /home/can-ai-know-when-its-wrong
uv run python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft --split tune
uv run python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft --split test
```

Then stop the server (`pkill -f "kev.serve"` on its own line). Check the runner's summary line: mean p near the actual rate (~0.22) is expected after training on the natural rate; many exactly-0-or-1 values aren't.

### 6.2 Laya (in-process)

```bash
cd /home/can-ai-know-when-its-wrong
uv run --group laya python training/laya_ft.py --out /home/runs/laya-smoke --limit 64 --epochs 1 --device cuda \
  && rm -rf /home/runs/laya-smoke /home/runs/laya-smoke-data.jsonl
uv run --group laya python training/laya_ft.py --out /home/runs/laya-ft --device cuda 2>&1 | tee /home/logs/laya-ft.log
uv run --group laya python runners/laya_classic.py --checkpoint /home/runs/laya-ft --split tune --device cuda
uv run --group laya python runners/laya_classic.py --checkpoint /home/runs/laya-ft --split test --device cuda
```

`laya.train` warns if predictions collapsed to the class prior (a near-constant output); if it does, stop and tell the author rather than adding epochs.

### 6.3 Kev-4B

The same as 6.1 with `REL=jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c`, its own `meta.base` (Qwen3.5-4B-Base), `/home/runs/kev-4b-ft`, and `runners/kev.py --size 4b`. Watch the first log lines for the step rate: one epoch is 2,500 optimizer steps (20,000 ÷ 8).

### 6.4 Qwen3.5-4B

```bash
cd /home/can-ai-know-when-its-wrong
G="--group runners --group training"
# 1. Readout check: the base model in-process must reproduce M3's zero-shot numbers (vLLM) on 50
#    tune customers. It writes results/checks/qwen_inprocess_check.json and stops on a FAIL.
uv run $G python runners/qwen_ft.py --zero-shot-check 50
# 2. Smoke, then the real run (one epoch, 1,250 optimizer steps of 16 customers).
uv run $G python training/qwen_lora.py --out /home/runs/qwen-smoke --limit 64 --max-steps 4 && rm -rf /home/runs/qwen-smoke
uv run $G python training/qwen_lora.py --out /home/runs/qwen-ft 2>&1 | tee /home/logs/qwen-ft.log
# 3. Score.
uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split tune
uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft --split test
```

If the readout check fails, don't score anything: commit the check file and tell the author.

### 6.5 Optional second epoch (PLAN.md §11, choice 1)

Only for a model whose first epoch took under an hour. Train one more epoch from the saved weights (Kev: `--init_from /home/runs/<name>-ft --out /home/runs/<name>-ft-e2`; Qwen: `--init-adapter /home/runs/qwen-ft --out /home/runs/qwen-ft-e2`), score **tune only** with `--out-dir results/checks/m6_epoch2`, and keep the second epoch only if its tune log loss is lower. If it is, move the first epoch's predictions file to `results/checks/m6_epoch1/` and score both splits with the second. Never look at test to decide.

### 6.7 Tabular foundation models (PLAN.md §11, Part B; added 2026-11-23)

Run after Qwen, before the wrap-up. They read the 19 raw fields with all 20,000 `train` customers in context: no training, no text. Each takes minutes.

```bash
cd /home/can-ai-know-when-its-wrong && git pull
uv sync --group runners --group laya --group training --group tabular
T="--group tabular"
uv run $T python baselines/tabular_fm.py --model kumo-tabular --smoke     # 500 context rows, prints only
uv run $T python baselines/tabular_fm.py --model kumo-tabular             # tune + test -> results/predictions
uv run $T python baselines/tabular_fm.py --model tabicl-v2 --smoke
uv run $T python baselines/tabular_fm.py --model tabicl-v2
```

- **Out of GPU memory with 20,000 context rows:** stop and tell the author. Don't cut the context; fewer rows than the baselines get would tilt the comparison.
- **TabPFN-3.5 is out** (the author's decision, 2026-11-24: non-commercial licence). Only Kumo-Tabular and TabICLv2 run.
- Commit `results/predictions/<model>.jsonl` and `results/baselines/<model>.json` with the rest in 6.6.

### 6.6 Wrap up

```bash
cd /home/can-ai-know-when-its-wrong
uv run --group runners python runners/manifest.py --m6 --allow-incomplete   # Jev/Sol come from the laptop
# Keep the trained weights for reproduction: one GitHub release, one archive per model.
cd /home/runs && for r in kev-0.8b-ft kev-4b-ft laya-ft qwen-ft; do tar czf /home/$r.tar.gz $r; done
gh release create m6-weights --repo nishzsche/can-ai-know-when-its-wrong --title "M6 trained weights" \
  --notes "Fine-tuned on the 20,000 train customers (PLAN.md §11)." /home/*-ft.tar.gz
cd /home/can-ai-know-when-its-wrong
git add results/predictions results/checks results/baselines results/run_manifest_m6.json
git commit -m "M6: fine-tuned Kev-0.8B, Laya, Kev-4B and Qwen on train; tune and test predictions"
git push
```

Each archive must be under 2 GB (Laya's full weights are ~1.7 GB). Then pause or delete the instance. `recalibrate.py` and the analysis run on the laptop.

## 6.8 Seed check: Kev-0.8B and Laya with seeds 1 and 2 (added 2026-11-24; author's decision)

The review panel asked how much a re-run moves the results (PLAN.md §11, choice 6 promised this "if GPU time allows"). Same data, settings and code as §6.1 and §6.2; only the training seed changes. Nothing replaces the main (seed 0) predictions: these go to `results/checks/m6_seeds/`. About 4 GPU hours (~$1.75): Kev-0.8B ~75 min per seed, Laya ~40 min per seed, plus a few minutes of scoring each.

Fresh instance: §6.0 blocks 1–3 first (the trained weights from the first session are in the `m6-weights` release if you need them; this check doesn't).

```bash
# Kev-0.8B, seeds 1 and 2: the seed-0 run's exact arguments (base model and its pinned revision
# included), with only --seed and --out changed. Needs the training files from 6.0 block 2.
cd /home/kev
ARGS="$(sed 's/--seed 0//' /home/can-ai-know-when-its-wrong/results/checks/m6_logs/kev-0.8b.args)"
echo "$ARGS"    # check: --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2... --init_from ...@788ddbdd...
for s in 1 2; do
  uv run python -m kev.train $ARGS --seed $s --out /home/runs/kev-0.8b-ft-s$s 2>&1 | tee /home/logs/kev-0.8b-ft-s$s.log
done
```

For each seed, serve it at temperature 1.0 (window 1) and score both splits (window 2), then stop the server:

```bash
# window 1
KEV_TEMPERATURE=1.0 uv run --extra serve python -m kev.serve --run /home/runs/kev-0.8b-ft-s1 --port 8009
# window 2
cd /home/can-ai-know-when-its-wrong
for split in tune test; do
  uv run python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft-s1 --split $split \
    --out-dir results/checks/m6_seeds/seed1
done
```

(Repeat with `-s2` and `seed2`.)

```bash
# Laya, seeds 1 and 2
cd /home/can-ai-know-when-its-wrong
for s in 1 2; do
  uv run --group laya python training/laya_ft.py --out /home/runs/laya-ft-s$s --device cuda --seed $s \
    2>&1 | tee /home/logs/laya-ft-s$s.log
  for split in tune test; do
    uv run --group laya python runners/laya_classic.py --checkpoint /home/runs/laya-ft-s$s --split $split \
      --device cuda --out-dir results/checks/m6_seeds/seed$s
  done
done
uv run python analysis/seeds.py        # spread across seeds 0, 1, 2 -> results/seeds.json
git add results/checks/m6_seeds results/seeds.json && git commit -m "M6: seed check (Kev-0.8B and Laya, seeds 1 and 2)" && git push
```

Commit the training logs too (`/home/logs/*-s1.log`, `*-s2.log` into `results/checks/m6_logs/`). No need to release these weights.

## 7. Session A: posts 4 and 5 (added 2026-11-25)

PLAN.md §12 has the specs. Post 4 runs the fine-tuned Kev-0.8B, Kev-4B and Qwen at six precisions (`bf16`, `bnb-int8`, `bnb-nf4`, `hqq-4`, `hqq-3`, `hqq-2`). Post 5 runs the zero-shot open models on four formats (`prose`, `markdown`, `csv`, `json`). Nothing here is tuned on `test`, and nothing needs an API key. Jev and Sol's format runs happen on the laptop.

**Running Claude Code here:** after §7.0 step 1, give it this prompt:

```text
You're on a fresh JarvisLabs L4 (24 GB) for can-ai-know-when-its-wrong: session A, the GPU runs
for posts 4 (quantization) and 5 (table formats). GPU time is billed per minute: follow the
runbook and avoid detours.

1. Read CLAUDE.md, the top section of STATUS.md ("After post 3: the plan") and PLAN.md §12 (the
   post 4 and post 5 specs). Those specs are fixed: ask me before changing any design choice.
2. Gate: PREDICTIONS.md must already have my post 4 and post 5 guesses committed. If it doesn't,
   stop and tell me.
3. Run L4_RUNBOOK.md §7 in order, one command at a time:
   - Kev and Qwen: run the bf16 reproduction check first. If it fails, stop and report.
   - If a precision won't load or run on a model, record it with
     `analysis/quantization.py --record-failure`, skip that pair and tell me. Never swap in
     another method or change its settings.
   - Before each full run, tell me its ETA in IST (the box's clock is UTC). If any single
     run's ETA is over 2 hours, stop and ask.
   - Keep only one model on the GPU at a time.
4. After each model finishes, commit and push its predictions and logs: no keys, no .env and
   no weights. At the end, run §7.5 and report what ran, what failed, the GPU hours, the cost
   and the finish time in IST. Then tell me it's safe to delete the instance, and remind me to
   revoke the GitHub token used here.
```

**Gate:** `PREDICTIONS.md` has the author's post 4 and post 5 guesses, committed before any run. That includes the smoke runs and the reproduction check, because both print numbers.

**Order and time** (estimates; check them in each run's first minutes): post 5's open models (~1 h), then Kev-0.8B (~30–60 min), Qwen (~1–2 h) and Kev-4B (~1.5–2.5 h). About 5–6 GPU hours in all. Post 5 goes first because its runners are already proven, so its results are banked even if a quantized model needs attention.

- **ETAs:** before each full run (one precision, one split), estimate its time from the smoke run or the first minutes. If any one run would take over 2 hours, stop and ask the author. Give the author times in IST; the box's clock is UTC.
- **A precision that fails** (it won't load, it errors, it returns NaN, or the smoke's predictions are all exactly 0 or 1): record it, then go on to the next precision. Never swap in another method or change its settings.
  ```bash
  uv run python analysis/quantization.py --record-failure kev-4b-ft-hqq-2 --stage load --error "<the error line>"
  ```

### 7.0 Setup (a fresh instance, about 30 billed minutes)

1. §6.0 blocks 1 and 2, with the `quant` group added to both uv commands: `uv sync --group runners --group laya --group training --group quant`, and the same groups for `pytest` (197 passed). Skip `render.py --split train` and `export.py`: nothing is trained here.
2. §6.0 block 3 (Kev's environment), then the two pinned libraries into it (`runners/quantize.py` refuses any other version):
   ```bash
   cd /home/kev && uv pip install bitsandbytes==0.50.2 hqq==0.2.8.post1
   ```
3. vLLM for Qwen's zero-shot format runs, at the version M3 used:
   ```bash
   uv venv /home/vllm-env --python 3.12 && VIRTUAL_ENV=/home/vllm-env uv pip install vllm==0.34.0
   ```
4. The M6 weights. The release is public, and each archive unpacks to `/home/runs/<model>/`:
   ```bash
   cd /home/runs
   for r in kev-0.8b-ft kev-4b-ft qwen-ft; do
     curl -sLO https://github.com/nishzsche/can-ai-know-when-its-wrong/releases/download/m6-weights/$r.tar.gz
     tar xzf $r.tar.gz && rm $r.tar.gz
   done
   ls /home/runs/kev-0.8b-ft/head.pt /home/runs/kev-4b-ft/head.pt /home/runs/qwen-ft/adapter_model.safetensors
   mkdir -p /home/logs /home/can-ai-know-when-its-wrong/results/checks/quantization
   ```

### 7.1 Post 5: the zero-shot open models on four formats (~1 h)

These are diagnostics on `test` only, out of the box, with the same servers and settings as M3. They write to `results/checks/formats/`.

```bash
cd /home/can-ai-know-when-its-wrong
F="prose markdown csv json"
# Laya, in-process on the GPU as in M3. The runner stops if Laya ever reports truncating a summary.
for f in $F; do
  uv run --group runners --group laya python runners/checks/disguise.py --contestant laya --variant $f --device cuda
done
```

- **Kev-4B:** start the zero-shot server exactly as in step 2 (window 1). Wait for `curl -s 127.0.0.1:8009/v1/models`, then run:
  ```bash
  for f in $F; do uv run --group runners python runners/checks/disguise.py --contestant kev-4b --variant $f; done
  ```
  Then `pkill -f "kev.serve"` on its own line.
- **Kev-0.8B:** the same, with its step 2 server and `--contestant kev-0.8b`.
- **Qwen:** start vLLM from `/home/vllm-env` with the exact command in STATUS.md ("Serving Qwen3.5-4B"). Wait for `curl -s 127.0.0.1:8000/v1/models`, then run:
  ```bash
  for f in $F; do uv run --group runners python runners/checks/disguise.py --contestant qwen-logprobs --variant $f; done
  ```
  Then `pkill -f "vllm serve"` on its own line.

Commit and push `results/checks/formats/`.

### 7.2 Post 4: Kev-0.8B at six precisions (~30–60 min)

For each precision: start the launcher in window 1, then run the smoke, tune and test in window 2, then stop the launcher. Run the commands one at a time, and stop if one fails. bf16 goes first, with the reproduction check.

```bash
# window 1, Kev's environment: one precision at a time
cd /home/kev
P=bf16    # then bnb-int8, bnb-nf4, hqq-4, hqq-3, hqq-2
KEV_TEMPERATURE=1.0 uv run --extra serve python /home/can-ai-know-when-its-wrong/runners/kev_quant_serve.py \
  --precision $P --run /home/runs/kev-0.8b-ft --port 8009 2>&1 | tee /home/logs/kev-0.8b-ft-$P-serve.log
```

```bash
# window 2, this repo, the same P
cd /home/can-ai-know-when-its-wrong
P=bf16
until curl -sf 127.0.0.1:8009/v1/quantization > /dev/null; do sleep 5; done
R="uv run --group runners python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft --precision $P"
$R --reproduce-check 50          # bf16 only: if it fails, stop; run no precision of this model
$R --smoke                       # 20 tune customers -> results/smoke/; read its summary line
$R --split tune 2>&1 | tee -a /home/logs/kev-0.8b-ft-$P-score.log
$R --split test 2>&1 | tee -a /home/logs/kev-0.8b-ft-$P-score.log
curl -s 127.0.0.1:8009/v1/quantization > results/checks/quantization/kev-0.8b-ft-$P.server.json
```

Then run `pkill -f kev_quant_serve` on its own line, and move to the next P.

- **The serve log:** the launcher prints how many linear layers it quantized and the GPU memory after loading.
- **The runner:** it refuses a server that reports another precision.

### 7.3 Post 4: Qwen at six precisions (~1–2 h)

Qwen runs in-process. Each command loads the base in fp32 on the CPU, merges the adapter, rounds once to bf16 and quantizes, which takes a few minutes. `--both-splits` then scores tune and test with that one load.

```bash
cd /home/can-ai-know-when-its-wrong
G="--group runners --group training --group quant"
R="uv run $G python runners/qwen_ft.py --adapter /home/runs/qwen-ft"
$R --precision bf16 --reproduce-check 50          # if it fails, stop; run no precision of Qwen
P=bf16    # then bnb-int8, bnb-nf4, hqq-4, hqq-3, hqq-2, one at a time
$R --precision $P --both-splits 2>&1 | tee /home/logs/qwen-ft-$P-score.log
```

For the ETA and a first sanity check, watch the first records: `wc -l results/predictions/qwen-ft-$P.jsonl` and `tail -2 results/predictions/qwen-ft-$P.jsonl`.

### 7.4 Post 4: Kev-4B at six precisions (~1.5–2.5 h)

The same as §7.2, with `--size 4b`, `--run /home/runs/kev-4b-ft`, `--finetuned /home/runs/kev-4b-ft` and `kev-4b-ft` in the log and JSON names.

### 7.5 Wrap up

```bash
cd /home/can-ai-know-when-its-wrong
uv run python analysis/recalibrate.py          # trained_calibrated for every precision (M3/M6 rows refresh with the same values)
uv run --group runners python runners/manifest.py --quant
uv run python analysis/quantization.py         # -> results/quantization.csv and .json
uv run --group runners python runners/checks/disguise.py --report   # post 5, the open models so far
mkdir -p results/checks/quant_logs && cp /home/logs/kev-*-ft-*.log /home/logs/qwen-ft-*.log results/checks/quant_logs/
git add results/predictions results/checks results/recalibration.json results/run_manifest_quant.json \
  results/quantization.csv results/quantization.json
git commit -m "Posts 4 and 5: fine-tuned models at six precisions; zero-shot open models on four formats"
git push
```

Then delete the instance. Pausing keeps storage billing, and nothing here needs to be kept: the predictions and logs are in git, and the weights are in the `m6-weights` release.

## 8. Session S: post 4's speed runs (added 2026-11-26)

PLAN.md §12, "Post 4 speed addendum". The review panel found post 4's speed numbers came from an untuned setup. This session measures the same three fine-tuned models on a tuned one, at five speed tags (`runners/speed.py`):

| Tag | What it is | Check | Records |
|---|---|---|---|
| `bf16-tuned` | Full size, fast setup: the speed reference | reproduce post 4's `bf16` on 50 tune customers | throughput only |
| `bnb-nf4-tuned` | Post 4's NF4, fast setup | reproduce post 4's `bnb-nf4` on 50 tune customers | throughput only |
| `hqq-4-aoint4` | Post 4's HQQ 4-bit weights on PyTorch's fused int4 matmul | smoke | tune + test, then throughput |
| `hqq-4-gemlite` | The same weights on GemLite (fp16 compute) | smoke | tune + test, then throughput |
| `fp8-dynamic` | FP8 weights and activations, dynamic scales (torchao), no data | smoke | tune + test, then throughput |

"Fast setup":
- the `causal-conv1d` and flash-linear-attention kernels are installed;
- Kev runs with CUDA graphs on, plus its fused kernels for `bf16-tuned` if its pinned FLA version imports;
- Qwen runs through `torch.compile`.

Throughput uses the same 500 tune customers at 1, 8 and 32 at a time (`runners/throughput.py`); its batch-1 time is the per-answer figure the post compares. Nothing here is tuned on `test`, and nothing needs an API key.

**Running Claude Code here:** after §8.0 step 1, give it this prompt:

```text
You're on a fresh JarvisLabs L4 (24 GB) for can-ai-know-when-its-wrong: session S, the speed runs
for post 4 (PLAN.md §12, "Post 4 speed addendum"). GPU time is billed per minute: follow the
runbook and avoid detours.

1. Read CLAUDE.md, the top of STATUS.md ("Post 4 draft") and PLAN.md §12's post 4 spec and its
   speed addendum. The design is fixed: ask me before changing any choice.
2. Gate: PREDICTIONS.md must already have my "Post 4 speed addendum" guesses committed (commit
   9f415ea). If it doesn't, stop and tell me.
3. Run L4_RUNBOOK.md §8 in order, one command at a time:
   - Install the fast libraries as §8.0 says, choosing each version to match that environment's
     torch, and write the versions you installed into STATUS.md before any run.
   - bf16-tuned and bnb-nf4-tuned: run the reproduction check first. If it fails, stop and report.
   - If a tag won't load, won't capture, errors, returns NaN, or its smoke run's answers are all
     exactly 0 or 1, record it with `analysis/quantization.py --record-failure`, skip it and tell
     me. Never swap in another method, backend or setting.
   - Before each full run, tell me its ETA in IST (the box's clock is UTC). If any single run's
     ETA is over 2 hours, stop and ask.
   - Keep only one model on the GPU at a time.
4. After each model, commit and push its predictions, throughput files and logs: no keys, no .env
   and no weights. At the end, run §8.4 and report what ran, what failed, the GPU hours, the cost
   and the finish time in IST. Then tell me it's safe to delete the instance, and remind me to
   revoke the GitHub token used here.
```

**Gate:** `PREDICTIONS.md` has the author's speed-run guesses, committed before any run (commit `9f415ea`). That includes smoke runs, reproduction checks and throughput, because all of them print numbers.

**Order and time** (estimates; check them in each run's first minutes): Kev-0.8B (~30–45 min), Qwen (~45–75 min), Kev-4B (~45–75 min). About 2–3 GPU hours plus 30 minutes of setup, about $1–1.50.

**A tag that fails** is recorded, then you go on to the next tag:
```bash
uv run python analysis/quantization.py --record-failure kev-4b-ft-fp8-dynamic --stage load --error "<the error line>"
```

### 8.0 Setup (a fresh instance, about 30 billed minutes)

1. §7.0 steps 1, 2 and 4: this repo with the `quant` group (tests: 213 passed), Kev's environment with the two post 4 pins, and the M6 weights. vLLM isn't needed.
2. **The fast libraries in this repo's environment.** Install them with `uv pip`, then run every command below with `uv run --no-sync`, so uv leaves them in place.
   - First check the environment's torch: `uv run --no-sync python -c "import torch; print(torch.__version__)"`.
   - Pick the torchao release whose notes list that torch, and the newest GemLite, `causal-conv1d` and `flash-linear-attention` that build and import with it.
   ```bash
   cd /home/can-ai-know-when-its-wrong
   uv pip install "torchao==<matching>" "gemlite==<v>" "causal-conv1d==<v>" "flash-linear-attention==<v>"
   uv run --no-sync python -c "import torchao, gemlite, causal_conv1d, fla; print(torchao.__version__, gemlite.__version__, fla.__version__)"
   ```
3. **The same in Kev's environment** (torch 2.8 in session A, which pairs with torchao 0.13).
   - **Flash-linear-attention:** try 0.5.2 first. It's the version Kev's fused kernels pin, and session A found 0.6.0 fails to import with Kev's Triton. If 0.5.2 fails too, go back to 0.5.0 (as M6 had); `bf16-tuned` then runs without fused kernels, and the records say so.
   - **causal-conv1d** may build from source; allow about 10 minutes.
   ```bash
   cd /home/kev
   uv pip install "torchao==0.13.0" "gemlite==<v>" "causal-conv1d==<v>" "flash-linear-attention==0.5.2"
   uv run --extra serve python -c "import torchao, gemlite, causal_conv1d, fla; print(fla.__version__)"
   ```
4. Write the installed versions into STATUS.md ("Session S"). Every record carries them in `model_version`, and the launcher reports any fast-layer kernel still missing.
   ```bash
   mkdir -p /home/logs /home/can-ai-know-when-its-wrong/results/checks/speed
   ```

### 8.1 Kev-0.8B (~30–45 min)

For each tag: start the launcher in window 1, run the commands in window 2, then stop the launcher. The tags go in table order. Run the commands one at a time, and stop if one fails.

```bash
# window 1, Kev's environment: one tag at a time (the launcher sets KEV_CUDA_GRAPHS)
cd /home/kev
T=bf16-tuned    # then bnb-nf4-tuned, hqq-4-aoint4, hqq-4-gemlite, fp8-dynamic
KEV_TEMPERATURE=1.0 uv run --extra serve python /home/can-ai-know-when-its-wrong/runners/kev_quant_serve.py \
  --precision $T --run /home/runs/kev-0.8b-ft --port 8009 2>&1 | tee /home/logs/kev-0.8b-ft-$T-serve.log
```

```bash
# window 2, this repo, the same T
cd /home/can-ai-know-when-its-wrong
T=bf16-tuned
until curl -sf 127.0.0.1:8009/v1/quantization > /dev/null; do sleep 5; done
R="uv run --no-sync python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft --precision $T"
# bf16-tuned and bnb-nf4-tuned: the reproduction check, then go straight to throughput
$R --reproduce-check 50
# hqq-4-aoint4, hqq-4-gemlite, fp8-dynamic: the smoke, then tune and test
$R --smoke
$R --split tune 2>&1 | tee -a /home/logs/kev-0.8b-ft-$T-score.log
$R --split test 2>&1 | tee -a /home/logs/kev-0.8b-ft-$T-score.log
# every tag: throughput on 500 tune customers at 1, 8 and 32 at a time
uv run --no-sync python runners/throughput.py --kev 0.8b --finetuned /home/runs/kev-0.8b-ft --precision $T \
  2>&1 | tee -a /home/logs/kev-0.8b-ft-$T-throughput.log
curl -s 127.0.0.1:8009/v1/quantization > results/checks/speed/kev-0.8b-ft-$T.server.json
```

Then run `pkill -f kev_quant_serve` on its own line, and move to the next tag.

- **The server JSON** has Kev's CUDA graph stats (captured, kept, pending, failed). Graphs that fail to capture keep running without them. If `failed` is above zero, say so in STATUS; it isn't a run failure.
- **The throughput log** prints, at each batch size, the largest difference from the one-at-a-time answers. Above 0.01, say so in STATUS.

### 8.2 Qwen (~45–75 min)

In-process, as in §7.3. Each command loads the base, merges, shrinks, sets up the tag and compiles, which takes a few minutes.

```bash
cd /home/can-ai-know-when-its-wrong
R="uv run --no-sync python runners/qwen_ft.py --adapter /home/runs/qwen-ft"
$R --precision bf16-tuned --reproduce-check 50
$R --precision bnb-nf4-tuned --reproduce-check 50
for T in hqq-4-aoint4 hqq-4-gemlite fp8-dynamic; do       # one at a time; stop if one fails
  $R --precision $T --both-splits 2>&1 | tee /home/logs/qwen-ft-$T-score.log
done
for T in bf16-tuned bnb-nf4-tuned hqq-4-aoint4 hqq-4-gemlite fp8-dynamic; do
  uv run --no-sync python runners/throughput.py --qwen /home/runs/qwen-ft --precision $T \
    2>&1 | tee /home/logs/qwen-ft-$T-throughput.log
done
```

If `torch.compile` fails for a tag, record the failure, and don't rerun with `--no-compile` unless the author agrees.

### 8.3 Kev-4B (~45–75 min)

The same as §8.1, with `--size 4b`, `--run /home/runs/kev-4b-ft`, `--finetuned /home/runs/kev-4b-ft`, `--kev 4b`, and `kev-4b-ft` in the log and JSON names.

### 8.4 Wrap up

```bash
cd /home/can-ai-know-when-its-wrong
uv run --no-sync python analysis/recalibrate.py      # trained_calibrated for the three full-run tags
uv run --no-sync python analysis/speed_report.py     # -> results/speed.csv, speed_throughput.csv, speed.json
mkdir -p results/checks/speed_logs && cp /home/logs/*-tuned-*.log /home/logs/*-aoint4-*.log \
  /home/logs/*-gemlite-*.log /home/logs/*-fp8-dynamic-*.log results/checks/speed_logs/
git add results/predictions results/checks results/recalibration.json results/speed*.csv results/speed.json STATUS.md
git commit -m "Post 4 speed runs: tuned full size, NF4, fast 4-bit and FP8; throughput at 1, 8 and 32"
git push
```

Then delete the instance, and revoke the GitHub token. The predictions, throughput files and logs are in git, and the weights are in the `m6-weights` release.

## 9. Session L: post 5's addendum, the table models and the trained models (added 2026-11-27)

PLAN.md §12, "Post 5 addendum". One short session, nothing tuned on `test`, no API key:

- **Part A (§9.1):** Kumo-Tabular and TabICL-v2 with the same 19 fields laid out 14 other ways (`data/arrange.py`). Each runs at the makers' defaults and again as a single estimator with the column shuffle off. The original runs again at four other seeds, for the noise floor. Records go to `results/checks/arrangements/`.
- **Part B (§9.2–9.4):** post 3's trained Kev-0.8B, Kev-4B, Qwen and Laya on post 5's five layouts, plus the original layout asked again, on the 2,000 test customers. Kev and Qwen use post 4's full-size path (`--precision bf16`), whose session A records are the reference. Records go to `results/checks/layouts_trained/`.

**Gate:** `PREDICTIONS.md` has the author's "Post 5 addendum" guesses, committed before any run. That includes smoke runs and reproduction checks, because all of them print numbers.

**Same setup as session A, on purpose.** Part B's references are session A's `-bf16` runs, so don't install post 4's speed libraries (torchao, GemLite, the fast kernels). The reproduction check confirms the path matches.

**Order and time** (estimates; check them in each run's first minutes):
- Setup: about 30 minutes.
- Part A: about 30–45 minutes. There are 38 runs per model, each a model load plus about a minute of Kumo or seconds of TabICL.
- Kev-0.8B: about 15 minutes.
- Kev-4B: about 30 minutes.
- Qwen: about 30 minutes.
- Laya: about 10 minutes.

That's about 2.5 hours in all, about $1.10.

**Running Claude Code here:** after §9.0 step 1, give it this prompt:

```text
You're on a fresh JarvisLabs L4 (24 GB) for can-ai-know-when-its-wrong: session L, post 5's
addendum (PLAN.md §12, "Post 5 addendum"). GPU time is billed per minute: follow the runbook and
avoid detours.

1. Read CLAUDE.md, the top of STATUS.md and PLAN.md §12's post 5 spec and its addendum. The design
   is fixed: ask me before changing any choice.
2. Gate: PREDICTIONS.md must already have my "Post 5 addendum" guesses committed. If it doesn't,
   stop and tell me.
3. Run L4_RUNBOOK.md §9 in order, one command at a time:
   - Don't install post 4's speed libraries: Part B's reference is session A's plain setup.
   - Kev and Qwen: run the reproduction check first. If it fails, stop and report.
   - If a run won't load, errors, runs out of memory, or its smoke answers are all exactly 0 or
     1, record it with `runners/manifest.py --record-failure`, skip it and tell me. Never cut
     the 20,000 context rows, and never swap in another model, setting or layout.
   - Before each part, tell me its ETA in IST (the box's clock is UTC). If any single run's ETA
     is over 1 hour, stop and ask.
   - Keep only one model on the GPU at a time.
4. After each part, commit and push its records and logs: no keys, no .env and no weights. At the
   end, run §9.5 and report what ran, what failed, the GPU hours, the cost and the finish time in
   IST. Then tell me it's safe to delete the instance, and remind me to revoke the GitHub token.
```

**A run that fails** is recorded, then you go on to the next one:
```bash
uv run --group runners python runners/manifest.py --record-failure tabicl-v2-rows-3-e1 --stage run --error "<the error line>"
```

### 9.0 Setup (a fresh instance, about 30 billed minutes)

Everything for this session, in one place. Don't use the "First-time setup" in STATUS.md: it's the M1 version, without the dependency groups, so its tests fail on missing `torch` and `jevper`. JarvisLabs, PyTorch template, one L4. Only `/home` survives a pause.

**1. Base tools**
```bash
cat > /home/env.sh <<'EOF'
export HF_HOME=/home/hf-cache
export UV_CACHE_DIR=/home/uv-cache
export UV_PYTHON_INSTALL_DIR=/home/uv-python
export GH_CONFIG_DIR=/home/gh-config
export PATH=/home/bin:$PATH
EOF
source /home/env.sh
mkdir -p /home/bin /home/runs /home/logs
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/home/bin sh
curl -sL https://github.com/cli/cli/releases/download/v2.105.0/gh_2.105.0_linux_amd64.tar.gz | tar xz -C /tmp   && cp /tmp/gh_2.105.0_linux_amd64/bin/gh /home/bin/
gh auth login -h github.com -p https -w      # only needed to push results back
tmux new -s l4                                # runs survive a dropped connection
```

**2. The project, with every group the session uses** (inside tmux)
```bash
source /home/env.sh
cd /home && git clone https://github.com/nishzsche/can-ai-know-when-its-wrong.git
cd can-ai-know-when-its-wrong && gh auth setup-git
G="--group runners --group laya --group training --group quant --group tabular"
uv sync $G
uv run $G pytest -q          # no failures (a few skips are fine); 317 tests on 2026-11-27
```
Already cloned with the old setup? Add the `GH_CONFIG_DIR` line to `/home/env.sh`, `source` it, `git pull`, then run the two `uv` lines.

**3. Kev's environment, as session A had it** (Part B's reference is session A's `-bf16` runs)
```bash
git clone https://github.com/jaredpalmer/kev.git /home/kev
cd /home/kev && git checkout 65b1e763943ace53fd3a4aae00d8d08dbd849772
uv sync --extra serve
uv pip install bitsandbytes==0.50.2 hqq==0.2.8.post1    # runners/quantize.py refuses other versions
```
Don't install flash-linear-attention, torchao, GemLite or `causal-conv1d` here or in the project: those are post 4's speed setup, not session A's.

**4. The four trained models** (the public `m6-weights` release; Laya's archive is 1.5 GB)
```bash
cd /home/runs
for r in kev-0.8b-ft kev-4b-ft qwen-ft laya-ft; do
  curl -sLO https://github.com/nishzsche/can-ai-know-when-its-wrong/releases/download/m6-weights/$r.tar.gz
  tar xzf $r.tar.gz && rm $r.tar.gz
done
ls /home/runs/kev-0.8b-ft/head.pt /home/runs/kev-4b-ft/head.pt    /home/runs/qwen-ft/adapter_model.safetensors /home/runs/laya-ft/model.safetensors
```

**After a resume or in a new shell:** `source /home/env.sh`, then `cd /home/can-ai-know-when-its-wrong && git pull`.

**Then** start Claude Code inside tmux and give it the prompt above, or run §9.1 onwards by hand.

### 9.1 Part A: the table models rearranged (~30–45 min)

Each command loads the model, puts all 20,000 `train` customers in context and answers. `original` answers tune and test, and every other arrangement answers test only.

```bash
cd /home/can-ai-know-when-its-wrong
T="uv run --group tabular python baselines/tabular_fm.py"
$T --model kumo-tabular --arrangement original --smoke     # prints only
$T --model tabicl-v2 --arrangement original --smoke
for M in kumo-tabular tabicl-v2; do
  for E in "" "--estimators 1"; do                         # the makers' defaults, then the diagnostic
    $T --model $M --arrangement original $E
    for K in 1 2 3 4; do $T --model $M --arrangement original -s $K $E; done
    for A in months-reversed by-month realigned shuffled-1 shuffled-2 shuffled-3 shuffled-4 \
             shuffled-5 rows-1 rows-2 rows-3 rows-4 rows-5 thousands; do
      $T --model $M --arrangement $A $E
    done
  done
done 2>&1 | tee /home/logs/arrangements.log
uv run python analysis/arrangements.py                     # -> results/arrangements.csv; read the repro columns
```

- **Reproduction:** `analysis/arrangements.py` reports how far this session's `original` is from M6's run (`repro_*` columns). Within 0.002 on average is expected. A larger gap isn't a stop: the in-session `original` is the reference, and the gap goes in STATUS.
- **Out of GPU memory with 20,000 context rows:** stop and tell the author. Don't cut the context.

Commit and push `results/checks/arrangements/`, `results/arrangements.*` and the log.

### 9.2 Part B: Kev-0.8B and Kev-4B (~45 min)

Kev's launcher at bf16 exactly as in §7.2, one model at a time.

```bash
# window 1, Kev's environment
cd /home/kev
KEV_TEMPERATURE=1.0 uv run --extra serve python /home/can-ai-know-when-its-wrong/runners/kev_quant_serve.py \
  --precision bf16 --run /home/runs/kev-0.8b-ft --port 8009 2>&1 | tee /home/logs/kev-0.8b-ft-layouts-serve.log
```

```bash
# window 2, this repo
cd /home/can-ai-know-when-its-wrong
until curl -sf 127.0.0.1:8009/v1/quantization > /dev/null; do sleep 5; done
R="uv run --group runners python runners/kev.py --size 0.8b --finetuned /home/runs/kev-0.8b-ft --precision bf16"
$R --reproduce-check 50                  # if it fails, stop
$R --smoke --layout json                 # 20 tune customers -> results/smoke/
$R --split test --layout all 2>&1 | tee /home/logs/kev-0.8b-ft-layouts.log   # five layouts, then again
```

Then `pkill -f kev_quant_serve` on its own line. Kev-4B is the same with `--run /home/runs/kev-4b-ft`, `--size 4b`, `--finetuned /home/runs/kev-4b-ft` and `kev-4b-ft` in the log names.

### 9.3 Part B: Qwen (~30 min)

In-process. With `--layout all` the model loads once and answers the five layouts, then the original again.

```bash
R="uv run --group runners python runners/qwen_ft.py --adapter /home/runs/qwen-ft --precision bf16"
$R --reproduce-check 50                  # if it fails, stop
$R --split test --layout all 2>&1 | tee /home/logs/qwen-ft-layouts.log
```

### 9.4 Part B: Laya (~10 min)

In-process on the GPU, as in M6 and post 5. The runner stops if Laya ever reports truncating a summary.

```bash
L="uv run --group runners --group laya python runners/laya_classic.py --checkpoint /home/runs/laya-ft --device cuda"
$L --smoke --layout json
$L --split test --layout all 2>&1 | tee /home/logs/laya-ft-layouts.log
```

### 9.5 Wrap up

```bash
cd /home/can-ai-know-when-its-wrong
uv run --group runners python runners/manifest.py --addendum      # -> results/run_manifest_addendum.json
uv run python analysis/layouts.py --trained                      # -> results/layouts_trained.csv
uv run python analysis/arrangements.py
mkdir -p results/checks/addendum_logs && cp /home/logs/arrangements.log /home/logs/*-layouts*.log results/checks/addendum_logs/
git add results/checks results/smoke results/arrangements.* results/layouts_trained.* results/run_manifest_addendum.json STATUS.md
git commit -m "Post 5 addendum: table models rearranged; trained models in five layouts"
git push
```

Then delete the instance, and revoke the GitHub token. The records and logs are in git, and the weights are in the `m6-weights` release.
