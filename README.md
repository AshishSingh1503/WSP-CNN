# WSP-CNN

Pre-registered study of whether an LLM can propose direct, structured weight edits to an ImageNet-pretrained
AlexNet adapted to CIFAR-10, and whether those edits add value over random editing, budget-matched random search,
a gradient-heuristic search and conventional fine-tuning. **Read `PROTOCOL.md` first** (the frozen protocol;
changes after the freeze appear only in its dated "Deviations" section). Negative results are valid outcomes.

## Setup (Windows PowerShell)

```powershell
cd C:\WSP-CNN
py -3.12 -m venv venv                        # Python 3.12 (3.11 also works)
.\venv\Scripts\Activate.ps1                  # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.lock.txt         # exact pinned environment (CUDA 12.4 wheels; ~2.5 GB download)
# or the looser pins: pip install -r requirements.txt
copy .env.example .env                       # then fill in LLM_PROVIDER, LLM_MODEL and the API key (never commit .env)
python scripts\record_env.py                 # writes results\env.json
python -m pytest -q                          # all tests (about 8 minutes on a 4 GB GPU laptop)
```

Notes: `data/`, `checkpoints/`, raw `results/` and `.env` are git-ignored. Keep the laptop plugged in and awake during
long runs (the runners ask Windows not to sleep). `num_workers` is 2 in `configs/config.yaml`; use 0 if workers hang.

## Reproduction, step by step

All settings live in `configs/config.yaml` (frozen Pydantic model; its hash is stored with every run).

```powershell
python main.py --experiment make_splits                 # stratified 45,000 / 2,500 / 2,500 split (split seed 2026) + diagnostic subset
python main.py --experiment baseline --seed 0           # E0 common checkpoint for a seed (repeat for seeds 1-4)
python scripts\run_all.py --plan-only                   # estimate of LLM calls, USD and hours; progress table
python scripts\run_all.py                               # E0, E1, PC, E2, E3, E4, E5, E6, E6-RS, E7 for seeds 0-4 (asks to confirm)
python main.py --experiment audit                       # budget audit; exits non-zero on any violation
```

`scripts\run_all.py` is resumable (a `done.json` with hashes marks each finished run), retries steps that fail before
writing output, and kills steps that go silent for 15 minutes. Single arms: `main.py --experiment
{pure_wsp,hybrid_wsp,random_edit,random_search,last_layer,last_layer_rs,grad_heuristic,finetune,positive_control} --seed S`
(add `--dry-run` for the scripted fake LLM; `positive_control --method all` runs Pure WSP, E5 and E7 on each fault).

### Freeze and the single test-set evaluation

```powershell
git add -A; git commit -m "Freeze"                      # the freeze refuses to run on a dirty tree
python scripts\freeze_protocol.py                       # writes results\FROZEN_PROTOCOL.lock and prints its SHA-256 (the token)
python main.py --experiment final_eval --token <sha256> # evaluates every frozen checkpoint once on the 10,000 test images
```

`final_eval` is the only module that may import `get_test_loader` (enforced by a static test), verifies every
checkpoint hash against the lock before touching the test set, writes read-only results to `results/test/`, and
refuses to re-run unless `--i-understand-this-is-a-deviation` is passed (which appends a dated entry to PROTOCOL.md).

### Statistics, figures and tables

```powershell
python scripts\reproduce_tables.py          # results\tables\*.csv, results\tables\results.md, results\figures\*.png|pdf, manifest.json
python scripts\reproduce_tables.py --check  # regenerates into a scratch folder; fails if any file differs from the manifest
```

Every number in the tables comes from `results/test/*`, `results/*/summary.json` and `results/PC/*`; nothing is typed by hand.
Statistics (`src/analysis/stats.py`): paired bootstrap (10,000 resamples, seeded), exact McNemar per seed, Holm over the
four primary claims, across-seed t-interval, Wilcoxon (low power with 5 seeds), Cohen's d, parameter efficiency.

### Ablations (validation only, exploratory; seeds 0-2)

```powershell
python scripts\run_ablations.py --plan-only            # list, run count and USD estimate
python scripts\run_ablations.py                        # A1-A3, A5-A9 as config overrides (asks to confirm)
```

## Layout

`configs/` YAML config and fault specs, `src/{data,models,diagnostics,wsp,training,experiments,analysis,utils}/`,
`prompts/` versioned LLM prompts (hashed into the lock), `docs/EDIT_DSL.md` the edit language given to the LLM,
`scripts/` entry points, `tests/`, `results/` (run summaries, audit trails, tables, figures), `results/llm_cache/`
(every LLM request and response, keyed by SHA-256), `main.py`.

## Release package

Version-controlled: code, configs, prompts, `PROTOCOL.md`, `requirements*.txt`, `results/env.json`, the registry and model card.
Archive separately (large or generated): `results/` (edit logs `history.jsonl`, `summary.json`, `llm_cache/`, `test/`, `tables/`,
`figures/`, `FROZEN_PROTOCOL.lock`) and `checkpoints/` (E0/E1 full checkpoints; editing arms are stored as sparse
`final_delta.npz` files rebuilt and hash-verified against their start model).

## Test-set isolation

The CIFAR-10 test set is reachable only through `get_test_loader(token, cfg)`, which raises `PermissionError` unless
`results/FROZEN_PROTOCOL.lock` exists and `token` is its SHA-256. `tests/test_test_isolation.py` and
`tests/test_stage8.py` statically scan the code to ensure no other module references it.
