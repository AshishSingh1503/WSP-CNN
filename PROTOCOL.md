# WSP-CNN Research Protocol (Pre-registration)

| Field | Value |
|---|---|
| Status | **DRAFT v0.1, awaiting Gate G0 approval** |
| Drafted | 2026-11-13 |
| Frozen on | _(fill at G0: date + git commit hash)_ |
| Amendments | Only in Section 16 (Deviations), dated, appended at the bottom. Nothing above Section 16 is edited after freeze. |

**Study in one sentence.** We test whether an LLM, returning schema-validated JSON edits that our own engine applies to a cloned ImageNet-pretrained AlexNet adapted to CIFAR-10, adds value beyond random editing, budget-matched random search, a gradient-heuristic search, and conventional fine-tuning. **Negative results are valid outcomes and will be reported as such.**

Source of the WSP idea: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; MIT CSAIL blog, Nov 2026). It is a blog post, not peer-reviewed; see LITERATURE.md.

### Freeze mechanics

At G0 and again before the first non-pilot run, a **freeze manifest** (`protocol_manifest.json`) is written. It contains the SHA-256 of: PROTOCOL.md, every file in `configs/`, every file in `prompts/`, the three split index files, and (once created) the hidden PC fault-spec files. At final evaluation (Stage 8) the manifest is written to `results/FROZEN_PROTOCOL.lock`; the SHA-256 of that file is the **final-eval token** that the guarded test loader requires (Section 13). The prompt hashes in Section 10 are filled into the manifest when the prompts exist; this is a freeze step, not a deviation, and must happen before any seed 0–4 run.

---

## 1. Research questions and hypotheses

Definitions used throughout: **E0** is the common checkpoint (Section 4); "accuracy" is top-1 accuracy on the official 10,000-image CIFAR-10 test set unless stated otherwise; a "run" is one arm, one seed, one budget; all hypotheses are one-sided (improvement) unless stated.

| RQ | Question | Hypothesis | Null hypothesis |
|---|---|---|---|
| RQ1 | Can an LLM with no gradient information (Pure WSP, E2) improve a common checkpoint? | **H1**: test accuracy of E2 > E0. | **H1₀**: E2 accuracy ≤ E0 accuracy. |
| RQ2 | Does Pure WSP beat budget-matched random search over the same edit DSL? | **H2**: test accuracy of E2 > E5. | **H2₀**: E2 accuracy ≤ E5 accuracy. |
| RQ3 | Does an LLM given gradient diagnostics (Hybrid WSP, E3) beat a gradient-heuristic search without an LLM (E7) at matched budget? | **H3**: test accuracy of E3 > E7. | **H3₀**: E3 accuracy ≤ E7 accuracy. |
| RQ4 | Can Pure WSP repair a deliberately injected, localized fault? | **H4**: repair success probability > 0.20 (success defined in Section 5). | **H4₀**: repair success probability ≤ 0.20. |
| RQ5 | Does WSP gain depend on the edit budget? | **H5**: gain of E2 over E0 increases monotonically with budget across 0.001%, 0.01%, 0.1%, 1%. | **H5₀**: no monotone dose-response; the budget level has no effect on gain. |
| RQ6 | How do WSP edits compare to fine-tuning (E1) and last-layer-only editing (E6) in accuracy per edited parameter and in compute? | _Descriptive; no formal hypothesis._ | _n/a_ |

H1–H4 are tested as primary claims (Section 2). H5 and RQ6 are exploratory.

## 2. Primary vs exploratory claims

**Primary claims (exactly four; Holm-corrected together, Section 12):**

| ID | Claim | Comparison | Arms, seeds |
|---|---|---|---|
| P-a | Pure WSP vs baseline | E2 vs E0, test accuracy | 5 seeds, default budget |
| P-b | WSP vs random search at equal candidate budget | E2 vs E5, test accuracy | 5 seeds, default budget |
| P-c | Hybrid WSP vs gradient-heuristic search | E3 vs E7, test accuracy | 5 seeds, default budget |
| P-d | PC fault repair rate | Pure WSP (E2 configuration) on PC, repair success rate vs 0.20 | 3 faults × 5 seeds = 15 runs |

"WSP" in P-b means Pure WSP (E2); "WSP" in P-c means Hybrid WSP (E3). These assignments are fixed here to avoid post-hoc choice of the better arm.

**Exploratory claims (everything else, labelled as exploratory in every table and figure):** E3 vs E2; E3 vs E5; E2 vs E7; E2/E3 vs E4; E6 vs E2; E1 as reference; A6 budget ablation (H5); accuracy per edited parameter; per-operation and per-layer analyses; per-fault PC breakdown; Hybrid and informed-prompt variants on PC; best-of-5 random search; cost comparisons; any analysis not listed above. Exploratory results are reported with unadjusted p-values and are never used to support a headline claim.

## 3. Data protocol

| Split | Size | Source | Used for |
|---|---|---|---|
| `train` | 45,000 (4,500 per class) | official CIFAR-10 train set | E0 warm-up, E1 |
| `val_edit` | 2,500 (250 per class) | official CIFAR-10 train set | WSP/search feedback and candidate acceptance |
| `val_select` | 2,500 (250 per class) | official CIFAR-10 train set | final checkpoint selection and stopping only; never shown to the LLM |
| `test` | 10,000 (1,000 per class) | official CIFAR-10 test set | evaluated once per frozen checkpoint (Section 13) |

- **Construction.** Two-stage sklearn `StratifiedShuffleSplit` on the official train labels, both stages with `random_state` = split seed **2026**: stage 1 holds out 5,000 images (500 per class) from the 50,000; stage 2 splits those into `val_edit` and `val_select` (250 per class each); the rest is `train` (4,500 per class). The split is identical for all arms and seeds.
- **Index files** are saved under `data/splits/` (`train_indices.npy`, `val_edit_indices.npy`, `val_select_indices.npy`, plus `manifest.json` with counts, class counts, library versions and SHA-256 of each file); hashes go in the freeze manifest. Existing files are verified, never regenerated. The disjointness and sizes are asserted by tests.
- **Preprocessing (all splits).** 32→224 bilinear resize, ImageNet mean/std normalization. Augmentation (train only): random crop 32 with padding 4, then horizontal flip, both before the resize.
- **Evaluation precision.** All val_edit, val_select and test evaluation runs in eval mode in **fp32** (no AMP), so that the 0.002 acceptance threshold is not comparable to autocast noise. AMP is used only for training.
- **Test-set access** is possible only through the guarded loader (Section 13). No WSP, search, tuning or selection code may import it.

## 4. Arms and what each may see

**Common checkpoint (E0).** torchvision AlexNet with `AlexNet_Weights.IMAGENET1K_V1` (weight file SHA-256 recorded). `classifier.6` is replaced by `Linear(4096, 10)`, initialized from the run seed. Warm-up trains **only `classifier.6`** (all other parameters frozen, model in train mode so dropout is active): 10 epochs, batch size 64, SGD momentum 0.9, lr 0.01, weight decay 5e-4, cosine annealing to 0, AMP fp16, no label smoothing; checkpoint = last epoch (no validation-based selection). Total parameters: **57,044,810**. The checkpoint's SHA-256 is recorded, and every arm refuses to start if the hash is missing or mismatched. One E0 per seed (5 total).

| Arm | Description | Information it may use | Gradients? | LLM? |
|---|---|---|---|---|
| **E0** | Common checkpoint, unedited | n/a | n/a | no |
| **E1** | Full fine-tuning from E0 (reference) | `train` (with augmentation); val_select for epoch selection | yes (optimizer) | no |
| **E2** | Pure WSP | Weight summaries (see below), val_edit aggregate metrics, history of its own prior candidates and outcomes, DSL spec, remaining budgets | **never** | yes |
| **E3** | Hybrid WSP | Everything in E2 **plus** gradient diagnostics of the val_edit loss at the current accepted state (per-layer gradient norms; per-output-unit gradient norms for the top 20 units network-wide) | diagnostics only | yes |
| **E4** | Random editing | Nothing (no feedback). Per trial: 10 random edits (one per iteration slot) drawn from the DSL sampler of E5 and applied in sequence with no evaluation or acceptance, subject to the edit budget. 5 independent trials per seed; the seed's value is the mean over trials. | no | no |
| **E5** | Random search, budget-matched | Same DSL, bounds, edit budget, acceptance rule and val_select final selection as E2; candidates drawn at random (layer uniform over the 8 weight layers, op uniform over the 4 non-trivial ops, unit/entry indices uniform, magnitudes log-uniform within bounds). 4 candidates × 10 iterations = 40 slots. 5 independent runs per seed; the seed's value is the mean over runs. | no | no |
| **E6** | Last-layer-only editing | E2's inputs; edits restricted to `classifier.6.weight` and `classifier.6.bias` (exploratory) | never | yes |
| **E7** | Gradient-heuristic search, no LLM | Same information as E3's diagnostics, same DSL, bounds, budgets, acceptance rule. Heuristic (frozen in `configs/e7.yaml`, hashed): at each iteration compute g = ∇θ CE on val_edit at the current accepted state; rank output units (conv channel / linear row) by ‖g_unit‖₂; take units greedily from the top until the remaining edit budget would be exceeded or 64 entries are reached; propose 4 `add_delta` candidates delta = −r·‖θ_units‖·g/‖g‖ for relative step r ∈ {0.005, 0.01, 0.02, 0.04}, clipped to the DSL elementwise bound. | yes (val_edit only) | no |
| **PC** | Positive control (Section 5) | Faulted checkpoint, same information and prompt as E2 | never (E2 config) | yes |

**What E2/E3 see in the prompt (fixed template, hash in Section 10).** For each of the 16 parameter tensors: name, shape, mean, std, min, max, L2 norm; per-output-unit L2 norms for the 5 conv layers and `classifier.6`; 5/25/50/75/95th percentile of per-row norms for `classifier.1` and `classifier.4`. val_edit loss, accuracy, per-class accuracy, and the 10×10 confusion matrix. No images, no per-example predictions, no activations, no val_select, no test data. E3 differs from E2 **only** by the added gradient-diagnostic block.

**Reference-only status of E1.** E1 is an unconstrained reference ("what conventional training achieves"), run for 10 epochs from E0 with batch size 64, SGD momentum 0.9, lr 0.001, weight decay 5e-4, cosine annealing to 0, AMP, best epoch by val_select accuracy. It is **not** a matched-budget competitor and no primary claim involves it. The main comparison is **WSP vs search under matched budgets** (E2 vs E5, E3 vs E7), plus **accuracy per edited parameter** (Section 7), reported against E1 and E6 descriptively.

**Matched-budget rule.** E5 and E7 receive the full 40 candidate slots and the 3-consecutive-unproductive early stop is **disabled** for them. WSP arms may stop early (Section 9). This is deliberately conservative: it can only give the baselines more search than WSP. Accuracy-vs-candidates-used curves are reported for all arms.

## 5. Positive control (PC)

A known, localized fault is injected into each seed's E0 to create a faulted checkpoint. WSP is then run on the faulted checkpoint with the same prompt as the main E2 (no mention that a fault exists, and no location). An "informed" variant ("a localized fault may exist") is exploratory.

| Fault | Definition | Scalars changed | Exactly invertible in the DSL? |
|---|---|---|---|
| **F1** | Negate (weight and bias) 8 output channels of `features.10` (256→256, 3×3; channels drawn with `np.random.default_rng(10000 + seed)`, without replacement, from 0–255) | 8 × 2,305 = 18,440 | Yes: `scale_output_channel` factor −1.0 |
| **F2** | Add −8.0 to `classifier.6.bias[3]` (class "cat") | 1 | Yes: `bias_shift` +8.0 |
| **F3** | Zero (weight and bias) 16 output channels of `features.3` (64→192, 5×5; channels drawn with `np.random.default_rng(20000 + seed)`, without replacement, from 0–191) | 16 × 1,601 = 25,616 | **No**: zeroed weights cannot be recovered from the checkpoint; only compensation is possible |

**Hidden fault spec.** For each (fault, seed) a JSON file under `checkpoints/pc/fault_specs/` records the fault type, tensor, indices and values. These files are never read by any prompt-building, LLM-provider or WSP code path (enforced structurally and tested), and their hashes are in the freeze manifest.

**Reporting.** For each (fault, seed): accuracy of clean E0, accuracy of the faulted checkpoint (the drop), and accuracy of the repaired final checkpoint, all on test, evaluated once (Section 13). **Precondition:** a fault is meaningful only if it drops val_select accuracy by ≥ 5.0 percentage points; this is checked at injection time before any WSP run. A fault failing the precondition is dropped from P-d and the drop is recorded as a deviation.

**Repair success** (per run): recovered fraction ρ = (acc_repaired − acc_faulted) / (acc_clean − acc_faulted) ≥ **0.80**, on test accuracy.
**Primary P-d statistic:** number of successes out of 15 runs (3 faults × 5 seeds), tested with an exact one-sided binomial test against H4₀: p ≤ 0.20. Runs are treated as independent, which is optimistic (same fault type and same LLM); this is flagged in Section 15. Per-fault repair rates, ρ distributions, and the same PC applied to E3, E5, E7 as controls are exploratory. Fault F3 is reported separately because exact repair is impossible by construction.

The edit budget for PC runs is counted relative to the **faulted** checkpoint (Section 7).

## 6. Edit DSL

The LLM returns JSON validated by a strict Pydantic v2 schema (unknown fields forbidden). A response contains ≤ **4** candidates; a candidate is a list of ≤ **3** operations. Targets are the 16 tensors: `features.{0,3,6,8,10}.{weight,bias}`, `classifier.{1,4,6}.{weight,bias}`. σ_T denotes the standard deviation of the target tensor at the current accepted state.

| Operation | Parameters | Bounds | Scalars changed |
|---|---|---|---|
| `add_delta` | tensor; list of (index, delta) | ≤ 64 entries; indices in range and unique; \|delta\| ≤ 0.5·σ_T | ≤ 64 |
| `scale_output_channel` | layer (one of the 8 weight layers); channel indices; factor | ≤ 8 distinct channels; factor ∈ [−2.0, 3.0], factor ≠ 1.0; scales that channel's weights **and** bias | ≤ 8 × (fan_in + 1) |
| `bias_shift` | bias tensor; indices; shift | ≤ 16 entries; \|shift\| ≤ 10.0 for `classifier.6.bias`, ≤ 1.0 for every other bias | ≤ 16 |
| `low_rank_update` | weight tensor (conv weights flattened to out × (in·kh·kw)); row indices; column indices; factors U, V; scalar α | ≤ 16 rows, ≤ 32 columns, rank ≤ 2, factor entries in [−1, 1]; composed update confined to the rows × columns block; every \|Δ\| ≤ 0.5·σ_T (reject, do not clip) | ≤ 16 × 32 = 512 |
| `no_edit` | none | none | 0 |

Any violation (out-of-range index, bound exceeded, unknown tensor/op, non-finite value, more than 4 candidates or 3 ops) makes the candidate invalid. An invalid candidate consumes its candidate slot, is logged with the reason, is not evaluated, and counts as rejected. Edits are applied to a clone of the current accepted model; the LLM never executes code or touches files.

## 7. Edit-counting rule

- **Edit size** = number of scalar parameters whose fp32 value differs (bitwise) between the edited and the reference model. Example: scaling one output channel of `features.10` (256 input channels, 3×3 kernel) changes up to 256·3·3 + 1 = **2,305** scalars (scaling by exactly 1.0, or scaling an exactly-zero value, changes nothing and is not counted). *(Correction: the briefing's example of 1,729 is the per-channel size for `features.6`, which has 192 input channels.)*
- **Cumulative edit size** of a run = number of scalars that differ between the current accepted state and the **start** checkpoint (the union of changed positions; re-editing an already-changed scalar does not count twice; reverting a scalar to its original value un-counts it). For PC runs the start checkpoint is the faulted checkpoint.
- A candidate whose cumulative size would exceed the edit budget is invalid.
- **Relative update norm** = ‖θ_final − θ_start‖₂ / ‖θ_start‖₂, reported **per tensor**, **per layer** (weight and bias together), and **globally** over all 57,044,810 parameters. For PC runs, also the distance to the clean checkpoint.
- **Accuracy per edited parameter** = (test accuracy gain over start, in percentage points) / (cumulative edit size), reported alongside the raw gain vs log₁₀(edit size). For E1 the edit size is the number of scalars that changed relative to E0.

## 8. Budgets

| Item | Value |
|---|---|
| Max iterations per run | 10 |
| Max candidates per iteration | 4 |
| Max candidate slots per run | 40 (every proposed candidate consumes a slot, valid or not) |
| Max accepted candidates per iteration | 1 |
| Edit budget (default) | 0.1% of 57,044,810 = ⌊57,044.81⌋ = **57,044** scalars |
| A6 ablation budgets | 0.001% = **570**; 0.01% = **5,704**; 0.1% = **57,044**; 1% = **570,448** scalars |
| LLM calls per iteration | 1, plus at most 1 retry if the reply is not valid JSON or fails schema validation (the retry is logged and does not consume candidate slots; if it also fails, the iteration is unproductive) |

A6 is exploratory: E2 and E5 at 0.001%, 0.01% and 1% for 5 seeds each; the 0.1% results are reused from the main runs. At 570 scalars a single `features.10` channel scale (2,305) is infeasible; this is inherent to the ablation.

## 9. Acceptance, selection and stopping

Each iteration: the LLM (or sampler) proposes up to 4 candidates; each is applied independently to a clone of the **current accepted state** and evaluated on val_edit (loss and accuracy).

- **Eligible:** val_edit cross-entropy **decreases by at least 0.002** relative to the current accepted state **and** val_edit accuracy is **≥** the current accepted state's val_edit accuracy.
- **Accepted:** among eligible candidates, the one with the lowest val_edit loss (ties: smaller edit size, then earlier index). At most one acceptance per iteration; the accepted candidate becomes the new current state. If none is eligible the iteration is **unproductive**.
- **val_select** is evaluated **only for accepted states and for the start state**, is logged, and is never used for acceptance or shown to the LLM. Rejected candidates log val_select as null.
- **Final checkpoint** = the state with the highest val_select accuracy among the start state and all accepted states (ties: the earlier state, i.e. the smaller edit).
- **Stopping:** the candidate-slot budget or iteration budget is exhausted, **or** 3 consecutive unproductive iterations (an iteration containing only `no_edit` candidates is unproductive). The early stop is disabled for E4, E5 and E7 (Section 4).

If the final checkpoint is the start state, the run is a "no-improvement" run: the accuracy difference to E0 is exactly 0 and the run is retained in all analyses.

## 10. LLM settings

| Setting | Value |
|---|---|
| Provider / model identifier | Anthropic API, **`claude-opus-5-5`** (same model for E2, E3, E6 and PC). The model string actually returned by the API is recorded for every call. |
| Temperature | **0**; if the API rejects it for this model, the lowest supported value is used and recorded |
| Max output tokens | **16,384**; a truncated response is treated as invalid JSON (retry rule, Section 8) |
| Tools / code execution / file access | none; JSON text in, JSON text out |
| Response format | single JSON object validated by the strict Pydantic schema |
| Prompt files | `prompts/system.md`, `prompts/pure.md`, `prompts/hybrid.md`, `prompts/schema.md` |
| Prompt SHA-256 | _recorded in the freeze manifest before the first non-pilot run_ (`system.md`: PENDING, `pure.md`: PENDING, `hybrid.md`: PENDING, `schema.md`: PENDING) |
| Prompt development | Only on pilot seeds **100–102** and only with val_edit feedback; never test or val_select; pilot runs are reported in an appendix. Prompts are frozen before any seed 0–4 run. |

The provider layer is abstracted so the model can be swapped, but swapping it after freeze is a deviation.

## 11. Seeds

- Primary arms use seeds **0, 1, 2, 3, 4**.
- The seed controls: E0 head initialization, data order and augmentation (E0 and E1), and every random control (E4 trials, E5 runs). The data split (seed 2026) stays fixed. The PC fault channel draws use seeds 10000+seed and 20000+seed (Section 5).
- The WSP run for seed s edits E0 of seed s. LLM sampling cannot be seeded; one LLM run per (arm, seed, budget).
- Determinism: all RNGs seeded; `torch.backends.cudnn.deterministic = True`, `benchmark = False`; `torch.use_deterministic_algorithms(True, warn_only=True)`; Windows `num_workers` fixed in config. Python, PyTorch, torchvision, CUDA, cuDNN, GPU name and driver versions are written into every results file. Bitwise GPU reproducibility is not guaranteed; the checkpoint hash, not the training process, defines the starting state.

## 12. Statistics

- **Unit of analysis for accuracy claims:** the 10,000 test examples. For arms A and B and each example i, d_i = mean over the 5 seeds of [correct_A,s,i − correct_B,s,i] (for E4/E5, correctness is first averaged over their independent trials/runs within the seed).
- **Effect size and CI:** mean of d_i in percentage points; **95% CI by paired bootstrap over test examples, 10,000 resamples** (bootstrap RNG seed 2026, percentile method).
- **p-value:** one-sided paired-bootstrap p = (1 + #{resamples with mean ≤ 0}) / (1 + 10,000).
- **McNemar test** (exact binomial on discordant pairs, two-sided) per seed for paired predictions, reported with the discordant counts (b, c) and odds ratio b/c. It is supporting evidence, not the Holm input.
- **Per-seed paired differences** and their mean ± SD (SD over 5 seeds, ddof = 1) in pp, plus per-seed raw tables for every arm (val_edit, val_select, test accuracy, edit size, relative norm, evaluations, cost).
- **P-d** uses the exact one-sided binomial test (Section 5), reported with a Clopper–Pearson 95% interval.
- **Multiplicity:** **Holm correction** across the 4 primary p-values (P-a, P-b, P-c, P-d) at familywise α = 0.05. Exploratory p-values are reported unadjusted and labelled.
- **Decision rule for P-a, P-b, P-c** ("supported" only if all three hold): (1) Holm-adjusted p < 0.05; (2) mean paired difference ≥ **+0.3 pp** (smallest effect of interest); (3) the per-seed difference is positive in **≥ 4 of 5 seeds**. Otherwise the claim is reported as **not supported**; failing (2) or (3) with p < 0.05 is reported as "statistically detectable but not practically or consistently significant". The bootstrap is conditional on the 5 trained checkpoints and does not capture between-seed variance, which is why condition (3) exists.
- Cohen's d_z of the 5 per-seed differences is reported descriptively. With 5 seeds no parametric seed-level test is used.
- **H5 (exploratory):** gain at each of the 4 budgets with CIs, and Spearman ρ between log₁₀ budget and per-seed gain over the 20 (budget, seed) points.

## 13. Test-set rule

1. The test set is reachable only via the guarded loader, which raises `PermissionError` unless `results/FROZEN_PROTOCOL.lock` exists and the supplied final-eval token equals its SHA-256 (Section "Freeze mechanics"). Tests assert that WSP, search, tuning and selection modules cannot reach it.
2. **Order:** all arms for all seeds (E0–E7, PC, A6 and every other planned run) are completed and their final checkpoints hashed and listed in the manifest. Only then is the test set evaluated, **once per frozen checkpoint**, in one pass; per-example predictions are saved to disk and all statistics are computed from the saved predictions, never from a re-evaluation.
3. Any re-run, re-training, re-selection or protocol change made after any test result has been seen is reported in Section 16 and flagged in every table that depends on it.
4. val_select is used only for final-checkpoint selection and E1 epoch selection; it is not used for prompt development, hyperparameter choice, or acceptance.

## 14. Compute and cost reporting

For every run: GPU time (CUDA-event / wall-clock seconds spent on GPU work), total wall time, number of val_edit evaluations, number of val_select evaluations, number of gradient passes, candidate slots used, number of LLM calls (including retries), input/output tokens, and USD cost (tokens × the provider's published prices at run time, with the price table recorded in the results file). Totals per arm are reported, and accuracy gain is reported per GPU-second, per validation evaluation, and per USD.

## 15. Threats to validity

1. **Validation overfitting.** Accepting edits on 2,500 val_edit images with a 0.002 loss threshold can fit noise; ≤ 40 candidates and the separate val_select check mitigate but do not remove this. The val_edit→test gap is reported per arm.
2. **Distributed-vs-localized mismatch.** WSP is reported to work best on small localized faults. The E0 deficit (head-only warm-up of a frozen ImageNet backbone) is a diffuse distribution shift, not a localized fault, so null results on P-a to P-c may reflect this mismatch rather than a failure of WSP in general. PC exists to separate these cases. F3 is not exactly repairable and F1/F2 are exactly invertible by DSL design; PC success therefore tests competence on easy cases.
3. **LLM nondeterminism and model drift.** Temperature 0 is not bitwise deterministic; API model snapshots can change; one LLM run per (arm, seed, budget).
4. **Small seed count.** 5 seeds limit seed-level inference; the bootstrap ignores seed variance; the 4-of-5 consistency rule partially compensates. P-d runs are treated as independent though they share one LLM and fault family.
5. **32→224 resize.** Upsampling CIFAR images to ImageNet resolution makes the setting unrepresentative of native-resolution CIFAR training and inflates compute.
6. **Prompt and DSL design.** Results depend on the chosen DSL bounds and prompt; the bounds (e.g. ≤ 64 delta entries) cap what any edit can do. Pilot tuning on seeds 100–102 could leak design choices; it is disclosed.
7. **Baseline strength.** E5/E7 are specific instantiations; a stronger search or a tuned heuristic could change conclusions. The E7 heuristic is frozen before runs and not tuned afterwards.
8. **Matched budget definition.** Candidate-slot matching ignores that LLM proposals are costlier in wall time and USD; Section 14 reports those separately.
9. **Evaluation-precision and hardware.** fp32 evaluation and deterministic flags reduce but do not remove numerical differences across GPUs/drivers.
10. **Frozen-backbone checkpoint.** Most E0 error may be addressable only by feature learning, favouring E1 trivially; this is why E1 is reference-only.

## 16. Deviations

_(Empty at freeze. Append dated entries only, newest last, in this format.)_

```
### YYYY-MM-DD — <short title>
- What changed (section, old value → new value):
- Why:
- Were any test results seen before this change? yes / no
- Affected claims / tables:
```

### 2026-11-13 — D1: E0 warm-up recipe (Section 4)
- What changed: E0 head warm-up now trains on cached 4096-d penultimate features (computed once, eval transform, NO augmentation, float16) with AdamW lr 1e-3, weight decay 1e-4, batch 256, 30 epochs, cosine schedule; the epoch with the best val_select accuracy is kept. Previously: SGD lr 0.01, 10 epochs, last epoch, with crop/flip augmentation.
- Why: the backbone is frozen, so features can be computed once; this makes warm-up take seconds. Full-model forward vs cached-feature head agreement is checked (>= 99.5%; observed 100%).
- Were any test results seen before this change? No (the test set has not been touched).
- Affected claims / tables: E0 itself, and therefore every arm that starts from it.

### 2026-11-13 — D2: val_select used for E0 epoch selection and E1 tuning (Sections 3 and 13)
- What changed: val_select is also used (a) to pick the E0 epoch (D1) and (b) to score the E1 learning-rate sweep (seed 0 only) and select E1 epochs. Section 13 limited val_select to final-checkpoint selection and E1 epoch selection.
- Why: val_select is the only held-out split besides test; the test set must stay untouched.
- Were any test results seen? No.
- Affected claims / tables: E0 and E1 only. No WSP or search arm uses val_select for acceptance (it is used only to select the final state among accepted ones, as in Section 9).

### 2026-11-13 — D3: E1 recipe (Section 4)
- What changed: E1 uses AdamW with two parameter groups (features at backbone_lr, classifier at classifier_lr), weight decay 0.01, 1 epoch linear warm-up then cosine decay, 15 epochs, AMP, batch 64, best epoch by val_select. Previously: SGD lr 0.001, 10 epochs, batch 128. The learning rates come from a seed-0 sweep over backbone_lr in {1e-5, 3e-5, 1e-4} and classifier_lr in {1e-4, 1e-3}, 8 epochs each, scored on val_select; results in results/E1/tuning_sweep.csv.
- Sweep result (seed 0, 8 epochs, best val_select accuracy per config): 1e-5/1e-4 0.9060; 1e-5/1e-3 0.9024; 3e-5/1e-4 0.9112; 3e-5/1e-3 0.9120; 1e-4/1e-4 0.9208; 1e-4/1e-3 0.9220 (backbone_lr/classifier_lr).
- Chosen values: backbone_lr = 1e-4, classifier_lr = 1e-3 (highest val_select accuracy; ties would have gone to the smaller rates). The margin over 1e-4/1e-4 is 0.12 points, within noise for 2,500 images; the winner sits at the upper edge of the backbone grid, so a higher backbone rate was not explored. Set in configs/config.yaml on 2026-11-13.
- Were any test results seen? No.
- Affected claims / tables: E1 only (reference arm, no primary claim).

### 2026-11-13 — D4: Edit DSL bounds replaced (Section 6)
- What changed: the Section 6 DSL is replaced by the Stage 5 DSL (configs/config.yaml `dsl:`, rendered in docs/EDIT_DSL.md): add_delta <= 512 indices with |delta| <= 0.05 and <= 0.5|value| + 0.01; scale_output_channel factors in [0.5, 1.5] (<= 8 channels, 0.0 only if allow_zero_channel); bias_shift <= 1.0 for classifier.6 and <= 0.2 elsewhere (<= 16 indices); low_rank_update rank <= 4 on a rows x cols block (<= 32 x 64) with ||update||_F <= 0.02 ||W||_F; <= 3 operations per proposal. Edit sizes are counted bitwise (a scalar counts if its stored bits change).
- Why: the Section 6 factor range [-2, 3] and bias bound were chosen before implementation; the narrower, value-relative bounds keep single edits small.
- Consequence: positive-control faults F1 (negate 8 channels: needs factor -1) and F2 (class bias -8: needs +8) can no longer be repaired exactly by one edit; F2 needs about 8 accepted bias shifts. F3 was never exactly repairable. PC results must be read with this in mind.
- Were any test results seen? No.
- Affected claims / tables: all editing arms; P-d.

### 2026-11-13 — D5: LLM settings (Section 10)
- What changed: temperature is NOT sent: since 2026-10-21 the Anthropic API rejects temperature, top_p and top_k on every model, so no sampling control exists ("temperature 0 or lowest supported" is not available). Effort is set to medium (the claude-opus-5-5 default); thinking cannot be disabled on this model. Replies are requested as JSON in the prompt and validated by Pydantic (no structured-output mode), with one repair retry. Per-run caps: 60 calls and USD 10. Every request/response is cached under results/llm_cache/ and replayed for identical requests.
- Providers: LLM_PROVIDER/LLM_MODEL/key come from .env. The Anthropic provider with claude-opus-5-5 is the study model. A Groq / xAI / generic OpenAI-compatible provider also exists for pilot-only runs; any run on a model other than claude-opus-5-5 must be labelled pilot-only and excluded from the primary claims.
- Were any test results seen? No.
- Affected claims / tables: E2, E3, E6, PC.

### 2026-11-13 — D6: Control arms (Section 4)
- E4: K = 10 random edit sets per seed (was 5 trials), matched to the seed's E2 final edit set: same operation types, same scalar counts per layer type (Conv2d / Linear), global relative update norm within +-10%; evaluated once on val_select, no selection. If E2 made no edit for a seed, the median E2 edit set across seeds is used and noted.
- E5: random valid proposals from the DSL (distributions in configs/config.yaml `random_search:`), same controller, 4 candidates x 10 iterations = 40 slots, same acceptance rule, no early stop (as in Section 4). E6-RS (E5 restricted to classifier.6) is added as the fair comparison for E6.
- E7: replaces the Section 4 E7 recipe. Each iteration ranks output channels by ||grad W_c||_2 / sqrt(numel_c) on the 1,000-image diagnostic subset and proposes three scale_output_channel candidates with factor clip(1 - alpha * sign(<grad_c, w_c>)) for alpha in {0.05, 0.1, 0.2} on the top-4 channels, plus one bias_shift on classifier.6 with shifts -alpha * grad_b / max|grad_b| (alpha cycling over {0.05, 0.1, 0.2}); if nothing was accepted the next iteration moves down the ranking. Same controller, budgets and acceptance rule, no early stop.
- E5 and E7 also run on the three PC faults with the same budgets.
- Were any test results seen? No.
- Affected claims / tables: P-b, P-c, exploratory comparisons.

### 2026-11-13 — D7: Evaluation implementation (Sections 3 and 8)
- What changed: val_edit and val_select are preprocessed once per process and held in RAM; every evaluation then runs the same fp32 model forward on those tensors. Numerics are identical to the loader path (same transform, same batches).
- Why: a full val_edit pass through the image loader takes about 30 s on this laptop; the cached path takes a few seconds.
- Were any test results seen? No.

### 2026-11-13 — D8: sparse checkpoints and a single-pass final evaluation (Sections 4, 13)
- What changed: editing arms (E2, E3, E5, E6, E6-RS, E7 and the PC variants) save their final state as the changed scalars relative to their start model (`final_delta.npz` + sidecar with the full-state SHA-256) instead of a full 230 MB checkpoint; E4 stores its proposals so each random model is rebuilt; PC faulted starts are rebuilt deterministically from E0 + `configs/faults.yaml`. Every rebuilt state is verified against its recorded SHA-256 before evaluation. E0 and E1 keep full checkpoints.
- Final evaluation opens the test set once and passes every frozen checkpoint over it in a single sweep (models are switched between batches by sparse overlays); all numerics are the same fp32 forward as validation. `results/test/eval_ledger.jsonl` records one line per evaluation.
- Why: 5 seeds of full checkpoints would need roughly 20 GB on a disk that has 2-13 GB free; the edits are at most 57,044 scalars.
- Were any test results seen? No.
- Affected claims / tables: none (storage and evaluation mechanics only).

### 2026-11-13 — D9: execution differences found while running seeds 0-4 (Sections 4, 11)
- E5 (random search) was run once per seed (1 run, 40 candidate slots), not 5 runs per seed averaged. E6-RS likewise.
- E2 made an edit on only 1 of 5 seeds (seed 4: a 40,970-scalar edit of the classifier.6 head); on the other 4 seeds its final state is the unedited start model because the accepted candidates tied the start on val_select. E4 therefore borrows seed 4's E2 edit set for all seeds, by the rule "if E2 made no edits for a seed, use the median E2 edit set across seeds" (n = 1 available). E4 matches scalars per layer type approximately for scale operations (a channel is a whole row) and the relative norm within +-10%.
- Positive control, seed 0: the three E5 runs were invalidated by a bug (a CUDA tensor converted with .numpy() crashed every add_delta iteration), quarantined in results/_quarantine_cuda_numpy_bug/, and rerun with the fixed code. E7 seed 4 was killed mid-run when the session ended and was rerun from scratch (partial output in results/_quarantine_interrupted_session/). E5 seed 3 failed at startup (transient CUDA out-of-memory while pinning cached validation images) and was rerun; pinning of cached validation batches was then switched off (no numerical effect).
- E1 seed 1 stalled twice (data-loader worker processes accumulating under memory pressure) and resumed from its epoch-4/5 resume file; E1 data loaders were then made non-persistent. Resuming restores model, optimizer, scaler, scheduler and RNG state, so the schedule is unchanged, but the resumed epochs are not bitwise-identical to an uninterrupted run.
- Were any test results seen before these changes? No.
- Affected claims / tables: P-b, P-c and exploratory comparisons involving E4/E5/E6-RS; P-d (the seed-0 E5 PC rows come from the rerun).

### 2026-11-13 — D10: post-freeze code fix in the final-evaluation helper (Sections 11, 14)
- What changed: `SwitchingModel.set_variant` (src/models/delta.py) indexed a CPU tensor with GPU indices, so the first final_eval attempt (started 21:28, after the freeze) crashed in its first batch with a RuntimeError. The fix indexes the CPU base copy with `idx.cpu()`. A CUDA regression test was added (tests/test_stage8.py). No frozen checkpoint, config, split, metric or analysis rule was changed; the lock file and all 150 checkpoint hashes are unchanged and were re-verified at the start of the run.
- Were any test results seen before this change? No. The failed attempt wrote nothing (results/test did not exist afterwards) and printed no metric. The fix was checked on validation (val_select) batches only, with all 150 variants.
- Affected claims / tables: none. This is not a re-evaluation: the test set is evaluated once, by the fixed code.

