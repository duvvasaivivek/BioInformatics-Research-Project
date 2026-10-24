# Run Log: Dual-Barrier Multi-Task GNN Implementation

**Date:** 2026-10-24
**Repo state at start:** HEAD `e13f3a5 MDs Added`, working tree clean
**Repo state at end:** HEAD unchanged — **nothing committed, nothing pushed**
**Task:** implement the multi-task GNN plan in the repo, make everything run on CUDA,
and log the run.

---

## 1. Scope correction applied mid-run

The plan as approved targeted `Initial Model/`. That was wrong, and was corrected during
the run:

- **`Initial Model/` is the base-paper reference implementation only. It was left
  untouched.** Two files had been created there (`Initial Model/gnn/__init__.py`,
  `Initial Model/gnn/device.py`); both were deleted and `git diff HEAD -- "Initial Model/"`
  confirmed empty. No file in that directory was modified at any point.
- **All work went into `B3DB Based Model/`**, the project's own novel section.

### Folder rename to "Novel Section": not done

Offered as optional. I kept the name, because `B3DB Based Model` is referenced **74 times
across 6 files**, including `RESEARCH_PAPER_MANUSCRIPT.md`, `README.md`,
`ARCHITECTURE_FLOW_BLOCK_DIAGRAM.md`, `NOVELTY_ASSESSMENT.md`,
`DATASET_AND_FILE_PROVENANCE_CATALOG.md` and `docs/SCIENTIFIC_DOCUMENTATION.md`. Renaming
is a one-command `git mv` plus a find-and-replace that can be done at any time; doing it
now would churn the manuscript and all the architecture docs for no functional gain. The
decision is reversible and costs nothing to defer.

---

## 2. What was built

A new `gnn/` package inside `B3DB Based Model/`, plus three scripts. Nothing pre-existing
was modified.

| File | Lines | Purpose |
|---|---:|---|
| `gnn/__init__.py` | 46 | Package surface |
| `gnn/device.py` | 100 | CUDA selection, GPU tuning, memory accounting |
| `gnn/featurizer.py` | 320 | The single canonical SMILES → graph path |
| `gnn/data_assembly.py` | 256 | Canonicalisation, InChIKey join, corpus merge, masks |
| `gnn/splits.py` | 229 | Molecule-level random + scaffold splits, integrity assertions |
| `gnn/dataset.py` | 164 | Graph construction, disk cache, loaders |
| `gnn/model.py` | 181 | `MultiTaskGNN` — shared trunk, one head per barrier |
| `gnn/losses.py` | 139 | Masked per-task cross-entropy |
| `gnn/metrics.py` | 127 | Metric suite + bootstrap CIs |
| `scripts/train_multitask_gnn.py` | 461 | Training entry point (CUDA) |
| `scripts/verify_multitask_gnn.py` | 294 | 29-check verification suite |
| `scripts/compare_gnn_experiments.py` | 205 | Consolidated comparison table |
| **total** | **2,522** | |

### Infrastructure that did not previously exist

- **A canonical SMILES / InChIKey join key.** `InChIKey` was generated nowhere in the
  repository, and `core/features.py` has no canonicalisation helper. Without it the two
  benchmarks cannot be merged at molecule level.
- **One shared featurizer.** The reference implementation has two copies that had already
  drifted apart on the valence API (`GetExplicitValence()` vs
  `GetValence(Chem.ValenceType.EXPLICIT)`). `gnn/featurizer.py` has a compatibility shim
  and is the only extractor.
- **Persisted per-molecule predictions.** None existed anywhere in the repo, which blocked
  any error-stratification analysis. Every run now writes a `*_test_predictions.csv`.

---

## 3. CUDA

Everything numeric runs on the GPU. Device selection is centralised in `gnn/device.py`;
no other module decides.

```
device            : cuda:0
GPU               : NVIDIA GeForce RTX 4050 Laptop GPU
memory            : 6.0 GB, compute capability 8.9, 20 SMs
torch             : 2.11.0+cu128   CUDA runtime 12.8   cuDNN 91900
TF32 matmul       : enabled
cudnn.benchmark   : enabled (graph sizes vary, so autotuning pays back over 200 epochs)
pinned memory     : enabled on all loaders when the device is CUDA
host->device      : non_blocking=True
mixed precision   : implemented behind --amp, OFF by default
peak GPU memory   : 0.061-0.065 GB allocated, ~0.10 GB reserved
```

`--amp` is off by default deliberately: these runs take two minutes, so reproducible
numerics are worth more than the speed. `BBB_GNN_FORCE_CPU=1` forces CPU for debugging.

Peak memory is ~1% of the card. The model is small and the bottleneck is Python-side graph
collation, not GPU compute — batch size could go far higher if needed.

---

## 4. Verification: 29/29 checks passed

`python scripts/verify_multitask_gnn.py`

The result that matters most:

> **The legacy featurizer reproduces the base paper's stored reference graphs
> bit-for-bit** — 50 graphs compared, 0 mismatches on `x`, `edge_index` and `edge_attr`.

That keeps the refactor and the feature change separable: any difference in results is
attributable to multi-task training or to the new encoding, not to a quietly different
graph. Legacy mode is confirmed as exactly the base paper's (9, 7) encoding.

Other checks: CUDA reachable and executing; declared feature dims match produced tensors in
both modes; edge lists bidirectional; single-atom molecules and invalid SMILES handled;
corpus label counts reconcile; InChIKeys unique; contingency table sums correctly; both
split protocols pass every integrity assertion with zero scaffold overlap; model forward
contract correct; architecture metadata self-describing.

**Gradient isolation** (the check that makes the masked loss trustworthy):

| Check | Result |
|---|---|
| Batch with no P-gp labels → P-gp head gradient L1 | **0.000e+00** |
| Same batch → BBB head gradient L1 | 3.081e+01 |
| Task weight 0 with 16 P-gp labels present → P-gp head gradient L1 | **0.000e+00** |

---

## 5. Data

| | Rows in | Unique structures | Positive / negative |
|---|---:|---:|---|
| B3DB (`B3DB_classification.tsv`) | 7,807 | 7,805 | 4,956 BBB+ / 2,849 BBB− |
| Broccatelli (`pgp_broccatelli.tab`) | 1,219 | **1,212** | 647 substrate / 565 non-substrate |
| **Merged corpus** | | **8,801** | — |

Reconciles exactly: 7,805 + 1,212 − 216 = 8,801.

- **Dual-labelled: 216** molecules. 97.5% of the corpus carries exactly one label, which
  is why the masked loss is not optional.
- **logBB available for 1,058** molecules — a regression head is possible later at no extra
  data cost.
- **The GNN recovers 7 P-gp molecules the XGBoost arm loses.** That arm's processed file has
  1,205 structures because the ETKDGv3 3D-conformer gate drops molecules that fail to embed.
  Graphs need no conformer, so the GNN trains on 1,212.

### Label independence on the dual-labelled subset

| | P-gp substrate | P-gp non-substrate |
|---|:---:|:---:|
| **BBB+** | 45 | 95 |
| **BBB−** | 19 | 57 |

χ² = 0.887, p = 0.346; Fisher OR = 1.42, 95% CI [0.758, 2.665]; Cramér's V = **0.064**.
P(P-gp+ | BBB+) = 32.1% vs P(P-gp+ | BBB−) = 25.0%.

Computed independently by `gnn/data_assembly.label_independence()` and matching the earlier
hand analysis to four decimals.

### Splits (molecule-level, both verified leak-free)

| Protocol | train / val / test | BBB labelled | P-gp labelled | dual-labelled |
|---|---|---|---|---|
| random | 6160 / 1320 / 1321 | 5463 / 1170 / 1172 | 848 / 182 / 182 | 151 / 32 / 33 |
| scaffold | 6160 / 1320 / 1321 | 5637 / 1132 / 1036 | 649 / 230 / 333 | 126 / 42 / 48 |

Scaffold split uses the DeepChem convention (largest clusters fill train, so rare chemotypes
land in test) and reports **zero** scaffold overlap between train and test. This is stricter
than the existing `run_validation.py` protocol, which assigns the largest clusters to test;
**the scaffold numbers below are therefore not comparable to the XGBoost arm's 0.9048.**

### Feature encodings

| Mode | Atom / bond features | Parameters (1 head) | Parameters (2 heads) |
|---|---|---:|---:|
| legacy (base-paper parity) | 9 / 7 | 345,514 | — |
| encoded (used here) | 66 / 14 | 364,608 | **372,898** |

The second barrier costs **+8,290 parameters, about 2.3%** of the model.

---

## 6. Experiments run (9 total, all on GPU)

| Run | Split | Tasks | Weights | Best epoch | Time |
|---|---|---|---|---:|---:|
| single_bbb | random | bbb | — | 45 | 217s |
| single_pgp | random | pgp | — | 17 | 39s |
| multitask | random | bbb,pgp | 1 / 1 | 35 | 147s |
| single_bbb | scaffold | bbb | — | 7 | 117s |
| single_pgp | scaffold | pgp | — | 19 | 43s |
| multitask | scaffold | bbb,pgp | 1 / 1 | 9 | 98s |
| mt_pgpw3 | random | bbb,pgp | 1 / 3 | 22 | 117s |
| mt_pgpw5 | random | bbb,pgp | 1 / 5 | 18 | 109s |
| sanity_pgpw0 | random | bbb,pgp | 1 / 0 | 63 | 174s |

### Core result

**RANDOM SPLIT**

| Task | Experiment | n | ROC-AUC | 95% CI | MCC | F1 | BalAcc |
|---|---|---:|---:|---|---:|---:|---:|
| bbb | single_bbb | 1172 | 0.9388 | [0.9246, 0.9531] | 0.7264 | 0.9012 | 0.8613 |
| bbb | multitask | 1172 | 0.9404 | [0.9270, 0.9533] | 0.7078 | 0.8915 | 0.8563 |
| pgp | single_pgp | 182 | 0.9390 | [0.8964, 0.9728] | 0.7460 | 0.8821 | 0.8727 |
| pgp | multitask | 182 | 0.9301 | [0.8810, 0.9688] | 0.7460 | 0.8821 | 0.8727 |

**SCAFFOLD SPLIT**

| Task | Experiment | n | ROC-AUC | 95% CI | MCC | F1 | BalAcc |
|---|---|---:|---:|---|---:|---:|---:|
| bbb | single_bbb | 1036 | 0.7583 | [0.7288, 0.7862] | 0.3927 | 0.7677 | 0.6783 |
| bbb | multitask | 1036 | 0.7467 | [0.7180, 0.7753] | 0.2879 | 0.7461 | 0.6130 |
| pgp | single_pgp | 333 | 0.8227 | [0.7763, 0.8696] | 0.5442 | 0.7990 | 0.7605 |
| pgp | multitask | 333 | 0.8171 | [0.7720, 0.8616] | 0.5389 | 0.7902 | 0.7660 |

**Multi-task effect (multi minus single):**

| Task | Split | ΔROC-AUC | ΔMCC | ΔF1 | Verdict |
|---|---|---:|---:|---:|---|
| bbb | random | +0.0016 | −0.0186 | −0.0097 | within noise (CIs overlap) |
| bbb | scaffold | −0.0116 | −0.1048 | −0.0216 | within noise (CIs overlap) |
| pgp | random | −0.0089 | +0.0000 | +0.0000 | within noise (CIs overlap) |
| pgp | scaffold | −0.0056 | −0.0053 | −0.0088 | within noise (CIs overlap) |

---

## 7. The finding: the pre-registered hypothesis is not supported

The plan pre-registered an asymmetric expectation — the small P-gp task should gain from
sharing an encoder with the large BBB task, while BBB might not gain. **Neither half holds.
Multi-task training is neutral to mildly harmful on both tasks, on both splits.**

This is a real negative result, not a bug, and three pieces of evidence establish that:

1. **The plumbing is verified.** `sanity_pgpw0` (multi-task architecture, P-gp weight 0)
   gives BBB ROC-AUC **0.9411** against the single-task baseline's 0.9388 — within noise —
   while its untrained P-gp head sits at **0.5626**, i.e. chance. Weight 0 genuinely removes
   a task, and the trunk trains exactly as it does single-task.
2. **The obvious confound is ruled out.** P-gp has 848 training molecules against BBB's
   5,463, so thin gradient signal was the likeliest innocent explanation. Upweighting P-gp
   does **not** rescue it (ROC-AUC 0.9295 at weight 3, 0.9299 at weight 5, versus 0.9390
   single-task) and progressively **damages BBB** (0.9388 → 0.9310 → 0.9086, MCC
   0.7264 → 0.6821 → 0.6467). The tasks compete for encoder capacity instead of helping
   each other.
3. **It coheres with the label measurement.** Cramér's V = 0.064 means the two labels share
   almost no information. Multi-task learning borrows strength from *correlated* auxiliary
   signal; near-independent labels have none to lend. The negative result is what the
   independence measurement predicts.

A 2025 review proposed exactly this architecture as future work — "multi-task learning
leverages auxiliary signals such as transporter affinity to guide the extraction of
disentangled, mechanism-relevant features." This run tests that proposal on the two standard
public benchmarks and finds it does not deliver, with a mechanistic explanation for why.

### Chemotype transfer

| Experiment | Task | random → scaffold ROC-AUC |
|---|---|---|
| single_bbb | bbb | 0.9388 → 0.7583 (**−0.1805**) |
| multitask | bbb | 0.9404 → 0.7467 (−0.1937) |
| single_pgp | pgp | 0.9390 → 0.8227 (**−0.1163**) |
| multitask | pgp | 0.9301 → 0.8171 (−0.1130) |

BBB degrades ~55% more than P-gp under chemotype shift (−0.18 vs −0.12), reproducing in a
GNN the asymmetry previously measured in the XGBoost arm. The permeability axis is the
chemotype-dependent one; the efflux axis transfers better.

---

## 8. Two bugs found and fixed during the run

**PyG silently corrupted the row pointer.** Graphs carried a `corpus_index` attribute to
trace predictions back to the corpus. PyG's default `Data.__inc__` increments *any*
attribute whose key contains `index` by the graph's node count when collating a batch — so
the pointer came out offset by cumulative node counts and `corpus.iloc[...]` raised
"positional indexers are out-of-bounds" only at the prediction-saving step, after training
had completed. Renamed to `corpus_row`, with a cache version bump so stale caches rebuild
rather than being silently reused. Had this gone unnoticed in a form that did not crash,
every saved prediction would have been attributed to the wrong molecule.

**`torch.cuda.reset_peak_memory_stats(int)` raised "Invalid device argument"** when called
before the CUDA context was initialised. Fixed by passing the `torch.device` object and
calling `torch.cuda.init()` first.

---

## 9. Pre-existing defects confirmed but NOT fixed

These are in `Initial Model/` and were left alone per the scope correction. They are
recorded because they affect any number quoted from that arm:

- `train_gnn.py` discards 269 Li molecules (14.2% of 1,890), so all its metrics are
  Adenot-only, and dropping Li makes imbalance *worse* (79.1% positive vs Li's 62.8%).
- `predict.py` cannot load its own default checkpoint: it hardcodes `HIDDEN_CHANNELS = 256`
  and ignores `checkpoint["hidden_channels"]`, while `best_bbb_gnn_model.pt` is a 128-dim
  artifact → `load_state_dict` size mismatch.
- `predict.py` also re-declares the model without dropout and substitutes
  `torch.mean(x, dim=0, keepdim=True)` for `global_mean_pool`.
- No metrics are persisted anywhere in that arm; `confusion_matrix` is imported and never
  called.

The new `gnn/` package avoids all four by construction: one shared featurizer, checkpoints
that describe their own architecture, and every run writing metrics to disk.

**The defects listed in `NOVELTY_ASSESSMENT.md` section 6 for the XGBoost arm (unscaled
SHAP, hard-coded docking probabilities, the two legacy evaluation scripts) are also still
unfixed.** This run did not touch them.

---

## 10. Artifacts produced

- **12 new source files** under `B3DB Based Model/gnn/` and `B3DB Based Model/scripts/`.
- **39 result files** in `B3DB Based Model/results/` — per run: a report JSON (full
  provenance: arguments, device, corpus stats, independence statistics, split report,
  architecture, metrics), a metrics CSV, an epoch-by-epoch history CSV, and a per-molecule
  test-prediction CSV. Plus `gnn_experiment_comparison.csv` and `gnn_multitask_delta.csv`.
- **9 checkpoints** in `B3DB Based Model/models/`, each ~1.5 MB and self-describing.
- **1 graph cache** at `B3DB Based Model/data/graph_cache/` (122 MB).

### One repo-hygiene decision left to you

`.gitignore:10` is `*.pt`, so **all 9 new GNN checkpoints and the graph cache are currently
ignored** — the same reproducibility gap the `Initial Model/` arm has. The 122 MB cache
should stay ignored. The checkpoints total ~13.5 MB and arguably should not. If you want
them tracked, add after the `*.pt` line:

```
!B3DB Based Model/models/gnn_*.pt
```

I have not changed `.gitignore` — what belongs in version control is your call.

---

## 11. What this means for the paper

The plan's decision gate fires: *"if the P-gp head does not improve, the headline shifts to
C2+C3+C4 — the measurement, the evaluation, and the attribution — with multi-task reported
as a negative result."*

So the paper is no longer "we built the first joint BBB/efflux GNN and it works." It is:

> **Blood-brain barrier and P-glycoprotein efflux labels are near-independent, and
> multi-task graph learning cannot bridge them.** We quantify the label independence on the
> 216 molecules the two standard benchmarks share, show that a shared-encoder multi-task GNN
> delivers no benefit to either task under either split protocol, demonstrate that
> upweighting the minority task causes measurable negative transfer rather than gain, and
> show the permeability axis degrades ~55% more than the efflux axis under chemotype shift.

That is a defendable contribution: it tests a published proposal, reports a clean null with
the confounds eliminated, and explains the null mechanistically. Negative results on a
proposal that a review named as promising are publishable in *Journal of Cheminformatics*,
and the released cross-labelled 216-molecule resource is the citable artifact.

**Still open and worth doing**, in order: the Kp,uu external validation (the strongest
remaining novelty per `PUBLICATION_STRATEGY.md` — no competitor validates against measured
brain exposure); dual-head atom attribution to test graphB3's untested P-gp hypothesis; the
encoded-vs-legacy feature ablation (one flag: `--features legacy`); and a logBB regression
head using the 1,058 available values.

**One caveat to carry forward:** every number here is a single seed. Before any of it goes
into a manuscript, repeat across seeds — the multi-task deltas are small enough that
seed variance could account for them, which is itself part of the negative-result argument
but needs to be measured rather than assumed.

---

## 12. Reproducing this run

```bash
cd "B3DB Based Model"
python scripts/verify_multitask_gnn.py                                   # 29 checks
python scripts/train_multitask_gnn.py --tasks bbb      --split random    --tag single_bbb
python scripts/train_multitask_gnn.py --tasks pgp      --split random    --tag single_pgp
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random    --tag multitask
python scripts/train_multitask_gnn.py --tasks bbb      --split scaffold  --tag single_bbb
python scripts/train_multitask_gnn.py --tasks pgp      --split scaffold  --tag single_pgp
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split scaffold  --tag multitask
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=3 --tag mt_pgpw3
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=5 --tag mt_pgpw5
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=0 --tag sanity_pgpw0
python scripts/compare_gnn_experiments.py
```

Total wall-clock for all nine runs plus verification: about 19 minutes on an RTX 4050.

---
---

# Session 2: Decision-Aware Metrics, K-Fold, and the Statistical Rigor Layer

**Date:** 2026-10-24 (same day, second work block)
**Repo state at start:** HEAD `e13f3a5`, Session 1 work uncommitted
**Repo state at end:** HEAD unchanged — **nothing committed, nothing pushed**
**Task:** add a novel metric contribution, plus the K-fold and seed-variance work needed
to make Session 1's null result defensible.

## 1. Why this session happened

Session 1 left two problems. The multi-task null rested on **one seed**, so deltas of
+0.0016 to −0.0116 could not be told from noise. And the project has no performance
story — its BBB ROC-AUC of 0.9388 sits below graphB3's 0.943 — so if there is a
contribution it has to be in *measurement*.

Four literature searches established what was unclaimed:

- **No paper proposes joint or quadrant metrics for multi-endpoint ADMET.** The field
  states the need itself ("ROC-AUC is a global metric that is not always useful for
  making decisions in a production setting"); TDC still uses per-task metrics only.
- **No prior art for efflux-stratified performance reporting.**
- **Nothing frames model selection as expected regret over a screening decision.** The
  closest is a 2026 preprint (Budget-Sensitive Discovery Scoring) whose cost-ratio
  parameterisation λ = c_FP/v_hit is adapted here and cited.
- **MDE-over-seeds reporting is NOT novel.** A 2026 PROTAC-activity preprint does exactly
  this, reporting a +0.0025 AUROC effect as "informative as a null result". It is adopted
  as a rigor layer and cited, never claimed.

## 2. What was built

| File | Lines | Purpose |
|---|---:|---|
| `gnn/training.py` | 283 | Shared training loop, extracted so three runners share one implementation |
| `gnn/decision_metrics.py` | 341 | M1–M5, the metric suite |
| `scripts/run_kfold_multitask.py` | 329 | K-fold runner producing out-of-fold predictions |
| `scripts/run_metric_analysis.py` | 248 | Scores the three decision strategies on M1–M5 |
| `scripts/run_seed_variance.py` | 300 | Multi-seed runs, MDE, P(A>B) |
| `scripts/verify_decision_metrics.py` | 262 | 31-check metric verification suite |
| `core/decision_matrix.py` | +33 | Added `quadrant_id`, `QUADRANT_NAMES`, `QUADRANT_IDS` |

**One refactor of existing code.** `scripts/train_multitask_gnn.py` went from 461 to 353
lines: its local copies of `set_all_seeds`, `parse_task_weights`, `move_batch`, `run_epoch`
and `quick_f1` were deleted and are now imported from `gnn/training.py`. Three copies of an
early-stopping loop would have drifted apart exactly the way the base paper's two
featurizers did.

**Quadrant convention is single-sourced.** `quadrant_id` was added to the *existing*
`core/decision_matrix.py` rather than reimplemented in the metrics module, and is verified
to agree with `determine_dual_barrier_quadrant` on all four inputs and to reproduce
`expected_quadrant` for all five `BENCHMARK_DRUGS`.

## 3. The metric suite

| | Metric | Definition |
|---|---|---|
| M1 | Quadrant exact-match | Both axes correct (subset accuracy) |
| M2 | Quadrant κ + 4×4 matrix | Chance-corrected 4-class agreement, with the error *kind* visible |
| M3 | Paradox recall / leakage | Recall on Q2, and the share of true Q2 molecules progressed as Q1 |
| M4 | Efflux-disparity gap | AUC_BBB(non-substrates) − AUC_BBB(substrates) |
| M5 | Decision regret | Expected cost of the progression decision, as a curve in λ |

The mathematics is standard. The contribution is the framing: these are the numbers that
decide a CNS progression call, and none of them is currently reported.

## 4. Verification: 31/31 metric checks, 29/29 regression

`python scripts/verify_decision_metrics.py` → **31/31**. The checks that matter:

| Check | Result |
|---|---|
| Perfect predictor | exact-match 1.0, κ 1.0, paradox recall 1.0, leakage 0.0, zero regret at every λ |
| Single-axis BBB model | paradox recall exactly **0**, leakage exactly **1.0** — by construction |
| All-Q2 predictor | recall 1.0, precision equals Q2 prevalence (0.2625 vs 0.2625) |
| Confusion matrix | sums to n; diagonal equals the exact-match count |
| EDG, equally informative strata | gap 0.0, CI spans zero |
| EDG, signal destroyed on substrates only | gap +0.5305, CI [0.4668, 0.5914] excludes zero |
| EDG, single-class stratum | returns a note instead of a number |
| Regret at λ=1 | equals the Q1-vs-rest error rate exactly |
| Regret at λ=10⁶ | dominated by false progressions (residual 1.1e-6) |
| MDE formula | matches hand calculation to 6 dp (0.016625) |
| P(A>B) | matches hand values exactly (0.7778, 0.5000, 1.0000) |

**Regressions after the refactor:** `verify_multitask_gnn.py` still **29/29**, and
`compare_gnn_experiments.py` reproduces the Session 1 table byte-identically.

**Quadrant ground truth reconciles:** Q1 = 95, Q2 = 45, Q3 = 57, Q4 = 19, sum **216**.

## 5. K-fold out-of-fold predictions

Session 1's single-split test partitions held only 33 (random) / 48 (scaffold) dual-labelled
molecules — too few to stratify. 5-fold stratified K-fold over all 8,801 molecules gives
out-of-fold predictions for **216 of 216**.

| Run | Folds | Integrity | Pooled OOF result | Time |
|---|---|---|---|---|
| single_bbb | 5 | 7,805 scored once, no leaks | BBB ROC-AUC **0.9379** [0.9323, 0.9433] | 771s |
| single_pgp | 5 | 1,212 scored once, no leaks | P-gp ROC-AUC **0.9011** [0.8838, 0.9190] | 142s |
| multitask | 5 | 8,801 scored once, no leaks, **216/216 dual** | BBB **0.9363** [0.9312, 0.9416], P-gp **0.9056** [0.8885, 0.9226] | 934s |

Every fold's assertions passed: no molecule scored more than once, every labelled molecule
covered, and no molecule scored by a fold that trained on it.

On the much larger pooled OOF sample the per-axis picture is the same as Session 1: BBB
−0.0016 and P-gp +0.0045 for multi-task, both with heavily overlapping intervals.

## 6. Decision-metric results

Three strategies scored on the 216 dual-labelled molecules:

- **A. BBB-only** — single-axis model; scoring it means assuming P-gp-negative throughout,
  which is what a BBB-only pipeline implies. This is the published status quo.
- **B. Composed two-model** — independent single-task BBB and P-gp models combined. Valid
  because both K-fold runs share an identical fold assignment (verified at runtime).
- **C. Multi-task** — one shared encoder, two heads.

### M1–M3

| Evaluation | Exact-match | κ | Paradox recall | Paradox precision | Leakage |
|---|---:|---:|---:|---:|---:|
| A BBB-only | 0.5556 [0.4769, 0.6204] | 0.2742 | **0.0000** | 0.0000 | **0.9333** |
| B Composed | **0.5787** [0.5093, 0.6481] | **0.3725** | 0.4444 [0.2954, 0.6047] | 0.4348 | 0.4889 |
| C Multi-task | 0.5602 [0.4954, 0.6251] | 0.3478 | **0.5111** [0.3657, 0.6667] | **0.4792** | **0.4222** |

**The headline: per-axis ROC-AUC is ~0.94, yet only 56–58% of molecules land in the correct
quadrant.** Four in ten joint decisions are wrong. No currently reported metric shows this.

**Paradox leakage** — true Q2 (permeant but effluxed) molecules progressed as Q1:

| | Leaked | Rate |
|---|---|---|
| A BBB-only | **42 of 45** | **93.3%** |
| B Composed | 22 of 45 | 48.9% |
| C Multi-task | 19 of 45 | 42.2% |

A single-axis model advances 93% of paradoxical compounds as though they reach the brain.
A second axis roughly halves that. B and C are indistinguishable from each other.

### M4 — the hypothesis was wrong, and the mechanism explains why

I predicted BBB prediction would be *worse* on efflux substrates. It is significantly
**better**:

| Evaluation | AUC non-substrate (n=152) | AUC substrate (n=64) | Gap | Verdict |
|---|---:|---:|---:|---|
| A / B | 0.7990 | 0.9368 | **−0.1379** [−0.2406, −0.0410] | gap detected |
| C | 0.7904 | 0.8760 | −0.0856 [−0.1992, +0.0247] | CI spans zero |

Why, measured rather than assumed. The two strata differ sharply in chemistry:

| | MolWt | LogP | TPSA | HDonors | RotBonds |
|---|---:|---:|---:|---:|---:|
| non-substrate | 281.4 | 2.2 | 60.3 | 1.4 | 3.9 |
| substrate | 370.9 | 3.8 | 47.8 | 0.8 | 5.2 |

And BBB permeability is *more* separable among substrates: the BBB+ vs BBB− TPSA gap is
**−28.9** there versus **−21.5** among non-substrates. P-gp substrates are a chemically
narrow, drug-like region where polarity cleanly determines permeability; non-substrates are
a heterogeneous mix including small polar molecules where permeability is genuinely harder
to call.

**This sharpens the paper rather than weakening it.** The harm from unmodelled efflux is not
that permeability prediction degrades — it is that permeability prediction is *accurate*, and
that accurate prediction is clinically misleading. The molecule really is permeant and really
is pumped back out. The 93.3% leakage is the harm; there is no accuracy deficit to point at.

### M5 — the method ranking depends on the cost ratio

| Evaluation | λ=0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A BBB-only | **0.1296** | **0.2083** | 0.3657 | 0.6806 | 1.3102 | 2.5694 | 5.0880 |
| B Composed | 0.1806 | 0.2268 | **0.3194** | **0.5046** | **0.8750** | **1.6157** | **3.0972** |
| C Multi-task | 0.1921 | 0.2407 | 0.3380 | 0.5324 | 0.9213 | 1.6991 | 3.2546 |

False progressions by true quadrant: A {Q2:42, Q3:21, Q4:5}, B {Q2:22, Q3:16, Q4:2},
C {Q2:19, Q3:19, Q4:4}.

Regret is linear in λ, so the crossover solves exactly: **λ = 0.643** against B and
**λ = 0.769** against C. **Below that, the single-axis model is genuinely the better
decision rule** — it progresses more compounds and incurs fewer missed opportunities
(11 vs 29 and 31). The efflux axis only pays once a false progression costs more than
roughly 0.7× a missed opportunity.

That nuance is the argument for reporting a curve. A single headline number would have
hidden it, and the crossover calculation is now part of the script rather than a one-off.

## 7. Seed variance and minimum detectable effect

5 seeds, paired by seed (the same seed drives the split *and* the initialisation for every
experiment), random split.

Per-seed ROC-AUC:

| Seed | BBB multi | BBB single | P-gp multi | P-gp single |
|---|---:|---:|---:|---:|
| 42 | 0.9443 | 0.9462 | 0.9464 | 0.9445 |
| 43 | 0.9331 | 0.9428 | 0.8734 | 0.8734 |
| 44 | 0.9340 | 0.9295 | 0.9195 | 0.9122 |
| 45 | 0.9163 | 0.9396 | 0.8841 | 0.9300 |
| 46 | 0.9286 | 0.9456 | 0.8933 | 0.8701 |

| | BBB | P-gp |
|---|---|---|
| multi-task mean | 0.9313 | 0.9033 |
| single-task mean | 0.9407 | 0.9060 |
| delta | **−0.00944** | **−0.00272** |
| delta 95% CI | [−0.02337, +0.00449] | [−0.03479, +0.02936] |
| sd of differences | 0.01122 | 0.02583 |
| **MDE at 5 seeds (80% power)** | **0.01865** | **0.04295** |
| abs(delta) below MDE | **yes** | **yes** |
| paired t-test | t = −1.881, p = 0.1331 | t = −0.235, p = 0.8256 |
| **P(multi > single)** | **0.200** | 0.540 |
| seeds needed for 80% power at this effect | **14** | **712** |

**What this licenses, stated precisely.** Neither delta is resolvable at 5 seeds, so the
correct conclusion is *"no effect detected at this sample size"* — never *"no effect
exists."* But the two tasks are not in the same position:

- **BBB:** P(multi > single) = **0.200**, meaning single-task wins four cross-seed pairings
  in five. By Bouthillier et al.'s own 0.75/0.25 criterion that is a reliable *ordering*,
  even though the paired t-test is non-significant at p = 0.133. The evidence leans toward
  multi-task being mildly harmful to BBB, and **14 seeds would settle it.**
- **P-gp:** P(multi > single) = **0.540**, indistinguishable from a coin flip, and resolving
  the −0.0027 delta would need **712 seeds**. An effect that small is practically zero
  whatever its sign. This is the strongest form the P-gp null can take.

A footnote worth recording: at seed 43 the two P-gp models produced identical ROC-AUC to
four decimals (0.8734). With n = 182 the AUC granularity is 1/8245, so a collision is
possible, and the other four seeds differ — there is no systematic identity. Not chased
further.

## 8. Two problems found and fixed

**Stored metrics were rounded to 4 decimals**, which makes exact delta arithmetic impossible
when the effects under study are of order 1e-3. Caught by a verification check that compared
the confusion-matrix diagonal against the exact-match rate and failed at 1e-9 tolerance. The
correct fix was to store at 6 dp and format to 4 only for display, not to loosen the test.

**The first background launch used `nohup` plus `&`**, which detached the child from the
tool's process management; it was killed immediately and reported success with an empty log.
Re-running through the tool's own backgrounding worked. A job that "succeeds" with no output
is worth checking rather than trusting.

## 9. Artifacts added this session

- **7 source files** (6 new, 1 modified) under `B3DB Based Model/gnn/` and `scripts/`.
- **3 OOF prediction files** covering every labelled molecule, with both axes for all 216
  dual-labelled ones.
- **3 K-fold report JSONs** with full provenance.
- **`results/gnn_decision_metrics.csv`** and **`_report.json`** — the M1–M5 tables.
- **4 seed-variance files** — per-run, summary, contrasts, report.

## 10. What this means for the paper

The claim is now three-part and each part is backed:

1. **BBB and P-gp labels are near-independent** (Cramér's V = 0.064, n = 216) — recomputed
   independently this session and matching to four decimals.
2. **Multi-task graph learning does not bridge them.** No effect detected on either task,
   on either split, on single-split or pooled K-fold; upweighting the minority task causes
   measurable negative transfer; and the plumbing is verified. For BBB the ordering favours
   single-task at P = 0.20 and 14 seeds would confirm it; for P-gp the effect is below any
   feasible detection threshold.
3. **Per-axis metrics hide the decision that matters.** ROC-AUC ~0.94 on both axes, yet
   56–58% quadrant accuracy and 93.3% paradox leakage for the single-axis status quo.

Contribution 3 is the new one and it is a *supporting* contribution: the mathematics is
standard, so the defence is that nobody evaluates this decision jointly. A determined
reviewer may still call it incremental, and it does not substitute for the Kp,uu validation
that `PUBLICATION_STRATEGY.md` identifies as the strongest remaining novelty.

**Still open**, in order: Kp,uu external validation; dual-head atom attribution to test
graphB3's untested P-gp hypothesis; the `--features legacy` encoding ablation; a logBB
regression head on the 1,058 available values; and 14 seeds on the BBB contrast to convert
"not detected" into "ordered."

**The scaffold-split decision metrics were not computed.** Only the random-split K-fold was
run. Repeating the K-fold under the scaffold protocol would show whether the quadrant and
leakage numbers survive chemotype shift, and should be done before any of these figures go
into a manuscript.

## 11. Reproducing this session

```bash
cd "B3DB Based Model"
python scripts/verify_decision_metrics.py                                    # 31 checks
python scripts/verify_multitask_gnn.py                                       # 29 checks
python scripts/run_kfold_multitask.py --tasks bbb     --folds 5 --tag single_bbb
python scripts/run_kfold_multitask.py --tasks pgp     --folds 5 --tag single_pgp
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py
python scripts/run_seed_variance.py --seeds 42,43,44,45,46 --split random
```

Wall-clock on the RTX 4050: K-fold 31 min, seed variance 18 min, analysis seconds.
