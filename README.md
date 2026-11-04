# Dual-Barrier CNS Drug Prediction

**Results refreshed 4 November 2026.** XGBoost validation, feature ablation,
global SHAP, GNN single-split experiments, five-fold out-of-fold predictions,
five-seed comparisons, and decision metrics use the regenerated November artifacts.
The pharmacopeia screen (20 October) and docking analysis (24 October) have not
been rerun with the new models; their sections are explicitly historical.
Clinical benchmark probabilities and local SHAP values were recomputed from the
current models on 4 November. Table checks verify numerical consistency, not
scientific validity or clinical efficacy.

Run `venv\Scripts\python -m pytest "Novelty Upgrade/tests"` for regression tests.
Install test dependencies with `pip install -r requirements-dev.txt`; docking
dependencies are in `requirements-docking.txt`. `requirements-lock.txt` pins the
tested environment's dependency versions (Windows, CUDA 12.8 PyTorch build).
The 3D model remains deployed for compatibility; the 2D ablation is an alternative
requiring its own validated model, rather than substituting zeros for failed 3D features.

Numerical result tables are generated from their named CSV artifacts and checked
by `scripts/sync_result_tables.py --check` and `scripts/verify_doc_consistency.py`.

---

## The problem

Whether a drug reaches the brain is two questions, not one:

1. Can it **cross** the blood-brain barrier by passive diffusion?
2. Will **P-glycoprotein** — an ATP-powered efflux pump on the blood-facing membrane of
   brain capillary endothelial cells — **pump it straight back out**?

```
Net brain exposure  ∝  passive influx / (active efflux + metabolism)
```

Published BBB benchmarks model only the numerator. **Loperamide** is the proof that the
denominator decides the outcome: a potent µ-opioid agonist, sold over the counter, that
produces no central effect at all because P-gp expels it. **Levetiracetam** is the mirror
image — it crosses *and* evades the pump, so it works in refractory epilepsy.

A model can therefore top a BBB leaderboard while being clinically wrong.

## The finding

We quantified how wrong. B3DB and the Broccatelli P-gp benchmark share **216 molecules with
unambiguous ground truth on both axes** (joined on full InChIKey):

|  | P-gp substrate | P-gp non-substrate | total |
|---|:---:|:---:|:---:|
| **BBB+ (permeant)** | **45** | 95 | **140** |
| **BBB− (non-permeant)** | 19 | 57 | 76 |
| **total** | **64** | **152** | **216** |

> **32.1% of the molecules B3DB labels "permeant" are confirmed P-glycoprotein substrates**
> (45/140, 95% CI [24.5%, 40.6%], Clopper-Pearson).

And the BBB label carries no usable information about efflux status: Cramér's V = **0.064**,
Fisher OR = 1.42 [0.76, 2.67], χ² p = 0.346. *The data are consistent with independence and
the shared variance is negligible; at n = 216 the interval still admits a weak association, so
the defensible statement is that the BBB label is **not a usable proxy** for efflux status.*

**Consequence, measured.**

Five-fold out-of-fold predictions are evaluated on the same 216 dual-labelled
molecules. A_bbb_only assumes P-gp-negative for every molecule;
B_composed_two_model combines independent BBB and P-gp predictions;
C_multitask uses the shared encoder. The BBB-only baseline cannot express Q2,
so zero paradox recall is a consequence of its definition.

<!-- results:decisions -->
| Strategy | N | Quadrant accuracy | Accuracy 95% CI | Kappa | Paradox recall | Paradox leakage | Leaked / true Q2 | Regret at lambda=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_bbb_only | 216 | 0.5185 | 0.4444 - 0.5833 | 0.2158 | 0.0000 | 0.9333 | 42 / 45 | 0.40278 |
| B_composed_two_model | 216 | 0.5741 | 0.5093 - 0.6390 | 0.3651 | 0.5111 | 0.4222 | 19 / 45 | 0.32407 |
| C_multitask | 216 | 0.5694 | 0.5046 - 0.6343 | 0.3598 | 0.4889 | 0.4444 | 20 / 45 | 0.31944 |
<!-- /results:decisions -->

BBB-only advances 42 of 45 true Q2 compounds as Q1 (93.3%). Adding the P-gp
axis reduces this to 19/45 (42.2%) for composed models and 20/45 (44.4%) for
multitask learning. Quadrant accuracy is 51.9%, 57.4%, and 56.9%, respectively.
These data support including the efflux axis, without demonstrating superiority
of the shared encoder over two independent models.

Pooled OOF per-axis AUCs are BBB 0.9339 (single) versus 0.9390 (multi), and
P-gp 0.9078 versus 0.9006. These use all labelled molecules for each endpoint,
whereas joint metrics use only the 216 dual-labelled molecules.

The efflux-disparity gap, AUC(non-substrates) minus AUC(substrates), is negative:
−0.164758 [−0.2698, −0.0668] for BBB-only/composed models and
−0.106741 [−0.2188, −0.0008] for multitask. Thus this subset does not show
worse BBB discrimination among substrates. At equal error costs, decision
regret is 0.40278, 0.32407, and 0.31944. These are computational decision errors,
not measured clinical outcomes.

---|---:|---:|
| Single-axis BBB model (the published status quo) | 55.6% | **93.3%** (42 of 45) |
| Two independent models, composed | 57.9% | 48.9% |
| Shared-encoder multi-task model | 56.0% | 42.2% |

**Paradox leakage** = molecules that are genuinely permeant-but-effluxed which the model
advances as though they reach the brain. A single-axis model leaks **93.3%** of them. Adding
an efflux axis roughly halves that. No metric in current use exposes this.

---

## What is in this repository

| | |
|---|---|
| **Two trained predictors** | BBB permeability (Optuna-tuned) and P-gp efflux (fixed parameters), identity-deduplicated |
| **Explainability layer** | TreeSHAP in log-odds margin space, Morgan bit → chemical fragment, applicability domain, reaction-SMARTS counterfactuals |
| **Historical pharmacopeia screen** | October output for 2,405 approved small molecules |
| **Historical docking analysis** | October output for 22 compounds docked into human P-gp (PDB 6QEX) |
| **Multi-task GNN arm** | shared GINEConv encoder, two heads, masked loss, paired comparisons and seed variability |
| **Decision-aware metrics** | five metrics (M1–M5) that score the *joint* dual-barrier decision |
| **Regression and consistency checks** | model behavior, split integrity, metrics, and result-table synchronization |

`Initial Model/` holds the base-paper reference
implementation and is deliberately unmodified.

---

## Results

### Model performance

Values are estimates with 1,000-resample bootstrap 95% confidence intervals.
The random test sets contain 1,171 BBB and 240 P-gp molecules after molecular
deduplication. Scaffold allocation places the largest scaffold clusters into
training first and has zero train/test scaffold overlap.

<!-- results:validation -->
| Metric | BBB random | BBB scaffold | P-gp random | P-gp scaffold |
| --- | --- | --- | --- | --- |
| Accuracy | 0.8958 [0.877 - 0.9129] | 0.6926 [0.6661 - 0.7182] | 0.9083 [0.8708 - 0.9417] | 0.8292 [0.7792 - 0.8792] |
| Balanced_Accuracy | 0.8896 [0.8691 - 0.9078] | 0.6622 [0.6389 - 0.6859] | 0.9097 [0.8725 - 0.9436] | 0.8233 [0.7702 - 0.8756] |
| Precision | 0.9226 [0.902 - 0.9413] | 0.6674 [0.6343 - 0.6972] | 0.9350 [0.8898 - 0.9748] | 0.7253 [0.6333 - 0.814] |
| Recall_Sensitivity | 0.9126 [0.8912 - 0.9319] | 0.9044 [0.8832 - 0.9256] | 0.8915 [0.8333 - 0.9426] | 0.8049 [0.7108 - 0.8873] |
| Specificity | 0.8665 [0.8311 - 0.8983] | 0.4199 [0.3781 - 0.4617] | 0.9279 [0.8739 - 0.9727] | 0.8418 [0.7826 - 0.8987] |
| F1_Score | 0.9176 [0.9022 - 0.9313] | 0.7680 [0.7441 - 0.7905] | 0.9127 [0.8745 - 0.9466] | 0.7630 [0.6853 - 0.8296] |
| ROC_AUC | 0.9563 [0.9452 - 0.9669] | 0.8112 [0.7862 - 0.8338] | 0.9607 [0.9321 - 0.9828] | 0.9001 [0.8559 - 0.9401] |
| PR_AUC | 0.9740 [0.9661 - 0.9813] | 0.8496 [0.8233 - 0.8741] | 0.9739 [0.9562 - 0.9884] | 0.8328 [0.7439 - 0.9047] |
| MCC | 0.7761 [0.7362 - 0.8114] | 0.3781 [0.3278 - 0.428] | 0.8173 [0.7417 - 0.8845] | 0.6321 [0.5236 - 0.7353] |
| Brier_Score | 0.0785 [0.0682 - 0.0893] | 0.1992 [0.1846 - 0.2158] | 0.0691 [0.0467 - 0.0927] | 0.1316 [0.0988 - 0.1664] |
<!-- /results:validation -->

Random-split AUC is 0.9563 for BBB and 0.9607 for P-gp; the scaffold benchmarks
yield 0.8112 and 0.9001. Specificity falls from 0.8665 to 0.4199 for BBB and
from 0.9279 to 0.8418 for P-gp. Both endpoints therefore perform worse under
these scaffold benchmarks; the earlier claim that P-gp performance is preserved
does not hold for the regenerated results.

The scaffold models are fresh fixed-configuration fits, whereas the BBB random
model is Optuna-tuned and the P-gp random model uses class weighting. Training
fractions also differ. These are independent benchmarks, not controlled estimates
of the causal effect of scaffold splitting. Brier scores are reported as probability
prediction errors; a low Brier score alone does not establish calibration.

### Feature ablation

Five-fold stratified CV uses identical folds across feature subsets and fits
each scaler within its training fold. Values are mean ± fold standard deviation,
not confidence intervals.

<!-- results:ablation -->
| Task | Configuration | Features | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| P-gp Substrate | 1. 2D Descriptors Only | 6 | 0.8623 ± 0.0137 | 0.8741 ± 0.0133 | 0.9152 ± 0.0083 |
| P-gp Substrate | 2. 3D Shape Only | 6 | 0.7797 ± 0.0317 | 0.7983 ± 0.0285 | 0.8428 ± 0.0277 |
| P-gp Substrate | 3. 2D Descriptors + 3D Shape | 12 | 0.8440 ± 0.0284 | 0.8560 ± 0.0254 | 0.9142 ± 0.0117 |
| P-gp Substrate | 4. Morgan FP + 2D Descriptors | 2054 | 0.8957 ± 0.0107 | 0.9027 ± 0.0075 | 0.9480 ± 0.0087 |
| P-gp Substrate | 5. Full Model (Morgan + 2D + 3D Shape) | 2060 | 0.8848 ± 0.0162 | 0.8928 ± 0.0127 | 0.9465 ± 0.0078 |
| BBB Permeability | 1. 2D Descriptors Only | 6 | 0.8555 ± 0.0113 | 0.8913 ± 0.0081 | 0.9174 ± 0.0081 |
| BBB Permeability | 2. Morgan Fingerprints Only | 2048 | 0.8588 ± 0.0078 | 0.8936 ± 0.0054 | 0.9393 ± 0.0054 |
| BBB Permeability | 3. Full Hybrid (Morgan + 2D Descriptors) | 2054 | 0.8738 ± 0.0072 | 0.9041 ± 0.0051 | 0.9467 ± 0.0035 |
<!-- /results:ablation -->

BBB Morgan + 2D achieves 0.9467 versus 0.9393 for Morgan alone and 0.9174 for
2D descriptors alone. This is a higher mean score; no paired significance test
is supplied by the ablation artifact.

For P-gp, Morgan + 2D achieves 0.9480 ± 0.0087 versus 0.9465 ± 0.0078 with
3D features. The added shape descriptors provide no observed mean AUC gain
in this experiment. Shape alone achieves 0.8428 ± 0.0277. These predictive
associations do not demonstrate a molecular binding mechanism. The deployed
3D pipeline is retained for compatibility, with failed conformers rejected.

### What drives each barrier

Mean absolute and mean signed TreeSHAP values are in log-odds margin units,
computed on held-out samples using the preprocessing metadata embedded in the
current model files. Exact native XGBoost contributions are checked for additivity.

<!-- results:shap -->
| Descriptor | BBB mean absolute SHAP | P-gp mean absolute SHAP | BBB mean signed SHAP | P-gp mean signed SHAP |
| --- | --- | --- | --- | --- |
| MolWt | 0.3991 | 0.6488 | -0.0240 | 0.0319 |
| LogP | 0.2855 | 1.3574 | 0.0388 | 0.0814 |
| TPSA | 1.2652 | 0.1431 | 0.1701 | 0.0364 |
| HDonors | 0.1694 | 0.0519 | 0.0219 | 0.0156 |
| HAcceptors | 0.0757 | 0.0656 | -0.0092 | 0.0060 |
| RotBonds | 0.1002 | 0.0561 | 0.0066 | 0.0066 |
<!-- /results:shap -->

Among the six shared descriptors, TPSA has the largest BBB mean absolute
attribution (1.2652), while LogP dominates P-gp (1.3574). Their importance ranking
correlation is Spearman rho=0.5429, p=0.2657 (six descriptors). This small analysis
does not establish a significant shared ranking. Mean absolute SHAP gives
importance, not direction; a mean signed attribution is not a causal dose-response
effect. Mechanistic interpretations and proposed descriptor windows remain hypotheses.

### Historical pharmacopeia screen — 20 October 2026

> The counts, probabilities, and applicability scores below belong to the October
> model and screening implementation. They are not results of the November models.
> Rerun screening to obtain current candidate rankings. Predictions are research
> hypotheses, not evidence of therapeutic suitability.

`S_CNS = P(BBB+) × (1 − P(P-gp+))`. This is a heuristic ranking score, not a calibrated joint probability.
Marginal label association does not establish conditional independence given a molecule.

| Quadrant | Meaning | Count |
|---|---|---|
| Q1 | BBB+ / P-gp− — reaches the brain | **710 (29.5%)** |
| Q2 | BBB+ / P-gp+ — **paradoxical** | **240 (10.0%)** |
| Q3 | BBB− / P-gp− — peripheral | 1,064 (44.2%) |
| Q4 | BBB− / P-gp+ — doubly restricted | 391 (16.3%) |

→ **273 non-CNS approved drugs in Q1** as repurposing candidates (Thalidomide, Betamethasone,
Linezolid, Milnacipran, Topotecan).
→ **105 marketed neuropsychiatric drugs in Q2**, a structural rationale for treatment
resistance — including **Everolimus** (P(P-gp+) = 0.976), prescribed for subependymal giant
cell astrocytoma, a brain tumour.

### Historical docking analysis — 24 October 2026

> These seeded docking results, group assignments, and ML correlations use the
> October model probabilities. They have not been reassessed with the new models.

| Group | n | ΔG_bind (kcal/mol) |
|---|:---:|:---:|
| Q2 predicted substrates | 9 | **−8.84 ± 0.89** |
| Q1 predicted evaders | 10 | **−6.91 ± 1.30** |

Welch *t* = −3.7933, **p = 1.6 × 10⁻³**; Mann-Whitney *U* = 11.0, p = 3.1 × 10⁻³;
Spearman ρ(P(P-gp+), ΔG) = **−0.5733**, p = 5.3 × 10⁻³.

A ligand-topology model trained only on cell-assay phenotypes agrees with rigid-receptor
binding thermodynamics it was never shown.

> **The separation does not transfer to individual compounds.** Four Q1 evaders bind as
> tightly as the weakest Q2 substrates, and the peripheral control **Penicillin G reaches
> −8.96 kcal/mol** at a predicted efflux probability of 0.020. Rigid-receptor ΔG is a
> population-level signal, not a per-molecule discriminator — consistent with published
> reports, and **precisely why a ligand model adds information the structure does not.**

### Multi-task GNN: mixed single-run effects and no across-seed AUC gain

The shared GINEConv encoder is compared with independent single-task models on
identical molecule partitions. Random-split training includes 5,463 BBB-labelled
and 848 P-gp-labelled molecules. Single-task models have 364,608 parameters;
the two-head model has 372,898. The following are ROC-AUC comparisons for seed 42:

<!-- results:gnn -->
| Task | Split | Single-task AUC | Multitask AUC | Delta | Paired delta 95% CI | Paired result |
| --- | --- | --- | --- | --- | --- | --- |
| BBB | random | 0.9391 | 0.9285 | -0.0106 | [-0.0189, -0.0024] | degraded |
| BBB | scaffold | 0.7661 | 0.7551 | -0.0110 | [-0.0273, +0.0034] | paired interval includes zero |
| PGP | random | 0.9323 | 0.9366 | +0.0042 | [-0.0178, +0.0257] | paired interval includes zero |
| PGP | scaffold | 0.8087 | 0.8365 | +0.0278 | [-0.0073, +0.0628] | paired interval includes zero |
<!-- /results:gnn -->

Paired bootstrap intervals resample the same test molecules for both models;
they are conditional on these fitted models and do not measure training-seed
variability. BBB random-split AUC decreases by 0.0106, with a paired interval
excluding zero. The other three intervals include zero, including the positive
P-gp scaffold estimate (+0.0278). Overlap of separate model intervals is not used
as a significance test.

The zero-P-gp-weight control yields BBB AUC 0.9383 versus single-task 0.9391,
while the untrained P-gp head yields 0.5077. Increasing P-gp weight to 3 and 5
yields P-gp AUCs 0.9346 and 0.9378, and BBB AUCs 0.9267 and 0.9215.
These point estimates show a trade-off; they do not establish a universal
negative-transfer mechanism or prove that weak marginal label association
precludes useful shared representations.

Five paired random-split seeds (42–46) give:

<!-- results:seed_variance -->
| Task | Metric | Seeds | Mean delta | Delta 95% CI | MDE | Paired p |
| --- | --- | --- | --- | --- | --- | --- |
| BBB | ROC_AUC | 5 | -0.006956 | [-0.018911, +0.004999] | 0.016007 | 0.1815 |
| BBB | MCC | 5 | -0.039378 | [-0.075128, -0.003627] | 0.047866 | 0.0377 |
| BBB | F1_Score | 5 | -0.015699 | [-0.025917, -0.005480] | 0.013681 | 0.013 |
| BBB | Balanced_Accuracy | 5 | -0.018726 | [-0.047316, +0.009865] | 0.03828 | 0.1431 |
| PGP | ROC_AUC | 5 | +0.001989 | [-0.023149, +0.027127] | 0.033657 | 0.8369 |
| PGP | MCC | 5 | +0.002344 | [-0.097909, +0.102596] | 0.134229 | 0.9514 |
| PGP | F1_Score | 5 | -0.000968 | [-0.048114, +0.046177] | 0.063124 | 0.9573 |
| PGP | Balanced_Accuracy | 5 | +0.001613 | [-0.049375, +0.052601] | 0.068268 | 0.9342 |
<!-- /results:seed_variance -->

Neither task's across-seed ROC-AUC contrast excludes zero: BBB mean delta
−0.006956 (p=0.1815), P-gp +0.001989 (p=0.8369). Both absolute deltas are below
their estimated MDEs (0.016007 and 0.033657). BBB MCC and F1 decrease at nominal
p<0.05, but these secondary tests are exploratory and unadjusted for multiplicity.
The evidence supports no detected across-seed AUC improvement, not equivalence.
No definitive required-seed count is inferred from these five runs.


---

## Datasets

| Dataset | File | Rows | Source |
|---|---|---|---|
| **B3DB** — BBB benchmark | `data/B3DB_classification.tsv` | 7,807 | Meng, Xi, Huang & Ayers, *Sci. Data* **8**, 289 (2021) |
| **Broccatelli** — P-gp benchmark | `data/pgp_broccatelli.tab` | 1,219 | Broccatelli et al. 2011, via Therapeutics Data Commons |
| **Drug Repurposing Hub** | `data/repurposing/*.txt` | — | Broad Institute, 2020-03-24 release |
| **Human P-gp** | `data/docking/6QEX.pdb` | — | RCSB PDB; Alam et al., *Science* **363**, 753 (2019) |

Derived, with losses accounted for:

```
7,807 B3DB      → 7,805  (2 SMILES RDKit cannot parse)
1,219 Broccatelli → 1,205 featurized rows → 1,198 unique XGBoost structures
                    (14 parsing/conformer exclusions; 7 duplicate identities removed)
Hub .txt        → 2,405  (clinical_phase == "Launched", RDKit-validated, deduplicated)
B3DB + Broccatelli → 8,801 unique structures (InChIKey join)
                      7,805 BBB-labelled · 1,212 P-gp-labelled · 216 BOTH
```

Feature spaces: **2,054 dims** for BBB (2048-bit Morgan ECFP4 r=2 ‖ 6 physicochemical) and
**2,060** for P-gp (the same ‖ 6 ETKDGv3 shape descriptors).

> The processed CSVs store **raw** descriptor values by design. Every consumer must apply the
> training-derived scaler — use `core.features.load_processed_dataset()`, which does it and
> **raises** if the scaler metadata is missing. Reading them directly feeds raw values to a
> model trained on z-scores.

---

## Install and run

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

The graph-neural-network arm needs `torch_geometric` and falls back to CPU without a GPU.
**Activate the venv before running anything under `gnn/` or `scripts/*multitask*`** — including
`verify_decision_metrics.py` and `verify_multitask_gnn.py`, which import the `gnn` package.
Install `requirements-dev.txt` into the same environment and run all tests there.
The current project virtual environment already includes pytest.

```bash
# Predict one molecule, with the full explainability report
python "Novelty Upgrade/inference.py" --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --explain

# A worked output is committed at results/loperamide_report.{csv,html}

# The five clinical reference drugs
python "Novelty Upgrade/inference.py" --test-benchmark
```

```bash
cd "Novelty Upgrade"

# 1. Featurise          2. Train
python training/bbb_data_preprocessing.py   ; python training/bbb_model_training.py
python training/pgp_data_preprocessing.py   ; python training/pgp_model_training.py

# 3. Validate
python scripts/run_validation.py            # random + scaffold, bootstrap CIs
python scripts/run_ablation_study.py        # feature-subset ablation
python scripts/run_explainability_analysis.py

# 4. Apply
python scripts/run_repurposing_screen.py          # 2,405-drug screen
python scripts/run_cryoem_docking_validation.py   # docking into 6QEX

# 5. Multi-task GNN
python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
# Generate matching single-task OOF predictions before joint decision analysis.
python scripts/run_kfold_multitask.py --tasks bbb --folds 5 --tag single_bbb
python scripts/run_kfold_multitask.py --tasks pgp --folds 5 --tag single_pgp
python scripts/run_metric_analysis.py
python scripts/run_seed_variance.py

# 6. Verify  — run after anything that changes a saved metric
python scripts/sync_result_tables.py --check  # all marked tables match CSVs
python scripts/verify_doc_consistency.py        # documentation and provenance checks
python scripts/verify_decision_metrics.py       # 31 checks: metric definitions
python scripts/verify_multitask_gnn.py          # 29 checks: splits, masking, CUDA
python -m pytest tests -q                     # behavior and audit regressions
```

---

## Layout

```
Novelty Upgrade/
  inference.py          single-molecule prediction with --explain
  core/                 features.py (featurisation + scaled loading)
                        decision_matrix.py (four quadrants, S_CNS)
  training/             preprocessing + training, one pair per barrier
  explainability/       TreeSHAP, substructure mapping, applicability domain, counterfactuals
  gnn/                  multi-task GNN: featurizer, model, masked loss, splits, decision metrics
  scripts/              validation, ablation, screening, docking, GNN experiments, 3 verifiers
  tests/                four-quadrant behavioural contracts
  data/ models/ results/ figures/
Initial Model/          base-paper reference implementation (unmodified, not in the pipeline)
                        GCN-BBB on the legacy corpus: 1,890 molecules from the Zhao 2007
                        supporting information, split Adenot (1,621) & Li (269)
docs/VIVA_PREP.md       defence brief: every figure traced to its source file
README.md  RESEARCH_PAPER_MANUSCRIPT.md  requirements.txt
```

---

## Limitations

- **The clinical benchmark drugs are in the training sets.** All five are in B3DB; three are
  also in Broccatelli. Those comparisons are **recall checks, not blind validation**. The
  scaffold split is the only generalisation evidence claimed. (Three showcase drugs *are*
  absent from both at connectivity-block level: Linezolid, Lurasidone, Pimavanserin — but
  three molecules is an illustration, not a validation set.)
- **Only 216 molecules carry both labels**, so every joint analysis rests on a small sample.
  The odds-ratio interval reaches 2.67; Q4 holds ~19 molecules.
- **3D conformer descriptors show no mean AUC gain** in the current P-gp ablation.
- **The dual-barrier framing is not new.** Adenot & Lahana (*J. Chem. Inf. Comput. Sci.* 2004,
  44, 239–248) built a BBB model and a P-gp model and proposed a combined CNS map. The
  contribution here is the **measurement** and its decision-level consequence, not the
  framework.
- **Predicting BBB+ / P-gp− does not guarantee efficacy.** Net exposure also depends on unbound
  brain fraction, plasma protein binding, BCRP and MRP efflux, and metabolic clearance. BCRP is
  the next transporter to add.

---

## Key references

1. **Löscher, W. & Potschka, H.** (2005) Blood-brain barrier active efflux transporters.
   *NeuroRx* **2**(1), 86–98.
2. **Meng, F., Xi, Y., Huang, J. & Ayers, P. W.** (2021) A curated diverse molecular database
   of blood-brain barrier permeability with chemical descriptors. *Sci. Data* **8**, 289.
3. **Broccatelli, F. et al.** (2011) Large-scale classification of P-glycoprotein substrates.
   *J. Med. Chem.* **54**(6), 1740–1751.
4. **Adenot, M. & Lahana, R.** (2004) Blood-brain barrier permeation models. *J. Chem. Inf.
   Comput. Sci.* **44**(1), 239–248.
5. **Alam, A. et al.** (2019) Structural insight into substrate and inhibitor discrimination by
   human P-glycoprotein. *Science* **363**(6428), 753–756.
6. **Seelig, A.** (1998) How does P-glycoprotein recognize its substrates? *Int. J. Clin.
   Pharmacol. Ther.* **36**(1), 50–54.
7. **Lundberg, S. M. et al.** (2020) From local explanations to global understanding with
   explainable AI for trees. *Nat. Mach. Intell.* **2**, 56–67.
8. **Pajouhesh, H. & Lenz, G. R.** (2005) Medicinal chemical properties of successful CNS
   drugs. *NeuroRx* **2**(4), 541–553.
