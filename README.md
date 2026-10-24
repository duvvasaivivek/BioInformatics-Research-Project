# Dual-Barrier CNS Drug Prediction

Predicting whether a small molecule reaches the brain requires two answers, not one:
can it **cross** the blood-brain barrier passively, and will **P-glycoprotein** pump it
straight back out. A molecule can do the first perfectly and still fail, which is why
Loperamide is a potent opioid with no central effect at all.

This repository models both barriers, explains its predictions, screens the approved
pharmacopeia through them, and checks the efflux axis against a cryo-EM structure of
human P-glycoprotein.

---

## Documentation

| Document | What it is for |
|---|---|
| **README.md** (this file) | What the project is, how to install it, how to run everything |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the models work: features, architectures, validation protocol |
| [`docs/BLOCK_DIAGRAM.md`](docs/BLOCK_DIAGRAM.md) | Exhaustive block-by-block specification with I/O shapes and loss functions |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | Decision record: novelty assessment, prior art, build logs, audit history |
| [`RESEARCH_PAPER_MANUSCRIPT.md`](RESEARCH_PAPER_MANUSCRIPT.md) | The manuscript draft |

---

## Results

All figures below are read from files in `Novelty Upgrade/results/` and are enforced
against those files by `scripts/verify_doc_consistency.py`.

**Model performance** (1,000-resample bootstrap 95% CI)

| | Random split | Bemis-Murcko scaffold split |
|---|---|---|
| BBB permeability, ROC-AUC | **0.9522** [0.9410, 0.9629] | 0.9048 [0.8835, 0.9240] |
| BBB specificity | 0.8478 | **0.5795** |
| P-gp efflux, ROC-AUC | **0.9587** [0.9330, 0.9806] | 0.9631 [0.9359, 0.9846] |

The scaffold split is the honest number. BBB permeability degrades sharply on unseen
chemotypes while P-gp efflux does not — the efflux axis transfers, the permeability
axis does not.

**Pharmacopeia screen** — 2,405 approved drugs across the dual-barrier quadrants:

| Quadrant | Meaning | Count |
|---|---|---|
| Q1 | BBB+ / P-gp− — reaches the brain | 710 (29.5%) |
| Q2 | BBB+ / P-gp+ — **paradoxical**: permeant but pumped out | 240 (10.0%) |
| Q3 | BBB− / P-gp− — peripheral | 1,064 (44.2%) |
| Q4 | BBB− / P-gp+ — doubly restricted | 391 (16.3%) |

This yields **273 non-CNS approved drugs in Q1** as repurposing candidates and
**105 marketed neuropsychiatric drugs in Q2** whose efflux liability offers a
structural rationale for treatment resistance.

**Cryo-EM validation** — 22 compounds docked into human P-gp (PDB 6QEX, seeded,
reproducible). Predicted efflux substrates bind the translocation cavity more tightly
than evaders: **−8.84 ± 0.89** vs **−6.91 ± 1.30 kcal/mol**, Welch *t* = −3.7933,
*p* = 1.6×10⁻³; Spearman ρ = −0.5733 between predicted efflux probability and binding
energy.

The group separation is real but does **not** transfer to individual compounds — four
Q1 evaders and the peripheral control Penicillin G bind as tightly as the weakest Q2
substrates. Rigid-receptor ΔG is a population-level signal, not a per-molecule
discriminator, which is precisely why a ligand-topology model adds information the
structure alone does not.

---

## Installation

```bash
git clone https://github.com/duvvasaivivek/BioInformatics-Research-Project.git
cd "BioInformatics Project"

python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Unix

pip install -r requirements.txt
```

A CUDA GPU is optional and used only by the graph-neural-network arm, which falls back
to CPU automatically.

---

## Running it

### Predict a single molecule

```bash
python "Novelty Upgrade/inference.py" --smiles "CN1CCC[C@H]1c2cccnc2" --explain
```

Returns the quadrant call, both probabilities, TreeSHAP attributions mapped to
substructures, applicability-domain status, and optional reaction-based counterfactuals.

### Reproduce the pipeline end to end

```bash
cd "Novelty Upgrade"

# 1. Featurise both benchmarks
python training/bbb_data_preprocessing.py
python training/pgp_data_preprocessing.py

# 2. Train both models
python training/bbb_model_training.py          # Optuna, 50 trials, scored on a held-out validation split
python training/pgp_model_training.py

# 3. Validate
python scripts/run_validation.py          # random + scaffold splits, bootstrap CIs
python scripts/run_ablation_study.py      # feature-subset ablation
python scripts/run_explainability_analysis.py

# 4. Apply
python scripts/run_repurposing_screen.py          # 2,405-drug pharmacopeia screen
python scripts/run_cryoem_docking_validation.py   # docking into PDB 6QEX
```

### The multi-task graph neural network

```bash
cd "Novelty Upgrade"

python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py             # decision-aware metrics
python scripts/run_seed_variance.py               # seed variance, MDE, P(A>B)
```

### Verify everything

```bash
cd "Novelty Upgrade"

python scripts/verify_doc_consistency.py    # 57 checks: docs match the artifacts
python scripts/verify_decision_metrics.py   # 31 checks: metric definitions
python scripts/verify_multitask_gnn.py      # 29 checks: splits, masking, CUDA
python tests/test_explainability.py         #  8 checks: four-quadrant contracts
```

**Run `verify_doc_consistency.py` after anything that changes a saved metric.** It
asserts that every number quoted in the prose matches the file it came from. Three
manuscript defects existed purely because figures were copied between documents
instead of re-read from the source.

---

## Where everything lives

Every path below is relative to `Novelty Upgrade/`. The three tables answer, in order:
*what data do we have*, *which script uses it*, and *which file holds a given result*.

### 1. Datasets

| Dataset | File | Rows | Upstream source |
|---|---|---|---|
| **B3DB** - blood-brain barrier benchmark | `data/B3DB_classification.tsv` | 7,807 | B3DB v1 (Meng et al., *Sci. Data* 2021); SMILES, logBB, BBB+/BBB- |
| **Broccatelli** - P-gp substrate benchmark | `data/pgp_broccatelli.tab` | 1,219 | Broccatelli et al. 2011, retrieved via TDC |
| B3DB, featurised | `data/processed_b3db_data_advanced.csv` | 7,805 | built by `training/bbb_data_preprocessing.py` |
| Broccatelli, featurised | `data/processed_pgp_data.csv` | 1,205 | built by `training/pgp_data_preprocessing.py` |
| **Drug Repurposing Hub**, approved subset | `data/repurposing/fda_approved_repurposing_compounds.csv` | 2,405 | built by `scripts/prepare_repurposing_library.py` |
| Drug Repurposing Hub, raw export | `data/repurposing/repurposing_drugs_20200324.txt`<br>`data/repurposing/repurposing_samples_20200324.txt` | - | Broad Institute, 2020-03-24 release |
| **Human P-glycoprotein**, cryo-EM structure | `data/docking/6QEX.pdb` | - | RCSB PDB entry 6QEX |
| P-gp receptor, prepared for docking | `data/docking/pgp_chain_a.pdb`, `.pdbqt` | - | chain A extracted from 6QEX, then Meeko |
| Docked ligands and output poses | `data/docking/poses/` | 44 files | regenerated by every docking run |

Two featurisations exist: **2,054 features** for BBB (2048-bit Morgan ECFP4 + 6 physicochemical
descriptors) and **2,060** for P-gp (the same plus 6 ETKDGv3 conformer shape descriptors).

> The processed CSVs store **raw** descriptor values. Anything reading them must apply the
> training-derived scaler - use `core.features.load_processed_dataset()`, which does it and raises
> if the scaler metadata is missing. Reading them directly feeds raw values to a model trained on
> z-scores.

### 2. Which script uses which dataset

Derived by parsing each script's path constants, so it cannot drift from the code.

| Script | Reads | Writes |
|---|---|---|
| `training/bbb_data_preprocessing.py` | `data/B3DB_classification.tsv` | `data/processed_b3db_data_advanced.csv` |
| `training/pgp_data_preprocessing.py` | `data/pgp_broccatelli.tab` | `data/processed_pgp_data.csv` |
| `training/bbb_model_training.py` | `data/processed_b3db_data_advanced.csv` | `models/bbb_xgboost.json`, `models/bbb_scaler_metadata.json` |
| `training/pgp_model_training.py` | `data/processed_pgp_data.csv` | `models/pgp_xgboost.json`, `models/pgp_scaler_metadata.json` |
| `scripts/run_validation.py` | both processed CSVs + both models | `results/validation_metrics_summary.csv`, 2 confusion-matrix figures |
| `scripts/run_ablation_study.py` | both processed CSVs | `results/ablation_study_results.csv`, `figures/feature_ablation_comparison.png` |
| `scripts/run_explainability_analysis.py` | both processed CSVs + both models | `results/shared_properties_quantitative_comparison.csv`, 4 figures |
| `scripts/prepare_repurposing_library.py` | the two raw Broad Hub `.txt` files | `data/repurposing/fda_approved_repurposing_compounds.csv` |
| `scripts/run_repurposing_screen.py` | the approved-drug CSV + both models | 3 screen CSVs, 2 figures |
| `scripts/run_cryoem_docking_validation.py` | `data/docking/6QEX.pdb` + both models | `results/cryoem_pgp_docking_validation.csv`, 2 figures, `data/docking/poses/` |
| `gnn/data_assembly.py` | `data/B3DB_classification.tsv` **and** `data/pgp_broccatelli.tab` | the merged 8,801-molecule corpus (in memory) |
| `scripts/train_multitask_gnn.py` | the merged corpus, cached under `data/graph_cache/` | `models/gnn_<tag>.pt`, 4 `results/gnn_<tag>_*` files |
| `scripts/run_kfold_multitask.py` | the merged corpus | `results/gnn_<tag>_kfold*_report.json` + OOF predictions |
| `scripts/run_metric_analysis.py` | the k-fold OOF predictions | `results/gnn_decision_metrics.csv`, `_report.json` |
| `scripts/run_seed_variance.py` | the merged corpus | `results/gnn_seedvar_*` (4 files) |
| `inference.py` | both models + both scalers | stdout, or `results/<name>_report.{csv,html}` |

The GNN arm is the only code that reads **both** benchmarks at once; that merge, and the 216
molecules carrying both labels, is what every dual-barrier analysis rests on.

### 3. Results index

| File in `results/` | What it answers |
|---|---|
| `validation_metrics_summary.csv` | headline ROC-AUC on random and scaffold splits, with bootstrap CIs |
| `ablation_study_results.csv` | whether each feature block earns its place (24 cells) |
| `shared_properties_quantitative_comparison.csv` | which descriptors drive BBB vs P-gp (TreeSHAP) |
| `fda_repurposing_dual_barrier_screen.csv` | all 2,405 approved drugs, assigned to quadrants |
| `top_cns_repurposing_candidates.csv` | the 273 non-CNS drugs landing in Q1 |
| `cns_drugs_efflux_resistance_profiles.csv` | the 105 marketed CNS drugs landing in Q2 |
| `cryoem_pgp_docking_validation.csv` | 22 compounds docked into 6QEX: dG, Kd, contact residues |
| `gnn_decision_metrics.csv` + `_report.json` | the five decision-aware metrics M1-M5 |
| `gnn_experiment_comparison.csv`, `gnn_multitask_delta.csv` | single-task vs multi-task, side by side |
| `gnn_seedvar_*` (4 files) | seed variance, minimum detectable effect, P(A>B) |
| `gnn_<tag>_*` (history / metrics / report / test_predictions) | one GNN run each; `<tag>` comes from `--tag` |
| `loperamide_report.{csv,html}` | a worked single-molecule `inference.py` example |

`figures/` holds 11 PNGs, each written by the script named in table 2.
`models/` holds the two XGBoost boosters, their two scaler metadata files, and 9 GNN checkpoints.

---

## Repository layout

```
Novelty Upgrade/          <- all novel work lives here
  inference.py            single-molecule prediction, with --explain
  core/                   features.py (featurisation + scaled loading)
                          decision_matrix.py (four quadrants, S_CNS)
  training/               preprocessing + training, one pair per barrier:
                            bbb_data_preprocessing.py / bbb_model_training.py
                            pgp_data_preprocessing.py / pgp_model_training.py
  gnn/                    multi-task graph neural network: featurizer, model,
                          masked loss, splits, training loop, decision metrics
  explainability/         TreeSHAP, substructure mapping, applicability domain,
                          counterfactuals
  scripts/                validation, ablation, screening, docking,
                          GNN experiments, and the four verifiers
  tests/                  four-quadrant behavioural contracts
  data/                   benchmarks - see table 1 above
    docking/              6QEX.pdb + prepared receptor; poses/ holds the output
    repurposing/          Drug Repurposing Hub, raw and approved-subset
    graph_cache/          generated graph tensors (gitignored)
  models/ results/ figures/
docs/                     ARCHITECTURE, BLOCK_DIAGRAM, RESEARCH_LOG
Initial Model/            base-paper reference implementation (unmodified)
README.md  RESEARCH_PAPER_MANUSCRIPT.md  requirements.txt
```

`Initial Model/` holds the reference implementation of the base paper (GCN-BBB) on the
legacy Adenot/Li corpus - 1,890 molecules drawn from the Zhao 2007 supporting
information, split Adenot (1,621) & Li (269). It is kept unmodified for provenance and
is not part of the pipeline above.

---

## Honest limitations

- **The clinical benchmark drugs are in the training sets.** All five are in B3DB and
  three are also in Broccatelli. Table 4 of the manuscript is a recall check, not blind
  validation; the scaffold split is the only generalisation evidence claimed.
- **Only 216 molecules carry both labels**, so every dual-barrier analysis rests on a
  small sample. The 95% interval on the label-independence odds ratio reaches 2.67.
- **3D conformer descriptors add nothing** once Morgan fingerprints are present
  (0.9481 with them, 0.9509 without). They are retained for mechanistic interpretability,
  not accuracy.
- **Predicting BBB+ and P-gp− does not guarantee efficacy.** Net brain exposure also
  depends on unbound fraction, plasma protein binding, BCRP and MRP efflux, and
  metabolic clearance.
