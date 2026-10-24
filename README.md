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

All figures below are read from files in `B3DB Based Model/results/` and are enforced
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
than evaders: **−8.84 ± 0.89** vs **−6.91 ± 1.30 kcal/mol**, Welch *t* = −3.79,
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
python "B3DB Based Model/inference.py" --smiles "CN1CCC[C@H]1c2cccnc2" --explain
```

Returns the quadrant call, both probabilities, TreeSHAP attributions mapped to
substructures, applicability-domain status, and optional reaction-based counterfactuals.

### Reproduce the pipeline end to end

```bash
cd "B3DB Based Model"

# 1. Featurise both benchmarks
python BBB/bbb_data_preprocessing.py
python Pgp/pgp_data_preprocessing.py

# 2. Train both models
python BBB/bbb_model_training.py          # Optuna, 50 trials, scored on a held-out validation split
python Pgp/pgp_model_training.py

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
cd "B3DB Based Model"

python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py             # decision-aware metrics
python scripts/run_seed_variance.py               # seed variance, MDE, P(A>B)
```

### Verify everything

```bash
cd "B3DB Based Model"

python scripts/verify_doc_consistency.py    # 45 checks: docs match the artifacts
python scripts/verify_decision_metrics.py   # 31 checks: metric definitions
python scripts/verify_multitask_gnn.py      # 29 checks: splits, masking, CUDA
python tests/test_explainability.py         #  8 checks: four-quadrant contracts
```

**Run `verify_doc_consistency.py` after anything that changes a saved metric.** It
asserts that every number quoted in the prose matches the file it came from. Three
manuscript defects existed purely because figures were copied between documents
instead of re-read from the source.

---

## Data

| File | Rows | Contents |
|---|---|---|
| `data/B3DB_classification.tsv` | 7,807 | B3DB blood-brain barrier benchmark — SMILES, logBB, BBB+/BBB− |
| `data/pgp_broccatelli.tab` | 1,219 | Broccatelli P-glycoprotein substrate benchmark (via TDC) |
| `data/processed_b3db_data_advanced.csv` | 7,805 | 2,054 features: 2048-bit Morgan + 6 physicochemical descriptors |
| `data/processed_pgp_data.csv` | 1,205 | 2,060 features: the above plus 6 ETKDGv3 conformer shape descriptors |
| `data/repurposing/fda_approved_repurposing_compounds.csv` | 2,405 | Broad Drug Repurposing Hub, approved subset |
| `data/docking/6QEX.pdb` | 1 | Cryo-EM structure of human P-glycoprotein |

> The processed CSVs store **raw** descriptor values. Anything reading them must apply
> the training-derived scaler — use `core.features.load_processed_dataset()`, which does
> it and raises if the scaler metadata is missing. Reading them directly feeds raw values
> to a model trained on z-scores.

`Initial Model/` holds the reference implementation of the base paper (GCN-BBB) on the
legacy Adenot/Li corpus — 1,890 molecules drawn from the Zhao 2007 supporting
information, split Adenot (1,621) & Li (269). It is kept unmodified for provenance and
is not part of the pipeline above.

---

## Repository layout

```
B3DB Based Model/
  core/          features.py (featurisation + scaled loading), decision_matrix.py (quadrants)
  BBB/ Pgp/      preprocessing and training for each barrier
  gnn/           multi-task graph neural network: featurizer, model, masked loss,
                 splits, training loop, decision-aware metrics
  explainability/ TreeSHAP, substructure mapping, applicability domain, counterfactuals
  scripts/       validation, ablation, screening, docking, GNN experiments, verifiers
  tests/         four-quadrant behavioural contracts
  data/ models/ results/ figures/
docs/            architecture, block diagram, research log
Initial Model/   base-paper reference implementation (unmodified)
```

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
