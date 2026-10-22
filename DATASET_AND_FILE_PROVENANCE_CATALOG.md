# Codebase Navigation & Dataset Provenance Guide
*A Simple, Zero-Jargon Guide to Every File, Dataset, and Pipeline in this Repository*

---

## 1. The 30-Second Mental Model: What is this Repo?

If you feel lost opening this repository, here is the entire project explained in 3 simple sentences:
1. **The Biological Goal**: We want to find drugs that can cross into the human brain to treat brain diseases (Alzheimer's, Parkinson's, Glioblastoma).
2. **The Two-Door Problem**: To get into the brain, a drug must pass **Door 1 (Passive Lipid Permeation)** AND avoid being thrown out by **Door 2 (The P-glycoprotein Efflux Pump)**.
3. **The Two Folders in the Repo**:
   - `Initial Model/` is **Paradigm 1 (Legacy Baseline)**: It only modeled Door 1 using a Graph Neural Network (GNN).
   - `B3DB Based Model/` is **Paradigm 2 (Active Discovery Engine)**: It models **both Door 1 and Door 2**, adds 3D conformer shapes, runs Cryo-EM molecular docking, and screens 2,405 FDA-approved drugs.

---

## 2. Cleaned Repository Architecture & File Locations

```
BioInformatics Project/
│
├── Initial Model/                                 [PARADIGM 1: Legacy Graph Neural Network Baseline]
│   ├── Dataset/                                   [raw_adenot_li.xls, cleaned_bbb_dataset.csv, graph_summary.csv]
│   ├── reference_papers/                          [adenot_2004, li_2005, zhao_2006, in_silico_bbb_2018 PDFs]
│   ├── graphs/                                    [1,890 PyG graph tensors (graph_0001.pt to graph_1890.pt)]
│   ├── clean_raw_data.py                          [ETL script: cleans raw_adenot_li.xls -> cleaned_bbb_dataset.csv]
│   ├── graph_generation.py                       [Converts cleaned CSV into 1,890 PyG graph tensors]
│   ├── check_graph.py                             [Inspector: prints tensor shapes of graph_0001.pt]
│   ├── gnn_dataset.py                             [PyG custom dataset loader class]
│   ├── train_gnn.py                               [10-Fold Cross-Validation GINEConv trainer]
│   ├── predict.py                                 [Interactive CLI predictor using trained GNN]
│   └── paper_gcnbbb_2025.pdf                      [Baseline GCNBBB paper from The AAPS Journal (2025)]
│
├── B3DB Based Model/                              [PARADIGM 2: Dual-Barrier Multi-Modal Engine]
│   ├── BBB/                                       [Door 1: Passive Permeation XGBoost Pipeline]
│   │   ├── bbb_data_preprocessing.py              [Featurizes B3DB_classification.tsv -> processed_b3db_data_advanced.csv]
│   │   ├── bbb_model_training.py                  [Optuna 50 trials tuning & trains bbb_xgboost_optuna_best.json]
│   │   └── bbb_model_evaluation.py                [Evaluates Door 1 model on test set & plots ROC curve]
│   │
│   ├── Pgp/                                       [Door 2: Active P-gp Efflux XGBoost Pipeline]
│   │   ├── pgp_data_preprocessing.py              [Featurizes pgp_broccatelli.tab with 3D shapes -> processed_pgp_data.csv]
│   │   ├── pgp_model_training.py                  [Trains pgp_xgboost_model.json for active efflux liability]
│   │   └── pgp_model_evaluation.py                [Evaluates Door 2 model on test set & plots ROC curve]
│   │
│   ├── core/                                      [Shared Foundations]
│   │   ├── features.py                            [Central featurizer: 2048-bit Morgan FP + 6 2D + 6 3D conformer shapes]
│   │   └── decision_matrix.py                     [Calculates 4-Quadrants (Q1-Q4) and Desirability S_CNS]
│   │
│   ├── data/                                      [Datasets & Structural Files]
│   │   ├── B3DB_classification.tsv                [Raw Door 1 benchmark dataset: 7,807 compounds]
│   │   ├── pgp_broccatelli.tab                    [Raw Door 2 benchmark dataset: 1,219 compounds]
│   │   ├── processed_b3db_data_advanced.csv       [Featurized Door 1 matrix: 2054 features]
│   │   ├── processed_pgp_data.csv                 [Featurized Door 2 matrix: 2060 features]
│   │   ├── repurposing/                           [Broad Institute clinical library & 2,405 FDA approved drugs]
│   │   └── docking/                               [Cryo-EM Human P-gp 6QEX receptor + 22 ligand PDBQT files]
│   │
│   ├── models/                                    [Trained Checkpoints & Deterministic Scalers]
│   │   ├── bbb_xgboost_optuna_best.json           [Trained Barrier 1 XGBoost model]
│   │   ├── bbb_scaler_metadata.json               [Exact training means/stds for 6 2D descriptors]
│   │   ├── pgp_xgboost_model.json                 [Trained Barrier 2 XGBoost model]
│   │   └── pgp_scaler_metadata.json               [Exact training means/stds for 12 descriptors (6 2D + 6 3D)]
│   │
│   ├── scripts/                                   [Scientific Discovery & Validation Runners]
│   │   ├── prepare_repurposing_library.py         [Filters Broad Hub down to 2,405 FDA approved drugs]
│   │   ├── run_repurposing_screen.py              [Screens all 2,405 FDA approved drugs through both models]
│   │   ├── run_cryoem_docking_validation.py       [AutoDock Vina docking into human P-gp (PDB: 6QEX)]
│   │   ├── run_ablation_study.py                  [Benchmarks 5 feature sets proving 3D shapes are required]
│   │   ├── run_validation.py                      [Calculates paper test set metrics & profiles 20 clinical drugs]
│   │   └── run_explainability_analysis.py         [Generates TreeSHAP feature attributions & Loperamide report]
│   │
│   ├── explainability/                            [Interpretability Suite]
│   │   ├── shap_explainer.py                      [TreeSHAP feature attribution calculator]
│   │   ├── substructure_mapper.py                 [Maps Morgan bits back to 2D chemical structure atoms]
│   │   ├── applicability_domain.py                [k-NN Tanimoto & Euclidean chemical confidence bounding]
│   │   ├── counterfactual.py                      [Generates medicinal chemistry optimizations (Q3 -> Q1)]
│   │   ├── shared_properties.py                   [Property distribution analysis across BBB and P-gp]
│   │   └── report_generator.py                    [HTML/CSV single-drug diagnostic report compiler]
│   │
│   ├── bin/                                       [AutoDock Vina Executable]
│   │   └── vina.exe                               [Molecular docking engine]
│   │
│   ├── results/                                   [Output Analytical Tables]
│   ├── figures/                                   [12 Publication-Quality Visualizations]
│   ├── tests/                                     [Automated Test Suite]
│   └── inference.py                               [Unified Dual-Barrier CLI Application]
│
├── ARCHITECTURE_DEEP_DIVE.md                      [Comprehensive Architectural & Biological Guide]
├── ARCHITECTURE_DEEP_DIVE.tex                     [Academic LaTeX Monograph with TikZ Architecture]
├── RESEARCH_PAPER_MANUSCRIPT.md                   [Full Scientific Manuscript for Publication]
├── B3DB_Model_Analysis.pdf                        [Project Analysis Report PDF]
├── DATASET_AND_FILE_PROVENANCE_CATALOG.md         [This File: Clean Navigation & File Directory]
├── requirements.txt                               [Python Environment Dependencies]
└── README.md                                      [Repository Entrypoint]
```

---

## 3. The Active Dataset Registry: "What Dataset is Used for What?"

| Dataset Filename | Exact Location | Rows | What's Inside | Which Script Uses It | What Output It Produces |
|---|---|---|---|---|---|
| **`B3DB_classification.tsv`** | `B3DB Based Model/data/` | 7,807 | SMILES, IUPAC name, logBB, BBB+/BBB- | `BBB/bbb_data_preprocessing.py` | `processed_b3db_data_advanced.csv` (2,048 Morgan bits + 6 2D descriptors) |
| **`processed_b3db_data_advanced.csv`** | `B3DB Based Model/data/` | 7,807 | Featurized matrix (2,054 features) | `BBB/bbb_model_training.py` | `models/bbb_xgboost_optuna_best.json` (Barrier 1 XGBoost model) |
| **`pgp_broccatelli.tab`** | `B3DB Based Model/data/` | 1,219 | SMILES, Drug ID, Efflux Activity (1/0) | `Pgp/pgp_data_preprocessing.py` | `processed_pgp_data.csv` (2,048 Morgan bits + 6 2D + 6 3D conformer shapes) |
| **`processed_pgp_data.csv`** | `B3DB Based Model/data/` | 1,219 | Featurized matrix (2,060 features) | `Pgp/pgp_model_training.py` | `models/pgp_xgboost_model.json` (Barrier 2 XGBoost model) |
| **`fda_approved_repurposing_compounds.csv`** | `B3DB Based Model/data/repurposing/` | 2,405 | Approved drug names, SMILES, indications | `scripts/run_repurposing_screen.py` | `results/fda_repurposing_dual_barrier_screen.csv` & `top_cns_repurposing_candidates.csv` |
| **`6QEX.pdb` & `pgp_chain_a.pdbqt`** | `B3DB Based Model/data/docking/` | 1 structure | Cryo-EM 3.6 Å structure of human P-gp | `scripts/run_cryoem_docking_validation.py` | Rigid receptor grid for AutoDock Vina |
| **22 Ligand `.pdbqt` files** | `B3DB Based Model/data/docking/` | 22 drugs | 3D conformers of clinical benchmark drugs | `scripts/run_cryoem_docking_validation.py` | `results/cryoem_pgp_docking_validation.csv` (Docking vs ML correlation $r = -0.732$) |
| **`raw_adenot_li.xls`** *(Legacy)* | `Initial Model/Dataset/` | 2 sheets | Raw Zhao 2006 literature Excel tables | `Initial Model/clean_raw_data.py` | Generates `cleaned_bbb_dataset.csv` |
| **`cleaned_bbb_dataset.csv`** *(Legacy)* | `Initial Model/Dataset/` | 1,890 | SMILES from Adenot (1,593) & Li (415) | `Initial Model/graph_generation.py` | `Initial Model/graphs/graph_XXXX.pt` (1,890 graph tensors) |

---

## 4. How to Run the Entire Project

```bash
# 1. Preprocess the two core training datasets
python "B3DB Based Model/BBB/bbb_data_preprocessing.py"
python "B3DB Based Model/Pgp/pgp_data_preprocessing.py"

# 2. Train both models
python "B3DB Based Model/BBB/bbb_model_training.py"
python "B3DB Based Model/Pgp/pgp_model_training.py"

# 3. Run validation benchmarks and Cryo-EM docking
python "B3DB Based Model/scripts/run_validation.py"
python "B3DB Based Model/scripts/run_ablation_study.py"
python "B3DB Based Model/scripts/run_cryoem_docking_validation.py"

# 4. Run the 2,405 FDA Drug Repurposing Discovery Screen
python "B3DB Based Model/scripts/run_repurposing_screen.py"

# 5. Predict any single molecule interactively
python "B3DB Based Model/inference.py" --smiles "CN1CCC[C@H]1c2cccnc2"
```
