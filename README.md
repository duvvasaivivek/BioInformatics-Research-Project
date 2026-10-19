# Dual-Barrier CNS Drug Predictor & Explainable AI Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost 2.0+](https://img.shields.io/badge/xgboost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![TreeSHAP](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-teal.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A scientifically grounded, explainable machine learning framework that **decouples passive Blood-Brain Barrier (BBB) permeability from active P-glycoprotein (P-gp / ABCB1) efflux transport**.

---

## 1. Scientific Motivation

Over 95% of neurotherapeutic candidates fail in clinical development due to inadequate brain exposure. Traditional QSAR methods evaluate BBB crossing purely as a function of passive lipophilicity and molecular size. This paradigm produces dangerous false positives: highly lipophilic molecules readily partition into brain capillary endothelial membranes, only to be actively extruded by **P-glycoprotein (P-gp / ABCB1)** back into the bloodstream.

This gives rise to **"paradoxical drugs"** (e.g., **Loperamide**), which possess high lipid solubility ($\log P > 5.0$) but produce zero central nervous system activity at therapeutic doses. Conversely, optimal CNS drugs (e.g., **Levetiracetam**) inhabit a narrow **"Golden Window"**: they permeate the membrane while remaining completely refractory to P-gp efflux.

```
                      PASSIVE MEMBRANE INFLUX (BBB+)
               ┌──────────────────────────────────────────┐
               │  MolLogP > 1.5                           │
               │  TPSA < 90 Å²                            │
               │  H-Bond Acceptors ≤ 6                    │
               └────────────────────┬─────────────────────┘
                                    │
                     THE "GOLDEN CNS WINDOW" (BBB+ / P-gp-)
                     ★ Moderate Lipophilicity: 1.5 ≤ LogP ≤ 3.0
                     ★ Low Molecular Weight: MW < 380 Da
                     ★ Restricted H-Bonding: H-Donors ≤ 1
                     ★ Compact 3D Envelope: Rg < 3.8 Å, Spherocity > 0.25
                                    │
               ┌────────────────────┴─────────────────────┐
               │  MolLogP > 3.5                           │
               │  MW > 450 Da                             │
               │  Rg > 4.5 Å                              │
               │  Polyvalent H-Bond Donor Pairs (Seelig)  │
               └──────────────────────────────────────────┘
                         ACTIVE EFFLUX (P-gp+)
```

---

## 2. Key Capabilities & Architecture

1. **Dual-Barrier Classifiers:**
   - **BBB Permeability Classifier ($\mathcal{M}_{\text{BBB}}$):** Optuna Bayesian-tuned XGBoost ensemble trained on the B3DB benchmark ($7,805$ small molecules) under a leakage-free 70/15/15 split. **In-Domain ROC-AUC: 0.9522 [95% CI: 0.9410 - 0.9629]**, **Accuracy: 0.8856 [95% CI: 0.8659 - 0.9035]**, **Bemis-Murcko Scaffold ROC-AUC: 0.9048**.
   - **P-gp Efflux Classifier ($\mathcal{M}_{\text{Pgp}}$):** 3D shape-integrated XGBoost classifier trained on the Broccatelli TDC benchmark ($1,205$ molecules) with 6 stereochemical conformer shape descriptors ($R_g$, Asphericity, Eccentricity, Spherocity, $NPR_1, NPR_2$) generated via RDKit ETKDGv3. **In-Domain ROC-AUC: 0.9587 [95% CI: 0.9330 - 0.9806]**, **Accuracy: 0.9295**, **Bemis-Murcko Scaffold ROC-AUC: 0.9631**. Standalone 3D shape descriptors alone achieve **0.8425 ROC-AUC**.
2. **Mathematically Rigorous TreeSHAP Explainability:**
   - Model attributions computed in log-odds margin space ($f(x) = \phi_0 + \sum \phi_i$) with base value reporting.
   - Automatic deconvolution of active 2048-bit Morgan fingerprint bits back to atomic subgraphs and functional group tags via `MorganSubstructureMapper`.
3. **Cross-Task Shared Molecular Properties Analysis:**
   - Evaluates the 6 common physicochemical descriptors across both models.
   - Exposes how properties diverge biologically (e.g., Lipophilicity promotes passive entry, but acts as the #1 recruiter for active P-gp efflux).
4. **Four-Quadrant Pharmacological Decision Matrix:**
   - Quadrant 1: $\text{BBB}^+ / \text{P-gp}^-$ [Optimal CNS Candidate]
   - Quadrant 2: $\text{BBB}^+ / \text{P-gp}^+$ [Paradoxical Drug / High Efflux Risk]
   - Quadrant 3: $\text{BBB}^- / \text{P-gp}^-$ [Peripheral Low-Permeation Agent]
   - Quadrant 4: $\text{BBB}^- / \text{P-gp}^+$ [Restricted Effluxed Peripheral Compound]
5. **Applicability Domain & Uncertainty Assessment:**
   - Structural Tanimoto maximum similarity against training benchmarks.
   - Normalized continuous descriptor leverage distance.
   - Quantitative dual-barrier uncertainty score.
6. **Reaction-Based Molecular Counterfactual Engine:**
   - Generates realistic medicinal chemistry analogs (HBD capping, N-methylation, fluorination) to flip predictions without breaking chemical sanity.
7. **Scientific Export Suite:**
   - Automated export of publication-grade standalone HTML reports, CSV summaries, and JSON dictionaries.

---

## 3. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/duvvasaivivek/BioInformatics-Research-Project.git
cd "BioInformatics Project"

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Usage & Execution Commands

### A. Run Interactive Inference & Clinical Benchmark Suite
```bash
# Run clinical benchmark suite (Levetiracetam, Loperamide, Caffeine, Diazepam, Penicillin G):
python "B3DB Based Model/inference.py" --test-benchmark

# Predict on a single SMILES string with full explainability:
python "B3DB Based Model/inference.py" --smiles "CC[C@@H](C(=O)N)N1CCCC1=O" --explain

# Predict with counterfactual suggestions and export to HTML & CSV:
python "B3DB Based Model/inference.py" --smiles "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O" --explain --counterfactual --export-html "B3DB Based Model/results/loperamide_report.html" --export-csv "B3DB Based Model/results/loperamide_report.csv"
```

### B. Run Automated Regression Test Suite
```bash
python "B3DB Based Model/tests/test_explainability.py"
```
*Executes 8 automated regression tests covering all 4 quadrants, exception handling, TreeSHAP alignment, and exports.*

### C. Reproduce Validation Metrics (1,000 Bootstrap 95% CI)
```bash
python "B3DB Based Model/scripts/run_validation.py"
```
*Calculates Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, MCC, Brier score, and Murcko scaffold diversity.*

### D. Reproduce Incremental Feature Ablation Study
```bash
python "B3DB Based Model/scripts/run_ablation_study.py"
```
*Runs 5-fold stratified cross-validation on 2D only, 3D only, Morgan only, and full hybrid models.*

### E. Generate Publication Figures & Global SHAP Analysis
```bash
python "B3DB Based Model/scripts/run_explainability_analysis.py"
```
*Generates figures in `B3DB Based Model/figures/`:*
- `bbb_shap_global_summary.png`
- `pgp_shap_global_summary.png`
- `shared_properties_cross_task_comparison.png`
- `dual_barrier_quadrant_matrix.png`

---

## 5. Repository Structure

```
BioInformatics Project/
├── B3DB Based Model/
│   ├── core/                        # Centralized feature extraction & domain logic
│   │   ├── __init__.py
│   │   ├── features.py              # 2D, Morgan FP, and 3D conformer extractors
│   │   └── decision_matrix.py       # 4-quadrant decision logic & benchmark definitions
│   ├── BBB/                         # BBB model training & evaluation
│   │   ├── bbb_data_preprocessing.py
│   │   ├── bbb_model_training.py
│   │   └── bbb_model_evaluation.py
│   ├── Pgp/                         # P-gp 3D model training & evaluation
│   │   ├── pgp_data_preprocessing.py
│   │   ├── pgp_model_training.py
│   │   └── pgp_model_evaluation.py
│   ├── explainability/              # Modular explainability package
│   │   ├── __init__.py
│   │   ├── shap_explainer.py        # TreeSHAP in log-odds margin space
│   │   ├── substructure_mapper.py   # Morgan bit-to-subgraph deconvolution
│   │   ├── shared_properties.py     # Cross-task shared properties divergence
│   │   ├── applicability_domain.py  # Tanimoto & leverage domain estimation
│   │   ├── counterfactual.py        # Reaction-based analog search
│   │   └── report_generator.py      # Standalone HTML, CSV, JSON export
│   ├── data/                        # Curated B3DB and TDC benchmarks
│   ├── models/                      # Serialized models & scaler metadata
│   ├── docs/                        # Consolidated scientific documentation
│   │   └── SCIENTIFIC_DOCUMENTATION.md
│   ├── figures/                     # High-resolution publication figures
│   ├── results/                     # Generated CSV metrics and HTML reports
│   ├── scripts/                     # Automated scientific reproduction scripts
│   │   ├── run_explainability_analysis.py
│   │   ├── run_validation.py
│   │   └── run_ablation_study.py
│   ├── tests/                       # Automated regression test suite
│   │   └── test_explainability.py
│   └── inference.py                 # Streamlined CLI prediction interface
├── Initial Model/                   # Base paper replication (GINEConv GNN)
├── RESEARCH_PAPER_MANUSCRIPT.md     # Full academic manuscript draft
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Key Empirical Results

### Test Set Performance (95% Bootstrap Confidence Intervals)
| Metric | BBB Permeability (B3DB) | P-gp Efflux Substrate (Broccatelli) |
|---|:---:|:---:|
| **Accuracy** | 0.8962 [0.8815 - 0.9116] | 0.9336 [0.9004 - 0.9627] |
| **Balanced Accuracy** | 0.8896 [0.8727 - 0.9066] | 0.9339 [0.9020 - 0.9625] |
| **Precision** | 0.9229 [0.9060 - 0.9381] | 0.9453 [0.9008 - 0.9778] |
| **Recall / Sensitivity** | 0.9137 [0.8963 - 0.9309] | 0.9308 [0.8824 - 0.9706] |
| **F1-Score** | 0.9183 [0.9058 - 0.9307] | 0.9380 [0.9061 - 0.9653] |
| **ROC-AUC** | **0.9544 [0.9449 - 0.9644]** | **0.9593 [0.9320 - 0.9822]** |
| **PR-AUC** | 0.9715 [0.9629 - 0.9789] | 0.9697 [0.9497 - 0.9850] |
| **MCC** | 0.7762 [0.7446 - 0.8099] | 0.8667 [0.8013 - 0.9249] |
| **Brier Score** | 0.0796 [0.0698 - 0.0893] | 0.0666 [0.0464 - 0.0921] |

---

## 7. Known Limitations & Clinical Perspective

1. **Unbound Fraction ($f_{u,\text{brain}}$):** A predicted $\text{BBB}^+ / \text{P-gp}^-$ result indicates favorable passive transcellular crossing and pump evasion, but does not measure non-specific brain tissue binding or metabolic turnover.
2. **Multi-Transporter Complexities:** While P-gp is the primary capillary efflux pump, BCRP (ABCG2) and MRPs (ABCCs) also mediate efflux for specific drug scaffolds.
3. **Preclinical Screening Scope:** This framework is designed for rational in silico lead optimization and drug repurposing, serving as a computational filter prior to in vivo pharmacokinetic testing.

---

## 8. References
1. Löscher, W., & Potschka, H. (2005). *NeuroRx*, 2(1), 86–98.
2. Martins, I. F., et al. (2012). *J. Chem. Inf. Model.*, 52(6), 1686–1697.
3. Broccatelli, F., et al. (2011). *J. Med. Chem.*, 54(6), 1740–1751.
4. Seelig, A. (1998). *Int. J. Clin. Pharmacol. Ther.*, 36(1), 50–54.
5. Lundberg, S. M., et al. (2020). *Nat. Mach. Intell.*, 2(1), 56–67.
