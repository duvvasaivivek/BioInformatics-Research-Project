# Reproducibility Guide & Execution Commands

This guide provides exact, deterministic commands to reproduce all evaluations, figures, tables, and inferences described in the study.

## 1. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Running Automated Regression & Validation Tests
To run the automated test suite verifying all 4 quadrants, TreeSHAP calculations, and exports:
```bash
.\venv\Scripts\python.exe "B3DB Based Model/tests/test_explainability.py"
```
Expected output: `Test Summary: 8/8 tests PASSED.`

---

## 3. Generating Research Visualizations & Global SHAP Analysis
To compute test-set TreeSHAP distributions, Spearman rank correlations, and generate all manuscript figures in `B3DB Based Model/figures/`:
```bash
.\venv\Scripts\python.exe "B3DB Based Model/scripts/run_explainability_analysis.py"
```
Generated artifacts:
- `B3DB Based Model/figures/bbb_shap_global_summary.png`
- `B3DB Based Model/figures/pgp_shap_global_summary.png`
- `B3DB Based Model/figures/shared_properties_cross_task_comparison.png`
- `B3DB Based Model/figures/dual_barrier_quadrant_matrix.png`
- `B3DB Based Model/results/shared_properties_quantitative_comparison.csv`

---

## 4. Reproducing Leakage-Resistant Validation Metrics (95% Bootstrap CI)
To calculate 1,000 bootstrap resamplings on test sets across 10 evaluation metrics:
```bash
.\venv\Scripts\python.exe "B3DB Based Model/scripts/run_validation.py"
```
Generated artifacts:
- `B3DB Based Model/results/validation_metrics_summary.csv`
- `B3DB Based Model/figures/bbb_test_confusion_matrix.png`
- `B3DB Based Model/figures/pgp_test_confusion_matrix.png`

---

## 5. Reproducing Incremental Feature Ablation Study
To execute 5-fold stratified cross-validation across all feature subsets:
```bash
.\venv\Scripts\python.exe "B3DB Based Model/scripts/run_ablation_study.py"
```
Generated artifacts:
- `B3DB Based Model/results/ablation_study_results.csv`
- `B3DB Based Model/figures/feature_ablation_comparison.png`

---

## 6. Running Interactive Inference & Clinical Benchmarks
```bash
# 1. Run clinical reference benchmark suite:
.\venv\Scripts\python.exe "B3DB Based Model/inference.py" --test-benchmark

# 2. Predict on a single SMILES with full XAI report:
.\venv\Scripts\python.exe "B3DB Based Model/inference.py" --smiles "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O" --explain

# 3. Predict with chemical counterfactuals and export to HTML & CSV:
.\venv\Scripts\python.exe "B3DB Based Model/inference.py" --smiles "CC[C@@H](C(=O)N)N1CCCC1=O" --explain --counterfactual --export-html "B3DB Based Model/results/sample_report.html" --export-csv "B3DB Based Model/results/sample_report.csv"
```
