"""
Ablation Study Script for BBB Permeability and P-gp Efflux Classifiers.
Rigorously evaluates the incremental benefit of each feature subset under identical
stratified 5-fold cross-validation protocol:
- BBB: 2D Descriptors Only vs Morgan FP Only vs Full Hybrid (Morgan + 2D)
- P-gp: 2D Descriptors Only vs Morgan + 2D vs 3D Shape Only vs Full 2D+Morgan+3D Model
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_feature_subset_cv(X: pd.DataFrame, y: pd.Series, feature_cols: List[str], n_splits: int = 5) -> Dict[str, float]:
    """Runs stratified 5-fold cross-validation on a specific feature subset."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, baccs, f1s, aucs, praucs, mccs = [], [], [], [], [], []

    X_sub = X[feature_cols].copy()

    for train_idx, val_idx in skf.split(X_sub, y):
        X_tr, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'
        )

        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, preds))
        baccs.append(balanced_accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, zero_division=0))
        aucs.append(roc_auc_score(y_val, probs))
        praucs.append(average_precision_score(y_val, probs))
        mccs.append(matthews_corrcoef(y_val, preds))

    return {
        "Accuracy_mean": np.mean(accs),
        "Accuracy_std": np.std(accs),
        "Balanced_Accuracy_mean": np.mean(baccs),
        "F1_Score_mean": np.mean(f1s),
        "F1_Score_std": np.std(f1s),
        "ROC_AUC_mean": np.mean(aucs),
        "ROC_AUC_std": np.std(aucs),
        "PR_AUC_mean": np.mean(praucs),
        "MCC_mean": np.mean(mccs),
    }


def run_ablation():
    print("=" * 80)
    print("   INCREMENTAL FEATURE ABLATION STUDY (5-FOLD STRATIFIED CV)   ")
    print("=" * 80)

    # 1. P-gp Feature Ablation
    print("\n--- Running P-gp Efflux Feature Set Ablations ---")
    df_pgp = pd.read_csv(os.path.join(DATA_DIR, "processed_pgp_data.csv"))
    X_pgp = df_pgp.drop(columns=['SMILES', 'Label'])
    y_pgp = df_pgp['Label']

    desc_2d = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
    desc_3d = ['RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']
    fp_cols = [c for c in X_pgp.columns if c.startswith('FP_')]

    pgp_configs = [
        ("1. 2D Descriptors Only", desc_2d),
        ("2. 3D Shape Only", desc_3d),
        ("3. 2D Descriptors + 3D Shape", desc_2d + desc_3d),
        ("4. Morgan FP + 2D Descriptors", fp_cols + desc_2d),
        ("5. Full Model (Morgan + 2D + 3D Shape)", fp_cols + desc_2d + desc_3d)
    ]

    pgp_records = []
    for name, cols in pgp_configs:
        print(f" * Evaluating config: {name} ({len(cols)} features)...")
        res = evaluate_feature_subset_cv(X_pgp, y_pgp, cols)
        print(f"   -> ROC-AUC: {res['ROC_AUC_mean']:.4f} ± {res['ROC_AUC_std']:.4f} | F1: {res['F1_Score_mean']:.4f} | Acc: {res['Accuracy_mean']:.4f}")
        pgp_records.append({
            "Task": "P-gp Substrate",
            "Configuration": name,
            "Num_Features": len(cols),
            "Accuracy": f"{res['Accuracy_mean']:.4f} ± {res['Accuracy_std']:.4f}",
            "F1_Score": f"{res['F1_Score_mean']:.4f} ± {res['F1_Score_std']:.4f}",
            "ROC_AUC": f"{res['ROC_AUC_mean']:.4f} ± {res['ROC_AUC_std']:.4f}",
            "PR_AUC": f"{res['PR_AUC_mean']:.4f}",
            "MCC": f"{res['MCC_mean']:.4f}",
            "Raw_AUC": res['ROC_AUC_mean']
        })

    # 2. BBB Feature Ablation
    print("\n--- Running BBB Permeability Feature Set Ablations ---")
    df_bbb = pd.read_csv(os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv"))
    X_bbb = df_bbb.drop(columns=['SMILES', 'Label'])
    y_bbb = df_bbb['Label']

    bbb_fp_cols = [c for c in X_bbb.columns if c.startswith('FP_')]

    bbb_configs = [
        ("1. 2D Descriptors Only", desc_2d),
        ("2. Morgan Fingerprints Only", bbb_fp_cols),
        ("3. Full Hybrid (Morgan + 2D Descriptors)", bbb_fp_cols + desc_2d)
    ]

    bbb_records = []
    for name, cols in bbb_configs:
        print(f" * Evaluating config: {name} ({len(cols)} features)...")
        res = evaluate_feature_subset_cv(X_bbb, y_bbb, cols)
        print(f"   -> ROC-AUC: {res['ROC_AUC_mean']:.4f} ± {res['ROC_AUC_std']:.4f} | F1: {res['F1_Score_mean']:.4f} | Acc: {res['Accuracy_mean']:.4f}")
        bbb_records.append({
            "Task": "BBB Permeability",
            "Configuration": name,
            "Num_Features": len(cols),
            "Accuracy": f"{res['Accuracy_mean']:.4f} ± {res['Accuracy_std']:.4f}",
            "F1_Score": f"{res['F1_Score_mean']:.4f} ± {res['F1_Score_std']:.4f}",
            "ROC_AUC": f"{res['ROC_AUC_mean']:.4f} ± {res['ROC_AUC_std']:.4f}",
            "PR_AUC": f"{res['PR_AUC_mean']:.4f}",
            "MCC": f"{res['MCC_mean']:.4f}",
            "Raw_AUC": res['ROC_AUC_mean']
        })

    # Save to CSV
    all_ablation_df = pd.DataFrame(pgp_records + bbb_records)
    csv_path = os.path.join(RESULTS_DIR, "ablation_study_results.csv")
    all_ablation_df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation study results to {csv_path}")

    # Plot Ablation Comparison Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # P-gp chart
    p_names = [r["Configuration"].split(". ")[1] for r in pgp_records]
    p_aucs = [r["Raw_AUC"] for r in pgp_records]
    ax1.barh(p_names, p_aucs, color='#e53e3e', edgecolor='#742a2a', height=0.55)
    ax1.set_xlim(0.65, 1.0)
    ax1.set_xlabel('5-Fold CV Mean ROC-AUC')
    ax1.set_title('P-glycoprotein (P-gp) Feature Set Ablation')
    for i, v in enumerate(p_aucs):
        ax1.text(v + 0.005, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=9)

    # BBB chart
    b_names = [r["Configuration"].split(". ")[1] for r in bbb_records]
    b_aucs = [r["Raw_AUC"] for r in bbb_records]
    ax2.barh(b_names, b_aucs, color='#3182ce', edgecolor='#1a365d', height=0.55)
    ax2.set_xlim(0.80, 1.0)
    ax2.set_xlabel('5-Fold CV Mean ROC-AUC')
    ax2.set_title('BBB Permeability Feature Set Ablation')
    for i, v in enumerate(b_aucs):
        ax2.text(v + 0.005, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "feature_ablation_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved ablation comparison figure to {fig_path}")


if __name__ == "__main__":
    run_ablation()
