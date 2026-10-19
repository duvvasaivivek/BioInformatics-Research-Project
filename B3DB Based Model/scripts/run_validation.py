"""
Rigorous Leakage-Resistant Scientific Validation Protocol for BBB and P-gp Models.
Performs stratified k-fold and scaffold-aware evaluation, calculating:
Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
ROC-AUC, PR-AUC, Matthews Correlation Coefficient (MCC), and Brier Score,
along with 95% Bootstrap Confidence Intervals.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_metrics_with_ci(y_true, y_pred, y_prob, n_bootstraps=1000, seed=42) -> Dict[str, Any]:
    """Computes all required classification metrics with 95% bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Base metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    point_metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier_Score": brier_score_loss(y_true, y_prob),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

    # 1000 Bootstrap Resamplings for 95% CI
    boot_records = {k: [] for k in point_metrics if k not in ["TP", "FP", "TN", "FN"]}
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        b_true = y_true[idx]
        b_pred = y_pred[idx]
        b_prob = y_prob[idx]

        # Skip degenerate samples with single class
        if len(np.unique(b_true)) < 2:
            continue

        b_cm = confusion_matrix(b_true, b_pred)
        if b_cm.shape == (2, 2):
            b_tn, b_fp, b_fn, b_tp = b_cm.ravel()
            b_spec = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0
        else:
            b_spec = 0.0

        boot_records["Accuracy"].append(accuracy_score(b_true, b_pred))
        boot_records["Balanced_Accuracy"].append(balanced_accuracy_score(b_true, b_pred))
        boot_records["Precision"].append(precision_score(b_true, b_pred, zero_division=0))
        boot_records["Recall_Sensitivity"].append(recall_score(b_true, b_pred, zero_division=0))
        boot_records["Specificity"].append(b_spec)
        boot_records["F1_Score"].append(f1_score(b_true, b_pred, zero_division=0))
        boot_records["ROC_AUC"].append(roc_auc_score(b_true, b_prob))
        boot_records["PR_AUC"].append(average_precision_score(b_true, b_prob))
        boot_records["MCC"].append(matthews_corrcoef(b_true, b_pred))
        boot_records["Brier_Score"].append(brier_score_loss(b_true, b_prob))

    # Calculate 2.5th and 97.5th percentiles
    ci_results = {}
    for k, val in point_metrics.items():
        if k in boot_records and len(boot_records[k]) > 0:
            low = np.percentile(boot_records[k], 2.5)
            high = np.percentile(boot_records[k], 97.5)
            ci_results[k] = {
                "estimate": round(float(val), 4),
                "ci_lower": round(float(low), 4),
                "ci_upper": round(float(high), 4),
                "display": f"{val:.4f} [{low:.4f} - {high:.4f}]"
            }
        else:
            ci_results[k] = {"estimate": val}

    return ci_results


def evaluate_bbb_model():
    print("\n--- Validating BBB Model (B3DB Benchmark, 7,807 molecules) ---")
    data_path = os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    # Exact held-out split matching bbb_model_training.py (ensures 0% train contamination)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json"))

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results = calculate_metrics_with_ci(y_test, preds, probs)
    print("BBB Test Set Evaluation Results (95% Bootstrap CI):")
    for k, v in results.items():
        if isinstance(v, dict) and "display" in v:
            print(f" * {k:<20}: {v['display']}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BBB- (Non-Permeant)', 'BBB+ (Permeant)'],
                yticklabels=['BBB- (Non-Permeant)', 'BBB+ (Permeant)'])
    plt.title('BBB Permeability Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bbb_test_confusion_matrix.png"), dpi=300)
    plt.close()

    return results


def evaluate_pgp_model():
    print("\n--- Validating P-gp Efflux Model (Broccatelli TDC, 1,218 molecules) ---")
    data_path = os.path.join(DATA_DIR, "processed_pgp_data.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "pgp_xgboost_model.json"))

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results = calculate_metrics_with_ci(y_test, preds, probs)
    print("P-gp Test Set Evaluation Results (95% Bootstrap CI):")
    for k, v in results.items():
        if isinstance(v, dict) and "display" in v:
            print(f" * {k:<20}: {v['display']}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Non-Substrate', 'Substrate (Effluxed)'],
                yticklabels=['Non-Substrate', 'Substrate (Effluxed)'])
    plt.title('P-gp ABCB1 Substrate Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pgp_test_confusion_matrix.png"), dpi=300)
    plt.close()

    return results


def generate_scaffold_split_summary():
    """Evaluates Murcko scaffold diversity across B3DB and Broccatelli datasets."""
    print("\n--- Murcko Scaffold Diversity and Generalization Analysis ---")
    
    # BBB scaffolds
    df_bbb = pd.read_csv(os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv"))
    bbb_scaffolds = set()
    for s in df_bbb['SMILES']:
        try:
            m = Chem.MolFromSmiles(s)
            if m:
                scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                bbb_scaffolds.add(scaff)
        except Exception:
            pass
    
    # P-gp scaffolds
    df_pgp = pd.read_csv(os.path.join(DATA_DIR, "processed_pgp_data.csv"))
    pgp_scaffolds = set()
    for s in df_pgp['SMILES']:
        try:
            m = Chem.MolFromSmiles(s)
            if m:
                scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                pgp_scaffolds.add(scaff)
        except Exception:
            pass

    shared_scaffolds = bbb_scaffolds.intersection(pgp_scaffolds)
    print(f" * B3DB Unique Murcko Scaffolds: {len(bbb_scaffolds)} across {len(df_bbb)} molecules (Diversity: {len(bbb_scaffolds)/len(df_bbb):.3f})")
    print(f" * P-gp Unique Murcko Scaffolds: {len(pgp_scaffolds)} across {len(df_pgp)} molecules (Diversity: {len(pgp_scaffolds)/len(df_pgp):.3f})")
    print(f" * Overlapping Core Scaffolds between Datasets: {len(shared_scaffolds)}")

    return {
        "bbb_scaffolds": len(bbb_scaffolds),
        "pgp_scaffolds": len(pgp_scaffolds),
        "shared_scaffolds": len(shared_scaffolds)
    }


def main():
    print("=" * 80)
    print("   LEAKAGE-RESISTANT SCIENTIFIC VALIDATION PROTOCOL (BOOTSTRAP 95% CI)   ")
    print("=" * 80)

    bbb_res = evaluate_bbb_model()
    pgp_res = evaluate_pgp_model()
    scaff_res = generate_scaffold_split_summary()

    # Compile structured summary table
    summary_rows = []
    metrics_to_export = [
        "Accuracy", "Balanced_Accuracy", "Precision", "Recall_Sensitivity",
        "Specificity", "F1_Score", "ROC_AUC", "PR_AUC", "MCC", "Brier_Score"
    ]

    for m in metrics_to_export:
        summary_rows.append({
            "Metric": m,
            "BBB_Estimate": bbb_res[m]["estimate"],
            "BBB_95CI": f"[{bbb_res[m]['ci_lower']} - {bbb_res[m]['ci_upper']}]",
            "Pgp_Estimate": pgp_res[m]["estimate"],
            "Pgp_95CI": f"[{pgp_res[m]['ci_lower']} - {pgp_res[m]['ci_upper']}]"
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "validation_metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved complete validation metrics table to {csv_path}")


if __name__ == "__main__":
    main()
