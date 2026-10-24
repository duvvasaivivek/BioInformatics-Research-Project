"""
Rigorous Scientific Validation Protocol for BBB and P-gp Models.
Performs:
1. Random Held-Out Test Set Validation (In-Distribution Generalization) with 95% Bootstrap CI.
2. Genuine Bemis-Murcko Scaffold Split Evaluation (Out-of-Distribution Chemotype Generalization).
3. Metric Suite: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity,
   F1-Score, ROC-AUC, PR-AUC, Matthews Correlation Coefficient (MCC), and Brier Score.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import BBB_DESCRIPTOR_NAMES, PGP_DESCRIPTOR_NAMES

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_metrics_with_ci(y_true, y_pred, y_prob, n_bootstraps=1000, seed=42) -> Dict[str, Any]:
    """Computes all classification metrics with 95% bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

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

    boot_records = {k: [] for k in point_metrics if k not in ["TP", "FP", "TN", "FN"]}
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        b_true = y_true[idx]
        b_pred = y_pred[idx]
        b_prob = y_prob[idx]

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


def generate_bemis_murcko_split(df: pd.DataFrame, smiles_col: str = 'SMILES', test_size: float = 0.20, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Partitions dataset by Bemis-Murcko scaffolds.
    Guarantees 0% scaffold overlap between train and test sets.
    """
    scaffold_map: Dict[str, List[int]] = {}
    for idx, smi in enumerate(df[smiles_col]):
        try:
            m = Chem.MolFromSmiles(str(smi))
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
        except Exception:
            scaff = ""
        scaffold_map.setdefault(scaff, []).append(idx)

    # Sort scaffold clusters by size
    clusters = sorted(scaffold_map.values(), key=len, reverse=True)
    target_test_size = int(len(df) * test_size)

    train_indices = []
    test_indices = []

    for cluster in clusters:
        if len(test_indices) + len(cluster) <= target_test_size:
            test_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    return train_indices, test_indices


def evaluate_bbb_model():
    print("\n--- Validating BBB Model (B3DB Benchmark, 7,807 molecules) ---")
    data_path = os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    # Exact held-out 15% test split matching bbb_model_training.py
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Scale continuous features using trained scaler metadata
    scaler_path = os.path.join(MODELS_DIR, "bbb_scaler_metadata.json")
    with open(scaler_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    means, stds = meta['means'], meta['stds']

    X_test_scaled = X_test.copy()
    for i, name in enumerate(BBB_DESCRIPTOR_NAMES):
        X_test_scaled[name] = (X_test_scaled[name] - means[i]) / (stds[i] if stds[i] != 0 else 1.0)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "bbb_xgboost.json"))

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results = calculate_metrics_with_ci(y_test, preds, probs)
    print("BBB Random Held-Out Test Set (95% Bootstrap CI):")
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

    # Bemis-Murcko Scaffold Evaluation for BBB
    print("\n--- Evaluating BBB Model on Bemis-Murcko Scaffold Split (Zero Chemotype Overlap) ---")
    tr_idx, te_idx = generate_bemis_murcko_split(df, smiles_col='SMILES', test_size=0.15, seed=42)
    X_tr_scaff, X_te_scaff = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_tr_scaff, y_te_scaff = y.iloc[tr_idx], y.iloc[te_idx]

    # Fit scaler strictly on scaffold train
    sc_scaff = StandardScaler()
    X_tr_scaff[BBB_DESCRIPTOR_NAMES] = sc_scaff.fit_transform(X_tr_scaff[BBB_DESCRIPTOR_NAMES])
    X_te_scaff[BBB_DESCRIPTOR_NAMES] = sc_scaff.transform(X_te_scaff[BBB_DESCRIPTOR_NAMES])

    scaff_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', tree_method='hist'
    )
    scaff_model.fit(X_tr_scaff, y_tr_scaff, verbose=False)
    scaff_preds = scaff_model.predict(X_te_scaff)
    scaff_probs = scaff_model.predict_proba(X_te_scaff)[:, 1]

    scaff_results = calculate_metrics_with_ci(y_te_scaff, scaff_preds, scaff_probs)
    print("BBB Bemis-Murcko Scaffold Generalization Results (95% Bootstrap CI):")
    for k, v in scaff_results.items():
        if isinstance(v, dict) and "display" in v:
            print(f" * {k:<20}: {v['display']}")

    return results, scaff_results


def evaluate_pgp_model():
    print("\n--- Validating P-gp Efflux Model (Broccatelli TDC, 1,218 molecules) ---")
    data_path = os.path.join(DATA_DIR, "processed_pgp_data.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale continuous features using trained scaler metadata
    scaler_path = os.path.join(MODELS_DIR, "pgp_scaler_metadata.json")
    with open(scaler_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    means, stds = meta['means'], meta['stds']

    X_test_scaled = X_test.copy()
    for i, name in enumerate(PGP_DESCRIPTOR_NAMES):
        X_test_scaled[name] = (X_test_scaled[name] - means[i]) / (stds[i] if stds[i] != 0 else 1.0)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "pgp_xgboost.json"))

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results = calculate_metrics_with_ci(y_test, preds, probs)
    print("P-gp Random Held-Out Test Set (95% Bootstrap CI):")
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

    # Bemis-Murcko Scaffold Evaluation for P-gp
    print("\n--- Evaluating P-gp Model on Bemis-Murcko Scaffold Split (Zero Chemotype Overlap) ---")
    tr_idx, te_idx = generate_bemis_murcko_split(df, smiles_col='SMILES', test_size=0.20, seed=42)
    X_tr_scaff, X_te_scaff = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_tr_scaff, y_te_scaff = y.iloc[tr_idx], y.iloc[te_idx]

    sc_scaff = StandardScaler()
    X_tr_scaff[PGP_DESCRIPTOR_NAMES] = sc_scaff.fit_transform(X_tr_scaff[PGP_DESCRIPTOR_NAMES])
    X_te_scaff[PGP_DESCRIPTOR_NAMES] = sc_scaff.transform(X_te_scaff[PGP_DESCRIPTOR_NAMES])

    scaff_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', tree_method='hist'
    )
    scaff_model.fit(X_tr_scaff, y_tr_scaff, verbose=False)
    scaff_preds = scaff_model.predict(X_te_scaff)
    scaff_probs = scaff_model.predict_proba(X_te_scaff)[:, 1]

    scaff_results = calculate_metrics_with_ci(y_te_scaff, scaff_preds, scaff_probs)
    print("P-gp Bemis-Murcko Scaffold Generalization Results (95% Bootstrap CI):")
    for k, v in scaff_results.items():
        if isinstance(v, dict) and "display" in v:
            print(f" * {k:<20}: {v['display']}")

    return results, scaff_results


def main():
    print("=" * 80)
    print("   LEAKAGE-RESISTANT SCIENTIFIC VALIDATION & SCAFFOLD BENCHMARK (95% CI)   ")
    print("=" * 80)

    bbb_rand, bbb_scaff = evaluate_bbb_model()
    pgp_rand, pgp_scaff = evaluate_pgp_model()

    summary_rows = []
    metrics_to_export = [
        "Accuracy", "Balanced_Accuracy", "Precision", "Recall_Sensitivity",
        "Specificity", "F1_Score", "ROC_AUC", "PR_AUC", "MCC", "Brier_Score"
    ]

    for m in metrics_to_export:
        summary_rows.append({
            "Metric": m,
            "BBB_Random_Estimate": bbb_rand[m]["estimate"],
            "BBB_Random_95CI": f"[{bbb_rand[m]['ci_lower']} - {bbb_rand[m]['ci_upper']}]",
            "BBB_Scaffold_Estimate": bbb_scaff[m]["estimate"],
            "BBB_Scaffold_95CI": f"[{bbb_scaff[m]['ci_lower']} - {bbb_scaff[m]['ci_upper']}]",
            "Pgp_Random_Estimate": pgp_rand[m]["estimate"],
            "Pgp_Random_95CI": f"[{pgp_rand[m]['ci_lower']} - {pgp_rand[m]['ci_upper']}]",
            "Pgp_Scaffold_Estimate": pgp_scaff[m]["estimate"],
            "Pgp_Scaffold_95CI": f"[{pgp_scaff[m]['ci_lower']} - {pgp_scaff[m]['ci_upper']}]",
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "validation_metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved comprehensive dual-split validation metrics table to {csv_path}")


if __name__ == "__main__":
    main()
