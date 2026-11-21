"""
Training Script for Stage 1: High-Throughput BBB Permeation Sieve (Novelty 2).

Trains the Stage 1 XGBoost model on the full B3DB benchmark (7,805 unique molecules).
Strictly separates training scaler statistics from validation and test sets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import (
    BBB_DESCRIPTOR_NAMES,
    calculate_morgan_fingerprint,
    calculate_2d_descriptors,
)
from rdkit import Chem

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_stage1(seed: int = 42):
    print("=" * 80)
    print("NOVELTY 2: TRAINING STAGE 1 BBB PERMEABILITY SIEVE")
    print("=" * 80)

    raw_path = os.path.join(DATA_DIR, "B3DB_classification.tsv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"B3DB dataset not found at {raw_path}")

    print(f"Loading raw B3DB dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path, sep="\t")

    # Filter invalid and parse
    print("Extracting Morgan fingerprints (ECFP4) and 2D descriptors...")
    records = []
    fps = []
    for _, row in df_raw.iterrows():
        smi = str(row["SMILES"])
        label_str = row["BBB+"]
        if pd.isna(label_str) or label_str not in ["BBB+", "BBB-"]:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        y = 1 if label_str == "BBB+" else 0
        fp = calculate_morgan_fingerprint(mol)
        desc = calculate_2d_descriptors(mol)
        fps.append(fp)
        rec = {"y": y}
        for name in BBB_DESCRIPTOR_NAMES:
            rec[name] = desc[name]
        records.append(rec)

    df_feats = pd.DataFrame(records)
    X_fp = np.vstack(fps)
    X_desc = df_feats[BBB_DESCRIPTOR_NAMES].values
    y = df_feats["y"].values

    print(f"Valid featurized molecules: {len(y)} (BBB+: {sum(y)}, BBB-: {len(y) - sum(y)})")

    # 70/15/15 Stratified Split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.17647, stratify=y[train_idx], random_state=seed)

    print(f"Splits: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    # Fit scaler strictly on training set
    means = X_desc[train_idx].mean(axis=0).tolist()
    stds = X_desc[train_idx].std(axis=0).tolist()
    stds = [s if s > 0 else 1.0 for s in stds]

    # Save scaler metadata
    scaler_path = os.path.join(MODELS_DIR, "bbb_scaler_metadata.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"features": BBB_DESCRIPTOR_NAMES, "means": means, "stds": stds}, f, indent=2)
    print(f"Saved scaler metadata to: {scaler_path}")

    # Scale continuous features
    X_desc_scaled = (X_desc - means) / stds
    X_all = np.hstack([X_fp, X_desc_scaled])

    cols = [f"FP_{i}" for i in range(X_fp.shape[1])] + BBB_DESCRIPTOR_NAMES
    X_train = pd.DataFrame(X_all[train_idx], columns=cols)
    y_train = y[train_idx]
    X_val = pd.DataFrame(X_all[val_idx], columns=cols)
    y_val = y[val_idx]
    X_test = pd.DataFrame(X_all[test_idx], columns=cols)
    y_test = y[test_idx]

    # Train XGBoost
    print("Training XGBoost classifier for Stage 1...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        early_stopping_rounds=30
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print("\nStage 1 BBB Test Performance:")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bacc:.4f}")
    print(f"  F1 Score          : {f1:.4f}")

    model_path = os.path.join(MODELS_DIR, "bbb_xgboost.json")
    model.save_model(model_path)
    print(f"Saved Stage 1 model to: {model_path}")
    return model, auc


if __name__ == "__main__":
    train_stage1()
