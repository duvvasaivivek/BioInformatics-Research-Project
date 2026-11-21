"""
Training Script for Stage 2: Targeted P-gp Efflux Gatekeeper (Novelty 2).

Trains the Stage 2 XGBoost model on the full Broccatelli benchmark (1,205 molecules).
Incorporates 2D descriptors, 3D ETKDGv3 stereochemical descriptors, and Morgan fingerprints.
Applies class-weight balancing and strictly isolates training scalers.
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
    PGP_DESCRIPTOR_NAMES,
    calculate_morgan_fingerprint,
    calculate_2d_descriptors,
    calculate_3d_descriptors,
)
from rdkit import Chem

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_stage2(seed: int = 42):
    print("=" * 80)
    print("NOVELTY 2: TRAINING STAGE 2 P-GP EFFLUX GATEKEEPER")
    print("=" * 80)

    raw_path = os.path.join(DATA_DIR, "pgp_broccatelli.tab")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Broccatelli dataset not found at {raw_path}")

    print(f"Loading raw Broccatelli dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path, sep="\t")

    print("Extracting 2048-bit Morgan FP, 2D descriptors, and 3D ETKDGv3 shape...")
    records = []
    fps = []
    for _, row in df_raw.iterrows():
        smi = str(row["Drug"])
        y_val = row["Y"]
        if pd.isna(y_val):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc_3d, _ = calculate_3d_descriptors(mol, seed=seed)
        if desc_3d is None:
            continue  # Exclude failed conformers
        y = int(y_val)
        fp = calculate_morgan_fingerprint(mol)
        desc_2d = calculate_2d_descriptors(mol)
        fps.append(fp)
        full_desc = {**desc_2d, **desc_3d}
        rec = {"y": y}
        for name in PGP_DESCRIPTOR_NAMES:
            rec[name] = full_desc[name]
        records.append(rec)

    df_feats = pd.DataFrame(records)
    X_fp = np.vstack(fps)
    X_desc = df_feats[PGP_DESCRIPTOR_NAMES].values
    y = df_feats["y"].values

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"Valid featurized molecules: {len(y)} (Efflux+: {n_pos}, Efflux-: {n_neg})")

    # 70/15/15 Stratified Split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.17647, stratify=y[train_idx], random_state=seed)

    print(f"Splits: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    # Fit scaler strictly on training set
    means = X_desc[train_idx].mean(axis=0).tolist()
    stds = X_desc[train_idx].std(axis=0).tolist()
    stds = [s if s > 0 else 1.0 for s in stds]

    scaler_path = os.path.join(MODELS_DIR, "pgp_scaler_metadata.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"features": PGP_DESCRIPTOR_NAMES, "means": means, "stds": stds}, f, indent=2)
    print(f"Saved scaler metadata to: {scaler_path}")

    # Scale continuous features
    X_desc_scaled = (X_desc - means) / stds
    X_all = np.hstack([X_fp, X_desc_scaled])

    cols = [f"FP_{i}" for i in range(X_fp.shape[1])] + PGP_DESCRIPTOR_NAMES
    X_train = pd.DataFrame(X_all[train_idx], columns=cols)
    y_train = y[train_idx]
    X_val = pd.DataFrame(X_all[val_idx], columns=cols)
    y_val = y[val_idx]
    X_test = pd.DataFrame(X_all[test_idx], columns=cols)
    y_test = y[test_idx]

    print("Training Stage 2 XGBoost classifier with class balancing...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
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

    print("\nStage 2 P-gp Test Performance:")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bacc:.4f}")
    print(f"  F1 Score          : {f1:.4f}")

    model_path = os.path.join(MODELS_DIR, "pgp_xgboost.json")
    model.save_model(model_path)
    print(f"Saved Stage 2 model to: {model_path}")
    return model, auc


if __name__ == "__main__":
    train_stage2()
