"""
Leakage-Free P-glycoprotein (P-gp / ABCB1) Efflux Model Training Pipeline.
Fits continuous scalers strictly on the training partition.
Evaluates on held-out stratified test set and exports scaler metadata.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore', category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import PGP_DESCRIPTOR_NAMES

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_pgp_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_METADATA_PATH = os.path.join(MODELS_DIR, "pgp_scaler_metadata.json")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "pgp_xgboost_model.json")


def train_pgp_model():
    print("Loading P-glycoprotein dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {dict(y.value_counts())}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Fit scaler strictly on X_train continuous descriptors
    print("Fitting StandardScaler strictly on training partition...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[PGP_DESCRIPTOR_NAMES] = scaler.fit_transform(X_train[PGP_DESCRIPTOR_NAMES])
    X_test_scaled[PGP_DESCRIPTOR_NAMES] = scaler.transform(X_test[PGP_DESCRIPTOR_NAMES])

    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata = {
        'means': scaler.mean_.tolist(),
        'stds': scaler.scale_.tolist(),
        'features': PGP_DESCRIPTOR_NAMES
    }
    with open(SCALER_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training-derived scaler metadata to {SCALER_METADATA_PATH}")

    # Compute scale_pos_weight
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    print("\nTraining XGBoost Classifier on P-gp Data...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=50
    )

    print("\n=== Evaluating Model on Held-Out Test Set ===")
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Test Accuracy:          {acc:.4f}")
    print(f"Test Balanced Accuracy: {bacc:.4f}")
    print(f"Test Precision:         {prec:.4f}")
    print(f"Test Recall:            {rec:.4f}")
    print(f"Test F1-Score:          {f1:.4f}")
    print(f"Test ROC-AUC:           {auc:.4f}")

    model.save_model(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pgp_model()
