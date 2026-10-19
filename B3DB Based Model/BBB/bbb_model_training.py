"""
Leakage-Free BBB Permeability Model Training & Optuna Bayesian Tuning.
Uses a strict 70/15/15 Train/Validation/Test stratified split.
Fits continuous feature scalers strictly on training data.
Isolates test data completely during hyperparameter search and early stopping.
"""

import os
import sys
import time
import json
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score
)

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import BBB_DESCRIPTOR_NAMES

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_b3db_data_advanced.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_METADATA_PATH = os.path.join(MODELS_DIR, "bbb_scaler_metadata.json")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json")


def get_optimal_device() -> str:
    """Checks if CUDA is available for XGBoost hist tree method."""
    try:
        test_model = xgb.XGBClassifier(device='cuda', n_estimators=2, max_depth=2)
        test_model.fit(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))
        return 'cuda'
    except Exception:
        return 'cpu'


def print_best_callback(study, trial):
    print(f"Trial {trial.number:02d} finished | Current Val F1: {trial.value:.4f} | Best Val F1: {study.best_value:.4f}")


def load_and_split_data():
    """
    Loads raw processed data and creates a strictly leakage-free 70/15/15 split.
    Scales continuous descriptors strictly using statistics from X_train.
    """
    print("Loading B3DB preprocessed dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']

    # Step 1: Hold out 15% Test set (completely isolated from tuning)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Step 2: Split remaining 85% into Train (70% overall) and Validation (15% overall)
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    print(f"Dataset Split Sizes:")
    print(f" * Train:      {X_train.shape[0]} molecules ({len(y_train[y_train==1])} BBB+, {len(y_train[y_train==0])} BBB-)")
    print(f" * Validation: {X_val.shape[0]} molecules ({len(y_val[y_val==1])} BBB+, {len(y_val[y_val==0])} BBB-)")
    print(f" * Test Set:   {X_test.shape[0]} molecules ({len(y_test[y_test==1])} BBB+, {len(y_test[y_test==0])} BBB-)")

    # Step 3: Fit scaler strictly on X_train continuous descriptors
    print("Fitting StandardScaler strictly on training set...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[BBB_DESCRIPTOR_NAMES] = scaler.fit_transform(X_train[BBB_DESCRIPTOR_NAMES])
    X_val_scaled[BBB_DESCRIPTOR_NAMES] = scaler.transform(X_val[BBB_DESCRIPTOR_NAMES])
    X_test_scaled[BBB_DESCRIPTOR_NAMES] = scaler.transform(X_test[BBB_DESCRIPTOR_NAMES])

    # Save scaler metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata = {
        'means': scaler.mean_.tolist(),
        'stds': scaler.scale_.tolist(),
        'features': BBB_DESCRIPTOR_NAMES
    }
    with open(SCALER_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training-derived scaler metadata to {SCALER_METADATA_PATH}")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def objective(trial, X_train, X_val, y_train, y_val, scale_pos_weight, device):
    """Optuna objective function: tunes strictly on Train/Val, never touching Test."""
    param = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight,
        'early_stopping_rounds': 30,
        'eval_metric': "logloss",
        'random_state': 42,
        'tree_method': 'hist',
        'device': device
    }

    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    return f1_score(y_val, preds)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    device = get_optimal_device()
    print(f"Using acceleration device: {device.upper()}")

    print("\nStarting Optuna Hyperparameter Optimization (50 trials on Validation set)...")
    study = optuna.create_study(direction="maximize")

    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val, scale_pos_weight, device),
        n_trials=50,
        callbacks=[print_best_callback]
    )

    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds.")
    print("=== Best Trial Hyperparameters ===")
    print(f"Best Validation F1-Score: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n--- Training Final Model with Best Parameters ---")
    best_params = study.best_params
    best_params['n_estimators'] = 2000
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['early_stopping_rounds'] = 50
    best_params['eval_metric'] = "logloss"
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['device'] = device

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    print("\n" + "=" * 60)
    print("   LOCKED VAULT EVALUATION: EVALUATING HELD-OUT TEST SET   ")
    print("=" * 60)
    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_test_pred)
    bacc = balanced_accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    print(f"Held-Out Test Accuracy:          {acc:.4f}")
    print(f"Held-Out Test Balanced Accuracy: {bacc:.4f}")
    print(f"Held-Out Test Precision:         {prec:.4f}")
    print(f"Held-Out Test Recall:            {rec:.4f}")
    print(f"Held-Out Test F1-Score:          {f1:.4f}")
    print(f"Held-Out Test ROC-AUC:           {roc_auc:.4f}")

    final_model.save_model(MODEL_SAVE_PATH)
    print(f"\nSaved final tuned model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
