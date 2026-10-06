import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings

# Suppress the harmless CPU/GPU device warning for clean terminal output
warnings.filterwarnings('ignore', category=UserWarning)

# Disable Optuna's default messy logger
optuna.logging.set_verbosity(optuna.logging.WARNING)

def print_best_callback(study, trial):
    print(f"Trial {trial.number:02d} finished | Current F1: {trial.value:.4f} | Best F1: {study.best_value:.4f}")

def load_data():
    print("Loading dataset...")
    df = pd.read_csv("../data/processed_b3db_data_advanced.csv")
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight):
    # Optuna will explore combinations of these parameters
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
        'device': 'cuda'
    }
    
    model = xgb.XGBClassifier(**param)
    
    # Train silently for Optuna
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predict and calculate F1 score (our target metric)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    
    return f1

def main():
    X_train, X_test, y_train, y_test = load_data()
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print("Starting Optuna Hyperparameter Tuning...")
    print("This will run 50 trials to find the absolute best settings.")
    
    # We want to MAXIMIZE the F1 score
    study = optuna.create_study(direction="maximize")
    
    start_time = time.time()
    # Run 50 trials (Should take a few minutes on your GPU)
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight), 
        n_trials=50,
        callbacks=[print_best_callback]
    )
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds.")
    print("\n=== Best Trial Parameters ===")
    print(f"Best F1-Score Found: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    print("\n--- Training Final Model with Best Parameters ---")
    best_params = study.best_params
    best_params['n_estimators'] = 2000 # Let it run longer since it's the final model
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['early_stopping_rounds'] = 50
    best_params['eval_metric'] = "logloss"
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50 # Print progress every 50 trees
    )
    
    print("\n=== Final Tuned Model Evaluation ===")
    y_pred = final_model.predict(X_test)
    
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    save_path = "../models/bbb_xgboost_optuna_best.json"
    final_model.save_model(save_path)
    print(f"\nSaved best tuned model to {save_path}")

if __name__ == "__main__":
    main()
