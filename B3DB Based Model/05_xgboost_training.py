import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def main():
    print("Loading data from processed_b3db_data_advanced.csv...")
    df = pd.read_csv("processed_b3db_data_advanced.csv")
    
    # Drop SMILES string and separate the target Label
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']
    
    print(f"Dataset shape: {X.shape}")
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight to handle any slight class imbalances
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    print("Initializing XGBoost Classifier...")
    # XGBoost is highly optimized for this kind of tabular fingerprint data
    model = xgb.XGBClassifier(
        n_estimators=1000,            # Max number of trees
        learning_rate=0.05,           # Slower learning rate to prevent overfitting
        max_depth=6,                  # Depth of each tree
        subsample=0.8,                # Use 80% of data per tree
        colsample_bytree=0.8,         # Use 80% of features per tree
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,     # Stop if test loss doesn't improve for 50 rounds
        eval_metric="logloss",
        random_state=42,
        tree_method='hist',           # Highly optimized histogram algorithm
        device='cuda'                 # Use GPU if available
    )
    
    print("Starting Training (Evaluating every 10 trees)...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    
    # Predictions
    print("Evaluating Best Model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- XGBoost Test Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    save_path = "bbb_xgboost_model.json"
    model.save_model(save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
