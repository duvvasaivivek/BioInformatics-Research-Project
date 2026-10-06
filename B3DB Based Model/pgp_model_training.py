import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def train_pgp_model():
    print("Loading P-glycoprotein dataset...")
    df = pd.read_csv("data/processed_pgp_data.csv")
    
    # Separate features and target
    X = df.drop(columns=['SMILES', 'Label'])
    y = df['Label']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # P-gp substrates are harder to predict, we will use robust hyper-parameters
    # These are generally good for small bioinformatics datasets
    print("\nTraining XGBoost Classifier on P-gp Data...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    print("\nEvaluating Model on Test Set...")
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save the model
    model_path = "models/pgp_xgboost_model.json"
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    


if __name__ == "__main__":
    train_pgp_model()
