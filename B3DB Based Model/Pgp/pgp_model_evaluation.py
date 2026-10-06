import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading dataset and model...")
df = pd.read_csv("../data/processed_pgp_data.csv")
X = df.drop(columns=['SMILES', 'Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier()
model.load_model("../models/pgp_xgboost_model.json")

print("Making predictions...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

out_dir = r"C:\Users\saivi\.gemini\antigravity-ide\brain\619012dd-ffbd-469e-bc73-d0122efef94e"
os.makedirs(out_dir, exist_ok=True)

# 1. Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Substrate', 'Substrate'],
            yticklabels=['Non-Substrate', 'Substrate'])
plt.title('P-gp 3D Model Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# 2. ROC Curve
plt.figure(figsize=(6,5))
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300)
plt.close()

print("Saved plots successfully!")
