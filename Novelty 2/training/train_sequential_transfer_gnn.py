"""
Sequential Transfer Learning Training Script (Novelty 2).

Extracts the pre-trained BBB GINEConv encoder trunk and fine-tunes
a downstream classification head on the Broccatelli P-gp dataset.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.sequential_transfer_gnn import (
    SequentialTransferGNN,
    load_pretrained_bbb_encoder,
)

# Reuse featurizer from Novelty 1
NOVELTY1_DIR = os.path.join(os.path.dirname(BASE_DIR), "Novelty 1")
if NOVELTY1_DIR not in sys.path:
    sys.path.insert(0, NOVELTY1_DIR)

from gnn.featurizer import smiles_to_data

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BBB_CKPT = os.path.join(MODELS_DIR, "gnn_single_bbb_random_encoded.pt")
SAVE_CKPT = os.path.join(MODELS_DIR, "gnn_sequential_transfer_pgp.pt")


def train_transfer_gnn(epochs: int = 40, lr: float = 0.001, seed: int = 42):
    print("=" * 80)
    print("NOVELTY 2: TRAINING SEQUENTIAL TRANSFER LEARNING GNN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pre-trained BBB Trunk
    print(f"Loading pre-trained BBB encoder trunk from: {BBB_CKPT}")
    trunk = load_pretrained_bbb_encoder(BBB_CKPT, device=device)

    # 2. Build Sequential Transfer Model (Frozen trunk initially)
    model = SequentialTransferGNN(trunk=trunk, readout_dim=32, num_classes=2, freeze_trunk=True).to(device)

    # 3. Load Broccatelli P-gp Dataset
    raw_path = os.path.join(DATA_DIR, "pgp_broccatelli.tab")
    df_raw = pd.read_csv(raw_path, sep="\t")

    print(f"Featurizing {len(df_raw)} P-gp molecules into molecular graphs...")
    graphs = []
    labels = []
    for _, row in df_raw.iterrows():
        smi = str(row["Drug"])
        y = row["Y"]
        if pd.isna(y):
            continue
        g = smiles_to_data(smi, mode="encoded")
        if g is None:
            continue
        g.y = torch.tensor([int(y)], dtype=torch.long)
        graphs.append(g)
        labels.append(int(y))

    labels = np.array(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([float(n_neg) / float(n_pos)], device=device)
    print(f"Graph dataset assembled: {len(graphs)} graphs (P-gp+: {n_pos}, P-gp-: {n_neg})")

    # 4. Stratified Split 70/15/15
    idx = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.17647, stratify=labels[train_idx], random_state=seed)

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=32, shuffle=False)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=32, shuffle=False)

    # 5. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = 0.0
    best_state = None

    print("\nPhase A: Training classification head with frozen BBB trunk...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=torch.tensor([1.0, pos_weight.item()], device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_probs, val_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_y.extend(batch.y.cpu().numpy())

        val_auc = roc_auc_score(val_y, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val ROC-AUC: {val_auc:.4f}")

    # 6. Evaluation on Test Set
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    test_probs, test_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_y.extend(batch.y.cpu().numpy())

    test_auc = roc_auc_score(test_y, test_probs)
    test_preds = (np.array(test_probs) >= 0.5).astype(int)
    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds)

    print("\n" + "=" * 80)
    print("SEQUENTIAL TRANSFER GNN TEST EVALUATION")
    print("=" * 80)
    print(f"Transferred Pre-trained Trunk: BBB (B3DB)")
    print(f"Downstream Adapted Task       : P-gp (Broccatelli)")
    print(f"Test ROC-AUC                  : {test_auc:.4f}")
    print(f"Test Accuracy                 : {test_acc:.4f}")
    print(f"Test F1 Score                 : {test_f1:.4f}")
    print("=" * 80)

    torch.save({
        "model_state_dict": best_state,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }, SAVE_CKPT)
    print(f"Saved fine-tuned sequential transfer model to: {SAVE_CKPT}")
    return test_auc


if __name__ == "__main__":
    train_transfer_gnn()
