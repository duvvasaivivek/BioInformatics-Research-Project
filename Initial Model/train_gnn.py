# ============================================================
# BBB GNN TRAINING
# Edge-Aware GINE + BatchNorm + Dropout
# CUDA + Scheduler + Early Stopping
# ============================================================

import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRAPH_FOLDER = "graphs"

BATCH_SIZE = 32
HIDDEN_CHANNELS = 256

EPOCHS = 200

LEARNING_RATE = 0.0005

DROPOUT = 0.30

WEIGHT_DECAY = 5e-4

PATIENCE = 20
NUM_FOLDS = 10

SEED = 42

BEST_MODEL = "best_bbb_gnn_model.pt"
FINAL_MODEL = "bbb_gnn_model.pt"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. CUDA CHECK
# ============================================================

print("=" * 60)
print("DEVICE CHECK")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "GPU            :",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("GPU            : Not available")
    print("Training will use CPU.")

print("Device         :", device)


# ============================================================
# 4. CHECK GRAPH FOLDER
# ============================================================

if not os.path.exists(GRAPH_FOLDER):

    raise FileNotFoundError(
        f"\nGraph folder not found: {GRAPH_FOLDER}\n"
        f"Make sure the 'graphs' folder exists."
    )


# ============================================================
# 5. LOAD GRAPH DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING GNN DATASET")
print("=" * 60)

files = sorted(
    [
        f
        for f in os.listdir(GRAPH_FOLDER)
        if f.endswith(".pt")
    ]
)

print("Graph files found:", len(files))

if len(files) == 0:

    raise RuntimeError(
        "No .pt graph files found inside the graphs folder."
    )


graphs = []

invalid_graphs = 0


for file in files:

    path = os.path.join(
        GRAPH_FOLDER,
        file
    )

    try:

        graph = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        
        # Only use Adenot dataset
        if graph.get("source", "") == "Li":
            continue

        # ----------------------------------------------------
        # Required graph components
        # ----------------------------------------------------

        x = graph["x"].float()

        edge_index = graph["edge_index"].long()

        edge_attr = graph["edge_attr"].float()

        y = graph["y"]

        # Make sure y is a single integer class
        if torch.is_tensor(y):

            y = y.reshape(-1)[0].long()

        else:

            y = torch.tensor(
                int(y),
                dtype=torch.long
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        # ----------------------------------------------------
        # Optional metadata
        # ----------------------------------------------------

        data.smiles = graph.get(
            "smiles",
            ""
        )

        data.name = graph.get(
            "name",
            ""
        )

        data.source = graph.get(
            "source",
            ""
        )

        graphs.append(data)

    except Exception as e:

        invalid_graphs += 1

        print(
            f"Skipping {file}: {e}"
        )


print("\nValid graphs:", len(graphs))
print("Invalid graphs:", invalid_graphs)

if len(graphs) == 0:

    raise RuntimeError(
        "No valid graphs could be loaded."
    )


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

labels = np.array(
    [
        int(graph.y.item())
        for graph in graphs
    ]
)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Total graphs:", len(graphs))

print("\nClass distribution:")

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    counts
):

    print(
        f"Class {label}: {count}"
    )


# ============================================================

# ============================================================
# 7. CROSS VALIDATION SETUP
# ============================================================

print("\n" + "=" * 60)
print("STARTING 10-FOLD CROSS VALIDATION")
print("=" * 60)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

fold_metrics = []

class BBB_GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # GINE LAYER 1
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(9, HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        )
        self.conv1 = GINEConv(nn1, edge_dim=7)

        # GINE LAYER 2
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        )
        self.conv2 = GINEConv(nn2, edge_dim=7)

        # GINE LAYER 3
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        )
        self.conv3 = GINEConv(nn3, edge_dim=7)

        # BATCH NORMALIZATION
        self.bn1 = torch.nn.BatchNorm1d(HIDDEN_CHANNELS)
        self.bn2 = torch.nn.BatchNorm1d(HIDDEN_CHANNELS)
        self.bn3 = torch.nn.BatchNorm1d(HIDDEN_CHANNELS)

        # CLASSIFIER
        self.fc1 = torch.nn.Linear(HIDDEN_CHANNELS, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GINE 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        # GINE 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        # GINE 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        # GLOBAL MEAN POOLING
        x = global_mean_pool(x, batch)

        # CLASSIFIER
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.fc2(x)
        return x

def train_epoch(model, loader, optimizer, class_weights):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(output, batch.y, weight=class_weights)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions = output.argmax(dim=1)
        
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(batch.y.detach().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return total_loss / len(loader), accuracy

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            all_labels.extend(batch.y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0

    return accuracy, precision, recall, f1, auc

for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print("\n" + "=" * 60)
    print(f"FOLD {fold + 1}/{NUM_FOLDS}")
    print("=" * 60)

    # --------------------------------------------------------
    # Fold Dataset Splits
    # --------------------------------------------------------
    # Further split train_val into train (90%) and val (10%)
    train_val_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=0.1, 
        random_state=SEED, 
        stratify=train_val_labels
    )

    train_dataset = [graphs[i] for i in train_idx]
    val_dataset = [graphs[i] for i in val_idx]
    test_dataset = [graphs[i] for i in test_idx]

    train_labels_array = np.array([int(g.y.item()) for g in train_dataset])
    class_counts = np.bincount(train_labels_array, minlength=2)
    
    # Class weights for this fold
    total_train = len(train_labels_array)
    class_weights_np = total_train / (2 * class_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------------
    # Model, Optimizer, Scheduler
    # --------------------------------------------------------
    model = BBB_GNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    counter = 0

    best_model_path = f"best_bbb_gnn_model_fold_{fold+1}.pt"

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, class_weights)
        val_acc, val_precision, val_recall, val_f1, val_auc = evaluate(model, val_loader)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "node_features": 9,
                "edge_features": 7,
                "hidden_channels": HIDDEN_CHANNELS,
                "classes": 2,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch
            }, best_model_path)
        else:
            counter += 1

        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
            break

    # --------------------------------------------------------
    # Fold Evaluation
    # --------------------------------------------------------
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_acc, test_precision, test_recall, test_f1, test_auc = evaluate(model, test_loader)
    fold_metrics.append({
        'acc': test_acc, 'precision': test_precision, 
        'recall': test_recall, 'f1': test_f1, 'auc': test_auc
    })

    print(f"Fold {fold+1} Test Results -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")


# ============================================================
# 8. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("10-FOLD CROSS VALIDATION RESULTS")
print("=" * 60)

accs = [m['acc'] for m in fold_metrics]
precs = [m['precision'] for m in fold_metrics]
recs = [m['recall'] for m in fold_metrics]
f1s = [m['f1'] for m in fold_metrics]
aucs = [m['auc'] for m in fold_metrics]

print(f"Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Precision : {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
print(f"Recall    : {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
print(f"F1 Score  : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"ROC-AUC   : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

if torch.cuda.is_available():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print("\nGPU MEMORY -> Allocated: {:.2f} MB | Reserved: {:.2f} MB".format(allocated, reserved))

print("\nTRAINING COMPLETE")
