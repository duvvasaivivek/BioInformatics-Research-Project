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
HIDDEN_CHANNELS = 128

EPOCHS = 200

LEARNING_RATE = 0.001

DROPOUT = 0.20

WEIGHT_DECAY = 1e-4

PATIENCE = 25

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
# 7. STRATIFIED DATASET SPLIT
# ============================================================

indices = np.arange(
    len(graphs)
)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)


temp_labels = labels[temp_indices]


val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)


train_dataset = [
    graphs[i]
    for i in train_indices
]

val_dataset = [
    graphs[i]
    for i in val_indices
]

test_dataset = [
    graphs[i]
    for i in test_indices
]


print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(
    "Training   :",
    len(train_dataset)
)

print(
    "Validation :",
    len(val_dataset)
)

print(
    "Testing    :",
    len(test_dataset)
)


# ============================================================
# 8. TRAINING CLASS DISTRIBUTION
# ============================================================

train_labels = np.array(
    [
        int(graph.y.item())
        for graph in train_dataset
    ]
)

class_counts = np.bincount(
    train_labels,
    minlength=2
)

print("\nTraining class distribution:")

print(
    "Class 0:",
    class_counts[0]
)

print(
    "Class 1:",
    class_counts[1]
)


# ============================================================
# 9. CLASS WEIGHTS
# ============================================================

# Helps compensate for class imbalance.

total_train = len(train_labels)

class_weights = total_train / (
    2 * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("\nClass weights:")

print(
    "Class 0:",
    class_weights[0].item()
)

print(
    "Class 1:",
    class_weights[1].item()
)


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 11. EDGE-AWARE GNN MODEL
# ============================================================

class BBB_GNN(torch.nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # GINE LAYER 1
        # ----------------------------------------------------

        nn1 = torch.nn.Sequential(

            torch.nn.Linear(
                9,
                HIDDEN_CHANNELS
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv1 = GINEConv(
            nn1,
            edge_dim=7
        )


        # ----------------------------------------------------
        # GINE LAYER 2
        # ----------------------------------------------------

        nn2 = torch.nn.Sequential(

            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv2 = GINEConv(
            nn2,
            edge_dim=7
        )


        # ----------------------------------------------------
        # GINE LAYER 3
        # ----------------------------------------------------

        nn3 = torch.nn.Sequential(

            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv3 = GINEConv(
            nn3,
            edge_dim=7
        )


        # ----------------------------------------------------
        # BATCH NORMALIZATION
        # ----------------------------------------------------

        self.bn1 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )

        self.bn2 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )

        self.bn3 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.fc1 = torch.nn.Linear(
            HIDDEN_CHANNELS,
            32
        )

        self.fc2 = torch.nn.Linear(
            32,
            2
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch
    ):

        # ----------------------------------------------------
        # GINE 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn1(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        # ----------------------------------------------------
        # GINE 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn2(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        # ----------------------------------------------------
        # GINE 3
        # ----------------------------------------------------

        x = self.conv3(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn3(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        # ----------------------------------------------------
        # GLOBAL MEAN POOLING
        # ----------------------------------------------------

        x = global_mean_pool(
            x,
            batch
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        x = self.fc1(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )

        x = self.fc2(x)

        return x


# ============================================================
# 12. CREATE MODEL
# ============================================================

model = BBB_GNN().to(device)


print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 14. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=8,
    min_lr=1e-6
)


# ============================================================
# 15. TRAIN FUNCTION
# ============================================================

def train():

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []


    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )


        # ----------------------------------------------------
        # Weighted Loss
        # ----------------------------------------------------

        loss = F.cross_entropy(
            output,
            batch.y,
            weight=class_weights
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient Clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )


        optimizer.step()


        total_loss += loss.item()


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = output.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            batch.y.detach().cpu().numpy()
        )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        total_loss / len(train_loader),
        accuracy
    )


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []


    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)


            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            predictions = output.argmax(
                dim=1
            )


            all_labels.extend(
                batch.y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        auc = 0.0


    return (
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# 17. TRAINING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

print("Epochs          :", EPOCHS)
print("Batch size      :", BATCH_SIZE)
print("Hidden channels :", HIDDEN_CHANNELS)
print("Learning rate   :", LEARNING_RATE)
print("Dropout         :", DROPOUT)
print("Weight decay    :", WEIGHT_DECAY)
print("Patience        :", PATIENCE)
print("Device          :", device)


# ============================================================
# EARLY STOPPING VARIABLES
# ============================================================

best_val_f1 = -1.0

best_epoch = 0

counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_acc = train()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = evaluate(
        val_loader
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Print epoch
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val F1 {val_f1:.4f} | "
        f"Val AUC {val_auc:.4f} | "
        f"LR {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch

        counter = 0


        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "node_features": 9,

                "edge_features": 7,

                "hidden_channels":
                    HIDDEN_CHANNELS,

                "classes": 2,

                "best_val_f1":
                    best_val_f1,

                "best_epoch":
                    best_epoch
            },
            BEST_MODEL
        )


        print(
            f"  -> BEST MODEL SAVED "
            f"(Val F1 = {best_val_f1:.4f})"
        )


    else:

        counter += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if counter >= PATIENCE:

        print("\n" + "=" * 60)
        print("EARLY STOPPING")
        print("=" * 60)

        print(
            "No validation F1 improvement for",
            PATIENCE,
            "epochs."
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Best validation F1:",
            f"{best_val_f1:.4f}"
        )

        break


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "node_features": 9,

        "edge_features": 7,

        "hidden_channels":
            HIDDEN_CHANNELS,

        "classes": 2,

        "epochs_trained": epoch
    },
    FINAL_MODEL
)


# ============================================================
# 19. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)


checkpoint = torch.load(
    BEST_MODEL,
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

print(
    "Best epoch:",
    checkpoint["best_epoch"]
)

print(
    "Best validation F1:",
    f"{checkpoint['best_val_f1']:.4f}"
)


# ============================================================
# 20. FINAL TEST EVALUATION
# ============================================================

(
    test_acc,
    test_precision,
    test_recall,
    test_f1,
    test_auc
) = evaluate(
    test_loader
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

model.eval()

all_test_labels = []
all_test_predictions = []


with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)


        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )


        predictions = output.argmax(
            dim=1
        )


        all_test_labels.extend(
            batch.y.cpu().numpy()
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )


cm = confusion_matrix(
    all_test_labels,
    all_test_predictions
)


# ============================================================
# 22. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {test_acc:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 23. MODEL FILES
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    BEST_MODEL
)

print(
    FINAL_MODEL
)


# ============================================================
# 24. GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**2
    )


    print("\n" + "=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print(
        f"Allocated : {allocated:.2f} MB"
    )

    print(
        f"Reserved  : {reserved:.2f} MB"
    )


# ============================================================
# 25. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)