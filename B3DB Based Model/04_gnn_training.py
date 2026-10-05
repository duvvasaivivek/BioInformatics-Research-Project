import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import train_test_split

# --- Metrics Function ---
def calculate_metrics_gpu(y_pred_logits, y_true):
    probs = torch.sigmoid(y_pred_logits)
    preds = (probs > 0.5).float()
    
    tp = (preds * y_true).sum()
    fp = (preds * (1 - y_true)).sum()
    tn = ((1 - preds) * (1 - y_true)).sum()
    fn = ((1 - preds) * y_true).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return accuracy.item(), precision.item(), recall.item(), f1.item()


# --- GNN Architecture ---
class BBB_GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BBB_GNN, self).__init__()
        
        # Graph Attention Layers (Message Passing with Multi-Head Attention + Edge Features)
        # Allows the model to learn the importance of different neighboring atoms and their bonds
        self.conv1 = GATv2Conv(num_node_features, 64, heads=4, edge_dim=6)
        self.conv2 = GATv2Conv(256, 64, heads=4, edge_dim=6) # 64 * 4 heads = 256 input
        self.conv3 = GATv2Conv(256, 256, heads=4, concat=False, edge_dim=6) # Average heads for final layer
        
        # Linear layers for final classification
        self.lin1 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Message Passing (GATv2 with Edge Features)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        
        # 2. Global Pooling (Collapse the whole graph into a single 256D vector per molecule)
        x = global_mean_pool(x, batch)
        
        # 3. Final MLP Classifier
        x = self.dropout(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return x


# --- Training Loop ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15

    # Load Graph Data
    print("Loading Graph Dataset (this may take a few seconds)...")
    # Setting weights_only=False is required in PyTorch 2.6+ to load custom objects like PyG Data
    graph_data_list = torch.load("b3db_graphs.pt", weights_only=False)
    
    # Train/Test Split
    train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)
    
    # PyG DataLoaders handle the complex math of batching lots of disconnected graphs together
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    num_node_features = graph_data_list[0].num_node_features
    print(f"Node feature dimension: {num_node_features}")

    # Initialize Model, Loss, Optimizer
    model = BBB_GNN(num_node_features=num_node_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\nStarting GNN Training...")
    best_test_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            # GNN needs node features (x), edge connections (edge_index), edge attributes (edge_attr), and batch
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            loss = criterion(outputs, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(train_loader.dataset)
        epoch_time = time.time() - start_time
        
        # --- EVALUATION ---
        model.eval()
        test_loss = 0.0
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(outputs, batch.y.unsqueeze(1))
                test_loss += loss.item() * batch.num_graphs
                
                all_preds.append(outputs)
                all_trues.append(batch.y.unsqueeze(1))
                
        test_loss /= len(test_loader.dataset)
        full_preds = torch.cat(all_preds)
        full_trues = torch.cat(all_trues)
        acc, prec, rec, f1 = calculate_metrics_gpu(full_preds, full_trues)
        
        # EARLY STOPPING
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1
            is_best = False
            
        best_marker = " (Best!)" if is_best else ""
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.4f}{best_marker}")
            
        if epochs_no_improve >= PATIENCE:
            pass # print(f"\nEarly stopping triggered at epoch {epoch}! No improvement for {PATIENCE} epochs.")
            # break

    print("\nTraining Complete! Loading best model weights...")
    model.load_state_dict(best_model_weights)
    
    # Final evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            all_preds.append(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
            all_trues.append(batch.y.unsqueeze(1))
    
    full_preds = torch.cat(all_preds)
    full_trues = torch.cat(all_trues)
    acc, prec, rec, f1 = calculate_metrics_gpu(full_preds, full_trues)
    
    print(f"\n--- Best GNN Test Metrics (Test Loss: {best_test_loss:.4f}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    save_path = "bbb_gnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
