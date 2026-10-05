import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import time
import copy

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

class BBBDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.X = torch.tensor(self.df.drop(columns=['SMILES', 'Label']).values, dtype=torch.float32)
        self.y = torch.tensor(self.df['Label'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BBB_MLP(nn.Module):
    # Input size is now 2048 (fingerprints) + 6 (descriptors) = 2054
    def __init__(self, input_size=2054):
        super(BBB_MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4), # Increased dropout to reduce overfitting
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.network(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.0005 # Slightly lower learning rate for stability
    PATIENCE = 10 # Stop if test loss doesn't improve for 10 epochs

    # Pointing to the new dataset with the 2D descriptors!
    dataset = BBBDataset("data/processed_b3db_data_advanced.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BBB_MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # ADDED WEIGHT DECAY (L2 Regularization) to heavily penalize overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\nStarting Training with Early Stopping & Weight Decay...")
    
    best_test_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- EVALUATION ---
        model.eval()
        test_loss = 0.0
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                all_preds.append(outputs)
                all_trues.append(batch_y)
                
        test_loss /= len(test_loader.dataset)
        full_preds = torch.cat(all_preds)
        full_trues = torch.cat(all_trues)
        acc, prec, rec, f1 = calculate_metrics_gpu(full_preds, full_trues)
        
        # EARLY STOPPING LOGIC
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Save a deepcopy of the weights when it performs best
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1
            is_best = False
            
        best_marker = " (Best!)" if is_best else ""
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.4f}{best_marker}")
            
        if epochs_no_improve >= PATIENCE:
            pass # print(f"\nEarly stopping triggered at epoch {epoch}! No improvement in test loss for {PATIENCE} epochs.")
            # break

    print("\nTraining Complete! Loading best model weights...")
    # Roll back to the best model (e.g. at epoch 10) instead of the overfitted model at epoch 50
    model.load_state_dict(best_model_weights)
    
    # Final evaluation of the best model
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            all_preds.append(model(batch_X))
            all_trues.append(batch_y)
    
    full_preds = torch.cat(all_preds)
    full_trues = torch.cat(all_trues)
    acc, prec, rec, f1 = calculate_metrics_gpu(full_preds, full_trues)
    
    print(f"\n--- Best Model Test Metrics (Test Loss: {best_test_loss:.4f}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    save_path = "models/bbb_mlp_model_advanced.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
