import os
import json
import logging
import sqlite3
import random
import shutil
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, global_mean_pool
import matplotlib.pyplot as plt
import optuna
import sys

# ---> IMPORT YOUR VISUALIZER <---
CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)
import common_function.save_visualizations as vnz 

# ==========================================
# 0. HARDWARE & REPRODUCIBILITY LOCK
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def lock_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lock_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Targeting GPU: {device}")

# ==========================================
# 1. THE SINGLE-HEAD ARCHITECTURE (Original)
# ==========================================
class DynamicGNN(torch.nn.Module):
    def __init__(self, model_type, num_layers, in_channels, hidden_dim, out_channels, heads=1, dropout=0.2):
        super(DynamicGNN, self).__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        current_dim = in_channels
        
        # Build all layers except the last one
        for i in range(num_layers - 1):
            if self.model_type == 'GAT':
                self.convs.append(GATv2Conv(current_dim, hidden_dim, heads=heads, edge_dim=1, concat=True))
                current_dim = hidden_dim * heads
            elif self.model_type == 'GCN':
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            elif self.model_type == 'SAGE':
                self.convs.append(SAGEConv(current_dim, hidden_dim))
                current_dim = hidden_dim
                
        # Build Final GNN Layer
        if self.model_type == 'GAT':
            self.convs.append(GATv2Conv(current_dim, hidden_dim, heads=1, edge_dim=1, concat=False))
        elif self.model_type == 'GCN':
            self.convs.append(GCNConv(current_dim, hidden_dim))
        elif self.model_type == 'SAGE':
            self.convs.append(SAGEConv(current_dim, hidden_dim))
            
        current_dim = hidden_dim
        
        # Shared Readout Head
        self.fc1 = torch.nn.Linear(current_dim, current_dim // 2)
        self.out = torch.nn.Linear(current_dim // 2, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        if self.model_type == 'GAT' and edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

        for i, conv in enumerate(self.convs):
            if self.model_type == 'GAT':
                x = F.relu(conv(x, edge_index, edge_attr=edge_weight))
            elif self.model_type == 'GCN':
                x = F.relu(conv(x, edge_index, edge_weight))
            elif self.model_type == 'SAGE':
                x = F.relu(conv(x, edge_index))
            
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.out(x)

# ==========================================
# 2. LOAD DATASET TO CPU MEMORY (FOR PINNING)
# ==========================================
print("Loading graphs from disk...")
train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)
test_graphs = torch.load('graph_data/test_graphs.pt', weights_only=False)

# Load metadata for unscaling and visualization
with open('graph_data/scalers.pkl', 'rb') as f:
    scaler_y = pickle.load(f)['scaler_y']
df_test_raw = pd.read_pickle('graph_data/df_test_raw.pkl')
y_test_loc = df_test_raw[['source_x', 'source_y']]
y_test_int = df_test_raw['I_0']
X_test_unscaled = df_test_raw.drop(columns=['source_x', 'source_y', 'I_0']) 

# ==========================================
# 3. THE OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # --- A. Define Hyperparameter Search Space ---
    model_type = trial.suggest_categorical("model_type", ["GAT", "GCN", "SAGE"])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    heads = trial.suggest_categorical("heads", [2, 4, 8]) if model_type == 'GAT' else 1
    
    # --- B. Setup Directory for this Trial ---
    trial_name = f"trial_{trial.number:03d}_{model_type}_{num_layers}L_{hidden_dim}D"
    trial_dir = os.path.join("optuna_trials", trial_name)
    os.makedirs(trial_dir, exist_ok=True)
    
    with open(os.path.join(trial_dir, "hyperparams.json"), 'w') as f:
        json.dump(trial.params, f, indent=4)

    # --- C. Initialize High-Performance CPU DataLoaders ---
    # RESTORED: Batch size 1024 for required gradient noise
    # ADDED: Asynchronous CPU-to-GPU transfer pipeline
    train_loader = DataLoader(
        train_graphs, batch_size=512, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=512, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_graphs, batch_size=512, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # --- D. Initialize Model & Training Tools ---
    model = DynamicGNN(model_type, num_layers, in_channels=6, hidden_dim=hidden_dim, out_channels=3, heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    criterion = torch.nn.MSELoss().to(device)
    
    epochs = 500 
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 75 
    
    # --- E. Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data in train_loader:
            # Asynchronous non_blocking DMA Transfer to A100
            batch_data = batch_data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            loss = criterion(pred, batch_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation Phase
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device, non_blocking=True)
                pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
                loss = criterion(pred, batch_data.y)
                running_val_loss += loss.item() * batch_data.num_graphs
                
        epoch_val_loss = running_val_loss / len(val_graphs)
        scheduler.step(epoch_val_loss)
        
        # Track and Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_weights.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Optuna Pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        if early_stop_counter >= patience:
            break

    # --- F. Final Inference & Plotting ---
    # EXPLICIT ROLLBACK: Load best weights before testing
    model.load_state_dict(torch.load(os.path.join(trial_dir, "best_weights.pth"), map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    running_test_loss = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            
            loss = criterion(pred, batch_data.y)
            running_test_loss += loss.item() * batch_data.num_graphs
            all_preds.append(pred.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    test_mse_scaled = running_test_loss / len(test_graphs)
    
    # Invert predictions back to spatial/physical units
    unscaled_preds = scaler_y.inverse_transform(all_preds)
    preds_loc = unscaled_preds[:, 0:2]
    preds_int = np.expm1(unscaled_preds[:, 2])
    
    with open(os.path.join(trial_dir, "results.json"), 'w') as f:
        json.dump({"best_val_mse_scaled": best_val_loss, "test_mse_scaled": test_mse_scaled}, f, indent=4)
        
    # Generate Output Plots 
    try:
        dict_pred_loc = {model_type: pd.DataFrame(preds_loc, columns=['source_x', 'source_y'], index=y_test_loc.index)}
        dict_pred_int = {model_type: pd.Series(preds_int, index=y_test_loc.index)}
        
        vnz.visualize_loc_results(dict_pred_loc, y_test_loc, X_test_unscaled)
        vnz.visualize_int_results(dict_pred_int, y_test_int, y_test_loc, X_test_unscaled)
        
        # Relocate generated visualization assets
        if os.path.exists('plots'):
            for filename in os.listdir('plots'):
                shutil.move(os.path.join('plots', filename), os.path.join(trial_dir, filename))
    except Exception as e:
        print(f"Warning: Plotting failed for trial {trial.number}. Error: {e}")

    return best_val_loss

# ==========================================
# 4. EXECUTE STUDY WITH SQLITE PERSISTENCE
# ==========================================
if __name__ == "__main__":
    TOTAL_TRIALS = 100
    
    print("\n" + "="*50)
    print(" INIT OPTUNA HYPERPARAMETER SEARCH ")
    print("="*50)
    
    study_name = "gnn_radiation_study"
    storage_name = "sqlite:///gnn_tuning.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30)
    )
    
    completed = len(study.trials)
    print(f"Loaded existing study. {completed} trials completed out of {TOTAL_TRIALS}.")
    
    if completed < TOTAL_TRIALS:
        study.optimize(objective, n_trials=TOTAL_TRIALS - completed)
        
    print("\n" + "="*50)
    print(" TUNING COMPLETE")
    print("="*50)
    print("Best Trial:")
    print(f"  Value (Validation MSE): {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")