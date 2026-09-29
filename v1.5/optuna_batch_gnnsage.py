import os
import json
import logging
import sqlite3
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import matplotlib.pyplot as plt
import optuna
import sys
import pickle

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
# 1. LOCKED SAGE ARCHITECTURE (From Phase 1)
# ==========================================
class Phase2LockedGNN(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_dim=512, out_channels=3, num_layers=5, dropout=0.00015923260230465551):
        super(Phase2LockedGNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # Layer 1
        self.convs.append(SAGEConv(in_channels, hidden_dim))
        # Intermediate Layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        # Final SAGE Layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Readout Layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = torch.nn.Linear(hidden_dim // 2, out_channels)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.out(x)

# ==========================================
# 2. LOAD RAW GRAPHS INTO CPU RAM FIRST
# ==========================================
print("Loading graphs into CPU memory...")
train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)
test_graphs = torch.load('graph_data/test_graphs.pt', weights_only=False)

# Metadata for scaling metrics
with open('graph_data/scalers.pkl', 'rb') as f:
    scaler_y = pickle.load(f)['scaler_y']

# ==========================================
# 3. PHASE 2 OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # --- A. Search Space Setup ---
    # Tuning the critical coupling of Batch Size vs Learning Rate
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 8192])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    # --- B. Directory Setup ---
    trial_name = f"phase2_trial_{trial.number:03d}_BS{batch_size}_LR{lr:.5f}"
    trial_dir = os.path.join("optuna_trials_phase2", trial_name)
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save parameters for this trial immediately
    with open(os.path.join(trial_dir, "hyperparams.json"), 'w') as f:
        json.dump(trial.params, f, indent=4)

    # --- C. Dynamic VRAM Data Loader Preparation ---
    # Batches are chunked and moved directly to VRAM dynamically per trial configuration
    gpu_train_batches = [batch.to(device) for batch in DataLoader(train_graphs, batch_size=batch_size, shuffle=True)]
    gpu_val_batches = [batch.to(device) for batch in DataLoader(val_graphs, batch_size=batch_size, shuffle=False)]
    gpu_test_batches = [batch.to(device) for batch in DataLoader(test_graphs, batch_size=batch_size, shuffle=False)]

    # --- D. Initialize Model with Locked Phase 1 Architecture ---
    model = Phase2LockedGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    criterion = torch.nn.MSELoss().to(device)
    
    epochs = 400  # Cap for faster tuning execution
    best_val_loss = float('inf')
    early_stop_counter = 0
    payout_patience = 50 
    
    # --- E. Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(gpu_train_batches) 
        
        for batch_data in gpu_train_batches:
            optimizer.zero_grad()
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(pred, batch_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for batch_data in gpu_val_batches:
                pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                loss = criterion(pred, batch_data.y)
                running_val_loss += loss.item() * batch_data.num_graphs
                
        epoch_val_loss = running_val_loss / len(val_graphs)
        scheduler.step(epoch_val_loss)
        
        # Checkpoint Saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_weights.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Pruning mechanism
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            # Clear VRAM allocations before exiting trial
            del gpu_train_batches, gpu_val_batches, gpu_test_batches
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
            
        if early_stop_counter >= payout_patience:
            break

    # --- F. Evaluate Best Trial Configuration ---
    model.load_state_dict(torch.load(os.path.join(trial_dir, "best_weights.pth"), weights_only=True))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch_data in gpu_test_batches:
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            all_preds.append(pred.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    test_mse_scaled = criterion(torch.tensor(all_preds).to(device), torch.cat([b.y for b in gpu_test_batches])).item()
    
    with open(os.path.join(trial_dir, "results.json"), 'w') as f:
        json.dump({"best_val_mse_scaled": best_val_loss, "test_mse_scaled": test_mse_scaled}, f, indent=4)
        
    # Prevent memory leaks by cleaning up explicit VRAM arrays at end of run
    del gpu_train_batches, gpu_val_batches, gpu_test_batches
    torch.cuda.empty_cache()

    return best_val_loss

# ==========================================
# 4. EXECUTE PHASE 2 TUNING
# ==========================================
if __name__ == "__main__":
    PHASE2_TOTAL_TRIALS = 30  # Batch/LR space converges much faster
    
    print("\n" + "="*50)
    print("?? INIT OPTUNA PHASE 2: BATCH SIZE & LR TUNING ??")
    print("="*50)
    
    # We point to the exact same database file, but use a distinct study name
    storage_name = "sqlite:///gnn_tuning.db"
    phase2_study_name = "gnn_radiation_study_phase2_batch_lr"
    
    study = optuna.create_study(
        study_name=phase2_study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
    )
    
    completed = len(study.trials)
    print(f"Loaded Phase 2 study. {completed} trials completed out of {PHASE2_TOTAL_TRIALS}.")
    
    if completed < PHASE2_TOTAL_TRIALS:
        study.optimize(objective, n_trials=PHASE2_TOTAL_TRIALS - completed)
        
    print("\n" + "="*50)
    print("?? PHASE 2 TUNING COMPLETE ??")
    print("="*50)
    print("Best Phase 2 Setup:")
    print(f"  Value (Validation MSE): {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")