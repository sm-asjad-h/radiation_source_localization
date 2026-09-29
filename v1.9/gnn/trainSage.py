import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader
from models import DynamicGNN 

# ==========================================
# 0. REPRODUCIBILITY & HARDWARE SETUP
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

print("Locking all random seeds to 42...")
lock_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Targeting device: {device}")

# ==========================================
# 1. LOAD DATASET & CONFIGURE DATALOADERS
# ==========================================
print("Loading graphs into CPU memory...")
train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)

# Adjust BATCH_SIZE based on your GPU memory capacity and graph size
BATCH_SIZE = 256  
NUM_WORKERS = 0  # The in-memory graph list exceeds shared-memory handles with worker processes.
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print(f"Loaded {len(train_graphs)} training graphs and {len(val_graphs)} validation graphs.")
print(f"Batch size: {BATCH_SIZE} | Train batches per epoch: {len(train_loader)}")

# ==========================================
# 2. RUNTIME INITIALIZATION
# ==========================================
model = DynamicGNN().to(device)

optuna_lr = 0.0028062749167200355
optimizer = torch.optim.Adam(model.parameters(), lr=optuna_lr, weight_decay=1e-4)

# Dynamic learning rate reduction on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30
)
criterion = torch.nn.MSELoss()

# ==========================================
# 3. TRAINING & VALIDATION LOOP
# ==========================================
epochs = 2000
best_val_loss = float('inf')
patience = 100
early_stop_counter = 0

history_train_loss = []
history_val_loss = []
total_train_graphs = len(train_graphs)
total_val_graphs = len(val_graphs)

print("Starting standard CPU-to-GPU streamed training...")

for epoch in range(1, epochs + 1):
    # --- TRAINING PHASE ---
    model.train()
    running_train_loss = 0.0
    
    for batch_data in train_loader:
        # Move mini-batch from CPU to GPU asynchronously
        batch_data = batch_data.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        predictions = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        loss = criterion(predictions, batch_data.y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_train_loss += loss.item() * batch_data.num_graphs
        
    epoch_train_loss = running_train_loss / total_train_graphs
    
    # --- VALIDATION PHASE ---
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            predictions = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            loss = criterion(predictions, batch_data.y)
            running_val_loss += loss.item() * batch_data.num_graphs
            
    epoch_val_loss = running_val_loss / total_val_graphs
    
    # Step scheduler with current validation loss
    scheduler.step(epoch_val_loss)
    
    history_train_loss.append(epoch_train_loss)
    history_val_loss.append(epoch_val_loss)
    
    # Checkpoint best model weights
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_sage_gnn.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        
    if epoch % 10 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f} (Best: {best_val_loss:.5f}) | LR: {current_lr:.6f}")

    if early_stop_counter >= patience:
        print(f"Early stopping triggered at Epoch {epoch}. Best Val MSE: {best_val_loss:.5f}")
        break

# ==========================================
# 4. SAVE TRAINING CURVE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(history_train_loss, label='Train MSE', color='blue')
plt.plot(history_val_loss, label='Val MSE', color='orange')
plt.title('SAGE Loss Curve (Dynamic Training)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss (Scaled)')
plt.legend()
plt.grid(True)
plt.savefig('sage_training_loss_curve.png', bbox_inches='tight')
plt.close()

print("Run complete. Checkpoint saved as 'best_sage_gnn.pth'.")
