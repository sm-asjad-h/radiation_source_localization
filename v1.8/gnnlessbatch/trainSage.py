import os
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# 1. Hardware Lockdown
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from models import DynamicGNN 

# ==========================================
# 0. REPRODUCIBILITY 
# ==========================================
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
print(f"Targeting isolated GPU. PyTorch device index: {device}")

# ==========================================
# 1. LOAD GRAPHS AND PACK INTO MASSIVE CHUNKS
# ==========================================
print("Loading graphs from storage...")
train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)

temp_train_loader = DataLoader(train_graphs, batch_size=512, shuffle=True)
temp_val_loader = DataLoader(val_graphs, batch_size=512, shuffle=False)

# ==========================================
# 2. TRANSFER EVERYTHING TO GPU VRAM 
# ==========================================
print(f"Transferring entire dataset directly into A100 VRAM...")
gpu_train_batches = [batch.to(device) for batch in temp_train_loader]
gpu_val_batches = [batch.to(device) for batch in temp_val_loader]
print(f"Success! {len(gpu_train_batches)} train chunks and {len(gpu_val_batches)} val chunks are locked in VRAM.")

# ==========================================
# 3. RUNTIME INITIALIZATION (The Winning Config)
# ==========================================
model = DynamicGNN().to(device)

# Exact Optuna Hyperparameters
optuna_lr = 0.0028062749167200355
optimizer = torch.optim.Adam(model.parameters(), lr=optuna_lr, weight_decay=1e-4)

# Crucial for preventing the "bouncing" effect
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
criterion = torch.nn.MSELoss().to(device)

# ==========================================
# 4. MONITORING ENGINE & TRAINING LOOP
# ==========================================
epochs = 2000
best_val_loss = float('inf')
patience = 100
early_stop_counter = 0

history_train_loss = []
history_val_loss = []
total_train_graphs = len(train_graphs)
total_val_graphs = len(val_graphs)

print("Executing purely In-Memory VRAM Training...")
for epoch in range(1, epochs + 1):
    # --- TRAINING PHASE ---
    model.train()
    running_train_loss = 0
    random.shuffle(gpu_train_batches)
    
    for batch_data in gpu_train_batches:
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
    running_val_loss = 0
    with torch.no_grad():
        for batch_data in gpu_val_batches:
            predictions = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            loss = criterion(predictions, batch_data.y)
            running_val_loss += loss.item() * batch_data.num_graphs
            
    epoch_val_loss = running_val_loss / total_val_graphs
    
    # Step the learning rate scheduler
    scheduler.step(epoch_val_loss)
    
    history_train_loss.append(epoch_train_loss)
    history_val_loss.append(epoch_val_loss)
    
    # Save best weights
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_sage_gnn.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d} | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f} (Best: {best_val_loss:.5f})")

    if early_stop_counter >= patience:
        print(f"Early stopping triggered at Epoch {epoch}. Best Val MSE: {best_val_loss:.5f}")
        break

# ==========================================
# 5. SAVE LOSS PLOT
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(history_train_loss, label='Train MSE', color='blue')
plt.plot(history_val_loss, label='Val MSE', color='orange')
plt.title('SAGE Loss Curve (5L - 512D)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss (Scaled)')
plt.legend()
plt.grid(True)
plt.savefig('sage_training_loss_curve.png', bbox_inches='tight')
plt.close()

print("Run complete. Checkpoint saved as 'best_sage_gnn.pth'.")
