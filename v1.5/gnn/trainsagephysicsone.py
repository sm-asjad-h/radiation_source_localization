import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

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

# ==========================================
# 1. LOAD GRAPHS & SCALERS
# ==========================================
print("Loading graphs and scalers...")
train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)

temp_train = DataLoader(train_graphs, batch_size=16384, shuffle=True)
temp_val = DataLoader(val_graphs, batch_size=16384, shuffle=False)

with open('graph_data/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

mean_y = torch.tensor(scalers['scaler_y'].mean_, dtype=torch.float32, device=device)
scale_y = torch.tensor(scalers['scaler_y'].scale_, dtype=torch.float32, device=device)

mean_x = torch.tensor(scalers['scaler_x'].mean_, dtype=torch.float32, device=device)
scale_x = torch.tensor(scalers['scaler_x'].scale_, dtype=torch.float32, device=device)

gpu_train = [batch.to(device) for batch in temp_train]
gpu_val = [batch.to(device) for batch in temp_val]
print("Dataset locked in VRAM.")

# ==========================================
# 2. PHYSICS-INFORMED LOSS FUNCTION
# ==========================================
# ==========================================
# 2. PHYSICS-INFORMED LOSS FUNCTION (LOG SPACE)
# ==========================================
class PIMLLoss(nn.Module):
    def __init__(self):
        super(PIMLLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))
        
    def forward(self, preds_scaled, targets_scaled, batch_x_scaled):
        # A. Data-Driven Loss
        mse_loc = F.mse_loss(preds_scaled[:, 0:2], targets_scaled[:, 0:2])
        mse_int = F.mse_loss(preds_scaled[:, 2], targets_scaled[:, 2])
        
        # B. Physical Unscaling (KEEP IN LOG SPACE)
        preds_unscaled = (preds_scaled * scale_y) + mean_y
        pred_x = preds_unscaled[:, 0]
        pred_y = preds_unscaled[:, 1]
        
        # Do NOT use expm1 here. Keep it as the Log Intensity.
        pred_log_i0 = preds_unscaled[:, 2] 
        
        x_unscaled = (batch_x_scaled * scale_x) + mean_x
        batch_size = preds_scaled.size(0)
        x_reshaped = x_unscaled.view(batch_size, 3, -1)
        
        sensor_x = x_reshaped[:, :, 0]  
        sensor_y = x_reshaped[:, :, 1]  
        
        # Grab the LOG intensity from the sensor (Index 3 from your data pipeline)
        actual_log_reading = x_reshaped[:, :, 3] 
        
        # C. Physics Violation Loss (IN LOG SPACE)
        dx = sensor_x - pred_x.unsqueeze(1)
        dy = sensor_y - pred_y.unsqueeze(1)
        d_sq = (dx**2) + (dy**2)
        
        # Physics Law: log(I / d^2) = log(I) - log(d^2)
        # Using ReLU to ensure we don't accidentally predict negative log radiation
        expected_log_reading = F.relu(pred_log_i0.unsqueeze(1) - torch.log(d_sq + 1e-6))
        
        # Now compare log vs log. No exploding Float32 limits!
        physics_loss = F.mse_loss(expected_log_reading, actual_log_reading)

        # D. Uncertainty Weighting
        prec_loc = torch.exp(-self.log_vars[0])
        prec_int = torch.exp(-self.log_vars[1])
        prec_phys = torch.exp(-self.log_vars[2])
        
        loss_loc = (prec_loc * mse_loc) + self.log_vars[0]
        loss_int = (prec_int * mse_int) + self.log_vars[1]
        loss_phys = (prec_phys * physics_loss) + self.log_vars[2]
        
        total_loss = loss_loc + loss_int + loss_phys
        
        return total_loss, mse_loc, mse_int, physics_loss# ==========================================
# 3. RUNTIME INITIALIZATION 
# ==========================================
model = DynamicGNN(model_type='SAGE', num_layers=5, in_channels=6, hidden_dim=512, out_channels=3, heads=1, dropout=0.0).to(device)
piml_criterion = PIMLLoss().to(device)

optimizer = torch.optim.Adam(list(model.parameters()) + list(piml_criterion.parameters()), lr=0.0028, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

# ==========================================
# 4. TRAINING LOOP & TRACKING
# ==========================================
epochs = 1000
best_val_loss = float('inf')

# History trackers
history_train_total = []
history_val_total = []
history_val_loc = []
history_val_int = []
history_val_phys = []

print("Executing PIML Training...")
for epoch in range(1, epochs + 1):
    model.train()
    piml_criterion.train()
    running_loss = 0
    random.shuffle(gpu_train)
    
    for batch in gpu_train:
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _, _, _ = piml_criterion(preds, batch.y, batch.x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
        
    epoch_train_loss = running_loss / len(train_graphs)
    
    # Validation
    model.eval()
    piml_criterion.eval()
    running_val_loss = 0
    running_val_loc = 0
    running_val_int = 0
    running_val_phys = 0
    
    with torch.no_grad():
        for batch in gpu_val:
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, m_loc, m_int, m_phys = piml_criterion(preds, batch.y, batch.x)
            
            running_val_loss += loss.item() * batch.num_graphs
            running_val_loc += m_loc.item() * batch.num_graphs
            running_val_int += m_int.item() * batch.num_graphs
            running_val_phys += m_phys.item() * batch.num_graphs
            
    epoch_val_loss = running_val_loss / len(val_graphs)
    epoch_val_loc = running_val_loc / len(val_graphs)
    epoch_val_int = running_val_int / len(val_graphs)
    epoch_val_phys = running_val_phys / len(val_graphs)
    
    scheduler.step(epoch_val_loss)
    
    # Save to history lists
    history_train_total.append(epoch_train_loss)
    history_val_total.append(epoch_val_loss)
    history_val_loc.append(epoch_val_loc)
    history_val_int.append(epoch_val_int)
    history_val_phys.append(epoch_val_phys)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_state_dict': piml_criterion.state_dict()
        }, 'best_piml_sage.pth')
        
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Total Val Loss: {epoch_val_loss:.4f} | "
              f"Loc MSE: {epoch_val_loc:.4f} | Int MSE: {epoch_val_int:.4f} | Phys MSE: {epoch_val_phys:.2f}")

# ==========================================
# 5. SAVE PIML DASHBOARD PLOTS
# ==========================================
print("Generating PIML training curves...")
plt.figure(figsize=(16, 10))

# Plot 1: Total Loss
plt.subplot(2, 2, 1)
plt.plot(history_train_total, label='Train Total', color='blue')
plt.plot(history_val_total, label='Val Total', color='orange')
plt.title('Total Uncertainty-Weighted Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Location Validation Error
plt.subplot(2, 2, 2)
plt.plot(history_val_loc, label='Val Location MSE', color='green')
plt.title('Location Target Error (Scaled)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Plot 3: Intensity Validation Error
plt.subplot(2, 2, 3)
plt.plot(history_val_int, label='Val Intensity MSE', color='red')
plt.title('Intensity Target Error (Scaled)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Plot 4: Physics Validation Error (Using Log Scale)
plt.subplot(2, 2, 4)
plt.plot(history_val_phys, label='Val Physics Violations', color='purple')
plt.title('Physics Equation Error (Unscaled)')
plt.xlabel('Epoch')
plt.ylabel('MSE (Log Scale)')
plt.yscale('log') # Log scale because unscaled physics loss starts massive
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('piml_training_dashboard.png', bbox_inches='tight')
plt.close()

print("PIML Training Complete. Checkpoint saved as 'best_piml_sage.pth'. Curves saved as 'piml_training_dashboard.png'.")