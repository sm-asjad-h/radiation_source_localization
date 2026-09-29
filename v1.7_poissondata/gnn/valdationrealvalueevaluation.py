import torch
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from models import DynamicGNN 

# ==========================================
# 1. SETUP & LOADING
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Targeting PyTorch device: {device}")

# Load the saved model
model = DynamicGNN().to(device)
model.load_state_dict(torch.load('best_sage_gnn.pth', map_location=device, weights_only=True))
model.eval()

# Load Validation Graphs
print("Loading validation graphs...")
val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)
val_loader = DataLoader(val_graphs, batch_size=512, shuffle=False)

# Load the Scaler
print("Loading scaler...")
with open('graph_data/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
scaler_y = scalers['scaler_y']

# ==========================================
# 2. RUN INFERENCE ON VALIDATION SET
# ==========================================
print("Running validation inference...")
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        
        # Get predictions
        predictions = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        
        # Store scaled predictions and scaled true targets
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(batch_data.y.cpu().numpy())

# Stack them into single numpy arrays
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# ==========================================
# 3. INVERT SCALING TO REAL-WORLD UNITS
# ==========================================
print("Inverting scaling to physical units...")
unscaled_preds = scaler_y.inverse_transform(all_preds)
unscaled_targets = scaler_y.inverse_transform(all_targets)

# Extract only the X and Y coordinates (Columns 0 and 1)
preds_loc = unscaled_preds[:, 0:2]
targets_loc = unscaled_targets[:, 0:2]

# ==========================================
# 4. CALCULATE REAL-WORLD METRICS
# ==========================================
# 1. Individual Error per Axis
errors_loc = preds_loc - targets_loc

# 2. MSE of X and Y coordinates individually
mse_x = np.mean(errors_loc[:, 0]**2)
mse_y = np.mean(errors_loc[:, 1]**2)

# 3. Euclidean Distance (Actual Physical Difference in Location)
# Formula: sqrt((X_pred - X_actual)^2 + (Y_pred - Y_actual)^2)
euclidean_distances = np.sqrt(errors_loc[:, 0]**2 + errors_loc[:, 1]**2)

mean_euclidean = np.mean(euclidean_distances)
median_euclidean = np.median(euclidean_distances)
mse_distance = np.mean(euclidean_distances**2) # This matches your 26.0 metric from earlier

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================
print("\n" + "="*50)
print(" PHYSICAL VALIDATION METRICS (UNSCALED)")
print("="*50)
print(f"Per-Axis Mean Squared Error (MSE):")
print(f"  - X-Axis MSE: {mse_x:.4f} units^2")
print(f"  - Y-Axis MSE: {mse_y:.4f} units^2")
print("-" * 50)
print(f"Physical Location Error (Euclidean Distance):")
print(f"  - Mean Distance Error:   {mean_euclidean:.4f} units (Average physical miss)")
print(f"  - Median Distance Error: {median_euclidean:.4f} units")
print(f"  - Distance MSE:          {mse_distance:.4f} units^2")
print("="*50)