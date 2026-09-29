import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. PATHS & CUSTOM MODULES
# ==========================================
CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset.csv')

SYS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(SYS_ROOT)

import common_function.feature_engineering_functions as fe
import common_function.split_and_scale as ss
import common_function.save_visualizations as vnz
from models import RadiationMLP


# ==========================================
# 1. LOAD MODEL & ARTIFACTS
# ==========================================
print("Loading Model and Test Data...")
with open('artifacts/scalers.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    scaler_y = artifacts['scaler_y']
    input_dim = artifacts['input_dim']

with open('artifacts/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Rebuild Model
model = RadiationMLP(input_dim=input_dim).to(device)
model.load_state_dict(torch.load('artifacts/best_mlp.pth', map_location=device, weights_only=True))
model.eval()

# Setup Test DataLoader
t_X_test = torch.tensor(test_data['X_test_scaled'], dtype=torch.float32)
test_loader = DataLoader(TensorDataset(t_X_test), batch_size=4096, shuffle=False)

# ==========================================
# 2. RUN INFERENCE
# ==========================================
print("Running Inference...")
all_preds = []
with torch.no_grad():
    for batch_x in test_loader:
        batch_x = batch_x[0].to(device)
        preds = model(batch_x)
        all_preds.append(preds.cpu().numpy())

all_preds = np.vstack(all_preds)

# ==========================================
# 3. POST-PROCESSING (Unscale & Anti-Explosion)
# ==========================================
print("Post-processing predictions...")
unscaled_preds = scaler_y.inverse_transform(all_preds)

# X and Y coords
preds_loc_np = unscaled_preds[:, 0:2]

# Log Intensity (Clipped to a max of 20 to prevent np.expm1 overflow crashes)
log_int_np = np.clip(unscaled_preds[:, 2], a_min=0.0, a_max=20.0)
preds_int_raw = np.expm1(log_int_np)

# Format for your visualizer
y_test_loc = test_data['y_test_loc']
y_test_int = test_data['y_test_int']
X_test_unscaled = test_data['X_test_unscaled']

loc_predictions = {"PyTorch_MLP": pd.DataFrame(preds_loc_np, columns=['source_x', 'source_y'], index=y_test_loc.index)}
int_predictions = {"PyTorch_MLP": pd.Series(preds_int_raw, index=y_test_int.index)}

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("Generating Plots...")
vnz.visualize_loc_results(loc_predictions, y_test_loc, X_test_unscaled)
vnz.visualize_int_results(int_predictions, y_test_int, y_test_loc, X_test_unscaled, SI_min=400.0, SI_max=8000.0)

print("\nTesting Complete! Check your plots directory.")