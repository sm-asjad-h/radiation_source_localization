import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Targeting isolated GPU. PyTorch device index: {device}")

from models import DynamicGNN 

CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)
import common_function.save_visualizations as vnz

# ==========================================
# 1. LOAD EVERYTHING FROM DISK
# ==========================================
print("Loading model, test graphs, and scalers...")

# Initialize the exact Optuna winner
model = DynamicGNN().to(device)
model.load_state_dict(torch.load('best_sage_gnn.pth', map_location=device, weights_only=True))
model.eval()

# Load Test Graphs & DataLoader
test_graphs = torch.load('graph_data/test_graphs.pt', weights_only=False)
test_loader = DataLoader(test_graphs, batch_size=512, shuffle=False)

# Load Scalers
with open('graph_data/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
scaler_y = scalers['scaler_y']

# Load Raw Test DataFrame
df_test_raw = pd.read_pickle('graph_data/df_test_raw.pkl')

# ==========================================
# 2. RUN INFERENCE ON TEST SET
# ==========================================
print("Running inferences...")
all_preds = []

with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        predictions = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        all_preds.append(predictions.cpu().numpy())

all_preds = np.vstack(all_preds)

# ==========================================
# 3. INVERSE TRANSFORM BACK TO PHYSICAL UNITS
# ==========================================
print("Inverting scaling and transformations...")

unscaled_preds = scaler_y.inverse_transform(all_preds)

preds_loc = unscaled_preds[:, 0:2]
preds_int = np.expm1(unscaled_preds[:, 2]) 

y_test_loc = df_test_raw[['source_x', 'source_y']]
y_test_int = df_test_raw['I_0']
X_test_unscaled = df_test_raw.drop(columns=['source_x', 'source_y', 'I_0']) 

# ==========================================
# 4. PASS TO VISUALIZER
# ==========================================
print("Passing combined results to vnz.visualize...")

dict_pred_loc = {
    "GraphSAGE": pd.DataFrame(preds_loc, columns=['source_x', 'source_y'], index=y_test_loc.index)
}
dict_pred_int = {
    "GraphSAGE": pd.Series(preds_int, index=y_test_loc.index)
}

vnz.visualize_loc_results(dict_pred_loc, y_test_loc, X_test_unscaled)
vnz.visualize_int_results(dict_pred_int, y_test_int, y_test_loc, X_test_unscaled)

print(" Testing and Visualization Complete!")