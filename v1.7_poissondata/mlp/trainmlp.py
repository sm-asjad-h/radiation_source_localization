import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1. DATA PIPELINE & FEATURE ENGINEERING
# ==========================================
print("Loading and Engineering Data...")
df = pd.read_csv(DATASET_PATH)

df = fe.sort_detectors_by_intensity(df)
df = fe.add_sensor_distances(df)
df = fe.add_relative_coordinates(df)
df = fe.add_intensity_ratios(df)
df = fe.add_weighted_centroid(df)
df = fe.apply_log(df, drop=True)

# Extract Targets
targets = df[['source_x', 'source_y']].copy()
targets['I_0_log'] = np.log1p(df['I_0'])

# Extract Features (Everything except the answers)
features = df.drop(columns=['source_x', 'source_y', 'I_0'])
input_dim = features.shape[1]

# ==========================================
# 2. TRAIN / VAL / TEST SPLIT
# ==========================================
# Split 1: 85% Train+Val, 15% Test
X_temp, X_test, y_temp, y_test = train_test_split(features, targets, test_size=0.15, random_state=42)
# Split 2: 70% Train, 15% Val (which is ~17.6% of the 85% chunk)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42)

print(f"Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scale Data (Fit ONLY on Train to prevent leakage)
scaler_x = StandardScaler().fit(X_train.values)
scaler_y = StandardScaler().fit(y_train.values)

X_train_s = scaler_x.transform(X_train.values)
X_val_s = scaler_x.transform(X_val.values)
y_train_s = scaler_y.transform(y_train.values)
y_val_s = scaler_y.transform(y_val.values)

# Save artifacts for testing
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y, 'input_dim': input_dim}, f)

# Save the exact unscaled test data so test.py doesn't have to re-process everything
test_data = {
    'X_test_scaled': scaler_x.transform(X_test.values),
    'X_test_unscaled': X_test,
    'y_test_loc': y_test[['source_x', 'source_y']],
    'y_test_int': np.expm1(y_test['I_0_log']) # Convert log back to raw for final testing metrics
}
with open('artifacts/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

# ==========================================
# 3. PYTORCH DATALOADERS
# ==========================================
batch_size = 512
train_dataset = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train_s, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
model = RadiationMLP(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
criterion = nn.MSELoss()

epochs = 300
best_val_loss = float('inf')
print("\nStarting Training...")
for epoch in range(1, epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    
    train_mse = train_loss / len(X_train)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            
    val_mse = val_loss / len(X_val)
    scheduler.step(val_mse)

    # Save Best Model
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        torch.save(model.state_dict(), 'artifacts/best_mlp.pth')

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Best: {best_val_loss:.4f}")

print("\nTraining Complete! Artifacts saved in 'artifacts/' folder.")