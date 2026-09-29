import os
import sys
import pandas as pd
import numpy as np

# CRITICAL FOR SERVER: Prevents display crash when running over SSH
import matplotlib
matplotlib.use('Agg') 

from autogluon.tabular import TabularPredictor

# ==========================================
# 1. DYNAMIC FOLDER ROUTING
# ==========================================
CURRENT_DIR = os.getcwd()
DATASET_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATASET_PATH = os.path.join(DATASET_ROOT, 'dataset', 'dataset.csv')
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

import common_function.feature_engineering_functions as fe
import common_function.split_and_scale as ss
import common_function.save_visualizations as vnz  # Your visualization file

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading Test Data...")
df = pd.read_csv(DATASET_PATH)
df = fe.sort_detectors_by_intensity(df)
df = fe.add_sensor_distances(df)
df = fe.add_relative_coordinates(df)
df = fe.add_intensity_ratios(df)
df = fe.add_weighted_centroid(df)

# Scale must be False, just like training!
_, X_test, _, y_test_loc, _, y_test_int, X_test_unscaled = ss.split_and_scale_data(df, scale=False)

if isinstance(y_test_int, pd.Series): y_test_int = y_test_int.to_frame()
test_data = pd.concat([X_test, y_test_loc, y_test_int], axis=1)
targets = list(y_test_loc.columns) + list(y_test_int.columns)

# ==========================================
# 3. LOAD MODELS & PREDICT
# ==========================================
MODEL_DIR = os.path.join(CURRENT_DIR, 'AG_Models')
predictions = {}

for target in targets:
    print(f"Loading model for {target}...")
    save_path = os.path.join(MODEL_DIR, target)
    
    # Loads the trained model directly from the folder
    predictor = TabularPredictor.load(save_path)
    
    # Predict (Must drop target columns from test data so it doesn't cheat)
    preds = predictor.predict(test_data.drop(columns=targets))
    predictions[target] = preds.to_numpy()

# ==========================================
# 4. FORMAT AND VISUALIZE (CORRECTED)
# ==========================================
print("\nFormatting predictions...")

# Fix: Change 'source_int' to 'I_0'
preds_loc = np.column_stack((predictions['source_x'], predictions['source_y']))
preds_int = predictions['I_0'].flatten() # Added .flatten() to ensure it is 1D

# Ensure y_test_int is also 1D for the visualizer
y_test_int_flat = y_test_int.to_numpy().flatten()

models_preds_loc = {'AutoGluon (Test Run)': preds_loc}
models_preds_int = {'AutoGluon (Test Run)': preds_int}

print("plotting...")
vnz.visualize_loc_results(models_preds_loc, y_test_loc, X_test_unscaled)
# Pass the flattened intensity target
vnz.visualize_int_results(models_preds_int, y_test_int_flat, y_test_loc, X_test_unscaled)

print("\n All plots generated and saved!")
