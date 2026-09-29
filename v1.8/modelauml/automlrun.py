import os
import sys
import pandas as pd
from autogluon.tabular import TabularPredictor
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["RAY_DASHBOARD_ENABLED"] = "0"
CURRENT_DIR = os.getcwd()

DATASET_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..')) 
DATASET_PATH = os.path.join(DATASET_ROOT, 'dataset', 'dataset.csv')

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

import common_function.feature_engineering_functions as fe
import common_function.split_and_scale as ss

print("Loading Data")
df = pd.read_csv(DATASET_PATH)

df = fe.sort_detectors_by_intensity(df)
df = fe.add_sensor_distances(df)
df = fe.add_relative_coordinates(df)
df = fe.add_intensity_ratios(df)
df = fe.add_weighted_centroid(df)

X_train_full, X_test, y_train_loc, y_test_loc, y_train_int, y_test_int, _ = ss.split_and_scale_data(df, scale=False)

if isinstance(y_train_int, pd.Series): 
    y_train_int = y_train_int.to_frame()

all_targets_df = pd.concat([y_train_loc, y_train_int], axis=1)
targets = list(all_targets_df.columns) 
print(f"Targets identified for training: {targets}")

# 48 hours (172800 seconds) per target
TIME_LIMIT = 36000 

for target in targets:
    print(f"\n" + "="*50)
    print(f" STARTING TRAINING: {target}")
    print("="*50)
    
    current_train_data = pd.concat([X_train_full, all_targets_df[target]], axis=1)
    
    save_path = os.path.join(CURRENT_DIR, f'AG_Models/{target}')
    
    predictor = TabularPredictor(
        label=target, 
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path=save_path
    ).fit(
        current_train_data,
        presets='best_quality',           
        time_limit=TIME_LIMIT,
        num_gpus=1,                      
        ag_args_fit={'num_gpus': 1}       
    )
    
    print(f"\n Leaderboard for {target} saved.")
    
print("\n Training complete! All models saved to AG_Models.")
