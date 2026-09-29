import os
import sys
import json
import time
import optuna
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
HP_TUNING_DIR = os.path.dirname(CURRENT_DIR)
V1_4_DIR = os.path.dirname(HP_TUNING_DIR)
PROJECT_ROOT = os.path.dirname(V1_4_DIR)

sys.path.append(PROJECT_ROOT)
DATASET_PATH = os.path.join(V1_4_DIR, 'dataset', 'dataset.csv')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')

os.makedirs(CONFIG_DIR, exist_ok=True)

import common_function.feature_engineering_functions as fe
import common_function.split_and_scale as ss

print(f" Tuning Script Running In: {CURRENT_DIR}")
print(f" Loading Dataset From: {DATASET_PATH}")
print(f" Config Will Be Saved To: {CONFIG_FILE}\n")

df_raw = pd.read_csv(DATASET_PATH)

df = fe.sort_detectors_by_intensity(df_raw)
df = fe.add_sensor_distances(df)
df = fe.add_relative_coordinates(df)
df = fe.add_intensity_ratios(df)
df = fe.add_weighted_centroid(df)
df = fe.apply_log(df, drop=False)

(X_train, X_test, 
 y_train_loc, y_test_loc, 
 y_train_int, y_test_int, 
 X_test_unscaled) = ss.split_and_scale_data(df, scale=True)

y_train_int_log = np.log(y_train_int)

class EarlyStoppingCallback:
    def __init__(self, patience=30):
        self.patience = patience
        self.stagnation_count = 0
        self.best_score = float('inf')

    def __call__(self, study, trial):
        current_best = study.best_value
        if current_best < self.best_score:
            self.best_score = current_best
            self.stagnation_count = 0  
        else:
            self.stagnation_count += 1 

        if self.stagnation_count >= self.patience:
            print(f"      Early Stopping: No improvement for {self.patience} trials.")
            study.stop()

def tune_with_optuna(model_name, X, y, is_location, n_trials=500):
    def objective(trial):
        if model_name == "Decision Tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)
            }
            base_model = DecisionTreeRegressor(random_state=42, **params)
            
        elif model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
            
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
            }
            base_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, **params)
            
        elif model_name == "SVM":
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e4, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"])
            }
            base_model = SVR(**params)

        if is_location and model_name in ["XGBoost", "SVM"]:
            model = MultiOutputRegressor(base_model)
        else:
            model = base_model

        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
        
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    
    early_stopper = EarlyStoppingCallback(patience=30)
    
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stopper])
    
    return study.best_params

models_to_tune = ["Random Forest"]
best_configurations = {"Location": {}, "Intensity": {}}

print("\n Starting tuning with optuna")
for model in models_to_tune:
    print(f" Tuning {model} ")
    start_time = time.time()
    
    print(f"    Location params...")
    loc_params = tune_with_optuna(model, X_train, y_train_loc, is_location=True)
    best_configurations["Location"][model] = loc_params
    print(f"      Location params: {loc_params}")
    
    print(f"   Intensity params")
    int_params = tune_with_optuna(model, X_train, y_train_int_log, is_location=False)
    best_configurations["Intensity"][model] = int_params
    print(f"      Intensity params: {int_params}")
    
    elapsed = (time.time() - start_time) / 60
    print(f"   {model} fully tuned in {elapsed:.2f} mins.\n")

with open(CONFIG_FILE, 'w') as f:
    json.dump(best_configurations, f, indent=4)

print(f"tuning done")
print(f"   {CONFIG_FILE}")
