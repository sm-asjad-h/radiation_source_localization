import os
import sys
import pandas as pd

from xgboost import XGBRegressor
model_class = XGBRegressor
model_params = {"random_state": 42,  "n_estimators": 450,
            "max_depth": 9,
            "learning_rate": 0.06251068599746087,
            "subsample": 0.8312134991545008,"device": "cuda",          # Enables GPU execution
    "tree_method": "hist"}#,
            #"colsample_bytree": 0.7941266337430576}

CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset.csv')
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..','..'))
sys.path.append(PROJECT_ROOT)
import common_function.feature_engineering_functions as fe
import common_function.split_and_scale as ss
import common_function.ML_model_training as mt
import common_function.save_visualizations as vnz

df_raw = pd.read_csv(DATASET_PATH)
df = fe.sort_detectors_by_intensity(df_raw)
df = fe.add_sensor_distances(df)
df = fe.add_relative_coordinates(df)
df = fe.add_intensity_ratios(df)
df = fe.add_weighted_centroid(df)
df = fe.apply_log(df, drop=False)

(X_train, X_test, y_train_loc, y_test_loc, y_train_int, y_test_int, X_test_unscaled) = ss.split_and_scale_data(df, scale=True)

loc_predictions = {}
int_predictions = {}

print(f"\nTRAINING {model_class.__name__}")

loc_model = model_class(**model_params)
loc_predictions["Current_Model"], _ = mt.train_generic_ml_model(
    model_name=f"{model_class.__name__} Location",
    model=loc_model,
    X_train=X_train, y_train=y_train_loc, X_test=X_test, is_location=True
)
model_params = {"random_state": 42,  "n_estimators": 450,
            "max_depth": 8,
            "learning_rate": 0.09035220468742607,
            "subsample": 0.8713077179712}#,
            #"colsample_bytree": 0.764221246232653}

int_model = model_class(**model_params)
int_predictions["Current_Model"], _ = mt.train_generic_ml_model(
    model_name=f"{model_class.__name__} Intensity",
    model=int_model,
    X_train=X_train, y_train=y_train_int, X_test=X_test, 
    is_location=False, is_log_target=True
)
vnz.visualize_loc_results(loc_predictions, y_test_loc, X_test_unscaled)
vnz.visualize_int_results(int_predictions, y_test_int, y_test_loc, X_test_unscaled, SI_min=400.0, SI_max=8000.0)
print(loc_predictions)
print(y_test_loc)
print(int_predictions)
print(y_test_int)