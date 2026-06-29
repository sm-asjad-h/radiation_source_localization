from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
def train_loc_models(X_train, y_train, X_test):
    print("Training Location Models...")
    dt_model_loc = DecisionTreeRegressor(random_state=42)
    rf_model_loc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model_loc = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
    svm_model_loc = MultiOutputRegressor(SVR(kernel='rbf'))

    dt_model_loc.fit(X_train, y_train)
    rf_model_loc.fit(X_train, y_train)
    xgb_model_loc.fit(X_train, y_train)
    svm_model_loc.fit(X_train, y_train)

    models_preds = {
        "Decision Tree": pd.DataFrame(dt_model_loc.predict(X_test), columns=['source_x', 'source_y'], index=X_test.index),
        "Random Forest": pd.DataFrame(rf_model_loc.predict(X_test), columns=['source_x', 'source_y'], index=X_test.index),
        "XGBoost": pd.DataFrame(xgb_model_loc.predict(X_test), columns=['source_x', 'source_y'], index=X_test.index),
        "SVM": pd.DataFrame(svm_model_loc.predict(X_test), columns=['source_x', 'source_y'], index=X_test.index)
    }
    return models_preds
def train_int_models(X_train, y_train, X_test):
    print("Training Intensity Models...")
    dt_model_intensity = DecisionTreeRegressor(random_state=42)
    rf_model_intensity = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model_intensity = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    svm_model_intensity = SVR(kernel='rbf')

    dt_model_intensity.fit(X_train, y_train)
    rf_model_intensity.fit(X_train, y_train)
    xgb_model_intensity.fit(X_train, y_train)
    svm_model_intensity.fit(X_train, y_train)

    models_preds = {
        "Decision Tree Intensity": dt_model_intensity.predict(X_test),
        "Random Forest Intensity": rf_model_intensity.predict(X_test),
        "XGBoost Intensity": xgb_model_intensity.predict(X_test),
        "SVM Intensity": svm_model_intensity.predict(X_test)
    }
    return models_preds
def train_int_models_log(X_train, y_train, X_test):
    print("Training Intensity Models...")
    dt_model_intensity = DecisionTreeRegressor(random_state=42)
    rf_model_intensity = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model_intensity = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
    svm_model_intensity = SVR(kernel='rbf')

    y_train_log = np.log(y_train)

    dt_model_intensity.fit(X_train, y_train_log)
    rf_model_intensity.fit(X_train, y_train_log)
    xgb_model_intensity.fit(X_train, y_train_log)
    svm_model_intensity.fit(X_train, y_train_log)

    models_preds = {
        "Decision Tree Intensity": pd.Series(np.exp(dt_model_intensity.predict(X_test)), index=X_test.index),
        "Random Forest Intensity": pd.Series(np.exp(rf_model_intensity.predict(X_test)), index=X_test.index),
        "XGBoost Intensity": pd.Series(np.exp(xgb_model_intensity.predict(X_test)), index=X_test.index),
        "SVM Intensity": pd.Series(np.exp(svm_model_intensity.predict(X_test)), index=X_test.index)
    }
    return models_preds