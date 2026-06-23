from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import pandas as pd
def train_loc_models(X_train, y_train, X_test):
    print("Training Location Models...")
    dt_model_loc = DecisionTreeRegressor(random_state=42)
    rf_model_loc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model_loc = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    svm_model_loc = MultiOutputRegressor(SVR(kernel='rbf'))

    dt_model_loc.fit(X_train, y_train)
    rf_model_loc.fit(X_train, y_train)
    xgb_model_loc.fit(X_train, y_train)
    svm_model_loc.fit(X_train, y_train)

    models_preds = {
        "Decision Tree": pd.DataFrame(dt_model_loc.predict(X_test), columns=['source_x', 'source_y']),
        "Random Forest": pd.DataFrame(rf_model_loc.predict(X_test), columns=['source_x', 'source_y']),
        "XGBoost": pd.DataFrame(xgb_model_loc.predict(X_test), columns=['source_x', 'source_y']),
        "SVM": pd.DataFrame(svm_model_loc.predict(X_test), columns=['source_x', 'source_y'])
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
