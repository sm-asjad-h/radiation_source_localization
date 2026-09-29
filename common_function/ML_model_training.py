import time
import pandas as pd
import numpy as np

def train_generic_ml_model(model_name, model, X_train, y_train, X_test, is_location=True, is_log_target=False):
    print(f" Training {model_name}")
    start_time = time.time()
    if is_log_target and not is_location:
        y_train_fit = np.log(y_train)
    else:
        y_train_fit = y_train

    model.fit(X_train, y_train_fit)

    preds = model.predict(X_test)

    if is_log_target and not is_location:
        preds = np.exp(preds)

    if is_location:
        preds_formatted = pd.DataFrame(preds, columns=['source_x', 'source_y'], index=X_test.index)
    else:
        preds_formatted = pd.Series(preds, name='I_0', index=X_test.index)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")

    return preds_formatted, model