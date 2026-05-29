# train_stack.py
import pandas as pd
import joblib
import warnings
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import config
import load_data as ld
import numpy as np

def train_meta_model():
    """
    Train a stacking meta-model from the base models' predictions on the
    validation set.
    """
    print("Training the stacking meta-model...")
    
    # --- 1. Load validation data (caches 1 and 2) ---
    try:
        print("Loading data (caches 1 and 2)...")
        X_train, X_test, y_train, y_test= ld.load_train_data(use_cache=True)
        print("Validation data loaded.")
    except FileNotFoundError:
        print("Error: data caches not found.")
        return

    # --- 2. Load base models ---
    try:
        model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        model_cat = joblib.load(config.CATBOOST_MODEL_PATH)
        X_train_nocat, X_test_nocat= ld.load_train_data_nocategories(use_cache=True)
        model_kernel = joblib.load(config.KERNEL_MODEL_PATH)
        print("Base models loaded.")
    except FileNotFoundError as e:
        print(f"Error: a base-model file is missing: {e}")
        return

    # --- 3. Base-model predictions on the validation set ---
    print("Generating base-model predictions...")



    pred_lgbm_val = model_lgbm.predict(X_test).ravel()
    pred_cat_val = model_cat.predict(X_test).ravel()
    pred_kernel_val = model_kernel.predict(X_test_nocat).ravel()


# -----------------------------------
    # --- 4. Build the meta-model training set ---
    X_meta_train = pd.DataFrame({
        'lgbm_pred': pred_lgbm_val,
        'cat_pred': pred_cat_val,
        'kernel_pred': pred_kernel_val
    })
    
    # The target is the real y
    y_meta_train = y_test
    
    # --- Meta-model: LGBMRegressor ---
    

    meta_model = LGBMRegressor(
        n_estimators=75,       # relatively few trees
        num_leaves=10,         # low complexity
        learning_rate=0.05,
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    
    meta_model.fit(X_meta_train, y_meta_train)

    # Evaluate the meta-model for reference
    print("Meta-model (LGBMRegressor) trained.")

    y_pred_meta = meta_model.predict(X_meta_train)

    meta_r2 = r2_score(y_meta_train, y_pred_meta)
    print(f"\nMeta-model R2 (validation): {meta_r2:.6f}")
    
    # --- Save the meta-model ---
    joblib.dump(meta_model, config.META_MODEL_PATH)
    print(f"Meta-model saved to: {config.META_MODEL_PATH}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train_meta_model()
