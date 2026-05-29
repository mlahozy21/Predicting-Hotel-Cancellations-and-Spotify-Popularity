# train_meta_model.py

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Importer nos configurations
import config

def train_l1_manager():
    """
    Load the L1 meta-features and train the final L1 meta-model.
    """
    print("--- Training the L1 meta-model... ---")

    # --- 1. Load the meta-features (L1 cache) ---
    print("  Step 1/3: loading meta-features (X_meta_train) and targets (y_full)...")
    try:
        X_meta_train = np.load(config.META_TRAIN_PATH)
        y_full = np.load(config.Y_FULL_PATH)
    except FileNotFoundError as e:
        print(f"Error: L1 cache file not found. {e}")
        print("Run 'train_base_models.py' first to generate the meta-features.")
        return

    print(f"  Meta-features loaded: X_meta_train shape = {X_meta_train.shape}")
    print(f"  Targets loaded: y_full shape = {y_full.shape}")

    # --- 2. Define the L1 meta-model ---
    print("  Step 2/3: initialising the L1 meta-model (LogisticRegression)...")
    final_estimator = LogisticRegression(**config.final_estimator_params)

    # --- 3. Train and save the L1 meta-model ---
    print("  Step 3/3: training and saving the meta-model...")
    final_estimator.fit(X_meta_train, y_full)
    
    # Save the final model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(final_estimator, config.META_MODEL_PATH)
    
    print("\n--- L1 training done ---")
    print(f"Final L1 meta-model saved to:")
    print(f"{config.META_MODEL_PATH}")

if __name__ == "__main__":
    train_l1_manager()