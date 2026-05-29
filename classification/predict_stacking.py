# predict_stacking.py

import pandas as pd
import numpy as np
import joblib
import os

# Importer nos configurations
import config

def generate_stacking_submission():
    """
    Load the L1 meta-model and the L1 test meta-features, predict the final
    results and write the submission file.
    """
    print("--- Stacking prediction (L1)... ---")

    # --- 1. Load the required components ---
    print("  Step 1/4: loading components (L1 meta-model, test meta-features, IDs, encoder)...")
    try:
        # Load the trained L1 meta-model
        meta_model = joblib.load(config.META_MODEL_PATH)
        
        # Load the L0 predictions on the test set (test meta-features)
        X_meta_test = np.load(config.META_TEST_PATH)
        
        # Load the label encoder (to map 0,1,2 back to the labels)
        le_y = joblib.load(config.LE_Y_PATH)
        
        # Load the test IDs for the submission file
        test_ids = joblib.load(config.TEST_IDS_PATH)
        
    except FileNotFoundError as e:
        print(f"Error: missing model or cache file. {e}")
        print("Run 'train_base_models.py' and 'train_meta_model.py' before predicting.")
        return
    
    print(f"  Components loaded. X_meta_test shape = {X_meta_test.shape}")

    # --- 2. Generate the final (encoded) predictions ---
    print("  Step 2/4: generating the final predictions...")
    predictions_encoded = meta_model.predict(X_meta_test)

    # --- 3. Decode the predictions ---
    print("  Step 3/4: decoding the labels...")
    predictions_labels = le_y.inverse_transform(predictions_encoded)

    # --- 4. Build and save the submission file ---
    print(f"  Step 4/4: writing the submission file '{config.SUBMISSION_FILE}'...")
    submission_df = pd.DataFrame({
        'row_id': test_ids, 
        'reservation_status': predictions_labels
    })
    
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)

    print("\n--- Prediction done ---")
    print(f"Submission written successfully: {config.SUBMISSION_FILE}")

if __name__ == "__main__":
    generate_stacking_submission()