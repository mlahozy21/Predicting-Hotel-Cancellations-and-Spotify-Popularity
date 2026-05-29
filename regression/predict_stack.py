# predict_stack.py
import pandas as pd
import joblib
import warnings
import config
import load_data as ld

def run_stacking_prediction():
    """
    Load all base models and the meta-model to generate the final
    stacking prediction.
    """
    print("Running the stacking prediction...")

    # --- 1. Load all models ---
    try:
        model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        model_cat = joblib.load(config.CATBOOST_MODEL_PATH)
        model_kernel=joblib.load(config.KERNEL_MODEL_PATH)
        meta_model = joblib.load(config.META_MODEL_PATH)
        print("All models (base and meta) loaded.")
    except FileNotFoundError as e:
        print(f"Error: a model file is missing: {e}")
        print("Make sure you ran train.py and train_stack.py first.")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- 2. Load raw test data ---
    dftest, row_ids = ld.load_test_data()
    dftest_nocat,_ =ld.load_test_data_nocat()
    print("Test data loaded.")
    # --- 3. Base-model predictions ---
    print("Generating base-model predictions...")
    pred_lgbm = model_lgbm.predict(dftest).ravel()
    pred_cat = model_cat.predict(dftest).ravel()
    pred_kernel = model_kernel.predict(dftest_nocat).ravel()


    
    # --- 4. Build the meta-model test set ---
    # Column order must match train_stack.py
    X_meta_test = pd.DataFrame({
        'lgbm_pred': pred_lgbm,
        'cat_pred': pred_cat,
        'kernel_pred': pred_kernel
    })

    # --- 5. Final prediction ---
    print("Generating the final prediction with the meta-model...")
    y_pred_final = meta_model.predict(X_meta_test)

    # --- 6. Build and save the submission ---
    submission_df = pd.DataFrame({
        'row_id': row_ids,
        config.VARIABLE_TO_PREDICT: y_pred_final
    })
    
    submission_df[config.VARIABLE_TO_PREDICT] = submission_df[config.VARIABLE_TO_PREDICT].clip(0, 100)
    
    # Save the submission
    stack_submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(stack_submission_path, index=False)
    
    print(f"Done. Stacking submission saved to: {stack_submission_path}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_stacking_prediction()
