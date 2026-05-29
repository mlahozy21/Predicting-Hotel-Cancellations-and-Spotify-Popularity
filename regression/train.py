import load_data as ld
import pandas as pd
import models as m
import os
import warnings
import config
def training():  # train the base models if they have not been trained yet
    # Load data
    X_train, X_test, y_train, y_test = ld.load_train_data(use_cache=True)
    print(f"Data (cache 1) loaded: {X_train.shape[0]} training rows.")
    # Combine data for the final training (LGBM/CatBoost)
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)  # same target for all models
    # Models
    print("\n--- LightGBM ---")
    if os.path.exists(config.LGBM_MODEL_PATH):
        print("LGBM model already exists. Skipping.")
    else:
        print("Training LightGBM...")
        optimal_n_lgbm = m.find_optimal_estimators_lgbm(X_train, y_train, X_test, y_test)
        print("Training the final LGBM...")
        m.train_final_lgbm(X_full, y_full, optimal_n_lgbm)

    print("\n--- CatBoost ---")
    if os.path.exists(config.CATBOOST_MODEL_PATH):
        print("CatBoost model already exists. Skipping.")
    else:
        print("Training CatBoost...")
        optimal_n_cat = m.find_optimal_iterations_catboost(X_train, y_train, X_test, y_test)
        print("Training the final CatBoost...")
        m.train_final_catboost(X_full, y_full, optimal_n_cat)
    print("\n--- Kernel model ---")
    X_train, X_test= ld.load_train_data_nocategories(use_cache=True)
    if os.path.exists(config.KERNEL_MODEL_PATH):
        print("Kernel model already exists. Skipping.")
    else:
        print("Training the kernel model...")
        m.train_kernel(X_train, y_train,X_test,  y_test)
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    training()
