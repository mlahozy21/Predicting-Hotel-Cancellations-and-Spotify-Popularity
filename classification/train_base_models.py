# train_base_models.py

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Import our own modules
import config
import data_loader as dl

# Ignorer les avertissements
warnings.filterwarnings('ignore')

def train_l0_experts():
    """
    Train the 3 level-0 base models (RF, LGBM, CatBoost) with an out-of-fold
    (OOF) scheme to produce the L1 meta-features, then save the final L0
    models and the meta-features.
    """
    
    # --- 1. Load the 3 prepared datasets ---
    print("--- Step 1/4: loading the 3 datasets... ---")
    datasets, y_full = dl.load_and_preprocess_data()
    if datasets is None:
        print("Data loading failed. Aborting.")
        return

    # Unpack the datasets
    X_A_full, X_A_test = datasets['A_rf']
    X_B_full, X_B_test, cats_B = datasets['B_lgbm']
    X_C_full, X_C_test, cats_C = datasets['C_cat']
    
    # Load the preprocessor for the RF pipeline
    preprocessor_rf = joblib.load(config.PREPROCESSOR_A_PATH)
    n_classes = len(np.unique(y_full))

    # --- 2. Define the L0 models and pipelines ---
    print("--- Step 2/4: initialising the L0 models... ---")
    
    # Model 1: RandomForest pipeline
    rf_model = RandomForestClassifier(**config.rf_params)
    rf_pipeline = make_pipeline(preprocessor_rf, rf_model)

    # Model 2: LGBMClassifier
    lgbm_king = lgb.LGBMClassifier(**config.lgbm_king_params)

    # Model 3: CatBoostClassifier
    catboost_challenger = CatBoostClassifier(**config.catboost_challenger_params)

    # --- 3. Out-of-fold training to build X_meta_train ---
    print(f"--- Step 3/4: out-of-fold training (K={config.N_SPLITS})... ---")
    
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    # Containers for the OOF predictions (L1 meta-features)
    meta_train_rf = np.zeros((len(X_A_full), n_classes))
    meta_train_lgbm = np.zeros((len(X_B_full), n_classes))
    meta_train_cat = np.zeros((len(X_C_full), n_classes))

    # Containers for the test predictions (to be averaged)
    meta_test_rf_folds = np.zeros((len(X_A_test), n_classes, config.N_SPLITS))
    meta_test_lgbm_folds = np.zeros((len(X_B_test), n_classes, config.N_SPLITS))
    meta_test_cat_folds = np.zeros((len(X_C_test), n_classes, config.N_SPLITS))

    # Best iterations per fold (LGBM and CatBoost)
    lgbm_best_iters = []
    catboost_best_iters = []

    # Cross-validation loop
    # Note: indices from X_B_full are used as reference,
    # since all datasets share the same length (len(y_full))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_B_full, y_full)):
        print(f"--- Fold {fold + 1}/{config.N_SPLITS} ---")
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # --- Model 1: RandomForest (dataset A) ---
        print("  Training RF...")
        X_A_train_fold, X_A_val_fold = X_A_full.iloc[train_idx], X_A_full.iloc[val_idx]
        rf_clone = clone(rf_pipeline)
        rf_clone.fit(X_A_train_fold, y_train_fold)
        meta_train_rf[val_idx] = rf_clone.predict_proba(X_A_val_fold)
        meta_test_rf_folds[:, :, fold] = rf_clone.predict_proba(X_A_test)

        # --- Model 2: LGBM (dataset B) ---
        print("  Training LGBM...")
        X_B_train_fold, X_B_val_fold = X_B_full.iloc[train_idx], X_B_full.iloc[val_idx]
        lgbm_clone = clone(lgbm_king)
        lgbm_clone.fit(
            X_B_train_fold, y_train_fold,
            eval_set=[(X_B_val_fold, y_val_fold)],
            eval_metric='multi_logloss',
            callbacks=config.lgbm_callbacks,
            categorical_feature=cats_B
        )
        meta_train_lgbm[val_idx] = lgbm_clone.predict_proba(X_B_val_fold)
        meta_test_lgbm_folds[:, :, fold] = lgbm_clone.predict_proba(X_B_test)
        lgbm_best_iters.append(lgbm_clone.best_iteration_ if lgbm_clone.best_iteration_ else config.lgbm_king_params['n_estimators'])

        # --- Model 3: CatBoost (dataset C) ---
        print("  Training CatBoost...")
        X_C_train_fold, X_C_val_fold = X_C_full.iloc[train_idx], X_C_full.iloc[val_idx]
        cat_clone = clone(catboost_challenger)
        cat_clone.fit(
            X_C_train_fold, y_train_fold,
            eval_set=(X_C_val_fold, y_val_fold),
            cat_features=cats_C
        )
        meta_train_cat[val_idx] = cat_clone.predict_proba(X_C_val_fold)
        meta_test_cat_folds[:, :, fold] = cat_clone.predict_proba(X_C_test)

        catboost_best_iters.append(cat_clone.get_best_iteration() if cat_clone.get_best_iteration() else config.catboost_challenger_params['iterations'])
    print("--- OOF L0 training done. ---")

    # --- 4. Finalisation et Sauvegarde ---
    print("--- Step 4/4: finalising and saving L0 models and meta-features... ---")

    # Combine the L1 meta-features for training
    X_meta_train = np.concatenate([meta_train_rf, meta_train_lgbm, meta_train_cat], axis=1)

    # Average the K-fold test predictions
    # Common way to build the test meta-features
    meta_test_rf = np.mean(meta_test_rf_folds, axis=2)
    meta_test_lgbm = np.mean(meta_test_lgbm_folds, axis=2)
    meta_test_cat = np.mean(meta_test_cat_folds, axis=2)
    
    # Combine the L1 meta-features for the test set
    X_meta_test = np.concatenate([meta_test_rf, meta_test_lgbm, meta_test_cat], axis=1)

    # Save the meta-features (L1 cache)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(config.META_TRAIN_PATH, X_meta_train)
    np.save(config.META_TEST_PATH, X_meta_test)
    np.save(config.Y_FULL_PATH, y_full)
    print(f"Meta-features (L1) saved in '{config.CACHE_DIR}'")

    # --- Train and save the final L0 models on 100% of the data ---
    # Reference models trained on the full data


    
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Final RF
    print("  Saving the final RF pipeline...")
    # Train a fresh model on 100% of the data


    print("  Training the final RF on 100% of the data...")
    rf_pipeline_final = clone(rf_pipeline)
    rf_pipeline_final.fit(X_A_full, y_full)
    joblib.dump(rf_pipeline_final, config.RF_MODEL_PATH)

    # 2. Final LGBM
    avg_lgbm_iters = int(np.mean(lgbm_best_iters) * 1.1)  # small buffer
    print(f"  Training the final LGBM ({avg_lgbm_iters} iterations)...")
    lgbm_king_final = clone(lgbm_king)
    lgbm_king_final.set_params(n_estimators=avg_lgbm_iters, early_stopping_round=None)
    lgbm_king_final.fit(X_B_full, y_full, categorical_feature=cats_B)
    joblib.dump(lgbm_king_final, config.LGBM_MODEL_PATH)
    
    # 3. Final CatBoost
    avg_cat_iters = int(np.mean(catboost_best_iters) * 1.1)  # small buffer
    print(f"  Training the final CatBoost ({avg_cat_iters} iterations)...")
    catboost_final = clone(catboost_challenger)
    catboost_final.set_params(iterations=avg_cat_iters, early_stopping_rounds=None)
    catboost_final.fit(X_C_full, y_full, cat_features=cats_C)
    joblib.dump(catboost_final, config.CAT_MODEL_PATH)
    
    print(f"Final L0 models saved in '{config.MODEL_DIR}'")
    print("--- L0 training process done. ---")


if __name__ == "__main__":
    train_l0_experts()