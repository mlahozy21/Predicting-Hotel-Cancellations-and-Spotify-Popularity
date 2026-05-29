# data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import os

# Import our config and feature-engineering helpers
import config
import feature_engineering as fe

def basic_clean(df, is_train=True):
    """Basic cleaning: drop unused columns and filter invalid rows."""
    df_clean = df.drop(['Unnamed: 0', 'row_id'], axis=1, errors='ignore')
    
    if is_train:
        # Apply filters only on the training set
        df_clean = df_clean[(df_clean['adults'] > 0) | (df_clean['children'] > 0) | (df_clean['babies'] > 0)]
        df_clean = df_clean[df_clean['adr'] < 5000].copy()
        
    return df_clean

def load_and_preprocess_data():
    """
    Load the raw data, clean it, and build the 3 data pipelines (A, B, C),
    one per base model (RF, LGBM, CatBoost).
    """
    print("--- Step: loading and basic cleaning ---")
    try:
        train_df_raw = pd.read_csv(config.TRAIN_FILE)
        test_df_raw = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error: data file not found. {e}")
        return None

    # Keep the test IDs for the final submission
    test_ids = test_df_raw['row_id']
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    joblib.dump(test_ids, config.TEST_IDS_PATH)

    # Basic cleaning
    train_df = basic_clean(train_df_raw, is_train=True)
    test_df_cleaned = basic_clean(test_df_raw, is_train=False)

    # Encode the target (y)
    le_y = LabelEncoder()
    y_full = le_y.fit_transform(train_df['reservation_status'].values)
    
    # Save the encoder for the final prediction
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(le_y, config.LE_Y_PATH)
    
    X_full_base = train_df.drop('reservation_status', axis=1)
    
    print(f"Data loaded. {len(y_full)} training samples.")
    
    # --- Build the 3 specialised datasets ---

    # --- 3.1 Dataset A: for RF (one-hot + scaler) ---
    print("--- Preparing dataset A (RF)... ---")
    X_A_full_raw = X_full_base.copy()
    X_A_test_raw = test_df_cleaned.copy()
    
    # Align columns
    X_A_full, X_A_test = X_A_full_raw.align(X_A_test_raw, join='inner', axis=1)
    
    numerical_cols_A = X_A_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_A = X_A_full.select_dtypes(include='object').columns.tolist()

    # Define the preprocessor
    preprocessor_ohe = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols_A),
            ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical_cols_A)
        ],
        remainder='passthrough'
    )
    # NOTE: this preprocessor is wrapped in a pipeline with the RF in the training script
    # Saved here for the OOF training
    joblib.dump(preprocessor_ohe, config.PREPROCESSOR_A_PATH)
    

    # --- 3.2 Dataset B: for LGBM (cyclical FE + label encoding) ---
    print("--- Preparing dataset B (LGBM)... ---")
    X_B_full_raw = fe.create_features_base_plus_cyclical(X_full_base)
    X_B_test_raw = fe.create_features_base_plus_cyclical(test_df_cleaned)
    
    X_B_full_le, X_B_test_le = X_B_full_raw.align(X_B_test_raw, join='inner', axis=1)
    
    X_B_full_le = X_B_full_le.copy()
    X_B_test_le = X_B_test_le.copy()
    
    numerical_cols_B = X_B_full_le.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_B = X_B_full_le.select_dtypes(include='object').columns.tolist()
    
    # Simple manual imputation (for label encoding)
    for col in numerical_cols_B:
        median_val = X_B_full_le[col].median()
        X_B_full_le[col] = X_B_full_le[col].fillna(median_val)
        X_B_test_le[col] = X_B_test_le[col].fillna(median_val)
        
    for col in categorical_cols_B:
        mode_series = X_B_full_le[col].mode()
        mode_val = "Unknown" if mode_series.empty else mode_series[0]
        X_B_full_le[col] = X_B_full_le[col].fillna(mode_val)
        X_B_test_le[col] = X_B_test_le[col].fillna(mode_val)
        
        # Label encoding
        le_col = LabelEncoder()
        combined_series = pd.concat([X_B_full_le[col], X_B_test_le[col]]).astype(str).unique()
        le_col.fit(combined_series)
        
        X_B_full_le[col] = le_col.transform(X_B_full_le[col].astype(str))
        X_B_test_le[col] = le_col.transform(X_B_test_le[col].astype(str))
        
    # Categorical feature names for LGBM
    categorical_features_names_B = categorical_cols_B.copy()
    if 'arrival_date_month' not in categorical_features_names_B:
         categorical_features_names_B.append('arrival_date_month')  # object column, but aligned
         
    # 'arrival_date_month' was label-encoded, so it is fine
    # Keep only columns that actually exist
    categorical_features_names_B = [col for col in categorical_features_names_B if col in X_B_full_le.columns]


    # --- 3.3 Dataset C: for CatBoost (interaction FE, no encoding) ---
    print("--- Preparing dataset C (CatBoost)... ---")
    X_C_full_raw = fe.create_features_catboost(X_full_base)
    X_C_test_raw = fe.create_features_catboost(test_df_cleaned)
    
    X_C_full, X_C_test = X_C_full_raw.align(X_C_test_raw, join='inner', axis=1)

    numerical_cols_C = X_C_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_C = X_C_full.select_dtypes(include='object').columns.tolist()

    # Simple imputation (CatBoost handles NaNs, but 'Missing' is more explicit)
    for col in numerical_cols_C:
        median_val = X_C_full[col].median()
        X_C_full[col] = X_C_full[col].fillna(median_val)
        X_C_test[col] = X_C_test[col].fillna(median_val)
        
    for col in categorical_cols_C:
        X_C_full[col] = X_C_full[col].fillna("Missing")
        X_C_test[col] = X_C_test[col].fillna("Missing")

    # Categorical feature names for CatBoost
    categorical_features_names_C = [col for col in categorical_cols_C if col in X_C_full.columns]

    print("--- Preprocessing done. ---")
    
    # Return all datasets and the info needed downstream
    datasets = {
        'A_rf': (X_A_full, X_A_test),
        'B_lgbm': (X_B_full_le, X_B_test_le, categorical_features_names_B),
        'C_cat': (X_C_full, X_C_test, categorical_features_names_C)
    }
    
    return datasets, y_full

if __name__ == "__main__":
    # Standalone test of this module
    print("Testing the data_loader module...")
    datasets, y_full = load_and_preprocess_data()
    if datasets:
        print("\n--- Data shapes ---")
        print(f"y_full: {y_full.shape}")
        
        X_A_full, X_A_test = datasets['A_rf']
        print(f"Dataset A (RF): Train={X_A_full.shape}, Test={X_A_test.shape}")
        
        X_B_full, X_B_test, cats_B = datasets['B_lgbm']
        print(f"Dataset B (LGBM): Train={X_B_full.shape}, Test={X_B_test.shape}")
        print(f"   LGBM categoricals: {cats_B}")
        
        X_C_full, X_C_test, cats_C = datasets['C_cat']
        print(f"Dataset C (CatBoost): Train={X_C_full.shape}, Test={X_C_test.shape}")
        print(f"   CatBoost categoricals: {cats_C}")