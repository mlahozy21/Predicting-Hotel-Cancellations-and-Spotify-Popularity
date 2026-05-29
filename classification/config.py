# config.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Global constants
RANDOM_STATE = 42
N_SPLITS = 5  # number of folds for OOF stacking

# Data paths
# Set DATA_DIR to the folder that holds the challenge data
# (e.g. "../data/" if the data is one level up)
DATA_DIR = "data/" 
TRAIN_FILE = DATA_DIR + "train_data2.csv"
TEST_FILE = DATA_DIR + "test_data2.csv"
SUBMISSION_FILE = DATA_DIR + "submission2.csv"

# Model and object paths
# Directory to store the models
MODEL_DIR = "models_stacking/"

# Base models (L0)
RF_MODEL_PATH = MODEL_DIR + "stack_rf_base.joblib"
LGBM_MODEL_PATH = MODEL_DIR + "stack_lgbm_base.joblib"
CAT_MODEL_PATH = MODEL_DIR + "stack_cat_base.joblib"

# Final model (L1)
META_MODEL_PATH = MODEL_DIR + "stack_meta_model.joblib"

# Preprocessors and encoders
PREPROCESSOR_A_PATH = MODEL_DIR + "stack_preprocessor_A_rf.joblib"
LE_Y_PATH = MODEL_DIR + "stack_label_encoder_y.joblib"

# Meta-features (cache)
CACHE_DIR = "cache_stacking/"
META_TRAIN_PATH = CACHE_DIR + "meta_train.npy"
META_TEST_PATH = CACHE_DIR + "meta_test.npy"
Y_FULL_PATH = CACHE_DIR + "y_full.npy"
TEST_IDS_PATH = CACHE_DIR + "test_ids.joblib"


# Model hyperparameters

# 1. RandomForest params (pipeline)
rf_params = {
    'n_estimators': 400, 
    'random_state': RANDOM_STATE, 
    'class_weight': 'balanced', 
    'n_jobs': -1
}

# 2. LightGBM params
lgbm_king_params = {
    'n_estimators': 3000, 
    'learning_rate': 0.03, 
    'num_leaves': 35,
    'min_child_samples': 100, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5, 
    'reg_lambda': 0.5,
    'class_weight': {0: 1.0, 1: 1.0, 2: 8.0},
    'random_state': RANDOM_STATE, 
    'n_jobs': -1
}
# LGBM callbacks
lgbm_callbacks = [lgb.early_stopping(100, verbose=False)]

# 3. CatBoost params
cat_class_weights = [1.0, 1.0, 0.7]
catboost_challenger_params = {
    'loss_function': 'MultiClassOneVsAll', 
    'eval_metric': 'TotalF1:average=Macro',
    'iterations': 20000, 
    'learning_rate': 0.05, 
    'depth': 8, 
    'l2_leaf_reg': 2.0,
    'early_stopping_rounds': 1000, 
    'bootstrap_type': 'No', 
    'rsm': 1.0,
    'random_strength': 0.0, 
    'border_count': 254, 
    'auto_class_weights': None,
    'class_weights': cat_class_weights, 
    'one_hot_max_size': 50,
    'max_ctr_complexity': 4, 
    'random_state': RANDOM_STATE, 
    'verbose': False,
}

# 4. L1 meta-model params
final_estimator_params = {
    'max_iter': 1000, 
    'random_state': RANDOM_STATE
}