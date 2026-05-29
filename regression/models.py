import lightgbm as lgbm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import config as config
import joblib 
import os
import pandas as pd


# Initial LightGBM to find the best n_estimators, then the final LightGBM
def find_optimal_estimators_lgbm(X_train, y_train, X_test, y_test):
    """Use early stopping to find the optimal n_estimators for LightGBM."""
    print("Searching for the optimal n_estimators for LightGBM...")
    
    eval_model = LGBMRegressor(**config.LGBM_EVAL_PARAMS)
    
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgbm.early_stopping(100, verbose=False)],
        categorical_feature=config.CAT_VARIABLES
    )
    
    optimal_n = eval_model.best_iteration_
    if optimal_n is None or optimal_n < 50:
        optimal_n = 50 
        
    print(f"Optimal LGBM n_estimators: {optimal_n}")
    
    y_pred = eval_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"LGBM (eval) R2: {r2:.4f} | MSE: {mse:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(eval_model, config.LGBMinicial_MODEL_PATH)
    return optimal_n

def train_final_lgbm(X_full, y_full, optimal_n_estimators):
    """Train the final LGBM on all the data and save it."""
    print("Training the final LGBM...")

    final_params = config.LGBM_FINAL_PARAMS.copy()
    final_params['n_estimators'] = optimal_n_estimators
    
    lgbm_model = LGBMRegressor(**final_params)
    
    lgbm_model.fit(
        X_full,
        y_full,
        categorical_feature=config.CAT_VARIABLES
    )
    
    print("Final LGBM trained.")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm_model, config.LGBM_MODEL_PATH)
    print(f"Model saved to: {config.LGBM_MODEL_PATH}")

    return lgbm_model

# Initial CatBoost to find the best iterations, then the final CatBoost

def find_optimal_iterations_catboost(X_train, y_train, X_test, y_test):
    """Use early stopping to find the optimal number of iterations for CatBoost."""
    print("Searching for the optimal number of iterations for CatBoost...")
    
    cat_model_eval = CatBoostRegressor(**config.CATBOOST_PARAMS)
    
    cat_model_eval.fit(
        X_train,
        y_train,
        cat_features=config.CAT_VARIABLES,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100
    )
    
    optimal_n = cat_model_eval.get_best_iteration()
    if optimal_n is None or optimal_n < 50:
        optimal_n = 50
        
    print(f"Optimal CatBoost iterations: {optimal_n}")
    
    y_pred = cat_model_eval.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"CatBoost (eval) R2: {r2:.4f} | MSE: {mse:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(cat_model_eval, config.CATBOOSTinicial_MODEL_PATH)
    return optimal_n

def train_final_catboost(X_full, y_full, optimal_n_iterations):
    """Train the final CatBoost on all the data and save it."""
    print("Training the final CatBoost...")

    final_params = config.CATBOOST_PARAMS.copy()
    final_params['iterations'] = optimal_n_iterations
    final_params['verbose'] = 0 
    final_params.pop('early_stopping_rounds', None) 

    cat_model_final = CatBoostRegressor(**final_params)
    
    cat_model_final.fit(
        X_full,
        y_full,
        cat_features=config.CAT_VARIABLES
    )
    
    print("Final CatBoost trained.")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(cat_model_final, config.CATBOOST_MODEL_PATH)
    print(f"Model saved to: {config.CATBOOST_MODEL_PATH}")

    return cat_model_final



def train_kernel(X_train, y_train, X_test, y_test, gamma: float = 1.0):

    """
    Apply an RBF (Gaussian) kernel approximation followed by a LinearSVR.

    Args:
        X_train, X_test, y_train, y_test: datasets.
        gamma (float): RBF kernel parameter.
    """
    
    print("Training RBF kernel approximation + LinearSVR (regression)...")

    # 1. Pipeline: RBF approximation -> linear regressor
    rbf_feature = RBFSampler(gamma='scale', random_state=config.RANDOM_STATE, n_components=2000)
    
    # LinearSVR is a fast large-scale linear regressor.
    linear_svm = LinearSVR(random_state=config.RANDOM_STATE)


    
    pipeline = Pipeline([
        ('rbf_sampler', rbf_feature),      # kernel approximation
        ('linear_svc', linear_svm)         # fast linear model
    ])

    # 2. Training
    pipeline.fit(X_train, y_train)

    # 3. Prediction
    y_pred = pipeline.predict(X_test)

    # 4. Evaluation (R2)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Kernel model (RBFSampler + LinearSVR) R2: {r2:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, config.KERNEL_MODEL_PATH)
    print(f"Model saved to: {config.KERNEL_MODEL_PATH}")
    
    return pipeline
