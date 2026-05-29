# Challenge 1 — Hotel booking status (classification)

Predict the final status of a hotel reservation among three classes: **check-out** (0),
**cancel** (1), **no-show** (2). The solution is a **stacked ensemble**.

## Pipeline

1. `train_base_models.py` — trains the three base (L0) models with an out-of-fold (OOF)
   scheme: **RandomForest**, **LGBMClassifier** and **CatBoostClassifier**. Each model has its
   own preprocessing (see `data_loader.py` / `feature_engineering.py`) and produces OOF
   predictions used as meta-features.
2. `train_meta_model.py` — trains the final (L1) meta-model, a **Logistic Regression** that
   takes the three base models' predictions as input.
3. `predict_stacking.py` — generates the final predictions and writes `submission`.

`config.py` defines global parameters (paths, hyperparameters). Set `DATA_DIR` to the folder
that contains the challenge data before running.
