# Challenge 2 — Spotify popularity (regression)

Predict the Spotify popularity (0–100) of a track from its audio features and metadata.
The solution is a **stacked ensemble**.

## Pipeline

1. `train.py` — trains the three base (L0) models: **CatBoostRegressor**, **LGBMRegressor**
   and an **RBF-kernel approximation + LinearSVR**. Trained models are cached on disk.
2. `train_stack.py` — trains the final (L1) meta-model, an **LGBMRegressor** that takes the
   three base models' predictions as input.
3. `predict_stack.py` — generates the final predictions and writes `submission.csv`.

`cache_functions.py` caches the preprocessed data and models so reruns are fast. `config.py`
defines global parameters; set `DATA_DIR` to the folder that contains the challenge data.
