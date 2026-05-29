# Data Challenges — Advanced Supervised Learning

Two end-to-end machine-learning competitions solved with **stacked ensembles**, from the
*Advanced Supervised Learning* course (M2 Mathematics & AI, Université Paris-Saclay, 2025).

| Challenge | Task | Target | Models (base → meta) |
|-----------|------|--------|----------------------|
| **Hotel bookings** (`classification/`) | Multiclass classification | Reservation status: check-out / cancel / no-show | RandomForest + LightGBM + CatBoost → Logistic Regression |
| **Spotify popularity** (`regression/`) | Regression | Track popularity (0–100) | LightGBM + CatBoost + RBF-kernel SVR → LightGBM |

Both use **stacking**: several diverse base models (L0) are trained with out-of-fold
predictions, and a meta-model (L1) learns to combine them.

## Challenge 1 — Hotel booking status (classification)

Cancellations and no-shows cause large losses in the hotel industry. The goal is to predict
the final status of a reservation among three classes: **check-out**, **cancel**, **no-show**.
Three complementary base learners are trained, each on its own preprocessing pipeline
(one-hot + scaling for RF, label-encoding + cyclical calendar features for LGBM, interaction
features for CatBoost), and stacked with a logistic-regression meta-model.

## Challenge 2 — Spotify popularity (regression)

The goal is to predict a track's Spotify popularity (0–100) from its audio features and
metadata. Engineered audio features (e.g. party/calm/feel-good scores) feed LightGBM, CatBoost
and an RBF-kernel SVR, combined by a LightGBM meta-model.

## Installation

```bash
git clone https://github.com/mlahozy21/Predicting-Hotel-Cancellations-and-Spotify-Popularity.git
cd Predicting-Hotel-Cancellations-and-Spotify-Popularity
pip install -r requirements.txt
```

Place the challenge data under `classification/data/` and `regression/data/` (not versioned),
then set `DATA_DIR` in the corresponding `config.py` if needed.

## Usage

Run each challenge from its own folder.

```bash
# Challenge 1 — classification
cd classification
python train_base_models.py     # train the L0 base models (OOF)
python train_meta_model.py      # train the L1 meta-model
python predict_stacking.py      # write submission

# Challenge 2 — regression
cd regression
python train.py                 # train the L0 base models
python train_stack.py           # train the L1 meta-model
python predict_stack.py         # write submission
```

Hyperparameters and paths live in each `config.py`.

## Project structure

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
├── classification/
│   ├── config.py                # paths and hyperparameters
│   ├── data_loader.py           # loading, cleaning, per-model preprocessing
│   ├── feature_engineering.py   # base, cyclical and interaction features
│   ├── train_base_models.py     # L0 training (out-of-fold)
│   ├── train_meta_model.py      # L1 meta-model
│   ├── predict_stacking.py      # final predictions / submission
│   └── README_CLASSIFICATION.md
└── regression/
    ├── config.py                # paths and hyperparameters
    ├── load_data.py             # data loading + caching
    ├── features.py              # feature engineering, preprocessing, clustering
    ├── models.py                # base-model definitions and training
    ├── cache_functions.py       # parquet cache helpers
    ├── train.py                 # L0 training
    ├── train_stack.py           # L1 meta-model
    ├── predict_stack.py         # final predictions / submission
    └── README_REGRESSION.md
```

## References

- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS.
- Wolpert (1992). *Stacked Generalization*. Neural Networks, 5(2):241–259.

## License

Released under the MIT License — see `LICENSE`.
