# feature_engineering.py

import pandas as pd
import numpy as np

# --- 1. Base features (used by the others) ---
def create_features_lgbm(df):
    """Create the base features (total_nights, total_guests, adr_per_person, ...)."""
    df_feat = df.copy()
    
    # Duration and party-size features
    df_feat['total_nights'] = df_feat['stays_in_weekend_nights'] + df_feat['stays_in_week_nights']
    df_feat['total_guests'] = df_feat['adults'] + df_feat['children'] + df_feat['babies']
    
    # Clean 'adr' (Average Daily Rate)
    # Median of valid 'adr' (> 0)
    adr_median = df_feat[df_feat['adr'] > 0]['adr'].median()
    # Guard against a zero/NaN median
    if pd.isna(adr_median) or adr_median == 0: 
        adr_median = 0.01  # small value to avoid division by zero
    
    df_feat['adr'] = df_feat['adr'].replace(0, adr_median)
    
    # Derived features
    # ADR per guest (1e-6 avoids division by zero when total_guests is 0)
    df_feat['adr_per_person'] = df_feat['adr'] / (df_feat['total_guests'] + 1e-6)
    df_feat['is_family'] = ((df_feat['children'] > 0) | (df_feat['babies'] > 0)).astype(int)
    
    # Ratio between lead_time and stay length
    # (1e-6 avoids division by zero when total_nights is 0)
    df_feat['lead_vs_stay_ratio'] = df_feat['lead_time'] / (df_feat['total_nights'] + 1e-6)
    
    return df_feat

# --- 2. LGBM features (with cyclical encodings) ---
def create_features_base_plus_cyclical(df):
    """Base features + cyclical (sin/cos) encodings for month and week."""
    df_feat = create_features_lgbm(df)
    
    # Map textual months to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_feat['arrival_month_num'] = df_feat['arrival_date_month'].map(month_map)
    
    # Cyclical features for the month
    df_feat['arrival_month_sin'] = np.sin(2 * np.pi * df_feat['arrival_month_num'] / 12)
    df_feat['arrival_month_cos'] = np.cos(2 * np.pi * df_feat['arrival_month_num'] / 12)
    
    # Cyclical features for the week (53 weeks max)
    df_feat['arrival_date_week_sin'] = np.sin(2 * np.pi * df_feat['arrival_date_week_number'] / 53)
    df_feat['arrival_date_week_cos'] = np.cos(2 * np.pi * df_feat['arrival_date_week_number'] / 53)
    
    return df_feat

# --- 3. CatBoost features (with interactions) ---
def create_features_catboost(df):
    """Base features + simple categorical interactions."""
    df_feat = create_features_lgbm(df)
    
    # Simple interactions CatBoost can exploit
    df_feat['hotel_market_segment'] = df_feat['hotel'].astype(str) + '_' + df_feat['market_segment'].astype(str)
    df_feat['hotel_customer_type'] = df_feat['hotel'].astype(str) + '_' + df_feat['customer_type'].astype(str)
    
    return df_feat