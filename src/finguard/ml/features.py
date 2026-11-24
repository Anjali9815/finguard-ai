# src/finguard/ml/features.py
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "hour", "day_of_week", "is_night", "log_amt", "age", "distance_km",
    "card_txn_count", "amt_ratio", "amt_zscore", "secs_since_last",
]
CATEGORICAL_FEATURES = ["category"]


def add_basic_features(df):
    df = df.copy()
    ts = df["trans_date_trans_time"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 3)).astype(int)
    df["log_amt"] = np.log1p(df["amt"])
    df["age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25

    # haversine distance (km) between customer home and merchant
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return df


def add_card_history_features(df):
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    g = df.groupby("cc_num")

    n = g.cumcount()                                   # transactions before this one
    s1 = g["amt"].cumsum() - df["amt"]                 # sum of PAST amounts only
    s2 = (df["amt"] ** 2).groupby(df["cc_num"]).cumsum() - df["amt"] ** 2

    mean_prev = s1 / n.replace(0, np.nan)
    std_prev = np.sqrt((s2 / n.replace(0, np.nan) - mean_prev ** 2).clip(lower=0))

    df["card_txn_count"] = n
    df["amt_ratio"] = (df["amt"] / mean_prev).fillna(1.0)
    df["amt_zscore"] = ((df["amt"] - mean_prev) / std_prev.replace(0, np.nan)).fillna(0)
    df["secs_since_last"] = g["trans_date_trans_time"].diff().dt.total_seconds()
    return df


def build_features(df):
    """Run the full feature pipeline: basic features first, then card history."""
    return add_card_history_features(add_basic_features(df))