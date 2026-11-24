"""Tests for src/finguard/ml/features.py.

The most important test is test_card_history_never_uses_future_data: it proves
that a transaction's history features do not change when later transactions
are added or removed, i.e. there is no target or future leakage.
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.finguard.ml.features import add_basic_features, build_features

HISTORY_COLS = ["card_txn_count", "amt_ratio", "amt_zscore", "secs_since_last"]


def make_transactions(n_cards=5, per_card=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_cards * per_card
    return pd.DataFrame({
        "trans_num": [f"t{i}" for i in range(n)],
        "cc_num": np.repeat(np.arange(1, n_cards + 1), per_card),
        "trans_date_trans_time": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 10**7, n), unit="s"),
        "amt": rng.gamma(2, 30, n).round(2) + 1,
        "dob": pd.Timestamp("1980-05-17"),
        "lat": 40.0, "long": -90.0, "merch_lat": 40.5, "merch_long": -90.5,
        "category": "home",
    })


def test_card_history_never_uses_future_data():
    df = make_transactions()
    cutoff = df["trans_date_trans_time"].median()

    full = build_features(df).set_index("trans_num")
    early = build_features(df[df["trans_date_trans_time"] <= cutoff]).set_index("trans_num")

    # same early transactions, computed with and without the later ones
    pdt.assert_frame_equal(
        full.loc[early.index, HISTORY_COLS].sort_index(),
        early[HISTORY_COLS].sort_index(),
        check_exact=False, rtol=1e-9,
    )


def test_leakage_check_has_power():
    """Sanity check on the test above: a deliberately leaky feature DOES change."""
    df = make_transactions()
    cutoff = df["trans_date_trans_time"].median()
    early = df[df["trans_date_trans_time"] <= cutoff]
    leaky_full = df.groupby("cc_num")["amt"].transform("mean")[early.index]
    leaky_early = early.groupby("cc_num")["amt"].transform("mean")
    assert not np.allclose(leaky_full, leaky_early)


def test_first_transaction_of_a_card_has_no_history():
    out = build_features(make_transactions())
    first = out.groupby("cc_num").head(1)
    assert (first["card_txn_count"] == 0).all()
    assert (first["amt_ratio"] == 1.0).all()
    assert (first["amt_zscore"] == 0.0).all()
    assert first["secs_since_last"].isna().all()


def test_amount_features_match_hand_calculation():
    # amounts 10, 20, 60: before the 3rd, mean = 15 and population std = 5
    df = pd.DataFrame({
        "trans_num": ["a", "b", "c"], "cc_num": 1,
        "trans_date_trans_time": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 11:00", "2020-01-01 13:00"]),
        "amt": [10.0, 20.0, 60.0], "dob": pd.Timestamp("1980-01-01"),
        "lat": 0.0, "long": 0.0, "merch_lat": 0.0, "merch_long": 0.0, "category": "home",
    })
    third = build_features(df).iloc[2]
    assert third["card_txn_count"] == 2
    assert np.isclose(third["amt_ratio"], 60 / 15)
    assert np.isclose(third["amt_zscore"], (60 - 15) / 5)
    assert third["secs_since_last"] == 2 * 3600


def test_is_night_boundaries():
    df = make_transactions(1, 4)
    df["trans_date_trans_time"] = pd.to_datetime(
        ["2020-01-01 22:00", "2020-01-01 03:59", "2020-01-01 04:00", "2020-01-01 21:59"])
    assert add_basic_features(df)["is_night"].tolist() == [1, 1, 0, 0]


def test_distance_one_degree_of_longitude_at_equator():
    df = make_transactions(1, 1)
    df[["lat", "long", "merch_lat", "merch_long"]] = [0.0, 0.0, 0.0, 1.0]
    assert np.isclose(add_basic_features(df)["distance_km"].iloc[0], 111.19, atol=0.1)