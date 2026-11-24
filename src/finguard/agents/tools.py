"""Tools the FinGuard agents call.

These are plain Python functions. Google ADK turns each function's name,
type hints and docstring into a tool the LLM can call, so the docstrings
matter: they are the instructions the agent reads.

Design rules:
  * The fraud label (is_fraud) is never returned. Agents must reason like a
    production system that does not know the answer.
  * Card history only uses transactions BEFORE the one being investigated.
  * Every tool returns JSON-safe dicts (no numpy / pandas types).
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from src.finguard.config import MODELS_DIR as MODELS
from src.finguard.config import SCORED_STORE as STORE


# ----------------------------------------------------------------- loaders
@lru_cache(maxsize=1)
def _store() -> pd.DataFrame:
    df = pd.read_parquet(STORE)
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    return df.set_index("trans_num", drop=False)


@lru_cache(maxsize=1)
def _config() -> dict:
    return json.loads((MODELS / "config.json").read_text())


@lru_cache(maxsize=1)
def _booster() -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(MODELS / "xgb_fraud.json"))
    return b


@lru_cache(maxsize=1)
def _sorted_anomaly() -> np.ndarray:
    return np.sort(_store()["anomaly_score"].to_numpy())


def _error(trans_num: str) -> dict:
    return {"status": "error", "message": f"Transaction '{trans_num}' not found."}


def _get_row(trans_num: str):
    df = _store()
    return df.loc[trans_num] if trans_num in df.index else None


# ------------------------------------------------------------------- tools
def list_flagged_transactions(limit: int = 5) -> dict:
    """Lists recent transactions that the fraud model flagged as suspicious.

    Use this to find transaction IDs to investigate.

    Args:
        limit: How many flagged transactions to return (default 5, max 20).

    Returns:
        A dict with a list of flagged transactions, highest fraud probability first.
    """
    limit = max(1, min(int(limit), 20))
    df = _store()
    thr = _config()["threshold"]
    flagged = df[(df["split"] == "test") & (df["fraud_prob"] >= thr)]
    top = flagged.sort_values("fraud_prob", ascending=False).head(limit)
    return {
        "status": "success",
        "threshold": thr,
        "total_flagged": int(len(flagged)),
        "transactions": [
            {
                "trans_num": str(r.trans_num),
                "amount": round(float(r.amt), 2),
                "category": str(r.category),
                "fraud_probability": round(float(r.fraud_prob), 4),
            }
            for r in top.itertuples()
        ],
    }


def get_transaction_details(trans_num: str) -> dict:
    """Gets the basic facts and model scores for one transaction.

    Args:
        trans_num: The transaction ID to look up.

    Returns:
        A dict with time, merchant, category, amount, location, the ML fraud
        probability, whether the model flagged it, and how anomalous it is
        (anomaly_percentile: 99 means more unusual than 99% of transactions).
    """
    r = _get_row(trans_num)
    if r is None:
        return _error(trans_num)
    anomaly_pct = 100 * np.searchsorted(_sorted_anomaly(), r["anomaly_score"]) / len(_sorted_anomaly())
    thr = _config()["threshold"]
    return {
        "status": "success",
        "trans_num": trans_num,
        "card_last4": str(r["cc_num"])[-4:],
        "time": str(r["trans_date_trans_time"]),
        "merchant": str(r["merchant"]),
        "category": str(r["category"]),
        "amount": round(float(r["amt"]), 2),
        "city": str(r.get("city", "")),
        "state": str(r.get("state", "")),
        "fraud_probability": round(float(r["fraud_prob"]), 4),
        "decision_threshold": thr,
        "model_flagged": bool(r["fraud_prob"] >= thr),
        "anomaly_percentile": round(float(anomaly_pct), 1),
    }


def explain_risk_factors(trans_num: str, top_k: int = 5) -> dict:
    """Explains WHY the model scored a transaction the way it did.

    Returns the features that pushed the fraud score up or down the most.
    Contributions are in log-odds: positive values push toward fraud,
    negative values push toward normal.

    Args:
        trans_num: The transaction ID to explain.
        top_k: How many of the strongest factors to return (default 5).

    Returns:
        A dict with the top factors, each with the feature name, its value for
        this transaction, its contribution, and the direction.
    """
    df = _store()
    if trans_num not in df.index:
        return _error(trans_num)
    cfg = _config()
    X = df.loc[[trans_num], cfg["features"]].copy()
    X["category"] = pd.Categorical(X["category"], categories=cfg["categories"])
    contrib = _booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0]
    pairs = list(zip(cfg["features"], X.iloc[0].tolist(), contrib[:-1].tolist()))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    factors = []
    for name, value, c in pairs[: max(1, int(top_k))]:
        if isinstance(value, (int, float, np.floating, np.integer)):
            value = None if pd.isna(value) else round(float(value), 3)
        factors.append({
            "feature": name,
            "value": value,
            "contribution": round(float(c), 3),
            "direction": "raises risk" if c > 0 else "lowers risk",
        })
    return {"status": "success", "trans_num": trans_num, "top_factors": factors}


def get_card_history(trans_num: str) -> dict:
    """Summarises this card's behaviour BEFORE the transaction being investigated.

    Use it to judge whether the transaction fits the cardholder's normal pattern.

    Args:
        trans_num: The transaction being investigated.

    Returns:
        A dict with the card's prior transaction count, typical and maximum
        amounts, where the current amount ranks in the card's history
        (amount_percentile_in_card_history: 99 = larger than 99% of its past
        purchases), what share of its past transactions happened near this hour
        and at night, top categories, recent velocity, whether the merchant and
        category were seen before, and the last few transactions.
    """
    df = _store()
    r = _get_row(trans_num)
    if r is None:
        return _error(trans_num)
    t = r["trans_date_trans_time"]
    card = df[df["cc_num"] == r["cc_num"]]
    past = card[card["trans_date_trans_time"] < t]
    if past.empty:
        return {"status": "success", "trans_num": trans_num, "prior_transactions": 0,
                "note": "No earlier history for this card."}
    last = past.sort_values("trans_date_trans_time").tail(3)

    # How often has this card transacted around the current hour / at night?
    # (More informative than a "top 3 usual hours" list, which makes any
    # card with a varied schedule look like it is acting outside its pattern.)
    h = int(r["hour"])
    near_hours = {(h - 1) % 24, h, (h + 1) % 24}
    is_night = lambda x: (x >= 22) | (x <= 3)
    return {
        "status": "success",
        "trans_num": trans_num,
        "prior_transactions": int(len(past)),
        "avg_amount": round(float(past["amt"].mean()), 2),
        "median_amount": round(float(past["amt"].median()), 2),
        "max_amount": round(float(past["amt"].max()), 2),
        "current_amount_vs_avg": round(float(r["amt"] / past["amt"].mean()), 2),
        "amount_percentile_in_card_history": round(float((past["amt"] < r["amt"]).mean() * 100), 1),
        "current_hour": h,
        "current_is_night": bool(is_night(h)),
        "share_of_past_near_this_hour": round(float(past["hour"].isin(near_hours).mean()), 3),
        "share_of_past_at_night": round(float(is_night(past["hour"]).mean()), 3),
        "top_categories": [str(c) for c in past["category"].value_counts().head(3).index],
        "category_seen_before": bool((past["category"] == r["category"]).any()),
        "merchant_seen_before": bool((past["merchant"] == r["merchant"]).any()),
        "transactions_last_1h": int((past["trans_date_trans_time"] >= t - pd.Timedelta(hours=1)).sum()),
        "transactions_last_24h": int((past["trans_date_trans_time"] >= t - pd.Timedelta(hours=24)).sum()),
        "last_transactions": [
            {"time": str(x.trans_date_trans_time), "merchant": str(x.merchant),
             "category": str(x.category), "amount": round(float(x.amt), 2)}
            for x in last.itertuples()
        ],
    }
