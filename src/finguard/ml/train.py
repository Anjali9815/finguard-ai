"""Train FinGuard models and save artifacts. Run from project root:
    python -m src.finguard.ml.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.finguard.ml.features import build_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

RAW, PROCESSED, MODELS = Path("data/raw"), Path("data/processed"), Path("models")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REVIEW_COST = 10  # ASSUMPTION: dollars of analyst time per flagged transaction


def load_and_build():
    cols = dict(index_col=0, parse_dates=["trans_date_trans_time"])
    train = pd.read_csv(RAW / "fraudTrain.csv", **cols).assign(split="train")
    test = pd.read_csv(RAW / "fraudTest.csv", **cols).assign(split="test")
    # build together so test cards keep their history from train
    return build_features(pd.concat([train, test], ignore_index=True))


def total_cost(y, p, amt, thr):
    flag = p >= thr
    return amt[(y == 1) & ~flag].sum() + REVIEW_COST * ((y == 0) & flag).sum()


def main():
    MODELS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_and_build()
    categories = sorted(df["category"].unique())
    df["category"] = pd.Categorical(df["category"], categories=categories)

    train_all = df[df.split == "train"]
    test = df[df.split == "test"]

    # validation = latest 15% of train by date
    cutoff = train_all["trans_date_trans_time"].quantile(0.85)
    fit = train_all[train_all["trans_date_trans_time"] <= cutoff]
    val = train_all[train_all["trans_date_trans_time"] > cutoff]

    # ---- XGBoost ----
    spw = (fit.is_fraud == 0).sum() / (fit.is_fraud == 1).sum()
    model = XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
        enable_categorical=True, eval_metric="aucpr", early_stopping_rounds=30,
        n_jobs=-1, random_state=42,
    )
    model.fit(fit[FEATURES], fit.is_fraud, eval_set=[(val[FEATURES], val.is_fraud)], verbose=100)

    # ---- threshold chosen on validation only ----
    val_p = model.predict_proba(val[FEATURES])[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(val.is_fraud.values, val_p, val.amt.values, t) for t in thresholds]
    threshold = float(thresholds[int(np.argmin(costs))])

    # ---- Isolation Forest (no labels) ----
    fill = fit[NUMERIC_FEATURES].median()
    iso = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
    iso.fit(fit[NUMERIC_FEATURES].fillna(fill))

    # ---- test evaluation (used once) ----
    test_p = model.predict_proba(test[FEATURES])[:, 1]
    test_anom = -iso.score_samples(test[NUMERIC_FEATURES].fillna(fill))
    y, amt = test.is_fraud.values, test.amt.values
    flag = test_p >= threshold
    metrics = {
        "xgb_pr_auc": float(average_precision_score(y, test_p)),
        "xgb_roc_auc": float(roc_auc_score(y, test_p)),
        "iso_pr_auc": float(average_precision_score(y, test_anom)),
        "random_pr_auc": float(y.mean()),
        "threshold": threshold,
        "caught": int(((y == 1) & flag).sum()),
        "missed": int(((y == 1) & ~flag).sum()),
        "false_alarms": int(((y == 0) & flag).sum()),
        "cost_with_model": float(total_cost(y, test_p, amt, threshold)),
        "cost_no_model": float(amt[y == 1].sum()),
        "review_cost_assumption": REVIEW_COST,
    }
    print(json.dumps(metrics, indent=2))

    # ---- save artifacts ----
    model.save_model(MODELS / "xgb_fraud.json")
    joblib.dump({"model": iso, "fill": fill.to_dict()}, MODELS / "iso_forest.joblib")
    (MODELS / "config.json").write_text(json.dumps(
        {"features": FEATURES, "numeric": NUMERIC_FEATURES, "categories": categories,
         "threshold": threshold}, indent=2))
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # scored transactions: this is the "database" the agents will query
    df["fraud_prob"] = model.predict_proba(df[FEATURES])[:, 1]
    df["anomaly_score"] = -iso.score_samples(df[NUMERIC_FEATURES].fillna(fill))
    df["category"] = df["category"].astype(str)
    df.to_parquet(PROCESSED / "transactions_scored.parquet", index=False)
    print("saved models/ and data/processed/transactions_scored.parquet")


if __name__ == "__main__":
    main()