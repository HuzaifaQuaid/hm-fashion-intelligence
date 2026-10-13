"""Censoring-aware repurchase model: will a customer return within N days?

Because there is no order id, "returning" means making another purchase visit
(customer + date) within the horizon after their first observed visit. Only
customers whose first visit is old enough to have the full horizon observed are
labelled, so recent customers are excluded rather than mislabelled.
"""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

HORIZON_DAYS = 90


def first_purchase_table(con, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    last_date = con.execute("SELECT max(t_dat) FROM fact_visits").fetchone()[0]
    frame = con.execute(f"""
        WITH first_visit AS (
            SELECT customer_key, min(t_dat) AS first_date FROM fact_visits GROUP BY 1
        ),
        later AS (
            SELECT f.customer_key,
                   count(v2.t_dat) FILTER (WHERE v2.t_dat > f.first_date
                                           AND v2.t_dat <= f.first_date + INTERVAL {int(horizon)} DAY) AS later_visits
            FROM first_visit f LEFT JOIN fact_visits v2 ON v2.customer_key = f.customer_key
            GROUP BY 1
        )
        SELECT f.customer_key, f.first_date, c.age, c.club_member_status, c.fashion_news_frequency,
               v.units AS first_units, v.scaled_value AS first_value, v.any_online,
               dayofweek(f.first_date) AS first_dow, l.later_visits
        FROM first_visit f
        JOIN fact_visits v ON v.customer_key = f.customer_key AND v.t_dat = f.first_date
        JOIN dim_customers c ON c.customer_key = f.customer_key
        JOIN later l ON l.customer_key = f.customer_key
    """).df()
    frame["matured"] = frame.first_date <= (pd.Timestamp(last_date) - pd.Timedelta(days=horizon))
    frame["repeat"] = (frame.later_visits > 0).astype(int)
    return frame


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["age"] = frame.age.fillna(frame.age.median())
    features["first_units"] = frame.first_units
    features["first_value"] = frame.first_value
    features["any_online"] = frame.any_online.fillna(0)
    features["first_dow"] = frame.first_dow
    features["first_month"] = pd.to_datetime(frame.first_date).dt.month
    for column in ["club_member_status", "fashion_news_frequency"]:
        dummies = pd.get_dummies(frame[column].fillna("unknown"), prefix=column, dtype=float)
        features = pd.concat([features, dummies], axis=1)
    return features


def train_repurchase(db_path: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path), read_only=True) as con:
        frame = first_purchase_table(con)
        source, built_at = con.execute("SELECT source, built_at::VARCHAR FROM metadata").fetchone()

    matured = frame.loc[frame.matured].copy()
    if matured.empty:
        raise ValueError("No matured customers to train on")
    matured = matured.sort_values("first_date").reset_index(drop=True)
    cutoff_index = int(len(matured) * 0.8)
    cutoff = matured.first_date.iloc[cutoff_index]
    train, test = matured.iloc[:cutoff_index], matured.iloc[cutoff_index:]
    X_train, X_test = _design(train), _design(test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train, y_test = train.repeat, test.repeat

    model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    decile = pd.qcut(probability, 10, labels=False, duplicates="drop")
    top_rate = float(y_test[decile == decile.max()].mean())
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision": float(average_precision_score(y_test, probability)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "top_decile_rate": top_rate,
        "lift_at_top_decile": float(top_rate / max(y_test.mean(), 1e-9)),
        "prevalence_baseline_ap": float(y_test.mean()),
    }
    predicted, actual = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
    calibration = pd.DataFrame({"bucket": [f"{i+1}" for i in range(len(predicted))],
                                "predicted": predicted, "actual": actual})
    importance = pd.DataFrame({"feature": X_train.columns, "coefficient": model.named_steps["logit"].coef_[0]})
    importance["abs"] = importance.coefficient.abs()
    importance = importance.sort_values("abs", ascending=False).head(12).drop(columns="abs")

    joblib.dump(model, artifact_dir / "repurchase.joblib")
    report = {
        "source": source, "database_built_at": str(built_at), "horizon_days": HORIZON_DAYS,
        "matured_customers": int(len(matured)), "train_customers": int(len(train)), "test_customers": int(len(test)),
        "split_date": str(cutoff), "overall_repeat_rate": float(matured.repeat.mean()),
        "test_repeat_rate": float(y_test.mean()), "metrics": {"logistic_regression": metrics},
        "calibration": calibration.to_dict("records"), "importance": importance.to_dict("records"),
        "limitations": [
            "A repeat is another purchase visit within the horizon, not a true order, because the dataset has no order id.",
            "The event is common in this two-year window, so the base rate is high; ranking still needs care.",
            "No delivery, review or marketing data exists, so those drivers cannot be modelled.",
            "This is historical H&M data, not a live customer feed.",
        ],
    }
    (artifact_dir / "repurchase_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def load_repurchase_report(artifact_dir: Path | None) -> dict | None:
    if not artifact_dir:
        return None
    path = Path(artifact_dir) / "repurchase_report.json"
    return json.loads(path.read_text()) if path.exists() else None
