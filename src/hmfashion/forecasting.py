"""Monthly purchase-event forecasting with rolling-origin backtests.

The target is monthly purchase events (articles sold), not currency, because H&M
anonymised prices. Boundary months that are only partly covered by observed days
are excluded so partial months are not read as growth or decline.
"""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HORIZON = 1


def monthly_series(con) -> pd.DataFrame:
    frame = con.execute("""
        SELECT date_trunc('month', t_dat) AS month, count(*) AS purchase_events
        FROM fact_transactions GROUP BY 1 ORDER BY 1
    """).df()
    frame["days"] = frame.month.dt.days_in_month
    observed = con.execute("""
        SELECT date_trunc('month', t_dat) AS month, count(DISTINCT t_dat) AS observed_days
        FROM fact_transactions GROUP BY 1
    """).df()
    frame = frame.merge(observed, on="month", how="left")
    frame["coverage"] = frame.observed_days / frame.days
    return frame.loc[frame.coverage >= 0.9].reset_index(drop=True)


def _features(values: np.ndarray, months: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"value": values})
    for lag in (1, 2, 3, 12):
        frame[f"lag{lag}"] = frame.value.shift(lag)
    frame["month"] = pd.to_datetime(months).dt.month.to_numpy()
    frame["sin"] = np.sin(2 * np.pi * frame.month / 12)
    frame["cos"] = np.cos(2 * np.pi * frame.month / 12)
    return frame


def _fit_predict(values: np.ndarray, months: pd.Series) -> float:
    features = _features(values, months).dropna()
    model = Ridge(alpha=1.0)
    model.fit(features.drop(columns="value"), features.value)
    latest = _features(values, months).iloc[[-1]].drop(columns="value")
    return float(model.predict(latest)[0])


def _backtest(values: np.ndarray, months: pd.Series) -> dict:
    errors = {"naive_last_value": [], "seasonal_naive": [], "calendar_lag_model": []}
    actuals = []
    for origin in range(12, len(values) - 1):
        history, history_months = values[: origin + 1], months.iloc[: origin + 1]
        target = values[origin + 1]
        actuals.append(target)
        errors["naive_last_value"].append(history[-1] - target)
        errors["seasonal_naive"].append(history[-12] - target)
        errors["calendar_lag_model"].append(_fit_predict(history, history_months) - target)
    metrics = {}
    for name, residuals in errors.items():
        residuals = np.array(residuals)
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "mape": float(np.mean(np.abs(residuals / np.array(actuals)))),
        }
    return metrics, {name: np.array(residuals) for name, residuals in errors.items()}


def train_forecast(db_path: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path), read_only=True) as con:
        series = monthly_series(con)
        source, built_at = con.execute("SELECT source, built_at::VARCHAR FROM metadata").fetchone()
    values = series.purchase_events.to_numpy(dtype=float)
    months = series.month
    if len(values) < 14:
        raise ValueError("Not enough complete months to backtest")
    metrics, residuals = _backtest(values, months)
    selected = min(metrics, key=lambda name: metrics[name]["rmse"])

    if selected == "naive_last_value":
        point = values[-1]
    elif selected == "seasonal_naive":
        point = values[-12]
    else:
        point = _fit_predict(values, months)
    # Empirical, symmetric intervals from the backtest's absolute errors, so the
    # range always brackets the point forecast and is easy to read.
    error = np.abs(residuals[selected])
    scale_80, scale_95 = float(np.percentile(error, 80)), float(np.percentile(error, 95))
    forecast = {
        "point": float(point),
        "low_80": float(point - scale_80), "high_80": float(point + scale_80),
        "low_95": float(point - scale_95), "high_95": float(point + scale_95),
    }
    next_month = (pd.Timestamp(months.iloc[-1]) + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    report = {
        "source": source, "database_built_at": str(built_at), "horizon_months": HORIZON,
        "complete_months": int(len(values)), "series_start": str(months.iloc[0].date()),
        "series_end": str(months.iloc[-1].date()), "next_month": next_month,
        "selected_method": selected, "metrics": metrics, "forecast": forecast,
        "recent_actuals": [{"month": str(m.date()), "purchase_events": float(v)} for m, v in zip(months, values)],
        "limitations": [
            "The target is purchase events, not currency, because H&M anonymised prices.",
            "Only 23 complete months exist, so long-term seasonality cannot be trusted.",
            "The forecast ignores promotions, pricing and marketing, which are not in the data.",
            "Backtest intervals come from past errors and may miss their stated coverage.",
        ],
    }
    (artifact_dir / "forecast_report.json").write_text(json.dumps(report, indent=2, default=str))
    backtest = pd.DataFrame({"month": months.iloc[13:].dt.strftime("%Y-%m").to_numpy(),
                             "actual": values[13:]})
    for name, residual in residuals.items():
        backtest[name + "_error"] = residual
    backtest.to_csv(artifact_dir / "forecast_backtest.csv", index=False)
    return report


def load_forecast(artifact_dir: Path | None) -> dict | None:
    if not artifact_dir:
        return None
    path = Path(artifact_dir) / "forecast_report.json"
    return json.loads(path.read_text()) if path.exists() else None
