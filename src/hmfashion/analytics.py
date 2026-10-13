"""Descriptive analytics for the H&M dataset: KPIs, trends, categories and cohorts.

Definitions used throughout:
- A purchase event is one (customer, article, date) row after exact duplicates are removed.
- A visit is a (customer, date) pair, used as an explicit proxy for a basket because
  the dataset has no order id.
- Value is relative and anonymised: H&M scaled prices, so there is no currency here.
"""
from pathlib import Path
import json

import duckdb
import pandas as pd

CATEGORY_COLUMNS = ["department_name", "product_group_name", "garment_group_name", "index_group_name", "section_name"]


def monthly_summary(con) -> pd.DataFrame:
    return con.execute("""
        SELECT date_trunc('month', t_dat) AS month,
               count(*) AS purchase_events,
               count(DISTINCT customer_key) AS customers,
               sum(price) AS scaled_value,
               avg(CASE WHEN sales_channel_id = 1 THEN 1.0 ELSE 0.0 END) AS online_share
        FROM fact_transactions
        GROUP BY 1 ORDER BY 1
    """).df()


def visits_summary(con) -> pd.DataFrame:
    return con.execute("""
        SELECT date_trunc('month', t_dat) AS month,
               count(*) AS visits,
               avg(units) AS avg_units_per_visit,
               avg(scaled_value) AS avg_scaled_value_per_visit
        FROM fact_visits
        GROUP BY 1 ORDER BY 1
    """).df()


def channel_split(con) -> pd.DataFrame:
    return con.execute("""
        SELECT CASE sales_channel_id WHEN 1 THEN 'Online' WHEN 2 THEN 'In store' END AS channel,
               count(*) AS purchase_events,
               count(DISTINCT customer_key) AS customers
        FROM fact_transactions GROUP BY 1 ORDER BY 2 DESC
    """).df()


def category_summary(con, column: str = "department_name", limit: int = 15) -> pd.DataFrame:
    if column not in CATEGORY_COLUMNS:
        raise ValueError(f"Unsupported category column: {column}")
    return con.execute(f"""
        SELECT a.{column} AS category,
               count(*) AS purchase_events,
               count(DISTINCT t.customer_key) AS customers
        FROM fact_transactions t JOIN dim_articles a USING(article_id)
        WHERE a.{column} IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT {int(limit)}
    """).df()


def colour_summary(con, limit: int = 12) -> pd.DataFrame:
    return con.execute(f"""
        SELECT a.colour_group_name AS colour,
               count(*) AS purchase_events
        FROM fact_transactions t JOIN dim_articles a USING(article_id)
        WHERE a.colour_group_name IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT {int(limit)}
    """).df()


def customer_overview(con) -> dict:
    row = con.execute("""
        SELECT count(*) AS customers,
               sum(CASE WHEN visits > 1 THEN 1 ELSE 0 END) AS repeat_customers,
               avg(visits) AS avg_visits,
               avg(age) AS avg_age,
               median(age) AS median_age
        FROM (SELECT customer_key, count(*) AS visits, avg(age) AS age
              FROM fact_visits v JOIN dim_customers c USING(customer_key)
              GROUP BY 1)
    """).df().iloc[0]
    return row.to_dict()


def age_distribution(con) -> pd.DataFrame:
    return con.execute("""
        SELECT CASE WHEN age < 20 THEN 'Under 20' WHEN age < 30 THEN '20-29'
                    WHEN age < 40 THEN '30-39' WHEN age < 50 THEN '40-49'
                    WHEN age < 60 THEN '50-59' ELSE '60+' END AS age_band,
               count(*) AS customers
        FROM dim_customers WHERE age IS NOT NULL GROUP BY 1 ORDER BY 1
    """).df()


def cohort_retention(con, max_month: int = 24) -> pd.DataFrame:
    """Monthly retention by acquisition cohort; blank (NaN) where unobserved."""
    frame = con.execute("""
        WITH first_seen AS (
            SELECT customer_key, date_trunc('month', min(t_dat)) AS cohort
            FROM fact_visits GROUP BY 1
        ),
        activity AS (
            SELECT v.customer_key, f.cohort,
                   date_diff('month', f.cohort, date_trunc('month', v.t_dat)) AS month_index
            FROM fact_visits v JOIN first_seen f USING(customer_key)
        )
        SELECT cohort, month_index, count(DISTINCT customer_key) AS active
        FROM activity WHERE month_index BETWEEN 0 AND %d
        GROUP BY 1, 2
    """ % int(max_month)).df()
    sizes = frame.loc[frame.month_index.eq(0)].set_index("cohort").active
    matrix = frame.pivot_table(index="cohort", columns="month_index", values="active", aggfunc="sum")
    matrix = matrix.div(sizes, axis=0)
    matrix.index = pd.to_datetime(matrix.index).strftime("%Y-%m")
    matrix.columns = [f"M{c}" for c in matrix.columns]
    return matrix


def repeat_by_acquisition(con) -> pd.DataFrame:
    return con.execute("""
        WITH per_customer AS (
            SELECT customer_key, min(t_dat) AS first_date, count(*) AS visits
            FROM fact_visits GROUP BY 1
        )
        SELECT strftime(first_date, '%Y-%m') AS cohort_month,
               count(*) AS customers,
               avg(CASE WHEN visits > 1 THEN 1.0 ELSE 0.0 END) AS repeat_rate
        FROM per_customer GROUP BY 1 ORDER BY 1
    """).df()


def generate_report(db_path: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path), read_only=True) as con:
        overview = customer_overview(con)
        channels = channel_split(con)
        monthly = monthly_summary(con)
        departments = category_summary(con, "department_name", 10)
        quality = con.execute("SELECT * FROM data_quality WHERE count > 0 ORDER BY count DESC").df()
        metadata = con.execute("SELECT * FROM metadata").df().iloc[0].to_dict()

    online = channels.loc[channels.channel.eq("Online"), "purchase_events"].sum()
    total = channels.purchase_events.sum()
    report = {
        "source": metadata["source"],
        "built_at": str(metadata["built_at"]),
        "purchase_events": int(total),
        "customers": int(overview["customers"]),
        "repeat_rate": float(overview["repeat_customers"] / overview["customers"]),
        "avg_visits": float(overview["avg_visits"]),
        "median_age": float(overview["median_age"]) if overview["median_age"] is not None else None,
        "online_share": float(online / total),
        "first_transaction": str(metadata["first_transaction"]),
        "last_transaction": str(metadata["last_transaction"]),
        "top_departments": departments.to_dict("records"),
        "quality": quality.to_dict("records"),
        "limitations": [
            "H&M anonymised and scaled the prices, so value here is relative, not a currency amount.",
            "The dataset has no order id, so a (customer, date) visit is a documented proxy for a basket, not a true order.",
            "Exact duplicate rows were removed; because there is no quantity column, some may have represented multiple units.",
            "There is no delivery, review, payment or seller information, so those analyses are not possible.",
            "This is historical data (2018-2020) and not a live feed.",
        ],
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# H&M Fashion Intelligence — findings",
        "",
        f"Historical H&M transactions, {report['first_transaction']} to {report['last_transaction']}. "
        "Prices are anonymised, so value is relative, not a currency.",
        "",
        "## Headline numbers",
        f"- Purchase events: **{report['purchase_events']:,}**",
        f"- Customers: **{report['customers']:,}**",
        f"- Repeat customers (more than one visit): **{report['repeat_rate']:.1%}**",
        f"- Average visits per customer: **{report['avg_visits']:.2f}**",
        f"- Median customer age: **{report['median_age']:.0f}**",
        f"- Online share of purchase events: **{report['online_share']:.1%}**",
        "",
        "## Top departments by purchase events",
    ]
    for row in report["top_departments"]:
        lines.append(f"- {row['category']}: {int(row['purchase_events']):,}")
    lines += ["", "## Data-quality observations"]
    for row in report["quality"]:
        lines.append(f"- {row['check']}: {int(row['count']):,} ({row['severity']})")
    lines += ["", "## Honest limitations"]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    (artifact_dir / "findings.md").write_text("\n".join(lines) + "\n")
    return report
