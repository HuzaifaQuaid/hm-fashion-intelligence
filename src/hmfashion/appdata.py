"""Pre-aggregated app tables, and a compact deployment snapshot.

The dashboard reads only these small tables, never the 28.8M-row fact table, so it
fits Streamlit Cloud's memory limit. The same tables are built locally and copied
into `deployment/hm.duckdb`.
"""
from pathlib import Path

import duckdb

APP_TABLES = ["app_overview", "app_monthly", "app_channel", "app_department", "app_colour",
              "app_age", "app_cohort", "app_repeat_cohort", "app_article",
              "app_department_monthly", "app_complete_months"]

APP_SQL = """
CREATE OR REPLACE TABLE app_complete_months AS
SELECT date_trunc('month', t_dat) AS month
FROM fact_transactions GROUP BY 1
HAVING count(DISTINCT t_dat) >= 0.9 * day(last_day(month));

CREATE OR REPLACE TABLE app_department_monthly AS
SELECT date_trunc('month', t.t_dat) AS month, a.department_name AS category, count(*) AS events
FROM fact_transactions t JOIN dim_articles a USING(article_id)
WHERE a.department_name IS NOT NULL GROUP BY 1, 2;

CREATE OR REPLACE TABLE app_overview AS
SELECT count(*) AS purchase_events,
       count(DISTINCT customer_key) AS customers,
       sum(price) AS scaled_value,
       (SELECT count(*) FROM fact_visits) AS visits,
       (SELECT count(*) FROM (SELECT customer_key FROM fact_visits GROUP BY 1 HAVING count(*) > 1)) AS repeat_customers,
       (SELECT median(age) FROM dim_customers WHERE age IS NOT NULL) AS median_age,
       avg(CASE WHEN sales_channel_id = 1 THEN 1.0 ELSE 0.0 END) AS online_share
FROM fact_transactions;

CREATE OR REPLACE TABLE app_monthly AS
SELECT date_trunc('month', t_dat) AS month, count(*) AS purchase_events,
       count(DISTINCT customer_key) AS customers, sum(price) AS scaled_value,
       avg(CASE WHEN sales_channel_id = 1 THEN 1.0 ELSE 0.0 END) AS online_share
FROM fact_transactions GROUP BY 1 ORDER BY 1;

CREATE OR REPLACE TABLE app_channel AS
SELECT CASE sales_channel_id WHEN 1 THEN 'Online' WHEN 2 THEN 'In store' END AS channel,
       count(*) AS purchase_events, count(DISTINCT customer_key) AS customers
FROM fact_transactions GROUP BY 1;

CREATE OR REPLACE TABLE app_department AS
SELECT a.department_name AS category, count(*) AS purchase_events, count(DISTINCT t.customer_key) AS customers
FROM fact_transactions t JOIN dim_articles a USING(article_id)
WHERE a.department_name IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;

CREATE OR REPLACE TABLE app_colour AS
SELECT a.colour_group_name AS colour, count(*) AS purchase_events
FROM fact_transactions t JOIN dim_articles a USING(article_id)
WHERE a.colour_group_name IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;

CREATE OR REPLACE TABLE app_age AS
SELECT CASE WHEN age < 20 THEN 'Under 20' WHEN age < 30 THEN '20-29' WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49' WHEN age < 60 THEN '50-59' ELSE '60+' END AS age_band,
       count(*) AS customers
FROM dim_customers WHERE age IS NOT NULL GROUP BY 1;

CREATE OR REPLACE TABLE app_cohort AS
WITH first_seen AS (
    SELECT customer_key, date_trunc('month', min(t_dat)) AS cohort FROM fact_visits GROUP BY 1
),
activity AS (
    SELECT v.customer_key, f.cohort, date_diff('month', f.cohort, date_trunc('month', v.t_dat)) AS month_index
    FROM fact_visits v JOIN first_seen f USING(customer_key)
),
sizes AS (SELECT cohort, count(*) AS size FROM first_seen GROUP BY 1),
agg AS (
    SELECT cohort, month_index, count(DISTINCT customer_key) AS active
    FROM activity WHERE month_index BETWEEN 0 AND 24 GROUP BY 1, 2
)
SELECT strftime(a.cohort, '%Y-%m') AS cohort, a.month_index, a.active, s.size,
       a.active * 1.0 / s.size AS rate
FROM agg a JOIN sizes s USING(cohort);

CREATE OR REPLACE TABLE app_repeat_cohort AS
WITH per_customer AS (SELECT customer_key, min(t_dat) AS first_date, count(*) AS visits FROM fact_visits GROUP BY 1)
SELECT strftime(first_date, '%Y-%m') AS cohort_month, count(*) AS customers,
       avg(CASE WHEN visits > 1 THEN 1.0 ELSE 0.0 END) AS repeat_rate
FROM per_customer GROUP BY 1 ORDER BY 1;

CREATE OR REPLACE TABLE app_article AS
SELECT a.article_id, a.prod_name, a.department_name, a.product_group_name, a.garment_group_name,
       a.index_group_name, a.section_name, a.colour_group_name, a.perceived_colour_master_name,
       count(*) AS units, count(DISTINCT t.customer_key) AS customers
FROM fact_transactions t JOIN dim_articles a USING(article_id)
GROUP BY ALL;

"""


def build_app_tables(db_path: Path) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(APP_SQL)


def build_deployment(db_path: Path, deployment_path: Path) -> Path:
    deployment_path.parent.mkdir(parents=True, exist_ok=True)
    if deployment_path.exists():
        deployment_path.unlink()
    with duckdb.connect(str(deployment_path)) as target:
        target.execute(f"ATTACH '{db_path}' AS source_db (READ_ONLY)")
        for table in APP_TABLES + ["metadata", "data_quality"]:
            target.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM source_db.{table}")
        target.execute("DETACH source_db")
    return deployment_path
