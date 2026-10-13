"""Validated ingestion and grain-safe analytical marts for the H&M dataset.

The H&M data is three related tables (transactions, customers, articles) rather
than Olist's nine. It has no order id, so a "visit" (customer + purchase date)
is used as an explicit, documented proxy for a basket. Prices are anonymised by
H&M, so money is reported as relative, scaled value rather than a currency.
"""
from pathlib import Path

import duckdb

RAW_FILES = {
    "transactions": "transactions_train.csv",
    "customers": "customers.csv",
    "articles": "articles.csv",
}

TRANSACTION_TYPES = {
    "t_dat": "DATE",
    "customer_id": "VARCHAR",
    "article_id": "INTEGER",
    "price": "DOUBLE",
    "sales_channel_id": "INTEGER",
}
CUSTOMER_TYPES = {
    "customer_id": "VARCHAR",
    "FN": "DOUBLE",
    "Active": "DOUBLE",
    "club_member_status": "VARCHAR",
    "fashion_news_frequency": "VARCHAR",
    "age": "DOUBLE",
    "postal_code": "VARCHAR",
}
ARTICLE_TYPES = {
    "article_id": "INTEGER",
    "product_code": "INTEGER",
    "prod_name": "VARCHAR",
    "product_type_no": "INTEGER",
    "product_type_name": "VARCHAR",
    "product_group_name": "VARCHAR",
    "graphical_appearance_no": "INTEGER",
    "graphical_appearance_name": "VARCHAR",
    "colour_group_code": "INTEGER",
    "colour_group_name": "VARCHAR",
    "perceived_colour_value_id": "INTEGER",
    "perceived_colour_value_name": "VARCHAR",
    "perceived_colour_master_id": "INTEGER",
    "perceived_colour_master_name": "VARCHAR",
    "department_no": "INTEGER",
    "department_name": "VARCHAR",
    "index_code": "VARCHAR",
    "index_name": "VARCHAR",
    "index_group_no": "INTEGER",
    "index_group_name": "VARCHAR",
    "section_no": "INTEGER",
    "section_name": "VARCHAR",
    "garment_group_no": "INTEGER",
    "garment_group_name": "VARCHAR",
    "detail_desc": "VARCHAR",
}


def _read_csv(con, path: Path, types: dict) -> None:
    columns = ", ".join(f"'{name}': '{kind}'" for name, kind in types.items())
    con.execute(f"CREATE OR REPLACE TABLE incoming AS SELECT * FROM read_csv('{path}', header=true, columns={{{columns}}})")


def build_database(raw_dir: Path, db_path: Path, source: str = "hm") -> dict:
    if source not in {"hm", "synthetic"}:
        raise ValueError("Source must be hm or synthetic")
    for name, filename in RAW_FILES.items():
        if not (raw_dir / filename).exists():
            raise FileNotFoundError(f"Missing {raw_dir / filename}. Download the H&M competition data first.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    checks: list[dict] = []

    def check(name, count, severity="warning"):
        checks.append({"check": name, "count": int(count), "severity": severity})

    try:
        con.execute("BEGIN TRANSACTION")
        for name, filename in RAW_FILES.items():
            types = {"transactions": TRANSACTION_TYPES, "customers": CUSTOMER_TYPES, "articles": ARTICLE_TYPES}[name]
            _read_csv(con, raw_dir / filename, types)
            con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT * FROM incoming")
            before = con.execute(f"SELECT count(*) FROM raw_{name}").fetchone()[0]
            con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT DISTINCT * FROM raw_{name}")
            after = con.execute(f"SELECT count(*) FROM raw_{name}").fetchone()[0]
            check(f"{name}: exact duplicate rows removed", before - after, "info")

        # ---- key and relationship validation -----------------------------
        for table, key in [("customers", "customer_id"), ("articles", "article_id")]:
            nulls = con.execute(f"SELECT count(*) FROM raw_{table} WHERE {key} IS NULL").fetchone()[0]
            if nulls:
                raise ValueError(f"{table}.{key}: {nulls} null keys")
            conflicting = con.execute(f"SELECT count(*) FROM (SELECT {key} FROM raw_{table} GROUP BY 1 HAVING count(*) > 1)").fetchone()[0]
            if conflicting:
                raise ValueError(f"{table}.{key}: {conflicting} conflicting keys; fix the source instead of dropping records")

        orphan_customers = con.execute("SELECT count(*) FROM raw_transactions t LEFT JOIN raw_customers c USING(customer_id) WHERE c.customer_id IS NULL").fetchone()[0]
        if orphan_customers:
            raise ValueError(f"transactions.customer_id: {orphan_customers} orphan records")
        orphan_articles = con.execute("SELECT count(*) FROM raw_transactions t LEFT JOIN raw_articles a USING(article_id) WHERE a.article_id IS NULL").fetchone()[0]
        if orphan_articles:
            raise ValueError(f"transactions.article_id: {orphan_articles} orphan records")

        # ---- data-quality observations -----------------------------------
        negative_price = con.execute("SELECT count(*) FROM raw_transactions WHERE price IS NULL OR price < 0").fetchone()[0]
        if negative_price:
            raise ValueError(f"transactions.price: {negative_price} missing or negative values")
        check("transactions: missing prices", con.execute("SELECT count(*) FROM raw_transactions WHERE price IS NULL").fetchone()[0])
        check("transactions: unknown sales channel", con.execute("SELECT count(*) FROM raw_transactions WHERE sales_channel_id NOT IN (1, 2)").fetchone()[0])
        check("customers: missing age", con.execute("SELECT count(*) FROM raw_customers WHERE age IS NULL").fetchone()[0])
        check("customers: age outside 16-100", con.execute("SELECT count(*) FROM raw_customers WHERE age < 16 OR age > 100").fetchone()[0])
        check("articles: missing product group", con.execute("SELECT count(*) FROM raw_articles WHERE product_group_name IS NULL").fetchone()[0])

        # ---- grain-safe marts --------------------------------------------
        con.execute(MART_SQL)
        check("transactions: articles never sold", con.execute("SELECT count(*) FROM raw_articles a LEFT JOIN (SELECT DISTINCT article_id FROM raw_transactions) t USING(article_id) WHERE t.article_id IS NULL").fetchone()[0])
        check("customers: never purchased", con.execute("SELECT count(*) FROM raw_customers c LEFT JOIN (SELECT DISTINCT customer_id FROM raw_transactions) t USING(customer_id) WHERE t.customer_id IS NULL").fetchone()[0])

        # Conservation: units and scaled value must survive the joins.
        source_units = con.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
        mart_units = con.execute("SELECT count(*) FROM fact_transactions").fetchone()[0]
        if source_units != mart_units:
            raise ValueError("Unit conservation failed: fact_transactions lost or multiplied rows")
        if con.execute("SELECT count(*) - count(DISTINCT customer_key || '|' || t_dat) FROM fact_visits").fetchone()[0]:
            raise ValueError("Visit mart contains duplicate keys")

        import pandas as pd

        quality = pd.DataFrame(checks, columns=["check", "count", "severity"])
        con.register("quality_incoming", quality)
        con.execute("CREATE OR REPLACE TABLE data_quality AS SELECT * FROM quality_incoming")
        con.unregister("quality_incoming")
        con.execute("DROP TABLE IF EXISTS incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    return {"source": source, "checks": checks}


MART_SQL = """
-- Surrogate integer keys keep the 31.8M-row fact table compact.
CREATE OR REPLACE TABLE dim_customers AS
SELECT row_number() OVER (ORDER BY customer_id) AS customer_key,
       customer_id, age, club_member_status, fashion_news_frequency, "Active" AS active_flag, "FN" AS fn_flag
FROM raw_customers;

CREATE OR REPLACE TABLE dim_articles AS
SELECT article_id, prod_name, product_type_name, product_group_name,
       graphical_appearance_name, colour_group_name, perceived_colour_value_name,
       perceived_colour_master_name, department_name, index_code, index_name,
       index_group_name, section_name, garment_group_name
FROM raw_articles;

CREATE OR REPLACE TABLE fact_transactions AS
SELECT t.t_dat, c.customer_key, t.article_id, t.price, t.sales_channel_id
FROM raw_transactions t
JOIN dim_customers c USING(customer_id);

CREATE OR REPLACE TABLE fact_visits AS
SELECT customer_key, t_dat,
       count(*) AS units,
       sum(price) AS scaled_value,
       max(CASE WHEN sales_channel_id = 1 THEN 1 ELSE 0 END) AS any_online,
       max(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0 END) AS any_store
FROM fact_transactions
GROUP BY customer_key, t_dat;

CREATE OR REPLACE TABLE metadata AS
SELECT 'hm' AS source, now() AS built_at,
       (SELECT min(t_dat) FROM raw_transactions) AS first_transaction,
       (SELECT max(t_dat) FROM raw_transactions) AS last_transaction,
       'Units sold and relative scaled value; H&M anonymised prices, so no currency' AS value_definition;
"""
