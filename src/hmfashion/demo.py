"""Generate a small synthetic dataset in the H&M schema, for tests and demos.

This never touches the real data: it writes to whatever directory it is given.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo(raw_dir: Path, n_customers: int = 800, n_articles: int = 200, n_events: int = 6000) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    departments = ["Menswear", "Ladieswear", "Sport", "Kids", "Divided"]
    groups = ["Garment Upper body", "Garment Lower body", "Shoes", "Accessories"]
    colours = ["Black", "White", "Blue", "Red", "Beige"]
    articles = pd.DataFrame({
        "article_id": np.arange(1, n_articles + 1),
        "product_code": rng.integers(100000, 999999, n_articles),
        "prod_name": [f"Item {i}" for i in range(1, n_articles + 1)],
        "product_type_no": rng.integers(100, 300, n_articles),
        "product_type_name": rng.choice(["T-shirt", "Dress", "Trousers", "Bag"], n_articles),
        "product_group_name": rng.choice(groups, n_articles),
        "graphical_appearance_no": rng.integers(1000000, 1001000, n_articles),
        "graphical_appearance_name": rng.choice(["Solid", "Stripe", "Print"], n_articles),
        "colour_group_code": rng.integers(1, 20, n_articles),
        "colour_group_name": rng.choice(colours, n_articles),
        "perceived_colour_value_id": rng.integers(1, 6, n_articles),
        "perceived_colour_value_name": rng.choice(["Dark", "Light", "Medium"], n_articles),
        "perceived_colour_master_id": rng.integers(1, 10, n_articles),
        "perceived_colour_master_name": rng.choice(colours, n_articles),
        "department_no": rng.integers(1000, 2000, n_articles),
        "department_name": rng.choice(departments, n_articles),
        "index_code": rng.choice(["A", "B"], n_articles),
        "index_name": rng.choice(["Ladieswear", "Menswear"], n_articles),
        "index_group_no": rng.integers(1, 5, n_articles),
        "index_group_name": rng.choice(departments, n_articles),
        "section_no": rng.integers(1, 50, n_articles),
        "section_name": rng.choice(["Everyday", "Basics", "Sport"], n_articles),
        "garment_group_no": rng.integers(1000, 1100, n_articles),
        "garment_group_name": rng.choice(groups, n_articles),
        "detail_desc": "synthetic item description",
    })
    customers = pd.DataFrame({
        "customer_id": [f"c{i:08d}" for i in range(n_customers)],
        "FN": rng.choice([np.nan, 1.0], n_customers),
        "Active": rng.choice([np.nan, 1.0], n_customers),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], n_customers),
        "fashion_news_frequency": rng.choice(["NONE", "Regularly", "Monthly"], n_customers),
        "age": rng.integers(18, 70, n_customers).astype(float),
        "postal_code": [f"{i:08d}" for i in range(n_customers)],
    })
    # 24 months so forecasting has enough complete months to backtest.
    all_days = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    rows = []
    for customer in customers.customer_id:
        first = all_days[rng.integers(0, int(len(all_days) * 0.85))]
        # Many customers buy only once, so the repeat label has both classes.
        extra = int(rng.choice([0, 0, 1, 1, 1, 2, 2, 3, 5]))
        days = [first] + [first + pd.Timedelta(days=int(x)) for x in rng.integers(1, 150, extra)]
        for day in days:
            if day <= all_days[-1]:
                rows.append((day, customer))
    # Guarantee daily coverage so every month counts as complete.
    for day in all_days:
        for _ in range(int(rng.integers(1, 3))):
            rows.append((day, customers.customer_id[rng.integers(0, n_customers)]))
    events = pd.DataFrame(rows, columns=["t_dat", "customer_id"])
    events["article_id"] = rng.choice(articles.article_id, len(events))
    events["price"] = np.round(rng.uniform(0.01, 0.2, len(events)), 6)
    events["sales_channel_id"] = rng.choice([1, 2], len(events))
    articles.to_csv(raw_dir / "articles.csv", index=False)
    customers.to_csv(raw_dir / "customers.csv", index=False)
    events.to_csv(raw_dir / "transactions_train.csv", index=False)
