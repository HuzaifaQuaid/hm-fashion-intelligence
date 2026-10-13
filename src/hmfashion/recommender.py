"""Popularity and content-based recommendations, evaluated on a temporal holdout.

The dataset has no ratings, so "relevant" means an article the customer actually
bought in the held-out window. Evaluation is strictly temporal: the catalogue and
popularity are fitted on earlier purchases only.
"""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

CATEGORY_FEATURES = ["department_name", "product_group_name", "garment_group_name",
                     "index_group_name", "section_name", "colour_group_name"]
CATEGORY_WEIGHT = 2.0
DEFAULT_K = 10
EVAL_CUSTOMERS = 6000


def build_catalog(con) -> pd.DataFrame:
    return con.execute("""
        SELECT a.article_id, a.prod_name, a.department_name, a.product_group_name,
               a.garment_group_name, a.index_group_name, a.section_name, a.colour_group_name,
               a.perceived_colour_master_name,
               count(*) AS units, count(DISTINCT t.customer_key) AS customers
        FROM fact_transactions t JOIN dim_articles a USING(article_id)
        GROUP BY ALL
    """).df()


class Recommender:
    """Cosine similarity over category attributes; no article-by-article matrix is stored."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        self.index = {article: row for row, article in enumerate(self.catalog.article_id)}
        blocks = []
        for column in CATEGORY_FEATURES:
            blocks.append(pd.get_dummies(self.catalog[column].fillna("unknown"), prefix=column, dtype=float).to_numpy())
        numeric = np.log1p(self.catalog.units.fillna(0).to_numpy(dtype=float)).reshape(-1, 1)
        numeric = (numeric - numeric.mean()) / (numeric.std() or 1)
        matrix = np.column_stack([block * CATEGORY_WEIGHT for block in blocks] + [numeric])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.popularity = self.catalog.sort_values(["units", "customers"], ascending=False).article_id.tolist()

    def similar_to_article(self, article_id: int, k: int = DEFAULT_K) -> pd.DataFrame:
        if article_id not in self.index:
            return self._popular(k, "unknown article; showing overall popularity")
        scores = self.matrix @ self.matrix[self.index[article_id]]
        order = np.argsort(-scores)
        picked = [row for row in order if row != self.index[article_id]][:k]
        result = self.catalog.iloc[picked].copy()
        result["score"] = scores[picked]
        result["reason"] = "similar attributes"
        return result

    def for_customer(self, purchased: list[int], k: int = DEFAULT_K) -> pd.DataFrame:
        seeds = [article for article in purchased if article in self.index]
        if not seeds:
            return self._popular(k, "no recognizable history; showing overall popularity")
        seed_rows = [self.index[article] for article in seeds]
        scores = self.matrix @ self.matrix[seed_rows].mean(axis=0)
        scores[seed_rows] = -np.inf
        order = np.argsort(-scores)[:k]
        result = self.catalog.iloc[order].copy()
        result["score"] = scores[order]
        result["reason"] = "similar to previously bought articles"
        return result

    def _popular(self, k: int, reason: str) -> pd.DataFrame:
        rows = [self.index[article] for article in self.popularity[:k]]
        result = self.catalog.iloc[rows].copy()
        result["score"] = np.nan
        result["reason"] = reason
        return result


def _average_precision(recommended: list, relevant: set) -> float:
    if not relevant:
        return np.nan
    hits, score = 0, 0.0
    for rank, article in enumerate(recommended, start=1):
        if article in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), len(recommended)) if hits else 0.0


def train_recommender(db_path: Path, artifact_dir: Path, k: int = DEFAULT_K) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path), read_only=True) as con:
        source, built_at = con.execute("SELECT source, built_at::VARCHAR FROM metadata").fetchone()
        last = con.execute("SELECT max(t_dat) FROM fact_transactions").fetchone()[0]
        cutoff = pd.Timestamp(last) - pd.Timedelta(weeks=8)
        # Aggregate in SQL so the 28.8M-row table never enters pandas.
        popularity = con.execute(f"SELECT article_id, count(*) AS units, count(DISTINCT customer_key) AS customers "
                                 f"FROM fact_transactions WHERE t_dat < '{cutoff.date()}' GROUP BY 1").df()
        attributes = con.execute("SELECT * FROM dim_articles").df()
        candidates = con.execute(f"""
            SELECT customer_key FROM (SELECT DISTINCT customer_key FROM fact_transactions WHERE t_dat < '{cutoff.date()}') a
            JOIN (SELECT DISTINCT customer_key FROM fact_transactions WHERE t_dat >= '{cutoff.date()}') b USING(customer_key)
        """).df()
        if len(candidates) > EVAL_CUSTOMERS:
            candidates = candidates.sample(n=EVAL_CUSTOMERS, random_state=0)
        con.register("sample_keys", candidates)
        train_pairs = con.execute(f"""
            SELECT customer_key, list(article_id) AS articles FROM fact_transactions
            WHERE t_dat < '{cutoff.date()}' AND customer_key IN (SELECT customer_key FROM sample_keys)
            GROUP BY 1
        """).df()
        test_pairs = con.execute(f"""
            SELECT customer_key, list(article_id) AS articles FROM fact_transactions
            WHERE t_dat >= '{cutoff.date()}' AND customer_key IN (SELECT customer_key FROM sample_keys)
            GROUP BY 1
        """).df()

    catalog = attributes.merge(popularity, on="article_id", how="inner")
    recommender = Recommender(catalog)
    train_catalog = set(catalog.article_id)

    train_by_customer = dict(zip(train_pairs.customer_key, train_pairs.articles))
    test_by_customer = {row.customer_key: set(row.articles) for row in test_pairs.itertuples()}
    evaluable = [customer for customer in test_by_customer
                 if customer in train_by_customer and test_by_customer[customer] & train_catalog]

    content_scores, popularity_scores, recall_content, recall_popularity = [], [], [], []
    popular_top = recommender.popularity[:k]
    for customer in evaluable:
        seen = train_by_customer[customer]
        relevant = test_by_customer[customer] & train_catalog
        content = list(recommender.for_customer(seen, k).article_id)
        content_scores.append(_average_precision(content, relevant))
        recall_content.append(len(set(content) & relevant) / len(relevant))
        baseline = [article for article in popular_top if article not in set(seen)][:k]
        popularity_scores.append(_average_precision(baseline, relevant))
        recall_popularity.append(len(set(baseline) & relevant) / len(relevant))

    metrics = {
        "content_based": {"map_at_k": float(np.nanmean(content_scores)), "recall_at_k": float(np.nanmean(recall_content))},
        "popularity": {"map_at_k": float(np.nanmean(popularity_scores)), "recall_at_k": float(np.nanmean(recall_popularity))},
    }
    report = {
        "source": source, "database_built_at": str(built_at), "cutoff": str(cutoff.date()), "k": k,
        "evaluable_customers": len(evaluable), "train_catalog_size": int(len(catalog)),
        "metrics": metrics,
        "limitations": [
            "Relevance is defined as an article the customer bought in the held-out 8 weeks, not a rating.",
            "Only customers active in both periods are evaluated; new customers are a cold-start problem.",
            "Content similarity uses category attributes only, because there are no user ratings.",
            "Exact duplicate rows were removed first, so repeat buys of the same article on one day count once.",
        ],
    }
    (artifact_dir / "recommender_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def load_recommender_report(artifact_dir: Path | None) -> dict | None:
    if not artifact_dir:
        return None
    path = Path(artifact_dir) / "recommender_report.json"
    return json.loads(path.read_text()) if path.exists() else None
