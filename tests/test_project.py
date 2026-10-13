import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hmfashion.analytics import cohort_retention, customer_overview, generate_report  # noqa: E402
from hmfashion.analyst import Analyst  # noqa: E402
from hmfashion.appdata import build_app_tables, build_deployment  # noqa: E402
from hmfashion.demo import generate_demo  # noqa: E402
from hmfashion.forecasting import monthly_series, train_forecast  # noqa: E402
from hmfashion.pipeline import build_database  # noqa: E402
from hmfashion.recommender import Recommender, train_recommender  # noqa: E402
from hmfashion.repurchase import train_repurchase  # noqa: E402


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.db = self.root / "test.duckdb"
        self.artifacts = self.root / "artifacts"
        generate_demo(self.raw, n_customers=300, n_articles=60, n_events=2500)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_database(self.raw, self.db, "synthetic")

    def test_grain_and_unit_conservation(self):
        self.build()
        with duckdb.connect(str(self.db)) as con:
            source = con.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
            facts = con.execute("SELECT count(*) FROM fact_transactions").fetchone()[0]
            self.assertEqual(source, facts)
            self.assertEqual(con.execute("SELECT count(*) - count(DISTINCT customer_key || '|' || t_dat) FROM fact_visits").fetchone()[0], 0)

    def test_conflicting_key_rejected(self):
        path = self.raw / "customers.csv"
        frame = pd.read_csv(path)
        duplicate = frame.iloc[[0]].copy()
        duplicate["age"] = 99
        pd.concat([frame, duplicate]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "conflicting keys"):
            self.build()

    def test_orphans_rejected(self):
        path = self.raw / "transactions_train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "article_id"] = 999999
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.build()


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.db, cls.artifacts = root / "test.duckdb", root / "artifacts"
        generate_demo(root / "raw", n_customers=400, n_articles=80, n_events=4000)
        build_database(root / "raw", cls.db, "synthetic")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_report_and_overview(self):
        report = generate_report(self.db, self.artifacts)
        self.assertGreater(report["purchase_events"], 0)
        self.assertTrue(0 <= report["online_share"] <= 1)
        self.assertTrue((self.artifacts / "findings.md").exists())
        with duckdb.connect(str(self.db), read_only=True) as con:
            overview = customer_overview(con)
            self.assertGreater(overview["customers"], 0)
            self.assertFalse(cohort_retention(con).empty)

    def test_app_tables(self):
        build_app_tables(self.db)
        with duckdb.connect(str(self.db), read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM app_overview").fetchone()[0], 1)
            self.assertGreater(con.execute("SELECT count(*) FROM app_article").fetchone()[0], 0)

    def test_compact_analyst(self):
        build_app_tables(self.db)
        compact = self.db.parent / "compact.duckdb"
        build_deployment(self.db, compact)
        analyst = Analyst(compact, self.artifacts)
        for question in analyst.suggestions():
            result = analyst.ask(question)
            self.assertTrue(result["answer"])
        analyst.con.close()

    def test_forecast_backtest(self):
        report = train_forecast(self.db, self.artifacts)
        self.assertIn(report["selected_method"], {"naive_last_value", "seasonal_naive", "calendar_lag_model"})
        self.assertLessEqual(report["forecast"]["low_80"], report["forecast"]["point"])
        self.assertGreaterEqual(report["forecast"]["high_80"], report["forecast"]["point"])

    def test_repurchase_model(self):
        report = train_repurchase(self.db, self.artifacts)
        metrics = report["metrics"]["logistic_regression"]
        self.assertTrue(0.3 <= metrics["roc_auc"] <= 1.0)
        self.assertGreater(report["matured_customers"], 0)

    def test_recommender_temporal(self):
        report = train_recommender(self.db, self.artifacts)
        self.assertIn("popularity", report["metrics"])
        self.assertIn("content_based", report["metrics"])
        self.assertGreaterEqual(report["metrics"]["popularity"]["map_at_k"], 0)

    def test_recommender_similarity(self):
        with duckdb.connect(str(self.db), read_only=True) as con:
            catalog = con.execute("SELECT * FROM app_article").df()
        recommender = Recommender(catalog)
        article = int(catalog.article_id.iloc[0])
        similar = recommender.similar_to_article(article, 5)
        self.assertFalse(similar.empty)
        self.assertNotIn(article, similar.article_id.tolist())

    def test_analyst_is_grounded(self):
        analyst = Analyst(self.db, self.artifacts)
        answer = analyst.ask("Is online or in-store bigger?")
        self.assertEqual(answer["tool"], "channel")
        refusal = analyst.ask("Can we calculate profit?")
        self.assertIsNone(refusal["tool"])
        self.assertIn("profit", refusal["answer"].lower())
        delivery = analyst.ask("How long is delivery?")
        self.assertIsNone(delivery["tool"])
        unknown = analyst.ask("zzzz nonsense")
        self.assertIsNone(unknown["tool"])


class DashboardTests(unittest.TestCase):
    def test_all_pages(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        if not app.sidebar.radio:
            self.skipTest("Build data/hm.duckdb or deployment/hm.duckdb first")
        for page in ["Business overview", "Channel", "Customers", "Repurchase model", "Forecasting",
                     "Recommendations", "AI analyst", "Data quality"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, f"Dashboard failed on {page}")


if __name__ == "__main__":
    unittest.main()
