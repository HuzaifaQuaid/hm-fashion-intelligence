"""Grounded analyst: a fixed set of approved tools, every answer shows its evidence.

Without an API key a deterministic keyword router still answers, so the dashboard
never depends on a paid service and never generates free-form SQL. Questions the
data cannot support are refused explicitly.
"""
from pathlib import Path
import json
import os

import duckdb
import numpy as np
import pandas as pd
import requests

from hmfashion import forecasting
from hmfashion.analytics import category_summary, channel_split, customer_overview
from hmfashion.recommender import Recommender, load_recommender_report
from hmfashion.repurchase import load_repurchase_report

MAX_EVIDENCE_ROWS = 8
REFUSALS = {
    "profit": "H&M anonymised the prices and there are no costs, so profit, margin and profitability cannot be calculated.",
    "margin": "Costs are not in the data, so margin cannot be computed.",
    "inventory": "Inventory levels and stock-outs are not recorded, so stock optimisation is not possible.",
    "stock": "Inventory levels are not recorded, so stock-outs and reorder points cannot be computed.",
    "delivery": "There is no shipping, carrier or delivery information in this dataset.",
    "shipping": "There is no shipping or delivery information in this dataset.",
    "review": "There are no customer reviews or ratings in this dataset.",
    "return": "Returns and refunds are not recorded, so return rates cannot be measured.",
    "gender": "Customer gender is not present; only age band and club membership are available.",
    "income": "Customer income is not present, so income segmentation is impossible.",
    "ad spend": "Marketing spend and channel attribution are absent.",
    "campaign": "Campaign and promotion data is absent, so promotion lift cannot be measured.",
}


def _brl(value):  # relative scaled value helper
    return f"{value:,.0f} units of scaled value"


class Analyst:
    def __init__(self, db_path: Path, artifact_dir: Path | None = None):
        self.con = duckdb.connect(str(db_path), read_only=True)
        self.compact = bool(self.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='app_overview'").fetchone()[0])
        self.metadata = self.con.execute(
            "SELECT source, built_at::VARCHAR AS built_at, first_transaction, last_transaction FROM metadata"
        ).df().iloc[0].to_dict()
        self.quality = self.con.execute("SELECT * FROM data_quality").df()
        self.artifact_dir = artifact_dir

    # ---- approved tools -------------------------------------------------
    def _sales_overview(self, params: dict) -> dict:
        row = self.con.execute("SELECT purchase_events AS events, customers, scaled_value AS value FROM app_overview" if self.compact else """
            SELECT count(*) AS events, count(DISTINCT customer_key) AS customers, sum(price) AS value
            FROM fact_transactions
        """).df().iloc[0]
        top = self.con.execute("SELECT * FROM app_department ORDER BY purchase_events DESC LIMIT 5").df() if self.compact else category_summary(self.con, "department_name", 5)
        leader = top.iloc[0]
        return {
            "answer": f"This dataset holds {row.events:,.0f} purchase events from {row.customers:,.0f} customers. "
                      f"{leader.category} is the largest department with {leader.purchase_events:,.0f} events.",
            "evidence": {"purchase_events": int(row.events), "customers": int(row.customers),
                         "top_departments": top.to_dict("records")},
            "caveats": ["Prices are anonymised, so value is relative, not a currency amount.",
                        "A purchase event is one customer buying one article on one day."],
        }

    def _channel(self, params: dict) -> dict:
        table = self.con.execute("SELECT * FROM app_channel").df() if self.compact else channel_split(self.con)
        online = float(table.loc[table.channel.eq("Online"), "purchase_events"].sum())
        total = float(table.purchase_events.sum())
        return {
            "answer": f"Online accounts for {online/total:.1%} of purchase events, with in-store at {1-online/total:.1%}.",
            "evidence": {"channels": table.to_dict("records")},
            "caveats": ["Channel is where the purchase happened; the dataset does not say how the customer discovered the product.",
                        "This is historical 2018-2020 behaviour, not a live feed."],
        }

    def _category_movers(self, params: dict) -> dict:
        series = self.con.execute("SELECT month, category, events FROM app_department_monthly" if self.compact else """
            SELECT date_trunc('month', t_dat) AS month, a.department_name AS category, count(*) AS events
            FROM fact_transactions t JOIN dim_articles a USING(article_id)
            WHERE a.department_name IS NOT NULL GROUP BY 1, 2
        """).df()
        # Only complete months, so a partial boundary month is not read as a decline.
        complete = set(self.con.execute("SELECT month FROM app_complete_months").df().month) if self.compact else set(forecasting.monthly_series(self.con).month)
        series = series.loc[series.month.isin(complete)]
        months = sorted(series.month.unique())
        if len(months) < 5:
            return {"answer": "Too few complete months to compare department momentum.", "evidence": {}, "caveats": []}
        pivot = series.pivot_table(index="category", columns="month", values="events", aggfunc="sum").fillna(0.0)
        latest = pivot[months[-1]]
        baseline = pivot[months[-4:-1]].mean(axis=1)
        change = (latest - baseline).dropna()
        worst = change.sort_values().index[0]
        best = change.sort_values().index[-1]
        table = pd.DataFrame({"category": change.index, "change": change.to_numpy(),
                              "last_month": latest.reindex(change.index).to_numpy(),
                              "prior_three_avg": baseline.reindex(change.index).to_numpy()})
        month_label = pd.Timestamp(months[-1]).strftime("%Y-%m")
        return {
            "answer": f"In {month_label}, {worst} moved most, changing {change[worst]:+,.0f} events against its prior three-month average. "
                      f"{best} rose the most at {change[best]:+,.0f}.",
            "evidence": {"compared_month": month_label,
                         "largest_declines": table.sort_values("change").head(MAX_EVIDENCE_ROWS).round(1).to_dict("records"),
                         "largest_gains": table.sort_values("change", ascending=False).head(MAX_EVIDENCE_ROWS).round(1).to_dict("records")},
            "caveats": ["The final month may be incomplete, so its change can be misleading.",
                        "A single month is noisy and can reflect seasonality or assortment changes, not demand."],
        }

    def _customers(self, params: dict) -> dict:
        overview = self.con.execute("SELECT *, visits * 1.0 / customers AS avg_visits FROM app_overview").df().iloc[0].to_dict() if self.compact else customer_overview(self.con)
        return {
            "answer": f"There are {overview['customers']:,.0f} customers, {overview['repeat_customers']:,.0f} of whom returned "
                      f"({overview['repeat_customers']/overview['customers']:.1%}). Median age is {overview['median_age']:.0f}.",
            "evidence": {"customers": int(overview["customers"]), "repeat_customers": int(overview["repeat_customers"]),
                         "repeat_rate": float(overview["repeat_customers"] / overview["customers"]),
                         "avg_visits": float(overview["avg_visits"]), "median_age": float(overview["median_age"])},
            "caveats": ["A 'return' is another purchase visit within the observed window, not a promise of loyalty.",
                        "The data starts mid-2018, so a first visit is a first observation, not necessarily a first-ever purchase."],
        }

    def _forecast(self, params: dict) -> dict:
        report = forecasting.load_forecast(self.artifact_dir) if self.artifact_dir else None
        if not report:
            return {"answer": "No forecast artifact is available. Run the forecasting step first.", "evidence": {}, "caveats": []}
        chosen = report.get("selected_method", "naive_last_value")
        forecast = report["forecast"]
        return {
            "answer": f"For {report['next_month']} the selected method ({chosen.replace('_',' ')}) expects about "
                      f"{forecast['point']:,.0f} purchase events (80% range {forecast['low_80']:,.0f} to {forecast['high_80']:,.0f}).",
            "evidence": {"selected_method": chosen, "metrics": report["metrics"], "forecast": forecast,
                         "complete_months": report["complete_months"]},
            "caveats": report["limitations"],
        }

    def _recommend(self, params: dict) -> dict:
        report = load_recommender_report(self.artifact_dir) if self.artifact_dir else None
        catalog = self.con.execute("SELECT article_id, prod_name, department_name, units FROM app_article ORDER BY units DESC LIMIT 5" if self.compact else """
            SELECT a.article_id, a.prod_name, a.department_name, count(*) AS units
            FROM fact_transactions t JOIN dim_articles a USING(article_id)
            GROUP BY 1, 2, 3 ORDER BY units DESC LIMIT 5
        """).df()
        metric_note = None
        if report:
            metrics = report["metrics"]
            metric_note = (f"Temporal holdout MAP@10: popularity {metrics['popularity']['map_at_k']:.4f}; "
                           f"content-based {metrics['content_based']['map_at_k']:.4f}.")
        return {
            "answer": "Top-selling articles by units: " + ", ".join(catalog.prod_name.head(5).tolist()) + ".",
            "evidence": {"top_articles": catalog.to_dict("records"), "evaluation": report.get("metrics") if report else None},
            "evaluation_note": metric_note,
            "caveats": report["limitations"] if report else ["No evaluation report is available yet."],
        }

    def _repurchase(self, params: dict) -> dict:
        report = load_repurchase_report(self.artifact_dir) if self.artifact_dir else None
        if not report:
            return {"answer": "No repurchase model artifact is available. Build it first.", "evidence": {}, "caveats": []}
        metrics = report["metrics"]["logistic_regression"]
        return {
            "answer": f"A repeat visit within {report['horizon_days']} days has a base rate of {report['overall_repeat_rate']:.1%}. "
                      f"The logistic model reaches ROC-AUC {metrics['roc_auc']:.2f} and {metrics['lift_at_top_decile']:.2f}x lift in the top decile.",
            "evidence": {"metrics": report["metrics"], "calibration": report["calibration"],
                         "matured_customers": report["matured_customers"], "horizon_days": report["horizon_days"]},
            "caveats": report["limitations"],
        }

    def _data_quality(self, params: dict) -> dict:
        nonzero = self.quality.loc[self.quality["count"].gt(0)]
        return {
            "answer": f"{len(nonzero)} data-quality checks have non-zero counts; they stay visible rather than hidden.",
            "evidence": {"checks": nonzero.to_dict("records"), "source": self.metadata.get("source")},
            "caveats": ["Exact duplicate rows were removed; with no quantity column some may have meant multiple units."],
        }

    # ---- routing --------------------------------------------------------
    def tools(self):
        return {
            "sales_overview": {"keywords": ["revenue", "sales", "sell", "turnover", "how much", "overview"],
                               "description": "Total purchase events, customers and leading departments.", "run": self._sales_overview},
            "channel": {"keywords": ["online", "store", "channel", "in-store", "ecommerce", "e-commerce"],
                        "description": "Online versus in-store share of purchases.", "run": self._channel},
            "category_movers": {"keywords": ["decline", "growth", "drop", "rise", "movers", "department", "category", "trend"],
                                "description": "Departments with the biggest change, last complete month versus prior three.", "run": self._category_movers},
            "customers": {"keywords": ["customer", "repeat", "loyal", "age", "retention", "return"],
                          "description": "Customer counts, repeat rate and age.", "run": self._customers},
            "forecast": {"keywords": ["forecast", "predict", "next month", "future", "projection", "outlook"],
                         "description": "Backtested next-month purchase-event forecast.", "run": self._forecast},
            "recommend": {"keywords": ["recommend", "suggest", "similar", "popular", "best selling", "top-selling",
                                       "top selling", "most sold", "top articles", "top products"],
                          "description": "Top-selling and similar articles with evaluation.", "run": self._recommend},
            "repurchase": {"keywords": ["repurchase", "buy again", "repeat", "churn", "retention"],
                           "description": "Repeat-visit model and its ranking quality.", "run": self._repurchase},
            "data_quality": {"keywords": ["quality", "missing", "duplicate", "trust", "clean", "validity"],
                             "description": "Validation checks and their non-zero counts.", "run": self._data_quality},
        }

    def _plan(self, question: str) -> dict:
        lowered = question.lower()
        for topic, message in REFUSALS.items():
            if topic in lowered:
                return {"planner": "policy", "tool": None, "params": {}, "refusal": message}
        llm = self._llm_plan(question)
        return llm or self._keyword_plan(question)

    def _keyword_plan(self, question: str) -> dict:
        lowered = question.lower()
        tools = self.tools()
        scores = {name: sum(len(keyword) + (4 if " " in keyword else 0) for keyword in tool["keywords"] if keyword in lowered)
                  for name, tool in tools.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return {"planner": "keyword", "tool": None, "params": {},
                    "refusal": "I could not map that question to an approved analysis. Try sales, channel, departments, customers, forecast, recommendations, repurchase or data quality."}
        return {"planner": "keyword", "tool": best, "params": {}}

    def _llm_plan(self, question: str) -> dict | None:
        tools = self.tools()
        catalogue = "\n".join(f"- {name}: {tool['description']}" for name, tool in tools.items())
        prompt = (f"Choose exactly one analysis tool for the question. Reply with JSON: "
                  f'{{"tool": "<name>", "params": {{}}}}.\nTools:\n{catalogue}\n\nQuestion: {question}')
        try:
            if os.environ.get("OPENAI_API_KEY"):
                response = requests.post("https://api.openai.com/v1/chat/completions",
                                         headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                                         json={"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
                                               "response_format": {"type": "json_object"},
                                               "messages": [{"role": "user", "content": prompt}]}, timeout=20)
                response.raise_for_status()
                choice = json.loads(response.json()["choices"][0]["message"]["content"])
            elif os.environ.get("ANTHROPIC_API_KEY"):
                response = requests.post("https://api.anthropic.com/v1/messages",
                                         headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                                         json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), "max_tokens": 200,
                                               "messages": [{"role": "user", "content": prompt}]}, timeout=20)
                response.raise_for_status()
                text = response.json()["content"][0]["text"]
                choice = json.loads(text[text.find("{"):text.rfind("}") + 1])
            else:
                return None
        except Exception:
            return None
        tool = choice.get("tool")
        if tool not in tools:
            return {"planner": "llm", "tool": None, "params": {},
                    "refusal": "The model suggested an analysis that is not on the approved list, so I did not run it."}
        return {"planner": "llm", "tool": tool, "params": choice.get("params") or {}}

    def ask(self, question: str) -> dict:
        plan = self._plan(question)
        if plan.get("tool") is None:
            return {"question": question, "planner": plan["planner"], "tool": None, "answer": plan["refusal"],
                    "evidence": {}, "caveats": ["Only approved analyses can be run; no SQL is generated from the question."]}
        result = self.tools()[plan["tool"]]["run"](plan["params"])
        return {"question": question, "planner": plan["planner"], "tool": plan["tool"], "params": plan["params"], **result}

    def suggestions(self) -> list[str]:
        return ["How much have we sold and where?", "Is online or in-store bigger?", "Which departments are declining?",
                "What do we know about customers?", "What sales can we expect next month?",
                "What are the top-selling articles?"]
