# H&M Data Lab

> **Independent educational project.** H&M Data Lab is not affiliated with, endorsed by, sponsored by or officially connected to H&M Group. It uses the publicly available H&M Personalized Fashion Recommendations dataset for historical analysis.

An evidence-first analytics app built on the real **H&M Personalized Fashion Recommendations** dataset: 28.8 million purchase events, 1.36 million customers and 104,547 articles from September 2018 to September 2020.

It takes three related CSV tables, validates and combines them safely into DuckDB marts, then answers business questions with a dashboard, evaluated models and a grounded analyst that shows its evidence.

## What this project can and cannot do

The dataset is real, but it is **not** a full marketplace dataset. Being explicit about that is the point.

**It has:** transactions (customer, article, date, price, channel), customer age and club status, and rich product attributes (department, colour, garment group).

**It does not have:** order ids, delivery, reviews, payments, freight, sellers, or costs.

So this project does **not** claim late-delivery prediction, seller performance, review analysis, profit or inventory optimisation. It reports **units and relative value** only, because H&M anonymised the prices.

## Results on the real data

| Measure | Result |
|---|---|
| Purchase events | 28,813,419 |
| Customers | 1,362,281 |
| Repeat customers (more than one visit) | 67.1% |
| Average visits per customer | 6.67 |
| Median customer age | 32 |
| Online share of purchase events | 31.7% |
| Repurchase model (90-day) ROC-AUC | 0.56 |
| Recommender MAP@10 — popularity | 0.0034 |
| Recommender MAP@10 — content-based | 0.0024 |
| Forecast method selected | seasonal naive |

Honest readings:

- The **repurchase model is weak** (ROC-AUC 0.56, barely above chance). It is reported as a weak ranking signal, not a predictive win.
- **Popularity beats content-based recommendations** on the temporal holdout. That is a real, useful negative result.
- The **seasonal-naive forecast wins** the backtest; the extra calendar/lag model did not help on 23 complete months.

## How the system works

```
raw CSVs  ->  validate  ->  DuckDB marts  ->  app tables  ->  Streamlit
                           fact_transactions   app_* tables    8 workspaces
                           fact_visits         (small)         + analyst
                           dim_customers
                           dim_articles
```

- **Ingestion** checks keys, orphans, dates and prices, and stops the build on hard failures.
- **Grain contracts:** `fact_transactions` is one row per customer-article-day; `fact_visits` is one row per customer-day, an explicit proxy for a basket since there is no order id.
- **Unit conservation:** the build fails if the joins change the number of purchase events.
- **The dashboard reads only small `app_*` aggregate tables**, never the 28.8M-row fact table, so it fits a small memory budget.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Real data: put transactions_train.csv, customers.csv, articles.csv in data/raw/
PYTHONPATH=src python -m hmfashion.cli ingest --raw-dir data/raw
PYTHONPATH=src python -m hmfashion.cli build
streamlit run app.py
```

The `build` step writes the analysis report, the repurchase model, the recommender evaluation and the forecast into `artifacts/`. Run `hmfashion.appdata.build_app_tables` and `build_deployment` to refresh the compact `deployment/hm.duckdb` snapshot.

Tests:

```bash
PYTHONPATH=src python -m unittest tests.test_project
```

## Workspaces

1. **Business overview** — purchase events, departments, colours and momentum.
2. **Channel** — online versus in-store.
3. **Customers** — repeat behaviour, age bands and cohort retention.
4. **Repurchase model** — a censoring-aware 90-day repeat-visit model.
5. **Forecasting** — backtested monthly purchase events.
6. **Recommendations** — popularity and content-based, evaluated temporally.
7. **AI analyst** — plain-English questions routed to approved tools only.
8. **Data quality** — validation checks and provenance.

## Data and licence

The data is the Kaggle competition dataset **H&M Personalized Fashion Recommendations** (H&M Group, 2022), provided for **non-commercial** use. Raw CSVs are git-ignored and must not be redistributed. The app shows derived aggregates only, and credits the source.

## Limitations

- Prices are anonymised, so value is relative, not a currency amount.
- No order id: a visit is a documented basket proxy.
- 2,974,905 exact duplicate rows were removed; with no quantity column, some may have meant multiple units.
- A "cohort" is a first *observed* month, because the data begins mid-history.
- Historical 2018–2020 data, not a live feed.
