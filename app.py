"""H&M Fashion Intelligence dashboard. Run with: streamlit run app.py"""
from pathlib import Path
import json
import sys
from html import escape

sys.path.insert(0, str(Path(__file__).parent / "src"))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hmfashion.analyst import Analyst
from hmfashion.recommender import Recommender

ROOT = Path(__file__).parent
st.set_page_config(page_title="H&M | Fashion Intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)

RED, CORAL, AMBER, ROSE = "#9B2F4A", "#D45465", "#E99A70", "#B8798E"


def connect(path: str, read_only: bool = True):
    connection = duckdb.connect(path, read_only=read_only)
    if read_only:
        connection.execute("SET memory_limit='700MB'")
        connection.execute("SET threads=2")
    return connection


@st.cache_resource
def load_tables(path: str, modified: float):
    with connect(path) as con:
        names = ["app_overview", "app_monthly", "app_channel", "app_department", "app_colour",
                 "app_age", "app_cohort", "app_repeat_cohort", "metadata", "data_quality"]
        return {name: con.execute(f"SELECT * FROM {name}").df() for name in names}


@st.cache_resource(show_spinner="Building the article catalogue…")
def load_recommender(path: str, modified: float):
    with connect(path) as con:
        catalog = con.execute("SELECT * FROM app_article").df()
    return Recommender(catalog)


@st.cache_resource(show_spinner="Loading the analyst…")
def load_analyst(path: str, modified: float, artifacts: str):
    return Analyst(Path(path), Path(artifacts))


def artifact(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def present_label(value, column: str | None = None):
    if not isinstance(value, str):
        return value
    text = value.replace("_", " ").strip()
    return text.title() if not (text.isupper() and len(text) <= 3) else text


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    shown.columns = [present_label(str(column)) for column in shown.columns]
    for column in frame.columns:
        if column in {"category", "department_name", "colour", "channel", "feature", "method", "age_band"}:
            shown[present_label(str(column))] = frame[column].map(present_label)
    return shown


def chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,18,25,0.74)",
                      margin=dict(l=16, r=16, t=55, b=20), font=dict(color="#F4EAF0", size=12),
                      title_font=dict(size=17, color="#FFF7FA"), colorway=["#FF5575", "#FF8A7A", "#F4B860", "#D991B0"],
                      hoverlabel=dict(bgcolor="#241A22", bordercolor="#B63A56", font_color="#FFF7FA"),
                      legend=dict(orientation="h", y=-.2, font=dict(color="#F4EAF0")))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.09)", zerolinecolor="rgba(255,255,255,0.14)", tickfont=dict(color="#E6D9E0"), title_font=dict(color="#F4EAF0"), showline=True, linecolor="#5B4650")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)", zerolinecolor="rgba(255,255,255,0.14)", tickfont=dict(color="#E6D9E0"), title_font=dict(color="#F4EAF0"), showline=True, linecolor="#5B4650")
    for index, trace in enumerate(fig.data):
        color = [RED, CORAL, AMBER, ROSE][index % 4]
        for attr in ("x", "y", "labels"):
            values = getattr(trace, attr, None)
            if values is not None and not isinstance(values, str):
                try:
                    setattr(trace, attr, [present_label(value) for value in values])
                except (TypeError, ValueError):
                    pass
        if trace.type == "scatter":
            trace.update(line=dict(color=color, width=3), marker=dict(color=color, size=7))
        elif trace.type in {"bar", "histogram"}:
            trace.update(marker_color=color, marker_line_width=0)
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False, "scrollZoom": False})


def caveats(items: list[str]):
    with st.expander("Limitations of this result", expanded=False):
        for item in items:
            st.markdown(f"- {item}")


def render_analyst_evidence(evidence: dict):
    if not evidence:
        st.caption("No supporting data was returned for this answer.")
        return
    scalars = [(k, v) for k, v in evidence.items() if isinstance(v, (str, int, float, bool)) and v is not None]
    if scalars:
        cols = st.columns(min(len(scalars), 4))
        for i, (key, value) in enumerate(scalars[:4]):
            shown = f"{value:,.3f}" if isinstance(value, float) and abs(value) < 1 else (
                f"{value:,.2f}" if isinstance(value, float) else str(value))
            cols[i % len(cols)].metric(present_label(key), shown)
    for key, value in evidence.items():
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            with st.expander(f"📊 {present_label(key)} · {len(value)} rows", expanded=True):
                st.dataframe(display_frame(pd.DataFrame(value)), hide_index=True, width="stretch")
        elif isinstance(value, dict):
            with st.expander(f"✦ {present_label(key)}", expanded=False):
                st.json(value, expanded=False)


st.sidebar.markdown('<div class="brand"><div class="brand-mark">H&amp;M</div><div><div class="brand-name">H&amp;M</div><div class="brand-sub">Fashion intelligence</div></div></div>', unsafe_allow_html=True)
candidates = [ROOT / "data/hm.duckdb", ROOT / "deployment/hm.duckdb"]
default_db = next((path for path in candidates if path.exists()), candidates[0])
db_path = Path(default_db)
if not db_path.exists():
    st.title("H&M fashion intelligence")
    st.info("Build the analytical database first.")
    st.code("python -m hmfashion.cli ingest --raw-dir data/raw\npython -m hmfashion.cli build", language="bash")
    st.stop()

data = load_tables(str(db_path), db_path.stat().st_mtime)
overview = data["app_overview"].iloc[0]
metadata = data["metadata"].iloc[0]
ARTIFACTS = ROOT / "artifacts"

st.markdown(f'''<div class="hero">
<div class="hero-eyebrow"><span class="status-dot"></span>Fashion intelligence / HISTORICAL H&M DATA</div>
<h1>A clearer view of fashion retail.</h1>
<p>What sold, who returned, and what might come next — built on 28.8 million real H&M purchase events, with evidence and honest limits at every step.</p>
<div class="hero-tags"><span>◈ {overview.purchase_events:,.0f} purchase events</span><span>{overview.customers:,.0f} customers</span><span>Online + in-store</span><span>2018–2020</span></div>
</div>''', unsafe_allow_html=True)

page = st.sidebar.radio("Workspace", ["Business overview", "Channel", "Customers", "Repurchase model",
                                      "Forecasting", "Recommendations", "AI analyst", "Data quality"])
st.sidebar.divider()
st.sidebar.caption("Prices are anonymised by H&M, so value is relative, not a currency. A purchase event is one customer buying one article on one day.")
st.sidebar.caption(f"Built {str(metadata.built_at)[:19]} · {str(metadata.source).upper()}")
st.sidebar.markdown("[H&M dataset / Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) · non-commercial")

details = {
    "Business overview": ("01", "The big picture", "Purchase events, departments and momentum."),
    "Channel": ("02", "Online versus in store", "How and where customers buy."),
    "Customers": ("03", "Who shops and who returns", "Repeat behaviour, age and cohorts."),
    "Repurchase model": ("04", "Will they come back?", "A censoring-aware repeat-visit model."),
    "Forecasting": ("05", "What comes next?", "Backtested monthly purchase-event forecast."),
    "Recommendations": ("06", "What to suggest", "Popularity and content-based, evaluated."),
    "AI analyst": ("07", "Ask in plain English", "A grounded analyst that shows its evidence."),
    "Data quality": ("08", "Evidence you can trust", "Validation checks and source provenance."),
}
number, title, subtitle = details[page]
st.markdown(f'<div class="section-heading"><span class="section-number">{number}</span><div><h2>{title}</h2><p>{subtitle}</p></div></div>', unsafe_allow_html=True)

monthly = data["app_monthly"]

if page == "Business overview":
    a, b, c, d = st.columns(4)
    a.metric("Purchase events", f"{overview.purchase_events:,.0f}")
    b.metric("Customers", f"{overview.customers:,.0f}")
    c.metric("Visits (customer-days)", f"{overview.visits:,.0f}")
    d.metric("Online share", f"{overview.online_share:.1%}")
    departments = data["app_department"]
    if not departments.empty:
        leader = departments.iloc[0]
        st.markdown(f'<div class="insight-strip"><span class="insight-icon">✧</span><div><strong>IN FOCUS · TOP DEPARTMENT</strong><p>{escape(present_label(leader.category))} leads with {leader.purchase_events:,.0f} purchase events from {leader.customers:,.0f} customers.</p></div></div>', unsafe_allow_html=True)
    metric = st.radio("Monthly trend", ["purchase_events", "customers", "online_share"], horizontal=True,
                      format_func=lambda value: {"purchase_events": "◈ Purchase events", "customers": "▦ Customers", "online_share": "◇ Online share"}[value])
    chart(px.line(monthly, x="month", y=metric, markers=True, title="Marketplace performance over time"))
    st.caption("The first and last months are partial in the source data and are excluded from modelling, but shown here for context.")
    left, right = st.columns(2)
    with left:
        chart(px.bar(departments.head(12).sort_values("purchase_events"), x="purchase_events", y="category", orientation="h", title="Leading departments"))
    with right:
        colours = data["app_colour"]
        chart(px.bar(colours.head(12).sort_values("purchase_events"), x="purchase_events", y="colour", orientation="h", title="Leading colours"))
    st.download_button("Download monthly KPIs", monthly.to_csv(index=False), "hm_monthly_kpis.csv", "text/csv")

elif page == "Channel":
    channels = data["app_channel"]
    total = channels.purchase_events.sum()
    a, b = st.columns(2)
    online = channels.loc[channels.channel.eq("Online"), "purchase_events"].sum()
    a.metric("Online purchase events", f"{online:,.0f}", f"{online/total:.1%} of total")
    b.metric("In-store purchase events", f"{total-online:,.0f}", f"{(total-online)/total:.1%} of total")
    chart(px.bar(channels, x="channel", y="purchase_events", color="channel", title="Purchase events by channel"))
    chart(px.line(monthly, x="month", y="online_share", markers=True, title="Online share over time"))
    st.caption("Channel is where the purchase happened. The data does not show how a customer discovered a product, so online versus store is not a funnel.")
    st.dataframe(display_frame(channels), hide_index=True, width="stretch")

elif page == "Customers":
    a, b, c = st.columns(3)
    a.metric("Customers", f"{overview.customers:,.0f}")
    a.caption("Distinct customers in the dataset")
    b.metric("Repeat customers", f"{overview.repeat_customers:,.0f}")
    b.caption("More than one purchase visit")
    c.metric("Median age", f"{overview.median_age:.0f}")
    st.caption("A 'visit' is a customer purchasing on one day. Repeat means more than one visit in the observed window, not guaranteed loyalty.")
    chart(px.bar(data["app_age"], x="age_band", y="customers", title="Customers by age band"))
    repeat = data["app_repeat_cohort"]
    chart(px.bar(repeat, x="cohort_month", y="repeat_rate", title="Share of customers who return, by first-observed month"))
    cohort = data["app_cohort"]
    matrix = cohort.pivot_table(index="cohort", columns="month_index", values="rate")
    matrix.columns = [f"M{c}" for c in matrix.columns]
    chart(px.imshow(matrix, labels=dict(x="Months since first observed purchase", y="First observed cohort", color="Purchase rate"),
                    aspect="auto", color_continuous_scale=[(0, "#D6A0AA"), (0.05, "#B96279"), (0.2, "#A64564"), (0.6, "#8F2945"), (1, "#76243F")],
                    zmin=0, zmax=1, title="Customer cohort retention"))
    st.caption("Because the dataset starts in September 2018, a 'cohort' is a first observed month, not necessarily a true acquisition. Blank cells are unobserved future months, not zero.")

elif page == "Repurchase model":
    report = artifact("repurchase_report.json")
    if not report:
        st.info("Build this model first: python -m hmfashion.cli build")
        st.stop()
    metrics = report["metrics"]["logistic_regression"]
    a, b, c, d = st.columns(4)
    a.metric("Base repeat rate", f"{report['overall_repeat_rate']:.1%}")
    b.metric("Held-out ROC-AUC", f"{metrics['roc_auc']:.2f}")
    c.metric("Average precision", f"{metrics['average_precision']:.3f}")
    d.metric("Top-decile lift", f"{metrics['lift_at_top_decile']:.2f}x")
    st.markdown(f"**Question:** will a customer make another visit within **{report['horizon_days']} days**? "
                f"Trained on **{report['train_customers']:,}** matured customers, tested on **{report['test_customers']:,}** later ones.")
    st.info(f"ROC-AUC {metrics['roc_auc']:.2f} is only slightly better than chance (0.50). The model is a weak ranking signal here, "
            "and that is reported honestly rather than overstated.")
    calibration = pd.DataFrame(report["calibration"])
    chart(px.bar(calibration.melt(id_vars="bucket", value_vars=["predicted", "actual"], var_name="series", value_name="rate"),
                 x="bucket", y="rate", color="series", barmode="group", title="Calibration on the held-out window"))
    st.dataframe(display_frame(pd.DataFrame(report["importance"])), hide_index=True, width="stretch")
    caveats(report["limitations"])

elif page == "Forecasting":
    report = artifact("forecast_report.json")
    if not report:
        st.info("Build the forecast first: python -m hmfashion.cli build")
        st.stop()
    metrics = report["metrics"]
    chosen = report.get("selected_method", min(metrics, key=lambda n: metrics[n]["rmse"]))
    forecast = report["forecast"]
    a, b, c = st.columns(3)
    a.metric(f"Forecast · {report['next_month']}", f"{forecast['point']:,.0f}")
    b.metric("80% range", f"{forecast['low_80']:,.0f} to {forecast['high_80']:,.0f}")
    c.metric("Selected by rolling RMSE", chosen.replace("_", " "))
    history = pd.DataFrame(report["recent_actuals"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.month, y=history.purchase_events, mode="lines+markers", name="Actual", fill="tozeroy", fillcolor="rgba(169,50,80,0.10)"))
    fig.add_trace(go.Scatter(x=[report["next_month"]], y=[forecast["point"]], mode="markers", name="Forecast", marker=dict(size=14, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[report["next_month"], report["next_month"]], y=[forecast["low_95"], forecast["high_95"]], mode="lines", name="95% range", line=dict(width=10, color="rgba(212,84,101,0.28)")))
    chart(fig.update_layout(title="Recent actuals and next-month forecast", yaxis_title="Purchase events"))
    st.markdown(f"**Selected method:** {chosen.replace('_',' ')} by lowest rolling-origin RMSE. With only **{report['complete_months']} complete months**, "
                f"forecast errors are substantial: naive RMSE {metrics['naive_last_value']['rmse']:,.0f}; "
                f"seasonal-naive RMSE {metrics['seasonal_naive']['rmse']:,.0f}.")
    st.dataframe(display_frame(pd.DataFrame(metrics).T.reset_index(names="method")), hide_index=True, width="stretch")
    caveats(report["limitations"])

elif page == "Recommendations":
    recommender = load_recommender(str(db_path), db_path.stat().st_mtime)
    left, right = st.columns(2)
    with left:
        st.markdown("**Similar articles to an article**")
        options = recommender.catalog.sort_values("units", ascending=False).head(400).article_id.tolist()
        article = st.selectbox("Article", options, format_func=lambda a: f"{a} · {recommender.catalog.set_index('article_id').loc[a, 'prod_name']}")
        if article:
            similar = recommender.similar_to_article(article, 8)
            st.dataframe(display_frame(similar[["article_id", "prod_name", "department_name", "units", "score"]].round(3)), hide_index=True, width="stretch")
    with right:
        st.markdown("**Popular articles**")
        popular = recommender.catalog.sort_values(["units", "customers"], ascending=False).head(8)
        st.dataframe(display_frame(popular[["article_id", "prod_name", "department_name", "units", "customers"]]), hide_index=True, width="stretch")
        st.caption("Customer-level histories are not included in the public deployment snapshot.")
    st.divider()
    report = artifact("recommender_report.json")
    if report:
        metrics = report["metrics"]
        a, b, c = st.columns(3)
        a.metric("Popularity MAP@10", f"{metrics['popularity']['map_at_k']:.4f}")
        b.metric("Content-based MAP@10", f"{metrics['content_based']['map_at_k']:.4f}")
        c.metric("Evaluable customers", f"{report['evaluable_customers']:,}")
        st.caption(f"Strict temporal holdout from {report['cutoff']}: catalogue and popularity fitted on earlier purchases only. "
                   "Relevance means the customer actually bought the article in the held-out 8 weeks. Popularity wins this split.")
        caveats(report["limitations"])
    else:
        st.info("Build the evaluation metrics first: python -m hmfashion.cli build")

elif page == "AI analyst":
    st.markdown("**Ask a question in plain English.** Get a clear takeaway, the numbers behind it, and the boundaries of what the data can tell us.")
    analyst = load_analyst(str(db_path), db_path.stat().st_mtime, str(ARTIFACTS))
    if "turns" not in st.session_state:
        st.session_state.turns = []
    question = st.text_input("Your question", placeholder="Is online or in-store bigger?", key="q")
    st.caption("Suggested questions")
    columns = st.columns(3)
    for index, suggestion in enumerate(analyst.suggestions()):
        if columns[index % 3].button(suggestion, key=f"s{index}"):
            question = suggestion
    if question and st.session_state.get("asked") != question:
        with st.spinner("Choosing an approved analysis…"):
            st.session_state.turns.append(analyst.ask(question))
        st.session_state.asked = question
    for turn in reversed(st.session_state.turns):
        with st.container(border=True):
            st.markdown(f"<div class='analyst-question'><span>YOU ASKED</span><br><strong>{escape(turn['question'])}</strong></div>", unsafe_allow_html=True)
            status = "Grounded in project data" if turn.get("tool") else "Outside the approved analyses"
            st.markdown(f"<div class='analyst-answer'><div class='analyst-answer-top'><span class='analyst-sparkle'>✦</span><span class='analyst-label'>THE TAKEAWAY</span><span class='analyst-status'>{escape(status)}</span></div><div class='analyst-answer-copy'>{escape(turn['answer'])}</div></div>", unsafe_allow_html=True)
            if turn.get("evaluation_note"):
                st.info(turn["evaluation_note"], icon="📌")
            if turn.get("evidence"):
                st.markdown("#### The evidence")
                render_analyst_evidence(turn["evidence"])
            if turn.get("caveats"):
                with st.expander("⚠️ How to interpret this", expanded=False):
                    for item in turn["caveats"]:
                        st.markdown(f"- {item}")
            with st.expander("About this answer", expanded=False):
                st.caption(f"Approved analysis: `{turn.get('tool') or 'none'}` · planner: `{turn.get('planner')}`. The analyst selects a fixed project tool; it does not generate SQL.")

else:
    st.subheader("Data checks and source details")
    quality = data["data_quality"].copy()
    quality = quality.loc[quality["count"].gt(0)]
    st.caption("These checks show records that were cleaned, excluded, or need context. A non-zero count is not automatically an error.")
    if quality.empty:
        st.success("No non-zero validation counts were recorded.")
    else:
        info = quality.severity.eq("info")
        if (~info).any():
            st.dataframe(display_frame(quality.loc[~info].rename(columns={"check": "Validation check"})), hide_index=True, width="stretch")
        if info.any():
            with st.expander(f"Informational clean-up steps · {int(info.sum())} checks", expanded=False):
                st.dataframe(display_frame(quality.loc[info].rename(columns={"check": "Validation check"})), hide_index=True, width="stretch")
    st.markdown("""**How the tables are structured**
- **Transactions:** one row per customer buying one article on one day. Exact duplicate rows are removed, because the source has no quantity column.
- **Visits:** one row per customer per day, used as a documented proxy for a basket since there is no order id.
- **Customers and articles:** one row each, joined to transactions by key. Orphan records would stop the build.
- **Value:** prices are anonymised by H&M, so value is relative, not a currency amount.

**Checks that stop the build:** conflicting keys, orphan records, negative prices, or units changing unexpectedly during a join.
""")
    st.markdown("**Dataset and refresh details**")
    details_frame = pd.DataFrame({
        "Detail": ["Dataset", "Database built", "First transaction", "Last transaction", "Value measure"],
        "Value": ["Historical H&M data (Kaggle competition)", str(metadata.built_at)[:19],
                  str(metadata.first_transaction), str(metadata.last_transaction),
                  "Purchase events and relative scaled value; H&M anonymised prices, so no currency"],
    })
    st.dataframe(details_frame, hide_index=True, width="stretch")
