"""Streamlit app: predict a developer's salary and explore the survey data.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from db import get_engine, recent_predictions, save_prediction  # noqa: E402
from features import EDUCATION_LEVELS, REMOTE, ROLES  # noqa: E402

st.set_page_config(page_title="Developer Salary Prediction", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "salary_model.joblib")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "salaries_2022.csv.gz")


@st.cache_resource
def engine():
    return get_engine()


bundle = load_model()
models, countries, metrics = bundle["models"], bundle["countries"], bundle["metrics"]

st.title("Developer salary prediction")
st.caption("Trained on the 2022 Stack Overflow Developer Survey (full-time professional developers, pay in USD).")

predict_tab, explore_tab, history_tab, about_tab = st.tabs(["Predict", "Explore", "History", "About the model"])

with predict_tab:
    with st.form("predict"):
        c1, c2 = st.columns(2)
        country = c1.selectbox("Country", countries, index=countries.index("United States") if "United States" in countries else 0,
                               help="Countries with fewer than 200 responses are grouped as Other.")
        role = c2.selectbox("Role", ROLES)
        education = c1.selectbox("Highest education", EDUCATION_LEVELS, index=1)
        remote = c2.selectbox("Work arrangement", REMOTE)
        years = st.slider("Years of professional coding experience", 0.0, 40.0, 3.0, 0.5)
        submitted = st.form_submit_button("Predict salary", type="primary")

    if submitted:
        row = pd.DataFrame([{"country": country, "education": education, "years_pro": years, "role": role, "remote": remote}])
        mid, lo, hi = (float(np.exp(models[k].predict(row)[0])) for k in ("median", "p25", "p75"))
        lo, hi = min(lo, mid), max(hi, mid)
        st.metric("Estimated annual salary", f"${mid:,.0f}")
        st.write(f"Half of similar developers in the survey earned roughly **\\${lo:,.0f} to \\${hi:,.0f}**.")
        try:
            save_prediction(engine(), row.iloc[0].to_dict(), mid, lo, hi)
        except Exception as exc:  # the prediction still shows if the database is down
            st.warning(f"Prediction not saved to history: {exc}")

with explore_tab:
    df = load_data()
    st.subheader("Median salary by country")
    st.bar_chart(df.groupby("country")["salary_usd"].median().sort_values(), horizontal=True, y_label="", x_label="Median USD")
    st.subheader("Median salary by role")
    st.bar_chart(df.groupby("role")["salary_usd"].median().sort_values(), horizontal=True, y_label="", x_label="Median USD")
    st.subheader("Median salary by years of experience")
    years_med = df.assign(years=df["years_pro"].clip(upper=30).round()).groupby("years")["salary_usd"].median()
    st.line_chart(years_med, x_label="Years of professional experience", y_label="Median USD")
    st.caption(f"{len(df):,} survey responses after cleaning.")

with history_tab:
    try:
        rows = recent_predictions(engine())
        if rows:
            st.dataframe(pd.DataFrame(rows).drop(columns=["id"]), hide_index=True)
        else:
            st.info("No predictions yet. Make one on the Predict tab and it will appear here.")
    except Exception as exc:
        st.error(f"Could not read prediction history: {exc}")

with about_tab:
    st.markdown(
        f"""
Gradient-boosted trees (scikit-learn, 300 trees of depth 3) predict the log of annual pay from country, role,
education, work arrangement, and years of professional experience. Two more models trained with quantile loss
give the 25th and 75th percentiles.

Results on {metrics['rows_test']:,} held-out responses:

| Metric | Value |
| --- | --- |
| Mean absolute error | \\${metrics['mae_usd']:,} |
| Median absolute error | \\${metrics['median_abs_error_usd']:,} |
| R² on log salary | {metrics['r2_log_salary']} |
| Baseline: each country's median salary, mean absolute error | \\${metrics['baseline_country_median_mae_usd']:,} |
| Actual salaries inside the predicted 25th–75th band | {metrics['share_inside_p25_p75']:.0%} |

Limits: 2022 pay converted to USD by Stack Overflow; self-reported; survey respondents are not a random sample
of developers; company, city, and skills are not in the model.

Data: [Stack Overflow Developer Survey 2022](https://survey.stackoverflow.co/2022/), Open Database License (ODbL).
"""
    )
