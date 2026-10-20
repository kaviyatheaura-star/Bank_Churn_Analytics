"""
Customer Segmentation & Churn Pattern Analytics in European Banking
Streamlit dashboard.

Run:
    pip install -r requirements.txt
    streamlit run app.py

The dataset (European_Bank.csv) can be uploaded in the sidebar, or placed in
the same folder as this file and it will be loaded automatically.
"""

import io
import os

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="European Bank - Churn Analytics",
    page_icon="🏦",
    layout="wide",
)

HIGH_VALUE_THRESHOLD = 100_000  # balance above this counts as high-value

REQUIRED_COLUMNS = [
    "CustomerId", "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited",
]

AGE_BINS = [0, 29, 45, 60, np.inf]
AGE_LABELS = ["<30", "30–45", "46–60", "60+"]

CREDIT_BINS = [-np.inf, 579, 739, np.inf]
CREDIT_LABELS = ["Low", "Medium", "High"]

TENURE_BINS = [-np.inf, 2, 6, np.inf]
TENURE_LABELS = ["New (0–2y)", "Mid-term (3–6y)", "Long-term (7–10y)"]

BAL_BINS = [-np.inf, 0, HIGH_VALUE_THRESHOLD, np.inf]
BAL_LABELS = ["Zero-balance", "Low-balance (≤100k)", "High-balance (>100k)"]

SALARY_BINS = [-np.inf, 50_000, 100_000, 150_000, np.inf]
SALARY_LABELS = ["<50k", "50k–100k", "100k–150k", "150k+"]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean and add segmentation fields."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw.copy()

    # Data ingestion & validation
    df = df.drop_duplicates(subset="CustomerId")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        df[col] = df[col].astype(int)
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0/1 values.")

    # Data cleaning: drop non-analytical fields
    df = df.drop(columns=[c for c in ["Surname", "Year"] if c in df.columns])

    # Derived segmentation fields
    df["AgeBand"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    df["CreditBand"] = pd.cut(df["CreditScore"], CREDIT_BINS, labels=CREDIT_LABELS)
    df["TenureGroup"] = pd.cut(df["Tenure"], TENURE_BINS, labels=TENURE_LABELS)
    df["BalanceSegment"] = pd.cut(df["Balance"], BAL_BINS, labels=BAL_LABELS)
    df["SalaryBand"] = pd.cut(df["EstimatedSalary"], SALARY_BINS, labels=SALARY_LABELS)
    df["HighValue"] = df["Balance"] > HIGH_VALUE_THRESHOLD

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------
def churn_rate(d: pd.DataFrame) -> float:
    """Churn % for a slice of customers (NaN when the slice is empty)."""
    return float(d["Exited"].mean() * 100) if len(d) else float("nan")


def churn_table(d: pd.DataFrame, by: str, baseline: float) -> pd.DataFrame:
    """Customers, churners, churn %, share of churners and risk index by a segment column."""
    grp = d.groupby(by, observed=True)["Exited"]
    out = pd.DataFrame({
        "Customers": grp.size(),
        "Churners": grp.sum().astype(int),
        "Churn %": grp.mean() * 100,
    })
    total_churners = d["Exited"].sum()
    out["Share of churners %"] = out["Churners"] / total_churners * 100 if total_churners else 0
    out["Risk index"] = out["Churn %"] / baseline if baseline else np.nan
    return out.round(2)


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "–"
    if abs(x) >= 1_000_000:
        return f"€{x / 1_000_000:,.1f}M"
    if abs(x) >= 1_000:
        return f"€{x / 1_000:,.1f}k"
    return f"€{x:,.0f}"


def fmt_pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1f}%"


# ---------------------------------------------------------------------------
# Sidebar: data source and filters
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload European_Bank.csv", type="csv")

local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "European_Bank.csv")
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
elif os.path.exists(local_path):
    with open(local_path, "rb") as fh:
        raw_bytes = fh.read()
else:
    st.title("Customer Segmentation & Churn Pattern Analytics")
    st.info("Upload the European_Bank.csv file in the sidebar to begin.")
    st.stop()

try:
    df = prepare(read_csv_bytes(raw_bytes))
except ValueError as err:
    st.error(f"Data validation failed: {err}")
    st.stop()

st.sidebar.header("Segment filters")

all_geo = sorted(df["Geography"].unique())
geo_sel = st.sidebar.multiselect("Geography", all_geo, default=all_geo)

gender_sel = st.sidebar.multiselect(
    "Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique())
)
age_sel = st.sidebar.multiselect("Age band", AGE_LABELS, default=AGE_LABELS)
tenure_sel = st.sidebar.multiselect("Tenure group", TENURE_LABELS, default=TENURE_LABELS)
credit_sel = st.sidebar.multiselect("Credit score band", CREDIT_LABELS, default=CREDIT_LABELS)
balance_sel = st.sidebar.multiselect("Balance segment", BAL_LABELS, default=BAL_LABELS)

all_products = sorted(df["NumOfProducts"].unique())
product_sel = st.sidebar.multiselect("Number of products", all_products, default=all_products)

activity_map = {"Active": 1, "Inactive": 0}
activity_labels = st.sidebar.multiselect(
    "Activity status", list(activity_map), default=list(activity_map)
)
activity_sel = [activity_map[a] for a in activity_labels]

mask = (
    df["Geography"].isin(geo_sel)
    & df["Gender"].isin(gender_sel)
    & df["AgeBand"].isin(age_sel)
    & df["TenureGroup"].isin(tenure_sel)
    & df["CreditBand"].isin(credit_sel)
    & df["BalanceSegment"].isin(balance_sel)
    & df["NumOfProducts"].isin(product_sel)
    & df["IsActiveMember"].isin(activity_sel)
)
view = df[mask]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(view):,}** of **{len(df):,}** customers")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European banking · France, Germany, Spain · Use the sidebar to filter segments.")

if view.empty:
    st.warning("No customers match the selected filters. Widen the selection to see results.")
    st.stop()

baseline_all = churn_rate(df)      # whole-portfolio churn rate (reference)
rate = churn_rate(view)            # churn rate for the current selection

tab_overview, tab_geo, tab_age, tab_hv, tab_profile = st.tabs([
    "Overall summary",
    "Geography",
    "Age & tenure",
    "High-value explorer",
    "Churned vs retained",
])

# ---------------------------------------------------------------------------
# 1. Overall summary
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Key performance indicators")

    active_rate = churn_rate(view[view["IsActiveMember"] == 1])
    inactive_rate = churn_rate(view[view["IsActiveMember"] == 0])
    hv = view[view["HighValue"]]
    hv_rate = churn_rate(hv)
    revenue_at_risk = view.loc[view["Exited"] == 1, "Balance"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall churn rate", fmt_pct(rate),
              delta=f"{rate - baseline_all:+.1f} pp vs portfolio" if baseline_all else None,
              delta_color="inverse")
    c2.metric("Churned customers", f"{int(view['Exited'].sum()):,}",
              help="Customers with Exited = 1 in the current selection.")
    c3.metric("Revenue at risk", fmt_money(revenue_at_risk),
              help="Total account balance held by churned customers in the selection.")
    c4.metric("High-value churn ratio", fmt_pct(hv_rate),
              help=f"Churn % among customers with balance above €{HIGH_VALUE_THRESHOLD:,}.")

    c5, c6, c7 = st.columns(3)
    c5.metric(
        "Engagement drop indicator",
        f"{inactive_rate - active_rate:+.1f} pp" if not (pd.isna(inactive_rate) or pd.isna(active_rate)) else "–",
        help="Churn % of inactive members minus churn % of active members. "
             "A positive value means inactive customers churn more.",
    )
    c6.metric("Active-member churn", fmt_pct(active_rate))
    c7.metric("Inactive-member churn", fmt_pct(inactive_rate))

    st.markdown("#### Churn rate by number of products")
    prod_tbl = churn_table(view, "NumOfProducts", rate)
    st.bar_chart(prod_tbl["Churn %"], y_label="Churn %", x_label="Number of products")
    st.dataframe(prod_tbl, use_container_width=True)

    st.markdown("#### Churn rate by gender and activity")
    gen_act = view.pivot_table(index="Gender", columns="IsActiveMember",
                               values="Exited", aggfunc="mean") * 100
    gen_act.columns = ["Inactive", "Active"]
    st.dataframe(gen_act.round(1), use_container_width=True)

    st.caption(
        "Risk index = segment churn ÷ selection churn (1.00 = average). "
        "Share of churners = segment's share of all churned customers in the selection."
    )

# ---------------------------------------------------------------------------
# 2. Geography
# ---------------------------------------------------------------------------
with tab_geo:
    st.subheader("Churn by geography")

    geo_tbl = churn_table(view, "Geography", rate)
    geo_tbl.index.name = "Country"
    st.bar_chart(geo_tbl["Churn %"], y_label="Churn %", x_label="Country")

    best = geo_tbl["Churn %"].idxmax()
    m1, m2 = st.columns(2)
    m1.metric("Highest-risk country", best,
              help="Country with the highest churn rate in the current selection.")
    m2.metric("Geographic risk index (highest)", f"{geo_tbl.loc[best, 'Risk index']:.2f}",
              help="Highest country churn ÷ selection churn.")

    st.dataframe(geo_tbl, use_container_width=True)

    st.markdown("#### Churn % by age band and country")
    age_geo = view.pivot_table(index="AgeBand", columns="Geography",
                               values="Exited", aggfunc="mean", observed=False) * 100
    st.dataframe(age_geo.round(1), use_container_width=True)

    st.markdown("#### Churn % by gender and country")
    gen_geo = view.pivot_table(index="Gender", columns="Geography",
                               values="Exited", aggfunc="mean") * 100
    st.dataframe(gen_geo.round(1), use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Age & tenure
# ---------------------------------------------------------------------------
with tab_age:
    st.subheader("Age, tenure and credit score comparison")

    age_tbl = churn_table(view, "AgeBand", rate)
    age_tbl.index.name = "Age band"

    ten_tbl = churn_table(view, "TenureGroup", rate)
    ten_tbl.index.name = "Tenure group"

    col_a, col_t = st.columns(2)
    with col_a:
        st.markdown("#### By age band")
        st.bar_chart(age_tbl["Churn %"], y_label="Churn %")
        st.dataframe(age_tbl, use_container_width=True)
    with col_t:
        st.markdown("#### By tenure group")
        st.bar_chart(ten_tbl["Churn %"], y_label="Churn %")
        st.dataframe(ten_tbl, use_container_width=True)

    st.markdown("#### By credit score band")
    cred_tbl = churn_table(view, "CreditBand", rate)
    cred_tbl.index.name = "Credit band"
    st.dataframe(cred_tbl, use_container_width=True)

    st.markdown("#### Churn % by age band and tenure group")
    age_ten = view.pivot_table(index="AgeBand", columns="TenureGroup",
                               values="Exited", aggfunc="mean", observed=False) * 100
    st.dataframe(age_ten.round(1), use_container_width=True)

# ---------------------------------------------------------------------------
# 4. High-value explorer
# ---------------------------------------------------------------------------
with tab_hv:
    st.subheader("High-value customer churn")

    bal_tbl = churn_table(view, "BalanceSegment", rate)
    bal_tbl.index.name = "Balance segment"

    hv_customers = view[view["HighValue"]]
    hv_churners = hv_customers[hv_customers["Exited"] == 1]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("High-value customers", f"{len(hv_customers):,}")
    h2.metric("High-value churn rate", fmt_pct(churn_rate(hv_customers)))
    h3.metric("High-value churners", f"{len(hv_churners):,}")
    h4.metric("High-value revenue at risk", fmt_money(hv_churners["Balance"].sum()))

    st.markdown("#### Churn by balance segment")
    st.bar_chart(bal_tbl["Churn %"], y_label="Churn %")
    st.dataframe(bal_tbl, use_container_width=True)

    st.markdown("#### Salary vs balance: churn by estimated salary band")
    sal_tbl = churn_table(view, "SalaryBand", rate)
    sal_tbl.index.name = "Salary band"
    st.bar_chart(sal_tbl["Churn %"], y_label="Churn %")
    st.dataframe(sal_tbl, use_container_width=True)

    st.markdown("#### High-value customers: churned vs retained")
    if len(hv_customers):
        hv_cmp = hv_customers.groupby("Exited").agg(
            Customers=("CustomerId", "size"),
            AvgBalance=("Balance", "mean"),
            AvgSalary=("EstimatedSalary", "mean"),
            AvgAge=("Age", "mean"),
            AvgProducts=("NumOfProducts", "mean"),
        ).round(1)
        hv_cmp.index = hv_cmp.index.map({0: "Retained", 1: "Churned"})
        st.dataframe(hv_cmp, use_container_width=True)

    st.markdown("#### Top high-value churners (by balance)")
    top_cols = ["CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance",
                "EstimatedSalary", "NumOfProducts", "IsActiveMember", "CreditScore"]
    top_hv = hv_churners.sort_values("Balance", ascending=False)[top_cols]
    st.dataframe(top_hv.head(50), use_container_width=True)
    st.download_button(
        "Download high-value churners (CSV)",
        data=top_hv.to_csv(index=False).encode("utf-8"),
        file_name="high_value_churners.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# 5. Churned vs retained profile
# ---------------------------------------------------------------------------
with tab_profile:
    st.subheader("Churned vs retained profile")

    profile = view.groupby("Exited").agg(
        Customers=("CustomerId", "size"),
        AvgAge=("Age", "mean"),
        AvgCreditScore=("CreditScore", "mean"),
        AvgTenure=("Tenure", "mean"),
        AvgBalance=("Balance", "mean"),
        AvgSalary=("EstimatedSalary", "mean"),
        AvgProducts=("NumOfProducts", "mean"),
        ActiveShare=("IsActiveMember", "mean"),
        HasCardShare=("HasCrCard", "mean"),
    ).round(2)
    profile.index = profile.index.map({0: "Retained", 1: "Churned"})
    profile["ActiveShare"] = (profile["ActiveShare"] * 100).round(1).astype(str) + "%"
    profile["HasCardShare"] = (profile["HasCardShare"] * 100).round(1).astype(str) + "%"
    st.dataframe(profile.T, use_container_width=True)

    st.markdown("#### Drill-down: filtered customer records")
    with st.expander("Show records in the current selection"):
        st.dataframe(view.drop(columns=["HighValue"]), use_container_width=True)
        st.download_button(
            "Download filtered data (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customers.csv",
            mime="text/csv",
        )
