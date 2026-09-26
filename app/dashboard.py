from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Voice Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
    }

    .insight-box {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data
def load_customer_voice_data():

    corpus_path = (
        DATA_DIR
        / "customer_voice_corpus.csv"
    )

    aspects_path = (
        DATA_DIR
        / "customer_voice_aspects.csv"
    )

    sentiment_path = (
        DATA_DIR
        / "aspect_level_sentiment.csv"
    )

    required_files = [
        corpus_path,
        aspects_path,
        sentiment_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing dashboard data files:\n"
            + "\n".join(missing_files)
        )

    corpus_df = pd.read_csv(
        corpus_path
    )

    aspects_df = pd.read_csv(
        aspects_path
    )

    sentiment_df = pd.read_csv(
        sentiment_path
    )

    if len(corpus_df) != 21021:
        raise ValueError(
            f"Unexpected corpus row count: "
            f"{len(corpus_df)}"
        )

    if len(aspects_df) != 21021:
        raise ValueError(
            f"Unexpected aspects row count: "
            f"{len(aspects_df)}"
        )

    if len(sentiment_df) != 21021:
        raise ValueError(
            f"Unexpected sentiment row count: "
            f"{len(sentiment_df)}"
        )

    return {
        "corpus": corpus_df,
        "aspects": aspects_df,
        "sentiment": sentiment_df,
    }


# =========================================================
# LOAD DATA
# =========================================================

data = load_customer_voice_data()

corpus_df = data["corpus"].copy()
aspects_df = data["aspects"].copy()
sentiment_df = data["sentiment"].copy()


# =========================================================
# NORMALIZATION
# =========================================================

corpus_df["sentiment"] = (
    corpus_df["sentiment"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

corpus_df["rating_value"] = pd.to_numeric(
    corpus_df["rating_value"],
    errors="coerce",
)


# =========================================================
# BUSINESS ASPECTS
# =========================================================

ASPECTS = [
    "delivery",
    "customer_service",
    "product_quality",
    "pricing",
    "returns_refunds",
    "website_app",
    "account_billing",
    "availability_selection",
]

ASPECT_LABELS = {
    "delivery": "Delivery",
    "customer_service": "Customer Service",
    "product_quality": "Product Quality",
    "pricing": "Pricing",
    "returns_refunds": "Returns & Refunds",
    "website_app": "Website & App",
    "account_billing": "Account & Billing",
    "availability_selection": "Availability & Selection",
}


# =========================================================
# COMMON METRICS
# =========================================================

total_reviews = len(corpus_df)

positive_count = int(
    (
        corpus_df["sentiment"]
        == "positive"
    ).sum()
)

negative_count = int(
    (
        corpus_df["sentiment"]
        == "negative"
    ).sum()
)

neutral_count = int(
    (
        corpus_df["sentiment"]
        == "neutral"
    ).sum()
)

positive_pct = (
    positive_count
    / total_reviews
    * 100
    if total_reviews
    else 0
)

negative_pct = (
    negative_count
    / total_reviews
    * 100
    if total_reviews
    else 0
)

neutral_pct = (
    neutral_count
    / total_reviews
    * 100
    if total_reviews
    else 0
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    "🎯 Customer Voice Intelligence Platform"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Advanced NLP-powered customer feedback analytics, "
    "business intelligence, complaint intelligence, "
    "and customer experience risk analysis."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Dashboard Controls"
)

sidebar_sentiment = st.sidebar.multiselect(
    "Sentiment Filter",
    options=[
        "positive",
        "negative",
        "neutral",
    ],
    default=[
        "positive",
        "negative",
        "neutral",
    ],
    key="sidebar_sentiment_filter",
)


# =========================================================
# STEP 89 — EXECUTIVE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">'
    "📊 Executive Overview"
    "</div>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Reviews",
        f"{total_reviews:,}",
    )

with col2:
    st.metric(
        "Positive Reviews",
        f"{positive_count:,}",
        f"{positive_pct:.1f}%",
    )

with col3:
    st.metric(
        "Negative Reviews",
        f"{negative_count:,}",
        f"{negative_pct:.1f}%",
    )

with col4:
    st.metric(
        "Neutral Reviews",
        f"{neutral_count:,}",
        f"{neutral_pct:.1f}%",
    )


# =========================================================
# DATASET STATUS
# =========================================================

st.markdown(
    '<div class="section-title">'
    "📁 Dataset Status"
    "</div>",
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success(
        f"Corpus loaded — {len(corpus_df):,} rows"
    )

with status_col2:
    st.success(
        f"Aspect data loaded — {len(aspects_df):,} rows"
    )

with status_col3:
    st.success(
        f"Sentiment data loaded — {len(sentiment_df):,} rows"
    )


# =========================================================
# STEP 89 — SENTIMENT ANALYTICS
# =========================================================

st.markdown(
    '<div class="section-title">'
    "💬 Sentiment Analytics"
    "</div>",
    unsafe_allow_html=True,
)

sentiment_counts = (
    corpus_df["sentiment"]
    .value_counts()
    .reindex(
        [
            "positive",
            "negative",
            "neutral",
        ],
        fill_value=0,
    )
    .reset_index()
)

sentiment_counts.columns = [
    "Sentiment",
    "Reviews",
]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    fig_sentiment_donut = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Reviews",
        hole=0.55,
        title="Overall Sentiment Distribution",
    )

    st.plotly_chart(
        fig_sentiment_donut,
        width="stretch",
    )

with chart_col2:

    fig_sentiment_bar = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Reviews",
        text="Reviews",
        title="Sentiment Volume",
    )

    st.plotly_chart(
        fig_sentiment_bar,
        width="stretch",
    )


# =========================================================
# STEP 90 — BUSINESS ASPECT INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🏢 Business Aspect Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

aspect_records = []

for aspect in ASPECTS:

    aspect_values = pd.to_numeric(
        aspects_df[aspect],
        errors="coerce",
    ).fillna(0)

    count = int(
        (
            aspect_values > 0
        ).sum()
    )

    coverage = (
        count
        / total_reviews
        * 100
        if total_reviews
        else 0
    )

    aspect_records.append(
        {
            "Aspect": ASPECT_LABELS[aspect],
            "Reviews": count,
            "Coverage %": coverage,
        }
    )

aspect_frequency_df = pd.DataFrame(
    aspect_records
)

aspect_col1, aspect_col2 = st.columns(2)

with aspect_col1:

    fig_aspects = px.bar(
        aspect_frequency_df.sort_values(
            "Reviews",
            ascending=True,
        ),
        x="Reviews",
        y="Aspect",
        orientation="h",
        title="Business Aspect Frequency",
        text="Reviews",
    )

    st.plotly_chart(
        fig_aspects,
        width="stretch",
    )

with aspect_col2:

    fig_coverage = px.bar(
        aspect_frequency_df.sort_values(
            "Coverage %",
            ascending=True,
        ),
        x="Coverage %",
        y="Aspect",
        orientation="h",
        title="Aspect Coverage %",
        text="Coverage %",
    )

    st.plotly_chart(
        fig_coverage,
        width="stretch",
    )


# Aspect-level sentiment

aspect_sentiment_records = []

for aspect in ASPECTS:

    label = ASPECT_LABELS[aspect]

    sentiment_column = (
        f"{aspect}_sentiment"
    )

    if sentiment_column not in sentiment_df.columns:
        continue

    sentiment_values = (
        sentiment_df[
            sentiment_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_sentiments = sentiment_values[
        sentiment_values.isin(
            [
                "positive",
                "negative",
                "neutral",
            ]
        )
    ]

    counts = (
        valid_sentiments
        .value_counts()
    )

    aspect_sentiment_records.extend(
        [
            {
                "Aspect": label,
                "Sentiment": "Positive",
                "Reviews": int(
                    counts.get(
                        "positive",
                        0,
                    )
                ),
            },
            {
                "Aspect": label,
                "Sentiment": "Negative",
                "Reviews": int(
                    counts.get(
                        "negative",
                        0,
                    )
                ),
            },
            {
                "Aspect": label,
                "Sentiment": "Neutral",
                "Reviews": int(
                    counts.get(
                        "neutral",
                        0,
                    )
                ),
            },
        ]
    )

aspect_sentiment_df = pd.DataFrame(
    aspect_sentiment_records
)

fig_aspect_sentiment = px.bar(
    aspect_sentiment_df,
    x="Aspect",
    y="Reviews",
    color="Sentiment",
    barmode="stack",
    title="Aspect-Level Sentiment",
)

st.plotly_chart(
    fig_aspect_sentiment,
    width="stretch",
)

st.dataframe(
    aspect_frequency_df,
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 91 — RATING INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "⭐ Rating Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

rated_df = corpus_df[
    corpus_df["rating_value"].notna()
].copy()

average_rating = (
    rated_df["rating_value"].mean()
    if len(rated_df)
    else 0
)

median_rating = (
    rated_df["rating_value"].median()
    if len(rated_df)
    else 0
)

rating_col1, rating_col2, rating_col3 = st.columns(3)

with rating_col1:
    st.metric(
        "Average Rating",
        f"{average_rating:.2f}",
    )

with rating_col2:
    st.metric(
        "Median Rating",
        f"{median_rating:.2f}",
    )

with rating_col3:
    st.metric(
        "Rated Reviews",
        f"{len(rated_df):,}",
    )

rating_distribution = (
    rated_df["rating_value"]
    .value_counts()
    .sort_index()
    .reset_index()
)

rating_distribution.columns = [
    "Rating",
    "Reviews",
]

rating_chart_col1, rating_chart_col2 = st.columns(2)

with rating_chart_col1:

    fig_rating = px.bar(
        rating_distribution,
        x="Rating",
        y="Reviews",
        text="Reviews",
        title="Rating Distribution",
    )

    st.plotly_chart(
        fig_rating,
        width="stretch",
    )

with rating_chart_col2:

    sentiment_rating = (
        corpus_df
        .groupby("sentiment")[
            "rating_value"
        ]
        .mean()
        .reset_index()
    )

    fig_sentiment_rating = px.bar(
        sentiment_rating,
        x="sentiment",
        y="rating_value",
        text="rating_value",
        title="Average Rating by Sentiment",
    )

    st.plotly_chart(
        fig_sentiment_rating,
        width="stretch",
    )


# =========================================================
# STEP 92 — TIME & TREND INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📈 Time & Trend Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

trend_df = corpus_df.copy()

trend_df["review_date"] = (
    pd.to_datetime(
        trend_df[
            "review_date_clean"
        ],
        errors="coerce",
        utc=True,
    )
    .dt
    .tz_localize(None)
)

trend_df = trend_df[
    trend_df["review_date"].notna()
].copy()

trend_df["month"] = (
    trend_df["review_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_volume = (
    trend_df
    .groupby("month")
    .size()
    .reset_index(
        name="Reviews"
    )
)

fig_monthly_volume = px.line(
    monthly_volume,
    x="month",
    y="Reviews",
    markers=True,
    title="Monthly Review Volume",
)

st.plotly_chart(
    fig_monthly_volume,
    width="stretch",
)

monthly_sentiment = (
    trend_df
    .groupby(
        [
            "month",
            "sentiment",
        ]
    )
    .size()
    .reset_index(
        name="Reviews"
    )
)

fig_monthly_sentiment = px.line(
    monthly_sentiment,
    x="month",
    y="Reviews",
    color="sentiment",
    markers=True,
    title="Monthly Sentiment Volume",
)

st.plotly_chart(
    fig_monthly_sentiment,
    width="stretch",
)

monthly_rating = (
    trend_df
    .groupby("month")[
        "rating_value"
    ]
    .mean()
    .reset_index()
)

fig_monthly_rating = px.line(
    monthly_rating,
    x="month",
    y="rating_value",
    markers=True,
    title="Monthly Average Rating",
)

st.plotly_chart(
    fig_monthly_rating,
    width="stretch",
)

yearly_summary = (
    trend_df
    .assign(
        year=trend_df[
            "review_date"
        ].dt.year
    )
    .groupby("year")
    .agg(
        Reviews=(
            "review_text",
            "count",
        ),
        Average_Rating=(
            "rating_value",
            "mean",
        ),
    )
    .reset_index()
)

yearly_summary["Positive %"] = (
    trend_df
    .assign(
        year=trend_df[
            "review_date"
        ].dt.year
    )
    .groupby("year")[
        "sentiment"
    ]
    .apply(
        lambda x:
        (
            x == "positive"
        ).mean()
        * 100
    )
    .values
)

yearly_summary["Negative %"] = (
    trend_df
    .assign(
        year=trend_df[
            "review_date"
        ].dt.year
    )
    .groupby("year")[
        "sentiment"
    ]
    .apply(
        lambda x:
        (
            x == "negative"
        ).mean()
        * 100
    )
    .values
)

yearly_summary["Neutral %"] = (
    trend_df
    .assign(
        year=trend_df[
            "review_date"
        ].dt.year
    )
    .groupby("year")[
        "sentiment"
    ]
    .apply(
        lambda x:
        (
            x == "neutral"
        ).mean()
        * 100
    )
    .values
)

st.dataframe(
    yearly_summary,
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 93 — COUNTRY & GEOGRAPHIC INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🌍 Country & Geographic Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

geo_df = corpus_df.copy()

geo_df["Country"] = (
    geo_df["Country"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

geo_countries = sorted(
    geo_df["Country"]
    .unique()
    .tolist()
)

selected_geo_country = st.selectbox(
    "Select Country",
    options=[
        "All Countries"
    ] + geo_countries,
    key="geo_country_selector",
)

if (
    selected_geo_country
    != "All Countries"
):

    selected_geo_df = geo_df[
        geo_df["Country"]
        == selected_geo_country
    ].copy()

else:

    selected_geo_df = geo_df.copy()

geo_total = len(selected_geo_df)

geo_negative = int(
    (
        selected_geo_df["sentiment"]
        == "negative"
    ).sum()
)

geo_positive = int(
    (
        selected_geo_df["sentiment"]
        == "positive"
    ).sum()
)

geo_avg_rating = (
    selected_geo_df[
        "rating_value"
    ].mean()
    if selected_geo_df[
        "rating_value"
    ].notna().any()
    else 0
)

geo_col1, geo_col2, geo_col3, geo_col4 = st.columns(4)

with geo_col1:
    st.metric(
        "Country Reviews",
        f"{geo_total:,}",
    )

with geo_col2:
    st.metric(
        "Positive",
        f"{geo_positive:,}",
    )

with geo_col3:
    st.metric(
        "Negative",
        f"{geo_negative:,}",
    )

with geo_col4:
    st.metric(
        "Average Rating",
        f"{geo_avg_rating:.2f}",
    )

country_volume = (
    geo_df
    .groupby("Country")
    .size()
    .reset_index(
        name="Reviews"
    )
    .sort_values(
        "Reviews",
        ascending=False,
    )
    .head(15)
)

fig_country_volume = px.bar(
    country_volume.sort_values(
        "Reviews"
    ),
    x="Reviews",
    y="Country",
    orientation="h",
    title="Top 15 Countries by Review Volume",
)

st.plotly_chart(
    fig_country_volume,
    width="stretch",
)

country_summary = (
    geo_df
    .groupby("Country")
    .agg(
        Reviews=(
            "review_text",
            "count",
        ),
        Negative=(
            "sentiment",
            lambda x:
            (
                x == "negative"
            ).sum(),
        ),
        Positive=(
            "sentiment",
            lambda x:
            (
                x == "positive"
            ).sum(),
        ),
        Average_Rating=(
            "rating_value",
            "mean",
        ),
    )
    .reset_index()
)

country_summary[
    "Complaint Rate %"
] = (
    country_summary["Negative"]
    / country_summary["Reviews"]
    * 100
)

country_summary_filtered = (
    country_summary[
        country_summary["Reviews"]
        >= 50
    ]
    .sort_values(
        "Complaint Rate %",
        ascending=False,
    )
)

country_sentiment = (
    geo_df
    .groupby(
        [
            "Country",
            "sentiment",
        ]
    )
    .size()
    .reset_index(
        name="Reviews"
    )
)

country_sentiment = (
    country_sentiment[
        country_sentiment[
            "Country"
        ].isin(
            country_summary_filtered[
                "Country"
            ]
        )
    ]
)

fig_country_sentiment = px.bar(
    country_sentiment,
    x="Country",
    y="Reviews",
    color="sentiment",
    barmode="stack",
    title="Country Sentiment — Minimum 50 Reviews",
)

st.plotly_chart(
    fig_country_sentiment,
    width="stretch",
)

st.dataframe(
    country_summary_filtered,
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 95 — ADVANCED CUSTOMER VOICE INSIGHTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🧠 Advanced Customer Voice Insights"
    "</div>",
    unsafe_allow_html=True,
)

insight_df = aspect_sentiment_df.copy()

insight_pivot = (
    insight_df
    .pivot(
        index="Aspect",
        columns="Sentiment",
        values="Reviews",
    )
    .fillna(0)
    .reset_index()
)

for column in [
    "Positive",
    "Negative",
    "Neutral",
]:

    if column not in insight_pivot.columns:
        insight_pivot[column] = 0

insight_pivot["Total"] = (
    insight_pivot["Positive"]
    + insight_pivot["Negative"]
    + insight_pivot["Neutral"]
)

insight_pivot["Positive %"] = (
    insight_pivot["Positive"]
    / insight_pivot["Total"]
    * 100
)

insight_pivot["Negative %"] = (
    insight_pivot["Negative"]
    / insight_pivot["Total"]
    * 100
)

insight_pivot["Neutral %"] = (
    insight_pivot["Neutral"]
    / insight_pivot["Total"]
    * 100
)

insight_pivot["Negative Impact"] = (
    insight_pivot["Negative"]
    * (
        insight_pivot["Negative %"]
        / 100
    )
)

pain_points = (
    insight_pivot
    .sort_values(
        [
            "Negative",
            "Negative %",
        ],
        ascending=False,
    )
    .head(5)
)

strengths = (
    insight_pivot
    .sort_values(
        [
            "Positive %",
            "Positive",
        ],
        ascending=False,
    )
    .head(5)
)

attention_required = (
    insight_pivot[
        insight_pivot["Total"] >= 100
    ]
    .sort_values(
        "Negative %",
        ascending=False,
    )
)

pain_col, strength_col = st.columns(2)

with pain_col:

    st.subheader(
        "🚨 Critical Customer Pain Points"
    )

    st.dataframe(
        pain_points[
            [
                "Aspect",
                "Negative",
                "Negative %",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with strength_col:

    st.subheader(
        "💚 Customer Strengths"
    )

    st.dataframe(
        strengths[
            [
                "Aspect",
                "Positive",
                "Positive %",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

st.subheader(
    "⚠️ Attention Required"
)

st.dataframe(
    attention_required[
        [
            "Aspect",
            "Total",
            "Negative",
            "Negative %",
            "Positive %",
        ]
    ],
    width="stretch",
    hide_index=True,
)

st.subheader(
    "🤖 Automated Business Insights"
)

most_discussed = (
    insight_pivot
    .sort_values(
        "Total",
        ascending=False,
    )
    .iloc[0]
)

highest_negative_volume = (
    insight_pivot
    .sort_values(
        "Negative",
        ascending=False,
    )
    .iloc[0]
)

highest_negative_share = (
    insight_pivot
    .sort_values(
        "Negative %",
        ascending=False,
    )
    .iloc[0]
)

strongest_positive_share = (
    insight_pivot
    .sort_values(
        "Positive %",
        ascending=False,
    )
    .iloc[0]
)

st.write(
    f"• Overall sentiment: "
    f"{positive_pct:.1f}% positive, "
    f"{negative_pct:.1f}% negative, "
    f"{neutral_pct:.1f}% neutral."
)

st.write(
    f"• Most frequently discussed area: "
    f"**{most_discussed['Aspect']}** "
    f"with {int(most_discussed['Total']):,} "
    f"aspect-level reviews."
)

st.write(
    f"• Highest negative review volume: "
    f"**{highest_negative_volume['Aspect']}** "
    f"with {int(highest_negative_volume['Negative']):,} "
    f"negative reviews."
)

st.write(
    f"• Highest negative sentiment share: "
    f"**{highest_negative_share['Aspect']}** "
    f"at {highest_negative_share['Negative %']:.1f}%."
)

st.write(
    f"• Strongest positive sentiment share: "
    f"**{strongest_positive_share['Aspect']}** "
    f"at {strongest_positive_share['Positive %']:.1f}%."
)

st.subheader(
    "📊 Aspect Intelligence Matrix"
)

st.dataframe(
    insight_pivot[
        [
            "Aspect",
            "Total",
            "Positive",
            "Negative",
            "Neutral",
            "Positive %",
            "Negative %",
            "Neutral %",
        ]
    ],
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 96 — CUSTOMER COMPLAINT INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🚨 Customer Complaint Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Deep-dive into negative customer feedback, "
    "complaint concentration, ratings, geographic patterns, "
    "and complaint trends."
)

complaint_df = corpus_df[
    corpus_df["sentiment"]
    == "negative"
].copy()

complaint_count = len(
    complaint_df
)

complaint_rate = (
    complaint_count
    / total_reviews
    * 100
    if total_reviews
    else 0
)

complaint_avg_rating = (
    complaint_df[
        "rating_value"
    ].mean()
    if complaint_df[
        "rating_value"
    ].notna().any()
    else 0
)

complaint_aspects = int(
    (
        insight_pivot["Negative"]
        > 0
    ).sum()
)

complaint_col1, complaint_col2, complaint_col3, complaint_col4 = (
    st.columns(4)
)

with complaint_col1:
    st.metric(
        "Negative Reviews",
        f"{complaint_count:,}",
    )

with complaint_col2:
    st.metric(
        "Complaint Rate",
        f"{complaint_rate:.1f}%",
    )

with complaint_col3:
    st.metric(
        "Avg Complaint Rating",
        f"{complaint_avg_rating:.2f}",
    )

with complaint_col4:
    st.metric(
        "Affected Business Areas",
        f"{complaint_aspects}",
    )


# Complaint volume by aspect

complaint_aspect_df = (
    insight_pivot[
        [
            "Aspect",
            "Negative",
        ]
    ]
    .sort_values(
        "Negative",
        ascending=False,
    )
    .copy()
)

total_negative_aspect = (
    complaint_aspect_df["Negative"]
    .sum()
)

complaint_aspect_df[
    "Complaint Share %"
] = (
    complaint_aspect_df["Negative"]
    / total_negative_aspect
    * 100
    if total_negative_aspect
    else 0
)

complaint_chart_col1, complaint_chart_col2 = st.columns(2)

with complaint_chart_col1:

    fig_complaint_aspect = px.bar(
        complaint_aspect_df.sort_values(
            "Negative"
        ),
        x="Negative",
        y="Aspect",
        orientation="h",
        text="Negative",
        title="Negative Reviews by Business Area",
    )

    st.plotly_chart(
        fig_complaint_aspect,
        width="stretch",
    )

with complaint_chart_col2:

    fig_complaint_share = px.bar(
        complaint_aspect_df.sort_values(
            "Complaint Share %"
        ),
        x="Complaint Share %",
        y="Aspect",
        orientation="h",
        text="Complaint Share %",
        title="Complaint Concentration by Aspect",
    )

    st.plotly_chart(
        fig_complaint_share,
        width="stretch",
    )

st.dataframe(
    complaint_aspect_df,
    width="stretch",
    hide_index=True,
)


# Complaint vs rating

complaint_rating_df = (
    complaint_df[
        complaint_df[
            "rating_value"
        ].notna()
    ]
    .groupby(
        "rating_value"
    )
    .size()
    .reset_index(
        name="Complaints"
    )
)

fig_complaint_rating = px.bar(
    complaint_rating_df,
    x="rating_value",
    y="Complaints",
    text="Complaints",
    title="Negative Reviews by Rating",
)

st.plotly_chart(
    fig_complaint_rating,
    width="stretch",
)

complaint_rating_summary = (
    complaint_df
    .groupby("rating_value")
    .agg(
        Complaints=(
            "review_text",
            "count",
        )
    )
    .reset_index()
)

complaint_rating_summary[
    "Complaint Share %"
] = (
    complaint_rating_summary[
        "Complaints"
    ]
    / complaint_rating_summary[
        "Complaints"
    ].sum()
    * 100
)

st.dataframe(
    complaint_rating_summary,
    width="stretch",
    hide_index=True,
)


# Complaint country patterns

complaint_country_df = complaint_df.copy()

complaint_country_df["Country"] = (
    complaint_country_df["Country"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

country_total = (
    geo_df
    .groupby("Country")
    .size()
    .reset_index(
        name="Reviews"
    )
)

country_complaints = (
    complaint_country_df
    .groupby("Country")
    .size()
    .reset_index(
        name="Complaints"
    )
)

complaint_country_summary = (
    country_total
    .merge(
        country_complaints,
        on="Country",
        how="left",
    )
)

complaint_country_summary[
    "Complaints"
] = complaint_country_summary[
    "Complaints"
].fillna(0)

complaint_country_summary[
    "Complaint Rate %"
] = (
    complaint_country_summary[
        "Complaints"
    ]
    / complaint_country_summary[
        "Reviews"
    ]
    * 100
)

complaint_country_summary = (
    complaint_country_summary[
        complaint_country_summary[
            "Reviews"
        ] >= 50
    ]
    .sort_values(
        "Complaint Rate %",
        ascending=False,
    )
)

fig_complaint_country = px.bar(
    complaint_country_summary.head(15),
    x="Country",
    y="Complaint Rate %",
    text="Complaint Rate %",
    title="Top Complaint Rates by Country — Minimum 50 Reviews",
)

st.plotly_chart(
    fig_complaint_country,
    width="stretch",
)

st.dataframe(
    complaint_country_summary,
    width="stretch",
    hide_index=True,
)


# Complaint trends

complaint_trend = (
    complaint_df.copy()
)

complaint_trend["review_date"] = (
    pd.to_datetime(
        complaint_trend[
            "review_date_clean"
        ],
        errors="coerce",
        utc=True,
    )
    .dt
    .tz_localize(None)
)

complaint_trend = complaint_trend[
    complaint_trend[
        "review_date"
    ].notna()
].copy()

complaint_trend["month"] = (
    complaint_trend[
        "review_date"
    ]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_complaints = (
    complaint_trend
    .groupby("month")
    .size()
    .reset_index(
        name="Complaints"
    )
)

all_trend = (
    corpus_df.copy()
)

all_trend["review_date"] = (
    pd.to_datetime(
        all_trend[
            "review_date_clean"
        ],
        errors="coerce",
        utc=True,
    )
    .dt
    .tz_localize(None)
)

all_trend = all_trend[
    all_trend[
        "review_date"
    ].notna()
].copy()

all_trend["month"] = (
    all_trend[
        "review_date"
    ]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_total = (
    all_trend
    .groupby("month")
    .size()
    .reset_index(
        name="Reviews"
    )
)

monthly_complaints = (
    monthly_complaints
    .merge(
        monthly_total,
        on="month",
        how="left",
    )
)

monthly_complaints[
    "Complaint Rate %"
] = (
    monthly_complaints[
        "Complaints"
    ]
    / monthly_complaints[
        "Reviews"
    ]
    * 100
)

trend_complaint_col1, trend_complaint_col2 = st.columns(2)

with trend_complaint_col1:

    fig_monthly_complaints = px.line(
        monthly_complaints,
        x="month",
        y="Complaints",
        markers=True,
        title="Monthly Complaint Volume",
    )

    st.plotly_chart(
        fig_monthly_complaints,
        width="stretch",
    )

with trend_complaint_col2:

    fig_monthly_complaint_rate = px.line(
        monthly_complaints,
        x="month",
        y="Complaint Rate %",
        markers=True,
        title="Monthly Complaint Rate",
    )

    st.plotly_chart(
        fig_monthly_complaint_rate,
        width="stretch",
    )

st.dataframe(
    monthly_complaints,
    width="stretch",
    hide_index=True,
)


# Complaint explorer

st.subheader(
    "🔎 Complaint Review Explorer"
)

complaint_search = st.text_input(
    "Search complaint text",
    key="complaint_explorer_search",
)

complaint_rating_options = sorted(
    complaint_df[
        "rating_value"
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_complaint_ratings = st.multiselect(
    "Complaint Ratings",
    options=complaint_rating_options,
    default=[],
    key="complaint_explorer_ratings",
)

complaint_country_options = sorted(
    complaint_df["Country"]
    .fillna("Unknown")
    .astype(str)
    .unique()
    .tolist()
)

selected_complaint_countries = st.multiselect(
    "Complaint Countries",
    options=complaint_country_options,
    default=[],
    key="complaint_explorer_countries",
)

selected_complaint_aspects = st.multiselect(
    "Complaint Business Areas",
    options=[
        ASPECT_LABELS[a]
        for a in ASPECTS
    ],
    default=[],
    key="complaint_explorer_aspects",
)

complaint_explorer = complaint_df.copy()

if complaint_search.strip():

    complaint_explorer = (
        complaint_explorer[
            complaint_explorer[
                "review_text"
            ]
            .fillna("")
            .str.contains(
                complaint_search,
                case=False,
                na=False,
            )
        ]
    )

if selected_complaint_ratings:

    complaint_explorer = (
        complaint_explorer[
            complaint_explorer[
                "rating_value"
            ].isin(
                selected_complaint_ratings
            )
        ]
    )

if selected_complaint_countries:

    complaint_explorer = (
        complaint_explorer[
            complaint_explorer[
                "Country"
            ]
            .fillna("Unknown")
            .isin(
                selected_complaint_countries
            )
        ]
    )

if selected_complaint_aspects:

    aspect_mask = pd.Series(
        False,
        index=complaint_explorer.index,
        dtype=bool,
    )

    for aspect in ASPECTS:

        label = ASPECT_LABELS[aspect]

        if (
            label
            in selected_complaint_aspects
        ):

            aspect_values = pd.to_numeric(
                aspects_df.loc[
                    complaint_explorer.index,
                    aspect,
                ],
                errors="coerce",
            ).fillna(0)

            aspect_mask = (
                aspect_mask
                |
                (
                    aspect_values > 0
                )
            )

    complaint_explorer = (
        complaint_explorer[
            aspect_mask
        ]
    )

complaint_explorer_count = len(
    complaint_explorer
)

complaint_explorer_avg_rating = (
    complaint_explorer[
        "rating_value"
    ].mean()
    if complaint_explorer[
        "rating_value"
    ].notna().any()
    else 0
)

ce1, ce2 = st.columns(2)

with ce1:

    st.metric(
        "Matching Complaints",
        f"{complaint_explorer_count:,}",
    )

with ce2:

    st.metric(
        "Average Rating",
        f"{complaint_explorer_avg_rating:.2f}",
    )

st.dataframe(
    complaint_explorer[
        [
            "review_text",
            "rating_value",
            "review_date_clean",
            "Country",
            "sentiment",
        ]
    ].head(100),
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 97 — CUSTOMER EXPERIENCE RISK & OPPORTUNITY
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🎯 Customer Experience Risk & Opportunity Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Aspect-level prioritization using review volume, "
    "negative sentiment, positive sentiment, and customer ratings."
)

risk_df = insight_pivot.copy()

aspect_rating_records = []

for aspect in ASPECTS:

    label = ASPECT_LABELS[aspect]

    aspect_flag = pd.to_numeric(
        aspects_df[aspect],
        errors="coerce",
    ).fillna(0)

    aspect_rows = corpus_df[
        aspect_flag > 0
    ].copy()

    aspect_rated = (
        aspect_rows[
            "rating_value"
        ]
        .dropna()
    )

    if len(aspect_rated) > 0:

        avg_aspect_rating = (
            aspect_rated.mean()
        )

    else:

        avg_aspect_rating = 0

    aspect_rating_records.append(
        {
            "Aspect": label,
            "Average Rating":
                avg_aspect_rating,
        }
    )

aspect_rating_df = pd.DataFrame(
    aspect_rating_records
)

risk_df = risk_df.merge(
    aspect_rating_df,
    on="Aspect",
    how="left",
)

max_negative = (
    risk_df["Negative"].max()
)

if max_negative > 0:

    risk_df[
        "Negative Volume Index"
    ] = (
        risk_df["Negative"]
        / max_negative
        * 100
    )

else:

    risk_df[
        "Negative Volume Index"
    ] = 0

risk_df[
    "Attention Score"
] = (
    0.60
    * risk_df["Negative %"]
    +
    0.40
    * risk_df[
        "Negative Volume Index"
    ]
)

risk_df[
    "Positive Strength %"
] = (
    risk_df["Positive %"]
)

st.subheader(
    "Business Aspect Attention Matrix"
)

st.dataframe(
    risk_df[
        [
            "Aspect",
            "Total",
            "Positive",
            "Negative",
            "Neutral",
            "Positive %",
            "Negative %",
            "Average Rating",
            "Attention Score",
        ]
    ].sort_values(
        "Attention Score",
        ascending=False,
    ),
    width="stretch",
    hide_index=True,
)

fig_attention = px.bar(
    risk_df.sort_values(
        "Attention Score"
    ),
    x="Attention Score",
    y="Aspect",
    orientation="h",
    text="Attention Score",
    title="Aspect Attention Score",
)

st.plotly_chart(
    fig_attention,
    width="stretch",
)

fig_risk_scatter = px.scatter(
    risk_df,
    x="Positive",
    y="Negative",
    size="Total",
    hover_name="Aspect",
    title="Positive vs Negative Aspect Volume",
)

st.plotly_chart(
    fig_risk_scatter,
    width="stretch",
)

indicator_col1, indicator_col2, indicator_col3, indicator_col4 = (
    st.columns(4)
)

highest_attention = (
    risk_df
    .sort_values(
        "Attention Score",
        ascending=False,
    )
    .iloc[0]
)

highest_rating = (
    risk_df
    .sort_values(
        "Average Rating",
        ascending=False,
    )
    .iloc[0]
)

largest_positive = (
    risk_df
    .sort_values(
        "Positive",
        ascending=False,
    )
    .iloc[0]
)

largest_negative = (
    risk_df
    .sort_values(
        "Negative",
        ascending=False,
    )
    .iloc[0]
)

with indicator_col1:

    st.metric(
        "Highest Attention",
        highest_attention["Aspect"],
    )

with indicator_col2:

    st.metric(
        "Highest Aspect Rating",
        highest_rating["Aspect"],
    )

with indicator_col3:

    st.metric(
        "Largest Positive Volume",
        largest_positive["Aspect"],
    )

with indicator_col4:

    st.metric(
        "Largest Negative Volume",
        largest_negative["Aspect"],
    )

fig_volume_attention = px.scatter(
    risk_df,
    x="Total",
    y="Attention Score",
    size="Negative",
    hover_name="Aspect",
    title="Review Volume vs Attention Score",
)

st.plotly_chart(
    fig_volume_attention,
    width="stretch",
)

st.subheader(
    "Executive Interpretation"
)

st.write(
    f"**{highest_attention['Aspect']}** has the "
    f"highest calculated Attention Score based on "
    f"negative sentiment share and negative review volume."
)

st.write(
    f"**{largest_negative['Aspect']}** generates the "
    f"largest number of negative aspect-level reviews."
)

st.write(
    f"**{largest_positive['Aspect']}** has the largest "
    f"positive aspect-level review volume."
)

st.caption(
    "Attention Score is a dashboard prioritization metric "
    "created from observed review data. It is not a predictive "
    "probability or statistical forecast."
)


# =========================================================
# STEP 94 — ADVANCED CUSTOMER REVIEW EXPLORER
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🔎 Advanced Customer Review Explorer"
    "</div>",
    unsafe_allow_html=True,
)

explorer_df = corpus_df.copy()

explorer_df["review_date"] = (
    pd.to_datetime(
        explorer_df[
            "review_date_clean"
        ],
        errors="coerce",
        utc=True,
    )
    .dt
    .tz_localize(None)
)

explorer_keyword = st.text_input(
    "Keyword Search",
    key="review_explorer_keyword",
)

explorer_sentiments = st.multiselect(
    "Sentiment",
    options=[
        "positive",
        "negative",
        "neutral",
    ],
    default=[
        "positive",
        "negative",
        "neutral",
    ],
    key="review_explorer_sentiments",
)

rating_values = sorted(
    explorer_df[
        "rating_value"
    ]
    .dropna()
    .unique()
    .tolist()
)

if rating_values:

    rating_min = int(
        min(rating_values)
    )

    rating_max = int(
        max(rating_values)
    )

else:

    rating_min = 1
    rating_max = 5

rating_range = st.slider(
    "Rating Range",
    min_value=rating_min,
    max_value=rating_max,
    value=(
        rating_min,
        rating_max,
    ),
    key="review_explorer_rating_range",
)

country_options = sorted(
    explorer_df["Country"]
    .fillna("Unknown")
    .astype(str)
    .unique()
    .tolist()
)

selected_countries = st.multiselect(
    "Country",
    options=country_options,
    default=[],
    key="review_explorer_countries",
)

selected_dates = st.date_input(
    "Review Date Range",
    value=None,
    key="review_explorer_date_range",
)

selected_aspects = st.multiselect(
    "Business Aspects",
    options=[
        ASPECT_LABELS[a]
        for a in ASPECTS
    ],
    default=[],
    key="review_explorer_aspects",
)

display_limit = st.selectbox(
    "Display Limit",
    options=[
        25,
        50,
        100,
        250,
        500,
    ],
    index=2,
    key="review_explorer_display_limit",
)

explorer_filtered = (
    explorer_df.copy()
)

if explorer_keyword.strip():

    explorer_filtered = (
        explorer_filtered[
            explorer_filtered[
                "review_text"
            ]
            .fillna("")
            .str.contains(
                explorer_keyword,
                case=False,
                na=False,
            )
        ]
    )

if explorer_sentiments:

    explorer_filtered = (
        explorer_filtered[
            explorer_filtered[
                "sentiment"
            ].isin(
                explorer_sentiments
            )
        ]
    )

explorer_filtered = (
    explorer_filtered[
        (
            explorer_filtered[
                "rating_value"
            ].isna()
        )
        |
        (
            explorer_filtered[
                "rating_value"
            ].between(
                rating_range[0],
                rating_range[1],
            )
        )
    ]
)

if selected_countries:

    explorer_filtered = (
        explorer_filtered[
            explorer_filtered[
                "Country"
            ]
            .fillna("Unknown")
            .isin(
                selected_countries
            )
        ]
    )

if (
    selected_dates is not None
    and len(selected_dates) == 2
):

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = (
        pd.Timestamp(
            selected_dates[1]
        )
        + pd.Timedelta(days=1)
    )

    explorer_filtered = (
        explorer_filtered[
            (
                explorer_filtered[
                    "review_date"
                ]
                >= start_date
            )
            &
            (
                explorer_filtered[
                    "review_date"
                ]
                < end_date
            )
        ]
    )

if selected_aspects:

    aspect_mask = pd.Series(
        False,
        index=explorer_filtered.index,
        dtype=bool,
    )

    for aspect in ASPECTS:

        label = ASPECT_LABELS[aspect]

        if (
            label
            in selected_aspects
        ):

            if aspect in aspects_df.columns:

                matching_aspect_values = pd.to_numeric(
                    aspects_df.loc[
                        explorer_filtered.index,
                        aspect,
                    ],
                    errors="coerce",
                ).fillna(0)

                aspect_mask = (
                    aspect_mask
                    |
                    (
                        matching_aspect_values
                        > 0
                    )
                )

    explorer_filtered = (
        explorer_filtered[
            aspect_mask
        ]
    )

explorer_count = len(
    explorer_filtered
)

explorer_avg_rating = (
    explorer_filtered[
        "rating_value"
    ].mean()
    if explorer_filtered[
        "rating_value"
    ].notna().any()
    else 0
)

explorer_positive = int(
    (
        explorer_filtered[
            "sentiment"
        ]
        == "positive"
    ).sum()
)

explorer_negative = int(
    (
        explorer_filtered[
            "sentiment"
        ]
        == "negative"
    ).sum()
)

explorer_col1, explorer_col2, explorer_col3, explorer_col4 = (
    st.columns(4)
)

with explorer_col1:
    st.metric(
        "Matching Reviews",
        f"{explorer_count:,}",
    )

with explorer_col2:
    st.metric(
        "Positive",
        f"{explorer_positive:,}",
    )

with explorer_col3:
    st.metric(
        "Negative",
        f"{explorer_negative:,}",
    )

with explorer_col4:
    st.metric(
        "Average Rating",
        f"{explorer_avg_rating:.2f}",
    )

explorer_display = explorer_filtered[
    [
        "review_text",
        "sentiment",
        "rating_value",
        "review_date_clean",
        "Country",
    ]
].head(
    display_limit
)

st.dataframe(
    explorer_display,
    width="stretch",
    hide_index=True,
)


# =========================================================
# STEP 98 — EXECUTIVE BUSINESS SUMMARY & EXPORT CENTER
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📋 Executive Business Summary & Export Center"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "A final executive layer that converts the analytical "
    "dashboard into concise business findings and reusable reports."
)


# ---------------------------------------------------------
# EXECUTIVE HEALTH INDICATORS
# ---------------------------------------------------------

overall_health = (
    positive_pct
    - negative_pct
)

rated_percentage = (
    len(rated_df)
    / total_reviews
    * 100
    if total_reviews
    else 0
)

executive_col1, executive_col2, executive_col3, executive_col4 = (
    st.columns(4)
)

with executive_col1:

    st.metric(
        "Positive Sentiment",
        f"{positive_pct:.1f}%",
    )

with executive_col2:

    st.metric(
        "Negative Sentiment",
        f"{negative_pct:.1f}%",
    )

with executive_col3:

    st.metric(
        "Average Rating",
        f"{average_rating:.2f}/5",
    )

with executive_col4:

    st.metric(
        "Rated Review Coverage",
        f"{rated_percentage:.1f}%",
    )


# ---------------------------------------------------------
# EXECUTIVE SNAPSHOT
# ---------------------------------------------------------

st.subheader(
    "📌 Executive Snapshot"
)

executive_snapshot = pd.DataFrame(
    [
        {
            "Metric": "Total Reviews",
            "Value": f"{total_reviews:,}",
        },
        {
            "Metric": "Positive Reviews",
            "Value": f"{positive_count:,}",
        },
        {
            "Metric": "Negative Reviews",
            "Value": f"{negative_count:,}",
        },
        {
            "Metric": "Neutral Reviews",
            "Value": f"{neutral_count:,}",
        },
        {
            "Metric": "Positive Share",
            "Value": f"{positive_pct:.1f}%",
        },
        {
            "Metric": "Negative Share",
            "Value": f"{negative_pct:.1f}%",
        },
        {
            "Metric": "Neutral Share",
            "Value": f"{neutral_pct:.1f}%",
        },
        {
            "Metric": "Average Rating",
            "Value": f"{average_rating:.2f}/5",
        },
        {
            "Metric": "Median Rating",
            "Value": f"{median_rating:.2f}/5",
        },
    ]
)

st.dataframe(
    executive_snapshot,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# TOP STRENGTHS
# ---------------------------------------------------------

st.subheader(
    "💚 Top Customer Strengths"
)

top_strengths = (
    risk_df
    .sort_values(
        [
            "Positive %",
            "Positive",
        ],
        ascending=False,
    )
    .head(5)
    [
        [
            "Aspect",
            "Total",
            "Positive",
            "Positive %",
            "Average Rating",
        ]
    ]
)

st.dataframe(
    top_strengths,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# TOP COMPLAINT AREAS
# ---------------------------------------------------------

st.subheader(
    "🚨 Top Complaint Areas"
)

top_complaints = (
    risk_df
    .sort_values(
        [
            "Negative",
            "Negative %",
        ],
        ascending=False,
    )
    .head(5)
    [
        [
            "Aspect",
            "Total",
            "Negative",
            "Negative %",
            "Average Rating",
        ]
    ]
)

st.dataframe(
    top_complaints,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# BUSINESS PRIORITY AREAS
# ---------------------------------------------------------

st.subheader(
    "🎯 Business Priority Areas"
)

business_priority = (
    risk_df
    .sort_values(
        "Attention Score",
        ascending=False,
    )
    [
        [
            "Aspect",
            "Total",
            "Negative",
            "Negative %",
            "Positive %",
            "Average Rating",
            "Attention Score",
        ]
    ]
)

st.dataframe(
    business_priority,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# EXECUTIVE NARRATIVE
# ---------------------------------------------------------

st.subheader(
    "🧠 Executive Narrative"
)

st.write(
    f"The dataset contains **{total_reviews:,} customer reviews**. "
    f"Overall sentiment consists of **{positive_pct:.1f}% positive**, "
    f"**{negative_pct:.1f}% negative**, and "
    f"**{neutral_pct:.1f}% neutral** feedback."
)

st.write(
    f"The average customer rating is "
    f"**{average_rating:.2f}/5**, based on "
    f"**{len(rated_df):,} rated reviews**."
)

st.write(
    f"The most frequently discussed business area is "
    f"**{most_discussed['Aspect']}**, with "
    f"**{int(most_discussed['Total']):,}** aspect-level reviews."
)

st.write(
    f"The largest concentration of negative feedback is "
    f"associated with **{largest_negative['Aspect']}**, "
    f"which contains "
    f"**{int(largest_negative['Negative']):,}** negative reviews."
)

st.write(
    f"The highest calculated attention score belongs to "
    f"**{highest_attention['Aspect']}**."
)

st.caption(
    "These findings summarize observed customer feedback in the "
    "available dataset. They should be interpreted as descriptive "
    "business intelligence rather than causal or predictive conclusions."
)


# ---------------------------------------------------------
# EXPORT CENTER
# ---------------------------------------------------------

st.subheader(
    "⬇️ Dashboard Data Export Center"
)

export_col1, export_col2, export_col3 = st.columns(3)


# Aspect intelligence export

aspect_export = risk_df[
    [
        "Aspect",
        "Total",
        "Positive",
        "Negative",
        "Neutral",
        "Positive %",
        "Negative %",
        "Neutral %",
        "Average Rating",
        "Attention Score",
    ]
].copy()

with export_col1:

    st.download_button(
        label="⬇️ Download Aspect Intelligence",
        data=aspect_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "customer_voice_aspect_intelligence.csv"
        ),
        mime="text/csv",
        key="download_aspect_intelligence",
    )


# Complaint export

complaint_export = complaint_df[
    [
        "review_text",
        "sentiment",
        "rating_value",
        "review_date_clean",
        "Country",
    ]
].copy()

with export_col2:

    st.download_button(
        label="⬇️ Download Complaint Data",
        data=complaint_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "customer_voice_complaints.csv"
        ),
        mime="text/csv",
        key="download_complaint_data",
    )


# Executive export

executive_export = pd.concat(
    [
        executive_snapshot,
        pd.DataFrame(
            [
                {
                    "Metric":
                        "Overall Sentiment Balance",
                    "Value":
                        f"{overall_health:.1f} percentage points",
                },
                {
                    "Metric":
                        "Highest Attention Aspect",
                    "Value":
                        highest_attention[
                            "Aspect"
                        ],
                },
                {
                    "Metric":
                        "Largest Negative Volume Aspect",
                    "Value":
                        largest_negative[
                            "Aspect"
                        ],
                },
                {
                    "Metric":
                        "Largest Positive Volume Aspect",
                    "Value":
                        largest_positive[
                            "Aspect"
                        ],
                },
            ]
        ),
    ],
    ignore_index=True,
)

with export_col3:

    st.download_button(
        label="⬇️ Download Executive Summary",
        data=executive_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "customer_voice_executive_summary.csv"
        ),
        mime="text/csv",
        key="download_executive_summary",
    )


# =========================================================
# FINAL FOOTER
# =========================================================

st.divider()

st.markdown(
    "### **Powered & Developed by Muzamil Rasul**"
)

st.markdown(
    "**Data Scientist | ML Engineer | NLP & AI Solutions**"
)

st.caption(
    "Customer Voice Intelligence Platform"
)