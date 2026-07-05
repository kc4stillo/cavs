from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "tweets_clean.csv"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#f43f5e",
}

st.set_page_config(
    page_title="Cavs Social Listening",
    page_icon="🏀",
    layout="wide",
)


def fmt_number(value: float) -> str:
    """Format dashboard numbers so the UI stays compact."""
    value = 0 if pd.isna(value) else float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def fmt_pct(value: float) -> str:
    value = 0 if pd.isna(value) else float(value)
    return f"{value:.1%}"


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]

    required_cols = {
        "tweet",
        "date",
        "query",
        "sentiment",
        "views",
        "engagement_total",
        "sentiment_score",
    }

    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "views",
        "likes",
        "comments",
        "retweets",
        "engagement_total",
        "word_count",
        "sentiment_confidence",
        "sentiment_negative_score",
        "sentiment_neutral_score",
        "sentiment_positive_score",
        "sentiment_score",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["query"] = df["query"].fillna("unknown").astype(str)
    df["sentiment"] = df["sentiment"].fillna("unknown").astype(str).str.lower()

    df["engagement_rate"] = (
        df["engagement_total"].div(df["views"].where(df["views"] > 0)).fillna(0)
    )

    return df


def clean_chart(chart: alt.Chart) -> alt.Chart:
    """Apply the same clean dark chart treatment everywhere."""
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#cbd5e1",
            gridColor="#263449",
            domainColor="#334155",
            tickColor="#334155",
        )
        .configure_legend(
            labelColor="#cbd5e1",
            titleColor="#cbd5e1",
            orient="bottom",
        )
        .configure_title(
            color="#e2e8f0",
            fontSize=15,
            fontWeight=400,
            anchor="start",
        )
    )


def available_sentiments(df: pd.DataFrame) -> list[str]:
    values = list(df["sentiment"].dropna().unique())
    ordered = [sentiment for sentiment in SENTIMENT_ORDER if sentiment in values]
    extras = sorted(
        [sentiment for sentiment in values if sentiment not in SENTIMENT_ORDER]
    )

    return ordered + extras


"""
# :material/forum: Cavs social listening
Track Cleveland Cavaliers conversation quality, fan sentiment, and engagement patterns from Twitter/X.
"""

""  # Keeps spacing similar to the Streamlit stockpeers demo.

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH}`. Make sure your clean CSV exists.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(str(exc))
    st.stop()


sentiment_options = available_sentiments(df)
query_options = sorted(df["query"].dropna().unique())

valid_dates = df["date"].dropna()
date_min = valid_dates.min().date() if not valid_dates.empty else None
date_max = valid_dates.max().date() if not valid_dates.empty else None


# -------------------------------------------------------------------
# Top layout: left controls, right main chart
# -------------------------------------------------------------------

top_cols = st.columns([1, 3])

with top_cols[0].container(
    border=True,
    height="stretch",
    vertical_alignment="center",
):
    st.markdown("#### Controls")

    selected_queries = st.multiselect(
        "Queries",
        options=query_options,
        default=query_options,
        placeholder="Choose search queries",
    )

    selected_sentiments = st.multiselect(
        "Sentiment",
        options=sentiment_options,
        default=sentiment_options,
        placeholder="Choose sentiment labels",
    )

    grain_label = st.pills(
        "Time grain",
        options=["Hourly", "Daily"],
        default="Hourly",
    )

    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    min_views = st.number_input(
        "Minimum views",
        min_value=0,
        value=0,
        step=100,
    )

    if date_min and date_max:
        selected_dates = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
    else:
        selected_dates = None


mask = (
    df["query"].isin(selected_queries)
    & df["sentiment"].isin(selected_sentiments)
    & (df["views"] >= min_views)
)

if "sentiment_confidence" in df.columns:
    mask = mask & (df["sentiment_confidence"] >= min_confidence)

if selected_dates and isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask = mask & df["date"].dt.date.between(start_date, end_date)

filtered = df.loc[mask].copy()

if filtered.empty:
    top_cols[0].info("No tweets match the current filters.", icon=":material/info:")
    st.stop()


grain = "h" if grain_label == "Hourly" else "D"


with top_cols[1].container(
    border=True,
    height="stretch",
    vertical_alignment="center",
):
    st.markdown("#### Sentiment over time")

    time_df = (
        filtered.dropna(subset=["date"])
        .set_index("date")
        .resample(grain)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            tweets=("tweet", "count"),
            engagement=("engagement_total", "sum"),
        )
        .reset_index()
    )

    if time_df.empty:
        st.info("No dated tweets available for this chart.", icon=":material/info:")
    else:
        line = (
            alt.Chart(time_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(
                    "avg_sentiment:Q",
                    title="Average sentiment",
                    scale=alt.Scale(domain=[-1, 1]),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Time"),
                    alt.Tooltip("avg_sentiment:Q", title="Avg sentiment", format=".3f"),
                    alt.Tooltip("tweets:Q", title="Tweets"),
                    alt.Tooltip("engagement:Q", title="Engagement"),
                ],
            )
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"baseline": [0]}))
            .mark_rule(strokeDash=[4, 4], opacity=0.45)
            .encode(y="baseline:Q")
        )

        st.altair_chart(
            clean_chart((zero_line + line).properties(height=410)),
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Metric card under the controls
# -------------------------------------------------------------------

with top_cols[0].container(
    border=True,
    height="stretch",
    vertical_alignment="center",
):
    metric_cols = st.columns(2)

    total_tweets = len(filtered)
    total_views = filtered["views"].sum()
    total_engagement = filtered["engagement_total"].sum()
    avg_sentiment = filtered["sentiment_score"].mean()
    engagement_rate = filtered["engagement_rate"].mean()
    positive_share = filtered["sentiment"].eq("positive").mean()

    metric_cols[0].metric("Tweets", fmt_number(total_tweets))
    metric_cols[1].metric("Views", fmt_number(total_views))

    metric_cols[0].metric("Engagement", fmt_number(total_engagement))
    metric_cols[1].metric("Avg sentiment", f"{avg_sentiment:.2f}")

    metric_cols[0].metric("Eng. rate", fmt_pct(engagement_rate))
    metric_cols[1].metric("Positive", fmt_pct(positive_share))


""

"""
## Sentiment and engagement
"""

mid_cols = st.columns(2)

with mid_cols[0].container(border=True):
    sentiment_counts = (
        filtered.groupby("sentiment", as_index=False)
        .agg(tweets=("tweet", "count"))
        .sort_values("tweets", ascending=False)
    )

    donut = (
        alt.Chart(sentiment_counts)
        .mark_arc(innerRadius=65, outerRadius=115)
        .encode(
            theta=alt.Theta("tweets:Q"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS.keys()),
                    range=list(SENTIMENT_COLORS.values()),
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("tweets:Q", title="Tweets"),
            ],
        )
        .properties(title="Sentiment mix", height=320)
    )

    st.altair_chart(clean_chart(donut), use_container_width=True)


with mid_cols[1].container(border=True):
    engagement_df = filtered.groupby("sentiment", as_index=False).agg(
        avg_engagement=("engagement_total", "mean"),
        avg_views=("views", "mean"),
        tweets=("tweet", "count"),
    )

    bar = (
        alt.Chart(engagement_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("sentiment:N", title=None, sort=SENTIMENT_ORDER),
            y=alt.Y("avg_engagement:Q", title="Avg engagement"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS.keys()),
                    range=list(SENTIMENT_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("avg_engagement:Q", title="Avg engagement", format=".2f"),
                alt.Tooltip("avg_views:Q", title="Avg views", format=".0f"),
                alt.Tooltip("tweets:Q", title="Tweets"),
            ],
        )
        .properties(title="Average engagement by sentiment", height=320)
    )

    st.altair_chart(clean_chart(bar), use_container_width=True)


"""
## Queries vs the conversation average
"""

unique_queries = sorted(filtered["query"].dropna().unique())

if len(unique_queries) <= 1:
    st.info(
        "Add more search queries to compare each topic against the rest of the conversation.",
        icon=":material/info:",
    )
else:
    query_cols = st.columns(4)

    for index, query in enumerate(unique_queries[:8]):
        query_data = filtered[filtered["query"] == query].dropna(subset=["date"])
        peer_data = filtered[filtered["query"] != query].dropna(subset=["date"])

        if query_data.empty or peer_data.empty:
            continue

        query_ts = (
            query_data.set_index("date")
            .resample(grain)["sentiment_score"]
            .mean()
            .rename(query)
        )

        peer_ts = (
            peer_data.set_index("date")
            .resample(grain)["sentiment_score"]
            .mean()
            .rename("Other queries")
        )

        plot_data = (
            pd.concat([query_ts, peer_ts], axis=1)
            .reset_index()
            .melt(id_vars="date", var_name="series", value_name="sentiment")
            .dropna()
        )

        card = query_cols[index % 4].container(border=True)
        card.write("")

        chart = (
            alt.Chart(plot_data)
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(
                    "sentiment:Q",
                    title=None,
                    scale=alt.Scale(domain=[-1, 1]),
                ),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[query, "Other queries"],
                        range=["#fdbb30", "#94a3b8"],
                    ),
                    title=None,
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Time"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("sentiment:Q", title="Sentiment", format=".3f"),
                ],
            )
            .properties(title=f"{query} vs other queries", height=250)
        )

        card.altair_chart(clean_chart(chart), use_container_width=True)


"""
## Top tweets
"""

with st.container(border=True):
    rank_by = st.pills(
        "Rank tweets by",
        options=["Engagement", "Views", "Likes", "Retweets", "Comments"],
        default="Engagement",
    )

    sort_map = {
        "Engagement": "engagement_total",
        "Views": "views",
        "Likes": "likes",
        "Retweets": "retweets",
        "Comments": "comments",
    }

    display_cols = [
        "tweet",
        "date",
        "query",
        "sentiment",
        "sentiment_score",
        "views",
        "likes",
        "comments",
        "retweets",
        "engagement_total",
    ]

    display_cols = [col for col in display_cols if col in filtered.columns]

    top_tweets = (
        filtered.sort_values(sort_map[rank_by], ascending=False)
        .head(50)
        .loc[:, display_cols]
    )

    st.dataframe(
        top_tweets,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tweet": st.column_config.TextColumn("Tweet", width="large"),
            "date": st.column_config.DatetimeColumn("Date"),
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment", format="%.3f"
            ),
            "views": st.column_config.NumberColumn("Views", format="%d"),
            "engagement_total": st.column_config.NumberColumn(
                "Engagement", format="%d"
            ),
        },
    )


"""
## Raw filtered data
"""

with st.expander("Show raw filtered dataset"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
