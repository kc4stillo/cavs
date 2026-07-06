import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

APP_PATH = Path(__file__).resolve()
BASE_DIR = (
    APP_PATH.parents[1]
    if APP_PATH.parent.name in {"app", "src", "pages"}
    else APP_PATH.parent
)
DATA_PATH = BASE_DIR / "data" / "clean" / "tweets_clean.csv"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#f43f5e",
}

MUTED = "#94a3b8"
GRID = "#263449"
ALL_OPTION = "All"

st.set_page_config(
    page_title="Cleveland Cavaliers on Twitter: Sentiment Analysis",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Selected option chips inside st.multiselect dropdowns */
    div[data-baseweb="select"] div[data-baseweb="tag"],
    div[data-baseweb="select"] div[data-baseweb="tag"] *,
    div[data-baseweb="select"] span[data-baseweb="tag"],
    div[data-baseweb="select"] span[data-baseweb="tag"] * {
        color: #0e1117 !important;
        -webkit-text-fill-color: #0e1117 !important;
    }

    /* Text inside selected chips */
    div[data-baseweb="select"] div[data-baseweb="tag"] span,
    div[data-baseweb="select"] div[data-baseweb="tag"] div,
    div[data-baseweb="select"] div[data-baseweb="tag"] p {
        color: #0e1117 !important;
        -webkit-text-fill-color: #0e1117 !important;
    }

    /* The x icon inside selected chips */
    div[data-baseweb="select"] div[data-baseweb="tag"] svg,
    div[data-baseweb="select"] span[data-baseweb="tag"] svg {
        color: #0e1117 !important;
        fill: #0e1117 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def multiselect_with_all(label, options, key, default=None):
    """Multiselect where a blank selection means All."""
    options = sorted([str(option) for option in options])

    if isinstance(default, str):
        default = [default]
    elif default is None:
        default = []

    option_lookup = {option.casefold(): option for option in options}
    resolved_default = [
        option_lookup[item.casefold()]
        for item in default
        if str(item).casefold() in option_lookup
    ]

    if key not in st.session_state:
        st.session_state[key] = resolved_default
    else:
        current = st.session_state.get(key, [])

        if not isinstance(current, (list, tuple, set)):
            current = [current]

        # Remove stale values from old sessions or changed data.
        st.session_state[key] = [item for item in current if item in options]

    selected = st.multiselect(
        label,
        options=options,
        key=key,
        placeholder="All",
        help="Leave blank to include all values.",
    )

    if not selected:
        return options

    return [item for item in selected if item in options]


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


def fmt_score(value: float) -> str:
    value = 0 if pd.isna(value) else float(value)
    return f"{value:+.2f}"


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)

    if mask.sum() == 0:
        return 0.0

    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def format_label(value: object) -> str:
    label = str(value).replace("_", " ").strip()

    if label.lower() in {"", "nan", "na", "n/a", "none", "null"}:
        label = "unclassified"

    label = label.title()
    label = label.replace("Lebron", "LeBron").replace("Nba", "NBA")
    return label


def sentiment_scale(labels: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=labels,
        range=[SENTIMENT_COLORS.get(label, MUTED) for label in labels],
    )


def log_slider_options(min_value: int, max_value: int, steps: int = 140) -> list[int]:
    """Create log-spaced integer options for a select_slider.

    Streamlit's regular range slider is linear, so large view outliers compress
    most useful values into a tiny portion of the slider. A select_slider with
    log-spaced values keeps the displayed values readable while making the
    control feel logarithmic.
    """
    min_value = max(0, int(min_value))
    max_value = max(min_value, int(max_value))

    if max_value == min_value:
        return [min_value]

    log_min = math.log10(min_value + 1)
    log_max = math.log10(max_value + 1)

    values = {
        int(round((10 ** (log_min + (log_max - log_min) * i / (steps - 1))) - 1))
        for i in range(steps)
    }
    values.update({min_value, max_value})

    return sorted(values)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]

    # Remove old score columns from the app entirely so they do not reappear in
    # the raw-data expander if they are still present in the CSV.
    df = df.drop(
        columns=["weighted_sentiment_score", "abs_weighted_sentiment_score"],
        errors="ignore",
    )

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
        "engagement_rate",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["query"] = df["query"].fillna("unknown").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("unknown").astype(str).str.lower()

    # The Cleveland Cavaliers query duplicates the Cavs query, so remove it
    # before any filter options or summaries are built.
    df = df[df["query"].str.casefold() != "cleveland cavaliers"].copy()

    if "narrative" not in df.columns:
        df["narrative"] = "unclassified"

    df["narrative"] = (
        df["narrative"]
        .fillna("unclassified")
        .astype(str)
        .str.strip()
        .replace(
            {
                "": "unclassified",
                "na": "unclassified",
                "n/a": "unclassified",
                "none": "unclassified",
                "null": "unclassified",
            }
        )
    )

    calculated_rate = df["engagement_total"].div(df["views"].where(df["views"] > 0))
    df["engagement_rate"] = df["engagement_rate"].where(
        df["engagement_rate"] > 0,
        calculated_rate,
    )
    df["engagement_rate"] = df["engagement_rate"].fillna(0)
    df["date_day"] = df["date"].dt.date

    return df


def clean_chart(chart: alt.Chart) -> alt.Chart:
    """Apply the same clean dark chart treatment everywhere."""
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#cbd5e1",
            gridColor=GRID,
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


def sorted_dimension_values(df: pd.DataFrame, column: str) -> list[str]:
    totals = (
        df.groupby(column, as_index=False)
        .agg(views=("views", "sum"), tweets=("tweet", "count"))
        .sort_values(["views", "tweets"], ascending=[False, False])
    )
    return totals[column].astype(str).tolist()


def summarize_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    total_views = df["views"].sum()
    total_engagement = df["engagement_total"].sum()
    rows = []

    for value, group in df.groupby(dimension, dropna=False):
        views = group["views"].sum()
        engagement = group["engagement_total"].sum()
        rows.append(
            {
                dimension: value,
                "label": format_label(value),
                "tweets": len(group),
                "views": views,
                "engagement": engagement,
                "aggregate_engagement_rate": safe_divide(engagement, views),
                "avg_engagement_rate": group["engagement_rate"].mean(),
                "avg_sentiment": group["sentiment_score"].mean(),
                "view_weighted_sentiment": weighted_average(
                    group["sentiment_score"], group["views"]
                ),
                "avg_confidence": group["sentiment_confidence"].mean(),
                "avg_word_count": group["word_count"].mean(),
                "positive_share": group["sentiment"].eq("positive").mean(),
                "negative_share": group["sentiment"].eq("negative").mean(),
                "conversation_share": safe_divide(views, total_views),
                "engagement_share": safe_divide(engagement, total_engagement),
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary["impact_score"] = (
        0.6 * summary["conversation_share"] + 0.4 * summary["engagement_share"]
    )
    return summary.sort_values("impact_score", ascending=False)


def build_time_summary(data: pd.DataFrame, grain: str) -> pd.DataFrame:
    dated = data.dropna(subset=["date"]).copy()

    if dated.empty:
        return pd.DataFrame()

    dated["period"] = dated["date"].dt.floor(grain)
    rows = []

    for (period, query), group in dated.groupby(["period", "query"], sort=True):
        rows.append(
            {
                "date": period,
                "query": query,
                "tweets": len(group),
                "views": group["views"].sum(),
                "engagement": group["engagement_total"].sum(),
                "avg_sentiment": group["sentiment_score"].mean(),
                "view_weighted_sentiment": weighted_average(
                    group["sentiment_score"], group["views"]
                ),
                "engagement_rate": safe_divide(
                    group["engagement_total"].sum(), group["views"].sum()
                ),
                "avg_confidence": group["sentiment_confidence"].mean(),
                "avg_word_count": group["word_count"].mean(),
            }
        )

    return pd.DataFrame(rows)


"""
# Cleveland Cavaliers on Twitter: Sentiment Analysis
By Kyle Castillo

An interactive dashboard tracking Cavs fan sentiment, engagement, and storyline momentum across Twitter/X.

* Built with Playwright, Streamlit, Python, and R
* Created as a quick social listening project for sports analytics and dashboard practice
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

if df.empty:
    st.info(
        "No tweets are available after removing the duplicate Cleveland Cavaliers query.",
        icon=":material/info:",
    )
    st.stop()

sentiment_options = available_sentiments(df)
query_options = sorted(df["query"].dropna().unique())
narrative_options = sorted_dimension_values(df, "narrative")

valid_dates = df["date"].dropna()
date_min = valid_dates.min().date() if not valid_dates.empty else None
date_max = valid_dates.max().date() if not valid_dates.empty else None

view_min = max(0, int(df["views"].min())) if not df.empty else 0
view_max = max(view_min, int(df["views"].max())) if not df.empty else view_min
view_slider_options = log_slider_options(view_min, view_max)

word_count_min = max(0, int(df["word_count"].min())) if not df.empty else 0
word_count_max = int(df["word_count"].max()) if not df.empty else word_count_min
word_count_max = max(word_count_min, word_count_max)

# -------------------------------------------------------------------
# Top layout: left controls, right main chart
# -------------------------------------------------------------------

top_cols = st.columns([1, 3])

with top_cols[0].container(border=True):
    st.markdown("#### Controls")

    selected_queries = multiselect_with_all(
        "Queries",
        query_options,
        key="selected_queries",
        default="cavs",
    )

    selected_narratives = multiselect_with_all(
        "Narratives",
        narrative_options,
        key="selected_narratives",
    )

    selected_sentiments = multiselect_with_all(
        "Sentiment",
        sentiment_options,
        key="selected_sentiments",
    )

    grain_label = st.pills(
        "Time grain",
        options=["Daily", "Hourly"],
        default="Daily",
    )

    main_metric = st.pills(
        "Main metric",
        options=["View-weighted sentiment", "Sentiment", "Eng. rate"],
        default="View-weighted sentiment",
    )

    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    if len(view_slider_options) > 1:
        selected_view_range = st.select_slider(
            "Tweet views",
            options=view_slider_options,
            value=(view_slider_options[0], view_slider_options[-1]),
            format_func=fmt_number,
            help="Log-spaced control so large view outliers do not dominate the slider.",
        )
        selected_view_range = (
            int(selected_view_range[0]),
            int(selected_view_range[1]),
        )
        st.caption(
            "Views filter: "
            f"**{fmt_number(selected_view_range[0])}** to "
            f"**{fmt_number(selected_view_range[1])}**"
        )
    else:
        selected_view_range = (view_slider_options[0], view_slider_options[0])
        st.caption(f"Tweet views: {fmt_number(view_slider_options[0])}")

    if word_count_max > word_count_min:
        tweet_length_range = st.slider(
            "Tweet length (words)",
            min_value=word_count_min,
            max_value=word_count_max,
            value=(word_count_min, word_count_max),
            step=1,
            help="Filters tweets by the word_count column.",
        )
    else:
        tweet_length_range = (word_count_min, word_count_max)
        st.caption(f"Tweet length: {word_count_min} words")

    min_engagement_rate = st.slider(
        "Minimum engagement rate",
        min_value=0.0,
        max_value=0.25,
        value=0.0,
        step=0.005,
        format="%.3f",
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
    & df["narrative"].isin(selected_narratives)
    & df["sentiment"].isin(selected_sentiments)
    & df["views"].between(selected_view_range[0], selected_view_range[1])
    & df["word_count"].between(tweet_length_range[0], tweet_length_range[1])
    & (df["sentiment_confidence"] >= min_confidence)
    & (df["engagement_rate"] >= min_engagement_rate)
)

if (
    selected_dates
    and isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
):
    start_date, end_date = selected_dates
    mask = mask & df["date"].dt.date.between(start_date, end_date)

filtered = df.loc[mask].copy()

if filtered.empty:
    top_cols[0].info("No tweets match the current filters.", icon=":material/info:")
    st.stop()

grain = "h" if grain_label == "Hourly" else "D"

metric_config = {
    "Sentiment": {
        "column": "avg_sentiment",
        "title": "Average sentiment",
        "domain": [-1, 1],
        "format": ".3f",
        "tooltip": "Avg sentiment",
    },
    "View-weighted sentiment": {
        "column": "view_weighted_sentiment",
        "title": "View-weighted sentiment",
        "domain": [-1, 1],
        "format": ".3f",
        "tooltip": "View-weighted sentiment",
    },
    "Eng. rate": {
        "column": "engagement_rate",
        "title": "Aggregate engagement rate",
        "domain": None,
        "format": ".2%",
        "tooltip": "Eng. rate",
    },
}

sentiment_labels = available_sentiments(filtered)

with top_cols[1].container(
    border=True,
    height="stretch",
    vertical_alignment="center",
):
    st.markdown(f"#### {metric_config[main_metric]['title']} over time")

    time_df = build_time_summary(filtered, grain)

    if time_df.empty:
        st.info("No dated tweets available for this chart.", icon=":material/info:")
    else:
        selected_col = metric_config[main_metric]["column"]
        y_scale = (
            alt.Scale(domain=metric_config[main_metric]["domain"])
            if metric_config[main_metric]["domain"]
            else alt.Scale(zero=False)
        )
        y_axis = (
            alt.Axis(format=".1%")
            if main_metric == "Eng. rate"
            else alt.Axis(format=".2f")
        )

        line = (
            alt.Chart(time_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "yearmonthdate(date):T",
                    title=None,
                    axis=alt.Axis(
                        format="%a %d",
                        labelAngle=-45,
                        values=sorted(
                            time_df["date"].dropna().drop_duplicates().tolist()
                        ),
                    ),
                ),
                y=alt.Y(
                    f"{selected_col}:Q",
                    title=metric_config[main_metric]["title"],
                    scale=y_scale,
                    axis=y_axis,
                ),
                color=alt.Color(
                    "query:N",
                    title="Query",
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("query:N", title="Query"),
                    alt.Tooltip("date:T", title="Time", format="%a %b %d, %Y"),
                    alt.Tooltip(
                        f"{selected_col}:Q",
                        title=metric_config[main_metric]["tooltip"],
                        format=metric_config[main_metric]["format"],
                    ),
                    alt.Tooltip("tweets:Q", title="Tweets"),
                    alt.Tooltip("views:Q", title="Views", format=",.0f"),
                    alt.Tooltip("engagement:Q", title="Engagement", format=",.0f"),
                    alt.Tooltip(
                        "avg_confidence:Q", title="Avg confidence", format=".2f"
                    ),
                    alt.Tooltip("avg_word_count:Q", title="Avg words", format=".1f"),
                ],
            )
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"baseline": [0]}))
            .mark_rule(strokeDash=[4, 4], opacity=0.45)
            .encode(y="baseline:Q")
        )

        chart = zero_line + line if main_metric != "Eng. rate" else line
        st.altair_chart(
            clean_chart(chart.properties(height=410)), use_container_width=True
        )

with top_cols[1].container(border=True):
    st.markdown("#### Message shape")

    scatter_df = filtered.copy()

    if len(scatter_df) > 3000:
        scatter_df = scatter_df.sample(3000, random_state=7)

    scatter_df["narrative_label"] = scatter_df["narrative"].map(format_label)

    scatter_x_min = int(tweet_length_range[0])
    scatter_x_max = int(tweet_length_range[1])

    # Avoid a zero-width Altair scale if the slider collapses to one word count.
    if scatter_x_max <= scatter_x_min:
        scatter_x_max = scatter_x_min + 1

    scatter = (
        alt.Chart(scatter_df)
        .mark_circle(opacity=0.62)
        .encode(
            x=alt.X(
                "word_count:Q",
                title="Word count",
                scale=alt.Scale(
                    domain=[scatter_x_min, scatter_x_max],
                    zero=False,
                    nice=False,
                ),
            ),
            y=alt.Y(
                "engagement_rate:Q",
                title="Engagement rate",
                axis=alt.Axis(format=".1%"),
            ),
            size=alt.Size(
                "views:Q",
                title="Views",
                scale=alt.Scale(type="sqrt", range=[20, 450]),
            ),
            color=alt.Color(
                "sentiment:N",
                scale=sentiment_scale(sentiment_labels),
                title="Sentiment",
            ),
            tooltip=[
                alt.Tooltip("tweet:N", title="Tweet"),
                alt.Tooltip("narrative_label:N", title="Narrative"),
                alt.Tooltip("word_count:Q", title="Words"),
                alt.Tooltip("engagement_rate:Q", title="Eng. rate", format=".2%"),
                alt.Tooltip("views:Q", title="Views", format=",.0f"),
                alt.Tooltip("sentiment_confidence:Q", title="Confidence", format=".1%"),
            ],
        )
        .properties(title="Tweet length, reach, and engagement efficiency", height=390)
    )

    st.altair_chart(clean_chart(scatter), use_container_width=True)

with top_cols[1].container(border=True):
    st.markdown("#### Top tweets")

    rank_by = st.pills(
        "Rank tweets by",
        options=[
            "Engagement",
            "Views",
            "Eng. rate",
            "Most negative",
            "Confidence",
            "Word count",
            "Likes",
            "Retweets",
            "Comments",
        ],
        default="Engagement",
    )

    sort_map = {
        "Engagement": ("engagement_total", False),
        "Views": ("views", False),
        "Eng. rate": ("engagement_rate", False),
        "Most negative": ("sentiment_score", True),
        "Confidence": ("sentiment_confidence", False),
        "Word count": ("word_count", False),
        "Likes": ("likes", False),
        "Retweets": ("retweets", False),
        "Comments": ("comments", False),
    }

    sort_col, ascending = sort_map[rank_by]

    display_cols = [
        "tweet",
        "date",
        "query",
        "narrative",
        "sentiment",
        "sentiment_score",
        "sentiment_confidence",
        "engagement_rate",
        "views",
        "likes",
        "comments",
        "retweets",
        "engagement_total",
        "word_count",
        "sentiment_negative_score",
        "sentiment_neutral_score",
        "sentiment_positive_score",
    ]

    display_cols = [col for col in display_cols if col in filtered.columns]

    top_tweets = (
        filtered.sort_values(sort_col, ascending=ascending)
        .head(50)
        .loc[:, display_cols]
        .copy()
    )
    top_tweets["narrative"] = top_tweets["narrative"].map(format_label)

    st.dataframe(
        top_tweets,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tweet": st.column_config.TextColumn("Tweet", width="large"),
            "date": st.column_config.DatetimeColumn("Date"),
            "narrative": st.column_config.TextColumn("Narrative", width="medium"),
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment", format="%.3f"
            ),
            "sentiment_confidence": st.column_config.NumberColumn(
                "Confidence", format="%.3f"
            ),
            "engagement_rate": st.column_config.NumberColumn(
                "Eng. rate", format="%.4f"
            ),
            "views": st.column_config.NumberColumn("Views", format="%d"),
            "engagement_total": st.column_config.NumberColumn(
                "Engagement", format="%d"
            ),
            "word_count": st.column_config.NumberColumn("Words", format="%d"),
            "sentiment_negative_score": st.column_config.NumberColumn(
                "Negative score", format="%.3f"
            ),
            "sentiment_neutral_score": st.column_config.NumberColumn(
                "Neutral score", format="%.3f"
            ),
            "sentiment_positive_score": st.column_config.NumberColumn(
                "Positive score", format="%.3f"
            ),
        },
    )

# -------------------------------------------------------------------
# Simple summary card under the controls
# -------------------------------------------------------------------

with top_cols[0].container(border=True):
    st.markdown("#### Quick summary")

    metric_cols = st.columns(2)

    total_tweets = len(filtered)
    total_views = filtered["views"].sum()
    total_likes = filtered["likes"].sum()
    total_comments = filtered["comments"].sum()
    total_retweets = filtered["retweets"].sum()
    total_interactions = total_likes + total_comments + total_retweets

    avg_views_per_tweet = safe_divide(total_views, total_tweets)
    avg_interactions_per_tweet = safe_divide(total_interactions, total_tweets)

    positive_tweets = filtered["sentiment"].eq("positive").sum()
    neutral_tweets = filtered["sentiment"].eq("neutral").sum()
    negative_tweets = filtered["sentiment"].eq("negative").sum()

    narrative_summary = summarize_dimension(filtered, "narrative")
    top_narrative = (
        format_label(narrative_summary.iloc[0]["narrative"])
        if not narrative_summary.empty
        else "N/A"
    )

    metric_cols[0].metric("Tweets collected", fmt_number(total_tweets))
    metric_cols[1].metric("Total views", fmt_number(total_views))

    metric_cols[0].metric("Likes", fmt_number(total_likes))
    metric_cols[1].metric("Comments", fmt_number(total_comments))

    metric_cols[0].metric("Retweets", fmt_number(total_retweets))
    metric_cols[1].metric("Total interactions", fmt_number(total_interactions))

    metric_cols[0].metric("Avg views per tweet", fmt_number(avg_views_per_tweet))
    metric_cols[1].metric(
        "Avg interactions per tweet",
        fmt_number(avg_interactions_per_tweet),
    )

    metric_cols[0].metric("Positive tweets", fmt_number(positive_tweets))
    metric_cols[1].metric("Negative tweets", fmt_number(negative_tweets))

    st.caption(
        f"Most common storyline: **{top_narrative}** · "
        f"Neutral tweets: **{fmt_number(neutral_tweets)}**"
    )
""

"""
## Sentiment model detail
"""
MODEL_CHART_HEIGHT = 300

model_cols = st.columns(3)

with model_cols[0].container(border=True):
    st.markdown("#### Sentiment mix")

    sentiment_counts = (
        filtered.groupby("sentiment", as_index=False)
        .agg(tweets=("tweet", "count"))
        .sort_values("tweets", ascending=False)
    )
    sentiment_labels = available_sentiments(filtered)

    donut = (
        alt.Chart(sentiment_counts)
        .mark_arc(innerRadius=58, outerRadius=100)
        .encode(
            theta=alt.Theta("tweets:Q"),
            color=alt.Color(
                "sentiment:N",
                scale=sentiment_scale(sentiment_labels),
                title=None,
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("tweets:Q", title="Tweets"),
            ],
        )
        .properties(
            height=MODEL_CHART_HEIGHT,
            padding={"top": 18, "bottom": 6, "left": 6, "right": 6},
        )
    )

    st.altair_chart(clean_chart(donut), use_container_width=True)

with model_cols[1].container(border=True):
    st.markdown("#### Average class probabilities")

    score_cols = {
        "sentiment_negative_score": "Negative",
        "sentiment_neutral_score": "Neutral",
        "sentiment_positive_score": "Positive",
    }
    probability_df = pd.DataFrame(
        {
            "score_type": list(score_cols.values()),
            "avg_score": [filtered[col].mean() for col in score_cols],
            "sentiment_key": ["negative", "neutral", "positive"],
        }
    )

    probability_bar = (
        alt.Chart(probability_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "score_type:N",
                title=None,
                sort=["Negative", "Neutral", "Positive"],
            ),
            y=alt.Y(
                "avg_score:Q",
                title="Average model score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "sentiment_key:N",
                scale=sentiment_scale(["negative", "neutral", "positive"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("score_type:N", title="Class"),
                alt.Tooltip("avg_score:Q", title="Avg score", format=".2%"),
            ],
        )
        .properties(height=MODEL_CHART_HEIGHT)
    )

    st.altair_chart(clean_chart(probability_bar), use_container_width=True)

with model_cols[2].container(border=True):
    st.markdown("#### Confidence by sentiment")

    confidence_df = filtered.groupby("sentiment", as_index=False).agg(
        avg_confidence=("sentiment_confidence", "mean"),
        avg_word_count=("word_count", "mean"),
        tweets=("tweet", "count"),
    )

    confidence_bar = (
        alt.Chart(confidence_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("sentiment:N", title=None, sort=SENTIMENT_ORDER),
            y=alt.Y(
                "avg_confidence:Q",
                title="Average confidence",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "sentiment:N",
                scale=sentiment_scale(sentiment_labels),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".2%"),
                alt.Tooltip("avg_word_count:Q", title="Avg words", format=".1f"),
                alt.Tooltip("tweets:Q", title="Tweets"),
            ],
        )
        .properties(height=MODEL_CHART_HEIGHT)
    )

    st.altair_chart(clean_chart(confidence_bar), use_container_width=True)

"""
## Raw filtered data
"""

with st.expander("Show raw filtered dataset", expanded=True):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
