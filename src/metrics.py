import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------

INPUT_CSV = "../data/int/tweets_int.csv"
OUTPUT_CSV = "../data/clean/tweets_clean.csv"

TEXT_COL = "tweet"
BATCH_SIZE = 64

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABELS = ["negative", "neutral", "positive"]

FINAL_COLUMNS = [
    "tweet",
    "date",
    "query",
    "views",
    "likes",
    "comments",
    "retweets",
    "engagement_total",
    "word_count",
    "sentiment",
    "sentiment_confidence",
    "sentiment_negative_score",
    "sentiment_neutral_score",
    "sentiment_positive_score",
    "sentiment_score",
    "engagement_rate",
    "weighted_sentiment_score",
    "narrative",
]

# -----------------------------
# CLEANING
# -----------------------------


def clean_tweet_for_model(text):
    """
    Clean tweet text in a Twitter-RoBERTa-friendly way.
    Do not over-clean. Keep emojis, punctuation, slang, and casing signals.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Replace URLs and mentions with placeholders.
    # This matches how many tweet models expect social text to look.
    text = re.sub(r"http\S+|www\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)


# -----------------------------
# LOAD MODEL
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------


def predict_sentiment_batch(texts):
    cleaned = [clean_tweet_for_model(t) for t in texts]

    encoded = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    scores = outputs.logits.detach().cpu().numpy()
    probs = softmax(scores, axis=1)

    results = []

    for prob in probs:
        best_idx = prob.argmax()

        results.append(
            {
                "sentiment": LABELS[best_idx],
                "sentiment_confidence": float(prob[best_idx]),
                "sentiment_negative_score": float(prob[0]),
                "sentiment_neutral_score": float(prob[1]),
                "sentiment_positive_score": float(prob[2]),
                "sentiment_score": float(prob[2] - prob[0]),
            }
        )

    return results


# -----------------------------
# ENGAGEMENT METRICS
# -----------------------------


def add_engagement_metrics(df):
    """
    Adds:
    - engagement_rate
    - weighted_sentiment_score

    weighted_sentiment_score keeps sentiment direction but weights it by tweet reach.
    log1p prevents a single high-view tweet from overpowering everything.
    """
    for col in ["views", "likes", "comments", "retweets"]:
        if col not in df.columns:
            df[col] = 0

        df[col] = to_numeric(df[col])

    if "word_count" not in df.columns:
        df["word_count"] = df[TEXT_COL].fillna("").astype(str).str.split().str.len()
    else:
        df["word_count"] = to_numeric(df["word_count"])

    # Recalculate this to guarantee consistency.
    df["engagement_total"] = df["likes"] + df["comments"] + df["retweets"]

    df["engagement_rate"] = np.where(
        df["views"] > 0,
        df["engagement_total"] / df["views"],
        0,
    )

    df["weighted_sentiment_score"] = df["sentiment_score"] * np.log1p(
        df["views"] + df["engagement_total"]
    )

    return df


# -----------------------------
# NARRATIVE CLASSIFICATION
# -----------------------------

CAVS_PATTERN = r"\b(cavs|cavaliers|cleveland)\b"
LEBRON_PATTERN = r"\b(lebron james|lebron|bron)\b"

MOVE_PATTERN = (
    r"\b(move|moves|moved|sign|signs|signed|signing|join|joins|joining|"
    r"goes|going|land|lands|landing|destination|destinations|landing spot|"
    r"landing spots|shortlist|sweepstakes|free agency|free agent)\b"
)

DESTINATION_PATTERNS = {
    # Eastern Conference
    "hawks": r"\b(hawks|atlanta|atlanta hawks|trae|trae young)\b",
    "celtics": r"\b(celtics|boston|boston celtics|tatum|jaylen brown|derrick white)\b",
    "nets": r"\b(nets|brooklyn|brooklyn nets)\b",
    "hornets": r"\b(hornets|charlotte|charlotte hornets|lamelo|lamelo ball)\b",
    "bulls": r"\b(bulls|chicago|chicago bulls)\b",
    "pistons": r"\b(pistons|detroit|detroit pistons|cade|cade cunningham)\b",
    "pacers": r"\b(pacers|indiana|indiana pacers|haliburton|tyrese haliburton)\b",
    "heat": r"\b(heat|miami|miami heat|bam|adebayo|spo|spoelstra)\b",
    "bucks": r"\b(bucks|milwaukee|milwaukee bucks|giannis|antetokounmpo)\b",
    "knicks": r"\b(knicks|new york knicks|nyk|msg|brunson|jalen brunson|kat|towns)\b",
    "magic": r"\b(magic|orlando|orlando magic|paolo|paolo banchero|franz)\b",
    "sixers": r"\b(sixers|76ers|philadelphia 76ers|philly|philadelphia|embiid|maxey|tyrese maxey)\b",
    "raptors": r"\b(raptors|toronto|toronto raptors)\b",
    "wizards": r"\b(wizards|washington|washington wizards)\b",
    # Western Conference
    "mavericks": r"\b(mavs|mavericks|dallas|dallas mavericks|ad|anthony davis|kyrie|cooper flagg)\b",
    "nuggets": r"\b(nuggets|denver|denver nuggets|jokic|nikola jokic|jamal|jamal murray)\b",
    "warriors": r"\b(warriors|golden state|golden state warriors|gsw|curry|steph|stephen curry|draymond|draymond green)\b",
    "rockets": r"\b(rockets|houston|houston rockets|amen|sengun|alperen sengun)\b",
    "lakers": r"\b(lakers|los angeles lakers|la lakers|luka|doncic|luka doncic)\b",
    "grizzlies": r"\b(grizzlies|memphis|memphis grizzlies|ja|ja morant|jaren jackson)\b",
    "timberwolves": r"\b(wolves|timberwolves|twolves|minnesota|minnesota timberwolves|ant|anthony edwards|rudy|gobert)\b",
    "pelicans": r"\b(pelicans|new orleans|new orleans pelicans|zion|zion williamson)\b",
    "thunder": r"\b(thunder|okc|oklahoma city|oklahoma city thunder|sga|shai|chet|jalen williams)\b",
    "suns": r"\b(suns|phoenix|phoenix suns|booker|devin booker)\b",
    "blazers": r"\b(blazers|trail blazers|portland|portland trail blazers|scoot|scoot henderson)\b",
    "kings": r"\b(kings|sacramento|sacramento kings|sabonis|domantas sabonis)\b",
    "spurs": r"\b(spurs|san antonio|san antonio spurs|wemby|wembanyama|victor wembanyama)\b",
    "jazz": r"\b(jazz|utah|utah jazz|lauri|markkanen|lauri markkanen)\b",
}


def count_destination_mentions(text):
    if pd.isna(text):
        return 0

    text = str(text).lower()

    count = 0

    for pattern in DESTINATION_PATTERNS.values():
        if re.search(pattern, text):
            count += 1

    return count


def assign_narrative(text):
    """
    Rule-based narrative classification.

    Main labels:
    - lebron_cavs
    - general_roster_changes
    - cavs_curr_core
    - lebron_non_cavs_move
    - player_move_destinations
    - na

    Order:
    - cavs-specific LeBron homecoming first
    - cavs roster/core discussion next
    - Non-Cavs LeBron movement next
    - Broader destination comparison after that
    - na for everything else
    """
    if pd.isna(text):
        return "na"

    text = str(text).lower()

    has_cavs = bool(re.search(CAVS_PATTERN, text))
    has_lebron = bool(re.search(LEBRON_PATTERN, text))
    has_move_language = bool(re.search(MOVE_PATTERN, text))

    non_cavs_destination_count = count_destination_mentions(text)

    # Cavs homecoming / final-chapter narrative.
    specific_cavs_homecoming = (
        r"\b(back to cleveland|back to the cavs|return to cleveland|"
        r"returns to cleveland|return to the cavs|returns to the cavs|"
        r"cavs reunion|cleveland reunion|go back to cleveland|coming home|"
        r"going back to cleveland|go back to the cavs|going back to the cavs)\b"
    )

    # LeBron move discussion that does NOT center on the Cavs.
    # Example: LeBron to Warriors, Sixers, Heat, Lakers, etc. without a Cavs angle.
    if has_lebron and non_cavs_destination_count >= 1:
        return "lebron_non_cavs_move"

    broader_homecoming_language = (
        r"\b(coming home|come home|back home|homecoming|"
        r"going back|go back|return|returns|reunion|"
        r"end his career|finish his career|retire|last dance|"
        r"storybook|story book)\b"
    )

    implied_lebron_homecoming = bool(
        has_cavs
        and re.search(
            r"\b(he|hes|he's|him).{0,40}(coming home|come home|back home|going back|go back|return|reunion|retire|end his career|finish his career)\b",
            text,
        )
    )

    if (has_lebron or implied_lebron_homecoming) and (
        re.search(specific_cavs_homecoming, text)
        or (has_cavs and re.search(broader_homecoming_language, text))
    ):
        return "lebron_cavs_rumor"

    # general roster changes.
    if has_cavs and re.search(
        r"\b(kuminga|jonathan kuminga|bronny|schroder|schröder|strus|roster|depth|bench|rotation)\b",
        text,
    ):
        return "general_roster_changes"

    # cavs draft discussion
    if (
        re.search(
            r"\b(thomas|will go|maleek thomas|udeh|nba draft|draft|drafted|draft pick|draft picks|lottery|rookie|rookies|prospect|prospects|summer league|first round|second round)\b",
            text,
        )
        and non_cavs_destination_count < 2
    ):
        return "nba_draft"

    # current cavs core
    if has_cavs and re.search(
        r"\b(harden|james harden|mobley|evan mobley|donovan mitchell|mitchell|spida|jarrett allen|allen|atkinson|kenny atkinson|kenny)\b",
        text,
    ):
        return "cavs_curr_core"

    return "na"


def add_narrative_labels(df):
    df["narrative"] = df[TEXT_COL].apply(assign_narrative)
    return df


# -----------------------------
# RUN
# -----------------------------

df = pd.read_csv(INPUT_CSV)

if TEXT_COL not in df.columns:
    raise ValueError(f"Could not find text column: {TEXT_COL}")

all_results = []

tweets = df[TEXT_COL].fillna("").tolist()

for i in tqdm(range(0, len(tweets), BATCH_SIZE), desc="Running sentiment analysis"):
    batch = tweets[i : i + BATCH_SIZE]
    batch_results = predict_sentiment_batch(batch)
    all_results.extend(batch_results)

sentiment_df = pd.DataFrame(all_results)

df_out = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

df_out = add_engagement_metrics(df_out)
df_out = add_narrative_labels(df_out)

# Keep only the clean final columns.
df_out = df_out[[col for col in FINAL_COLUMNS if col in df_out.columns]]

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved sentiment results to {OUTPUT_CSV}")
print()
print("Sentiment distribution:")
print(df_out["sentiment"].value_counts(normalize=True))
print()
print("Narrative distribution:")
print(df_out["narrative"].value_counts())
print()
print("Average sentiment by narrative:")
print(df_out.groupby("narrative")["sentiment_score"].mean().sort_values())
print()
print("Average weighted sentiment by narrative:")
print(df_out.groupby("narrative")["weighted_sentiment_score"].mean().sort_values())

# -----------------------------
# PREVIEW NARRATIVES
# -----------------------------

print()
print("=" * 80)
print("NARRATIVE PREVIEW")
print("=" * 80)

if "narrative" in df_out.columns:
    narratives = sorted(df_out["narrative"].dropna().unique())

    for narrative in narratives:
        narrative_df = df_out[df_out["narrative"] == narrative].copy()

        if narrative_df.empty:
            continue

        print()
        print("-" * 80)
        print(f"Narrative: {narrative}")
        print(f"Tweet count: {len(narrative_df)}")
        print("-" * 80)

        # Show the most impactful tweets in this narrative.
        # Uses absolute value so it surfaces both strongly positive and strongly negative tweets.
        preview_df = narrative_df.copy()
        preview_df["abs_weighted_sentiment_score"] = preview_df[
            "weighted_sentiment_score"
        ].abs()

        preview_df = preview_df.sort_values(
            by="abs_weighted_sentiment_score",
            ascending=False,
        ).head(5)

        for _, row in preview_df.iterrows():
            print()
            print(f"Sentiment: {row['sentiment']} ({row['sentiment_score']:.3f})")
            print(f"Weighted sentiment: {row['weighted_sentiment_score']:.3f}")
            print(f"Engagement: {row['engagement_total']} | Views: {row['views']}")
            print(f"Tweet: {row['tweet']}")
else:
    print("No narrative column found.")
