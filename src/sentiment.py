import re

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
        cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt"
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

df_out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved sentiment results to {OUTPUT_CSV}")
print(df_out["sentiment"].value_counts(normalize=True))
