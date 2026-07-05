# cleaning.R

library(tidyverse)
library(lubridate)
library(textclean)
library(stringr)
library(cld3)

# -----------------------------
# file paths
# -----------------------------

raw_path <- "../../data/raw/tweets_raw.csv"
out_path <- "../../data/int/tweets_int.csv"

# -----------------------------
# helper: clean tweet text
# -----------------------------

clean_tweet_text <- function(text) {
  text %>%
    iconv("UTF-8", "UTF-8", sub = "") %>%              # preserve valid utf-8 + emojis
    replace_url(replacement = " ") %>%                 # remove urls
    replace_html() %>%                                 # convert html entities
    str_replace_all("&amp;", "and") %>%                # common html leftover
    str_replace_all("&", " and ") %>%                  # convert standalone &
    str_remove_all("@[A-Za-z0-9_]+") %>%               # remove mentions
    str_replace_all("#([A-Za-z0-9_]+)", "\\1") %>%     # keep hashtag text
    str_replace_all("\\bRT\\b", " ") %>%               # remove retweet token
    str_replace_all("[\"“”]", "") %>%                  # remove double quotes
    str_replace_all("[’‘']", "") %>%                   # remove apostrophes
    str_replace_all("…", " ") %>%                      # remove ellipsis character
    str_replace_all("/", " ") %>%                      # split joined names like ant/melo/rudy
    str_replace_all("\\s+", " ") %>%                   # normalize whitespace
    str_squish() %>%
    str_to_lower()
}

clean_query_text <- function(query) {
  query %>%
    iconv("UTF-8", "UTF-8", sub = "") %>%
    replace_html() %>%
    str_replace_all('""', '"') %>%
    str_replace_all('^"+|"+$', "") %>%
    str_replace_all("#([A-Za-z0-9_]+)", "\\1") %>%     # #cavs -> cavs
    str_replace_all("[\"“”]", "") %>%
    str_replace_all("[’‘']", "") %>%
    str_replace_all("\\s+", " ") %>%
    str_squish() %>%
    str_to_lower()
}

query_to_regex <- function(query) {
  # Escape regex characters, then allow flexible spaces between words.
  query %>%
    str_replace_all("([\\^$.|?*+()\\[\\]{}\\\\])", "\\\\\\1") %>%
    str_replace_all("\\s+", "\\\\s+")
}

# -----------------------------
# read raw data
# -----------------------------

tweets_raw <- read_csv(raw_path, show_col_types = FALSE)

# -----------------------------
# clean data
# -----------------------------

tweets_clean <- tweets_raw %>%
  mutate(
    tweet = clean_tweet_text(tweet),
    date = ymd_hms(date, quiet = TRUE),
    query = clean_query_text(query),
    stats = str_remove_all(stats, "[()]")
  ) %>%
  separate(
    stats,
    into = c("views", "likes", "comments", "retweets"),
    sep = ",\\s*",
    convert = TRUE,
    remove = TRUE
  ) %>%
  mutate(
    language = detect_language(tweet),

    # standard timestamp string for pandas
    date = format(date, "%Y-%m-%d %H:%M:%S"),

    across(
      c(views, likes, comments, retweets),
      ~ replace_na(as.integer(.x), 0L)
    ),

    query_regex = map_chr(query, query_to_regex),
    query_in_tweet = str_detect(tweet, query_regex),

    engagement_total = likes + comments + retweets,
    word_count = str_count(tweet, "\\S+")
  ) %>%
  filter(
    !is.na(tweet),
    tweet != "",
    !is.na(query),
    query != "",
    language == "en",
    query_in_tweet
  ) %>%
  select(
    tweet,
    date,
    query,
    views,
    likes,
    comments,
    retweets,
    engagement_total,
    word_count
  )

# -----------------------------
# save cleaned file
# -----------------------------

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

write_csv(tweets_clean, out_path)

print(glimpse(tweets_clean))
cat("\ncleaned english-only data saved to:", out_path, "\n")