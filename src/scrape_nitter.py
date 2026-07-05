# %%
import csv
from datetime import datetime

from playwright.sync_api import sync_playwright

from utilities import random_long_wait

# %%
START_DATE = "2026-06-01"
END_DATE = "2026-07-01"


# %%
def main(p):
    page = open_page(p)
    queries = txt_to_list()

    for query in queries:
        search_query = f"{query}"

        page = search(page, search_query)

        scrape(page, search_query)

        print(f"Done scraping query: {query}")

        random_long_wait()


def txt_to_list():
    """read queries from file"""
    file_path = "ref/queries.txt"

    queries = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            queries.append(stripped_line)

    return queries


def open_page(p):
    browser = p.chromium.launch(headless=False)

    context = browser.new_context()
    page = context.new_page()

    page.goto("http://localhost:8080/")

    return page


def search(page, query):
    """
    Search local Nitter for a query while optionally filtering dates
    and excluding native retweets.

    Parameters
    ----------
    page : playwright.sync_api.Page
        Active Playwright page on your local Nitter instance.
    query : str
        Search query, e.g. "Cavs", "#LetEmKnow", "Donovan Mitchell".

    Returns
    -------
    page : playwright.sync_api.Page
        The same page after the search has been submitted.
    """
    # Fill search query
    search_input = page.locator('input[name="q"]')
    search_input.click()
    search_input.fill(query)

    # Submit search
    page.keyboard.press("Enter")

    # Open the advanced search / filter panel
    page.locator('label[for="search-panel-toggle"]').click()

    # Exclude native retweets if the checkbox exists and is not already checked
    page.locator('label[title="e-nativeretweets"]').click()
    page.locator('label[title="e-replies"]').click()

    # Optional date filters
    if START_DATE:
        page.locator('.search-panel input[name="since"][type="date"]').fill(START_DATE)
    if END_DATE:
        page.locator('.search-panel input[name="until"][type="date"]').fill(END_DATE)

    page.locator('form:has(input[name="q"]) button[type="submit"]').click()

    return page


def scrape(page, query):
    while True:
        random_long_wait()

        if page.locator("h2.timeline-end").count():
            print("NO MORE TWEETS, BREAKING NOW")
            break

        content = read_tweet_content(page)
        write_to_csv(content, query)
        print(f"appended {len(content)} to csv")

        random_long_wait()

        page = scroll_and_click(page)


def read_tweet_content(page):
    """read all tweets within page"""
    timeline_items = page.locator(".timeline-item")
    timeline_items.first.wait_for(state="attached")

    content = []
    count = timeline_items.count()

    for i in range(count):
        item = timeline_items.nth(i)

        text = grab_tweet_content(item)
        date = grab_tweet_date(item)
        stats = grab_tweet_stats(item)

        if text is None or date is None:
            continue

        tweet_data = (text, date, stats)
        content.append(tweet_data)

    return content


def grab_tweet_content(item):
    text_locator = item.locator(".tweet-content.media-body")

    if text_locator.count() == 0:
        return None

    text = text_locator.first.text_content()

    if not text:
        return None

    return " ".join(text.split())


def grab_tweet_date(item):
    date_locator = item.locator(".tweet-date a")

    if date_locator.count() == 0:
        return None

    date_str = date_locator.first.get_attribute("title")

    if not date_str:
        return None

    return datetime.strptime(date_str.strip(), "%b %d, %Y · %I:%M %p UTC")


def grab_tweet_stats(item):
    def get_stat_value(icon_class):
        stat = item.locator(f".tweet-stat:has(.{icon_class})")

        if stat.count() == 0:
            return 0

        text = stat.first.text_content()
        if not text:
            return 0

        text = text.strip()

        # remove the icon label and normalize
        parts = text.split()
        for part in reversed(parts):
            cleaned = part.replace(",", "")
            if cleaned.isdigit():
                return int(cleaned)

        return 0

    return (
        get_stat_value("icon-views"),
        get_stat_value("icon-heart"),
        get_stat_value("icon-comment"),
        get_stat_value("icon-retweet"),
    )


def scroll_and_click(page):
    page.mouse.wheel(0, 5000)

    load_more = page.get_by_role("link", name="Load more")

    if load_more.count() > 0:
        load_more.first.click()

    random_long_wait()
    random_long_wait()

    return page


def write_to_csv(tweets, query, filename="tweets_raw"):
    """
    write (tweet, date, stats) tuples to a csv file.
    """
    filepath = f"../data/raw/{filename}.csv"
    expected_header = ["tweet", "date", "stats", "query"]

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        first_row = next(csv.reader(f), None)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if first_row != expected_header:
            writer.writerow(expected_header)

        rows = [(tweet, date, stats, query) for tweet, date, stats in tweets]
        writer.writerows(rows)


if __name__ == "__main__":
    with sync_playwright() as p:
        main(p)
