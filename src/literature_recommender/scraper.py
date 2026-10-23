import json
import random
import re
from time import sleep

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from literature_recommender.config import (
    ADDITIONAL_LINKS_FILE,
    HEADERS,
    RAW_DATA_FILE,
    REQUEST_DELAY_MULTIPLIER,
    SCRAPING_DATA_DIR,
    STARTING_URLS,
    TARGET_ARTICLE_COUNT,
)


def clean_content(content: BeautifulSoup) -> None:
    """
    Removes Wikipedia noise elements (references, infoboxes, navboxes).
    """
    unwanted_selectors = [
        "sup.reference",  # Citation numbers like [1], [2]
        "span.mw-editsection",  # [edit] links
        "div.reflist",  # Reference blocks at the bottom
        "div.navbox",  # Footer navigation boxes
        "table.infobox",  # Sidebars (infoboxes)
        "table.metadata",  # Stub/disambiguation notices
        "div.thumb",  # Image captions
        "div.noprint",  # Print-hidden elements
        "style",
        "script",
    ]
    for sel in unwanted_selectors:
        for element in content.select(sel):
            element.decompose()


def is_valid_book_link(href: str | None) -> bool:
    """
    Filters out footnotes, system pages, and publication year links.
    """
    if not href or not href.startswith("https://en.wikipedia.org/wiki/"):
        return False
    if "#" in href:
        return False

    disallowed_prefixes = (
        "/wiki/Wikipedia:",
        "/wiki/Category:",
        "/wiki/Special:",
        "/wiki/Help:",
        "/wiki/File:",
        "/wiki/Talk:",
        "/wiki/Portal:",
        "/wiki/Template:",
    )
    if any(href.startswith("https://en.wikipedia.org" + p) for p in disallowed_prefixes):
        return False

    return not re.search(r"/(1[789]\d\d|20\d\d)(_in_literature)?$", href)


def scrape_article(url: str) -> dict | None:
    """
    Scrapes and parses a single Wikipedia article.
    """
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        parsed = BeautifulSoup(response.text, "html.parser")

        title_tag = parsed.find("h1", {"id": "firstHeading"})
        title = title_tag.get_text().strip() if title_tag else ""

        content = parsed.select_one("#mw-content-text > div.mw-parser-output")
        if not content:
            content = parsed.select_one("#bodyContent div.mw-parser-output")

        if not content:
            print(f"No content found for {url}")
            return None

        clean_content(content)

        paragraphs = []
        for p in content.find_all("p"):
            txt = p.get_text().strip()
            if len(txt) > 30:
                txt = re.sub(r"\s+", " ", txt)
                paragraphs.append(txt)

        if not paragraphs:
            print(f"No valid paragraphs found for {url}")
            return None
        
        summary = paragraphs[0]
        full_text = "\n\n".join(paragraphs)

        categories = []
        cat_links = parsed.select("#mw-normal-catlinks ul li a")
        for cat_link in cat_links:
            categories.append(cat_link.get_text().strip())

        return {
            "title": title,
            "url": url,
            "summary": summary,
            "text": full_text,
            "categories": categories,
            "num_characters": len(full_text),
            "num_paragraphs": len(paragraphs),
        }

        
    except AttributeError as e:
        print(f"Error scraping {url}: {e}")
        return None


def extract_book_links_from_author(author_link: str) -> set[str]:
    works_links = set()
    try:
        response = requests.get(author_link, headers=HEADERS)
        parsed = BeautifulSoup(response.text, "html.parser")

        infobox = parsed.find("table", class_="infobox")
        if infobox:
            for tr in infobox.find_all("tr"):
                label = tr.find(["th", "td"], class_="infobox-label")
                clean_row_text = re.sub(
                        r"\s+", " ", tr.get_text()
                    ).strip().lower()
                if label and "notable work" in clean_row_text:
                    data = tr.find("td", class_="infobox-data")
                    if data:
                        for a in data.find_all("a"):
                            if is_valid_book_link(a.get("href")):
                                works_links.add(a.get("href"))
    except AttributeError as e:
        print(f"Error extracting works from {author_link}: {e}")
    return works_links
    

def extract_links(rows, lists) -> None:
    for row in rows:
        cols = row.find_all("td")
        if not cols:
            continue
        try:
            if len([cols[col_index].find("a").get("href") for col_index, _ in lists]) == len(lists):
                for col_index, link_list in lists:
                    link = cols[col_index].find("a").get("href")
                    link_list.append(link) 
        except AttributeError:
            continue


def get_links_to_scrape() -> tuple[list[str], list[str]]:
    links_to_books_lm = []
    links_to_authors_lm = []
    links_to_authors_n = []
    links_to_books_p = []
    links_to_authors_p = []
    for source_name, url in STARTING_URLS.items():
        sleep(0.3)
        response = requests.get(url, headers=HEADERS)
        parsed = BeautifulSoup(response.text, "html.parser")

        match source_name:
            case "Le Monde":
                rows = parsed.find("tbody").find_all("tr")
                extract_links(rows, [(1, links_to_books_lm), (2, links_to_authors_lm)])
                
            case "Nobel Prize":
                rows = parsed.find("section", {"aria-labelledby": "Laureates"}).find("tbody").find_all("tr")
                extract_links(rows, [(2, links_to_authors_n)])

            case "Pulitzer Prize":
                rows = parsed.find("section", {"aria-labelledby": "Winners"}).find('table', {"class": "wikitable"}).find_all("tr")
                extract_links(rows, [(0, links_to_authors_p), (1, links_to_books_p)])

    links_to_authors = links_to_authors_lm + links_to_authors_n + links_to_authors_p
    links_to_books = links_to_books_lm + links_to_books_p

    return links_to_books, links_to_authors


def scrape_main(debug=False):
    """
    Executes the full srcraping process: collects links, scrapes articles, and saves them to a file.
    """
    SCRAPING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    scraped_urls = set()

    if RAW_DATA_FILE.exists():
        with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    scraped_urls.add(data["url"])
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(scraped_urls)} existing records in {RAW_DATA_FILE}. Continuing the scraping.")

    if len(scraped_urls) >= TARGET_ARTICLE_COUNT:
        print(f"Already scraped {len(scraped_urls)} articles. Target {TARGET_ARTICLE_COUNT} reached. Exiting.")
        return

    if debug:
        print("STEP 1: Getting links to scrape")

    links_to_books, links_to_authors = get_links_to_scrape()
    links_to_scrape = set(links_to_books + links_to_authors)
    links_to_authors = list(set(links_to_authors))
    
    if len(links_to_scrape) < TARGET_ARTICLE_COUNT:
        if debug:
            print(f"Found only {len(links_to_scrape)} unique links.")

        if ADDITIONAL_LINKS_FILE.exists():
            with open(ADDITIONAL_LINKS_FILE, "r") as f:
                prev_len = len(links_to_scrape)
                additional_links = [line.strip() for line in f if line.strip()]
                links_to_scrape.update(additional_links)
                if debug:
                    print(f"Loaded {len(links_to_scrape) - prev_len} additional unique links from {ADDITIONAL_LINKS_FILE}.")

        else:
            with open(ADDITIONAL_LINKS_FILE, "a", encoding="utf-8") as f:
                prev_len = len(links_to_scrape)
                for author_link in tqdm(links_to_authors, desc="Expanding links from authors"):
                    works_links = extract_book_links_from_author(author_link)
                    for link in works_links:
                        links_to_scrape.add(link)
                        f.write(link + "\n")
                    f.flush()
                    sleep(random.random() * REQUEST_DELAY_MULTIPLIER)

                if debug:
                    print(f"Expanded from {prev_len} to {len(links_to_scrape)} unique links and saved additional links to {ADDITIONAL_LINKS_FILE}.")

    if debug:
        print(f"Got {len(links_to_scrape)} unique links to scrape. Continuing")

    if debug:
        print(f"STEP 2: Scraping articles [{len(scraped_urls)}/{TARGET_ARTICLE_COUNT}]")

    with open(RAW_DATA_FILE, "a", encoding="utf-8") as out_file:
        progress_bar = tqdm(
            total=TARGET_ARTICLE_COUNT, initial=len(scraped_urls), desc="Scraping"
        )

        for link in links_to_scrape:
            if len(scraped_urls) >= TARGET_ARTICLE_COUNT:
                if debug:
                    print(f"Reached target of {TARGET_ARTICLE_COUNT} articles. Stopping.")
                break

            if link in scraped_urls:
                continue

            article_data = scrape_article(link)

            if article_data and article_data["num_characters"] >= 300:
                out_file.write(json.dumps(article_data, ensure_ascii=False) + "\n")
                out_file.flush()
                scraped_urls.add(link)
                progress_bar.update(1)

            sleep(random.random() * REQUEST_DELAY_MULTIPLIER)
        progress_bar.close()

    if debug:
        print(f"Scraping completed. Successfully saved {len(scraped_urls)} articles to '{RAW_DATA_FILE}'.")