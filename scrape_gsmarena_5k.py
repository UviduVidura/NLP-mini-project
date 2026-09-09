"""
GSMArena 5,000 Reviews Scraper
-------------------------------
Collects ~5,000 user reviews from GSMArena across top flagship smartphone models
(Samsung Galaxy S24, S24 Ultra, S23, S23 Ultra, iPhone 15 Pro Max).

Key Features:
- Ultra-Fast Resume: Starts directly from the last scraped page (max_page + 1)
  for each phone model, avoiding re-fetching earlier pages.
- Robust File Locking: Retry loop with atomic write on Windows prevents PermissionError (Errno 13).
- Auto-Skips Completed Models: Skips phones that are fully crawled or already in seed.
- Session Pooling: High-performance keep-alive HTTP requests.
- UTF-8 Console: Safe emoji handling on Windows.

Usage:
    python scrape_gsmarena_5k.py
"""

import sys
import os
import csv
import re
import time
import random
import requests
from bs4 import BeautifulSoup

# Ensure Windows stdout handles emojis cleanly without crashing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

TARGET_COUNT = 5000
OUTPUT_CSV = "gsmarena_reviews_5k.csv"
SEED_CSV = "gsmarena_samsung_galaxy_s24_reviews.csv"

DEVICES = [
    {
        "model": "Samsung Galaxy S24",
        "first_page": "https://www.gsmarena.com/samsung_galaxy_s24-reviews-12773.php",
        "base_url": "https://www.gsmarena.com/samsung_galaxy_s24-reviews-12773p{page}.php",
    },
    {
        "model": "Samsung Galaxy S24 Ultra",
        "first_page": "https://www.gsmarena.com/samsung_galaxy_s24_ultra-reviews-12771.php",
        "base_url": "https://www.gsmarena.com/samsung_galaxy_s24_ultra-reviews-12771p{page}.php",
    },
    {
        "model": "Samsung Galaxy S23",
        "first_page": "https://www.gsmarena.com/samsung_galaxy_s23-reviews-12082.php",
        "base_url": "https://www.gsmarena.com/samsung_galaxy_s23-reviews-12082p{page}.php",
    },
    {
        "model": "Samsung Galaxy S23 Ultra",
        "first_page": "https://www.gsmarena.com/samsung_galaxy_s23_ultra-reviews-12024.php",
        "base_url": "https://www.gsmarena.com/samsung_galaxy_s23_ultra-reviews-12024p{page}.php",
    },
    {
        "model": "Apple iPhone 15 Pro Max",
        "first_page": "https://www.gsmarena.com/apple_iphone_15_pro_max-reviews-12548.php",
        "base_url": "https://www.gsmarena.com/apple_iphone_15_pro_max-reviews-12548p{page}.php",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.gsmarena.com/",
}


def load_existing_dataset():
    """
    Loads existing reviews and tracks the maximum page scraped per model.
    """
    reviews = []
    seen_ids = set()
    model_max_pages = {}
    completed_models = set()

    files_to_check = []
    if os.path.exists(OUTPUT_CSV):
        files_to_check.append(OUTPUT_CSV)
    if os.path.exists(SEED_CSV):
        files_to_check.append(SEED_CSV)

    for fpath in files_to_check:
        print(f"Loading existing reviews from '{fpath}' ...")
        try:
            with open(fpath, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("review_text", "").strip()
                    op_id = row.get("opinion_id", "").strip()
                    model = row.get("phone_model", "Samsung Galaxy S24").strip()
                    if not model:
                        model = "Samsung Galaxy S24"

                    page_str = str(row.get("page", 1)).strip()
                    page_num = int(page_str) if page_str.isdigit() else 1

                    if text and len(text) >= 10:
                        key = op_id if op_id else text[:50]
                        if key not in seen_ids:
                            seen_ids.add(key)
                            reviews.append({
                                "opinion_id": op_id,
                                "phone_model": model,
                                "username": row.get("username", "Anonymous"),
                                "date_posted": row.get("date_posted", ""),
                                "review_text": text,
                                "page": page_num,
                            })
                            curr_max = model_max_pages.get(model, 0)
                            if page_num > curr_max:
                                model_max_pages[model] = page_num
        except Exception as e:
            print(f"Notice reading {fpath}: {e}")

    model_counts = {}
    for r in reviews:
        m = r["phone_model"]
        model_counts[m] = model_counts.get(m, 0) + 1

    print(f"Total reviews in memory: {len(reviews)} / {TARGET_COUNT}")
    for m, c in model_counts.items():
        max_p = model_max_pages.get(m, 0)
        print(f"  - {m}: {c} reviews (latest page recorded: {max_p})")

    # Mark S24 as complete if loaded from seed
    if model_counts.get("Samsung Galaxy S24", 0) >= 1300:
        completed_models.add("Samsung Galaxy S24")
        print("  -> 'Samsung Galaxy S24' marked as COMPLETE (70 pages parsed).")

    return reviews, seen_ids, model_max_pages, completed_models


def parse_page_reviews(soup, page_num, phone_model):
    """Extract review items from a single GSMArena page soup."""
    reviews = []
    threads = soup.select(".user-thread")

    for thread in threads:
        uopin = thread.select_one(".uopin")
        if not uopin:
            continue

        for quoted in uopin.select(".uinreply"):
            quoted.decompose()

        review_text = uopin.get_text(separator=" ", strip=True)
        if not review_text or len(review_text) < 10:
            continue

        uname_tag = thread.select_one(".uname, .uname2, .uname3")
        username = uname_tag.get_text(strip=True) if uname_tag else "Anonymous"

        upost_tag = thread.select_one(".upost")
        date_posted = upost_tag.get_text(strip=True) if upost_tag else ""

        reply_link = thread.select_one("a[href*='idOpinion=']")
        opinion_id = ""
        if reply_link and reply_link.get("href"):
            m = re.search(r"idOpinion=(\d+)", reply_link["href"])
            if m:
                opinion_id = m.group(1)

        reviews.append({
            "opinion_id": opinion_id,
            "phone_model": phone_model,
            "username": username,
            "date_posted": date_posted,
            "review_text": review_text,
            "page": page_num,
        })

    return reviews


def save_csv(reviews, filename=OUTPUT_CSV, max_retries=5):
    """Saves review list to CSV with retry handling against Windows file locks."""
    if not reviews:
        return

    fieldnames = ["opinion_id", "phone_model", "username", "date_posted", "review_text", "page"]
    tmp_filename = filename + ".tmp"

    for attempt in range(1, max_retries + 1):
        try:
            with open(tmp_filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reviews)

            # Replace atomic on Windows
            if os.path.exists(filename):
                os.replace(tmp_filename, filename)
            else:
                os.rename(tmp_filename, filename)
            return
        except PermissionError:
            time.sleep(0.6)
        except Exception as e:
            time.sleep(0.5)

    # Fallback to direct write if temp file replacement was locked
    try:
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)
    except Exception as e:
        print(f"Warning: Could not save checkpoint ({e}), will retry next flush.")


def scrape_to_target(target=TARGET_COUNT, output_csv=OUTPUT_CSV):
    all_reviews, seen_keys, model_max_pages, completed_models = load_existing_dataset()

    if len(all_reviews) >= target:
        print(f"\n🎉 Target already met! Have {len(all_reviews)} reviews (>= {target}).")
        save_csv(all_reviews, output_csv)
        return all_reviews

    print(f"\nTarget: {target} reviews. Remaining to scrape: {target - len(all_reviews)}.")

    session = requests.Session()
    session.headers.update(HEADERS)

    for dev in DEVICES:
        model = dev["model"]
        if model in completed_models:
            print(f"\n[SKIP] {model} is fully scraped.")
            continue

        # Resume from next page after max recorded page
        last_page = model_max_pages.get(model, 0)
        start_page = last_page + 1 if last_page > 0 else 1

        print(f"\n{'='*65}")
        print(f"SCRAPING {model} (Resuming from Page {start_page})")
        print(f"{'='*65}")

        page = start_page
        consecutive_empty = 0

        while len(all_reviews) < target and consecutive_empty < 3:
            url = dev["first_page"] if page == 1 else dev["base_url"].format(page=page)

            try:
                resp = session.get(url, timeout=12)
                if resp.status_code == 429:
                    print(f"Rate limited on page {page}. Sleeping 10 seconds...")
                    time.sleep(10)
                    continue
                elif resp.status_code != 200:
                    print(f"HTTP {resp.status_code} on page {page}. Ending {model} crawl.")
                    break
            except Exception as e:
                print(f"Network glitch on page {page}: {e}. Retrying in 2s...")
                time.sleep(2)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            page_reviews = parse_page_reviews(soup, page, model)

            if not page_reviews:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print(f"Reached end of review pages for {model} at page {page}.")
                    break
            else:
                consecutive_empty = 0
                added = 0
                for r in page_reviews:
                    key = r["opinion_id"] if r["opinion_id"] else r["review_text"][:50]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_reviews.append(r)
                        added += 1

                print(f"[{model}] Page {page}: +{added} new reviews (Total: {len(all_reviews)} / {target})", flush=True)

            # Save checkpoint every 5 pages
            if page % 5 == 0:
                save_csv(all_reviews, output_csv)

            if len(all_reviews) >= target:
                print(f"\n🎉 TARGET ACHIEVED: Successfully collected {len(all_reviews)} reviews!", flush=True)
                break

            page += 1
            time.sleep(random.uniform(0.4, 0.8))

        completed_models.add(model)
        save_csv(all_reviews, output_csv)
        print(f"Checkpoint saved for {model}. Total reviews: {len(all_reviews)}.", flush=True)

        if len(all_reviews) >= target:
            break

    save_csv(all_reviews, output_csv)
    print(f"\nAll done! Successfully saved {len(all_reviews)} reviews to '{output_csv}'.")
    return all_reviews


if __name__ == "__main__":
    scrape_to_target(TARGET_COUNT, OUTPUT_CSV)
