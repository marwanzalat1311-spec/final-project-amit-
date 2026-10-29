"""Command-line scraper: the quickest way to build the dataset.

Examples
    python scrape.py --limit 10          # quick test on 10 new conditions
    python scrape.py                     # scrape everything not saved yet (resumes)
    python scrape.py --retry-failed      # only the pages that failed last time
    python scrape.py --all               # re-scrape every page, even saved ones
    python scrape.py --urls data/rescrape_urls.txt   # re-scrape a chosen list
    python scrape.py --delay 3           # slower, if the site keeps refusing
    python scrape.py --no-mongo          # only write data/conditions.json

Every page is saved as soon as it is downloaded, so you can stop (Ctrl+C) and
run the same command again: it continues where it stopped.
"""
import argparse
import json
import logging
import os
from collections import Counter

import config
from scraper import nhs_scraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FIELDS = ("symptoms", "causes", "warnings", "recommendations")
FAILED_FILE = "data/failed_urls.txt"


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return {d["url"]: d for d in json.load(f)}
    except FileNotFoundError:
        return {}


def save_json(path, by_url):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(by_url.values(), key=lambda d: d["condition"].lower()),
                  f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="scrape at most N pages")
    p.add_argument("--retry-failed", action="store_true", help=f"only retry URLs in {FAILED_FILE}")
    p.add_argument("--all", action="store_true", help="re-scrape pages that are already saved")
    p.add_argument("--urls", default=None, help="file with one URL per line to (re-)scrape, even if saved")
    p.add_argument("--delay", type=float, default=None, help="seconds between pages (default 1)")
    p.add_argument("--no-mongo", action="store_true", help="skip MongoDB, save JSON only")
    p.add_argument("--out", default="data/conditions.json")
    args = p.parse_args()

    if args.delay is not None:
        config.REQUEST_DELAY_SECONDS = args.delay

    saved = load_json(args.out)

    use_mongo = not args.no_mongo
    if use_mongo:
        import db
        db.init_indexes()
        mongo_urls = {d["url"] for d in db.conditions().find({}, {"url": 1})}
    else:
        mongo_urls = set()

    # Which pages to scrape
    urls = None
    if args.retry_failed:
        try:
            with open(FAILED_FILE, encoding="utf-8") as f:
                urls = [line.split("\t")[0].strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise SystemExit(f"{FAILED_FILE} not found - nothing to retry.")
        print(f"Retrying {len(urls)} failed pages")
    if args.urls:
        with open(args.urls, encoding="utf-8") as f:
            urls = [line.split("\t")[0].strip() for line in f if line.strip().startswith("http")]
        print(f"Re-scraping {len(urls)} pages from {args.urls}")
    skip = None if (args.all or args.urls) else (set(saved) | mongo_urls)

    def on_doc(doc):
        saved[doc["url"]] = doc
        save_json(args.out, saved)                 # page by page: nothing lost on Ctrl+C
        if use_mongo:
            db.upsert_conditions([doc])

    def progress(i, total, url, ok):
        reason = "" if ok else f"  ({nhs_scraper.last_error})"
        print(f"[{i}/{total}] {'OK  ' if ok else 'FAIL'} {url}{reason}")

    try:
        docs, failed = nhs_scraper.scrape_all(limit=args.limit, urls=urls, skip=skip,
                                              on_doc=on_doc, on_progress=progress)
    except KeyboardInterrupt:
        print("\nStopped. Everything downloaded so far is saved; run the same command to continue.")
        return

    # Failed list for --retry-failed
    os.makedirs("data", exist_ok=True)
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for url, reason in failed:
            f.write(f"{url}\t{reason}\n")

    all_docs = list(saved.values())
    print("\n" + "=" * 60)
    print(f"Downloaded this run : {len(docs)}")
    print(f"Failed this run     : {len(failed)}")
    if failed:
        print("Failure reasons     :", dict(Counter(r for _, r in failed)))
        print(f"Failed URLs saved to {FAILED_FILE} -> run: python scrape.py --retry-failed")
    print(f"Total conditions    : {len(all_docs)}  (in {args.out})")
    if use_mongo:
        print(f"Total in MongoDB    : {db.conditions().count_documents({})}")
    print("Empty fields        :", {f: sum(1 for d in all_docs if not d.get(f)) for f in FIELDS})


if __name__ == "__main__":
    main()
