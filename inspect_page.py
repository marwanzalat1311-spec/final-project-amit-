"""Debug tool: show how the parser reads one NHS Inform page.

    python inspect_page.py acne
    python inspect_page.py https://www.nhsinform.scot/illnesses-and-conditions/...
    python inspect_page.py flu --save-html     # also save the page HTML to data/debug/

Prints every heading and text item with the field it was assigned to,
so wrong or missing sections are easy to spot.
"""
import json
import sys

from scraper import nhs_scraper


def find_url(term):
    if term.startswith("http"):
        return term
    try:
        with open("data/conditions.json", encoding="utf-8") as f:
            for d in json.load(f):
                if term.lower() in d["condition"].lower() or term.lower() in d["url"]:
                    return d["url"]
    except FileNotFoundError:
        pass
    for url in nhs_scraper.get_condition_links():
        if term.lower() in url.lower():
            return url
    sys.exit(f"No condition found matching '{term}'")


def main():
    args = [a for a in sys.argv[1:] if a != "--save-html"]
    if not args:
        sys.exit("Usage: python inspect_page.py <name or url> [--save-html]")
    url = find_url(" ".join(args))
    html = nhs_scraper.fetch(url)
    if html is None:
        sys.exit(f"Could not download {url}")
    if "--save-html" in sys.argv:
        import os
        os.makedirs("data/debug", exist_ok=True)
        path = "data/debug/" + url.rstrip("/").split("/")[-1] + ".html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved raw HTML to {path}")
    trace = []
    doc = nhs_scraper.parse_condition_page(html, url, trace=trace)
    print(f"URL: {url}\nCondition: {doc['condition']}\n")
    for tag, text, field in trace:
        short = text if len(text) <= 70 else text[:67] + "..."
        print(f"{tag:<5} [{field:<15}] {short}")
    print("\nCounts:", {k: len(doc[k]) for k in ("symptoms", "causes", "warnings", "recommendations")})


if __name__ == "__main__":
    main()
