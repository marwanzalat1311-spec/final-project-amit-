"""Scraper for NHS Inform A-Z condition pages.

Step 1 of the project: collect condition links from the A-Z index, open each
condition page and extract four fields:
    symptoms, causes, warnings, recommendations
Each field is stored as a list of strings (one item per bullet/sentence block).
"""
import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- HTTP layer

_session = requests.Session()
_session.headers.update(config.HEADERS)

# Reason for the most recent failed fetch, e.g. "HTTP 403" or "Timeout".
last_error = None

# 403/429 from this site usually means "slow down": wait longer, then retry.
_RETRY_STATUS = {403, 429, 500, 502, 503, 504}


def fetch(url):
    """GET a page with retries and back-off. Returns HTML or None."""
    global last_error
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            r = _session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
            if r.status_code == 200:
                last_error = None
                return r.text
            last_error = f"HTTP {r.status_code}"
            if r.status_code in _RETRY_STATUS:
                wait = 15 * attempt if r.status_code in (403, 429) else 3 * attempt
                log.warning("%s on %s, retrying in %ss", last_error, url, wait)
                time.sleep(wait)
                continue
            return None   # 404 etc.: retrying will not help
        except requests.RequestException as e:
            last_error = type(e).__name__   # Timeout, ConnectionError, ...
            log.warning("%s on %s (attempt %d)", last_error, url, attempt)
            time.sleep(5 * attempt)
    return None


# ---------------------------------------------------------------- A-Z index

_CONDITION_PATH = re.compile(r"^/illnesses-and-conditions/(?!a-to-z)[^/]+/.+")


def parse_az_links(html, base_url=config.NHS_BASE_URL):
    """Return a de-duplicated, ordered list of condition page URLs."""
    soup = BeautifulSoup(html, "lxml")
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"].strip())
        parsed = urlparse(url)
        if parsed.netloc and "nhsinform.scot" not in parsed.netloc:
            continue
        path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
        if not _CONDITION_PATH.match(path):
            continue
        clean = f"{config.NHS_BASE_URL}{path}"
        if clean not in seen:
            seen.add(clean)
            links.append(clean)
    return links


def get_condition_links():
    html = fetch(config.NHS_AZ_URL)
    if html is None:
        raise RuntimeError(
            "Could not download the A-Z index. Check your internet connection "
            "or whether the site is blocking requests."
        )
    return parse_az_links(html)


# ---------------------------------------------------------------- Condition page

# Heading keywords -> field. Order matters: warnings is checked first because
# headings like "When to seek help for symptoms" mention symptoms too.
SECTION_RULES = [
    ("warnings", ["when to", "urgent", "emergency", "999", "111", "a&e",
                  "seek", "speak to your gp", "contact your gp", "see a gp",
                  "get help", "immediate", "call", "phone"]),
    ("symptoms", ["symptom", "signs", "how do i know", "what does it look like"]),
    ("causes", ["cause", "risk factor", "at risk", "why does", "who is affected",
                "who's affected", "triggers", "how it spreads", "how is it spread"]),
    ("recommendations", ["treat", "self-care", "self care", "self help",
                         "self-help", "manag", "prevent", "living with",
                         "things you can do", "medicine", "recover",
                         "how to ease", "relief", "what you can do", "looking after",
                         "advice"]),
]

# Text at the start of a callout box that marks it as a warning, even
# without a matching heading.
WARNING_PREFIXES = ("phone 999", "call 999", "dial 999", "phone 111", "call 111",
                    "go to a&e", "speak to your gp", "contact your gp",
                    "see your gp", "phone your gp", "seek immediate",
                    "get urgent", "emergency")

NOISE_SELECTORS = ["nav", "header", "footer", "aside", "script", "style",
                   "noscript", "form", "button", "svg", "iframe",
                   ".breadcrumb", ".breadcrumbs", ".feedback", ".print",
                   ".share", ".cookie", "[role=navigation]",
                   # Interactive "self-help guide" widget (Flu, UTI, Sore throat...):
                   # a JavaScript template above the article, with its own hidden
                   # "Last Updated:" line that would otherwise end parsing early.
                   "#selfHelpGuideMain", ".self-help-guide", "#shg-container", ".modal"]


# Headings that start an unrelated section: they stop a sub-heading from
# inheriting its parent's field.
NEUTRAL_KEYWORDS = ["complication", "diagnos", "test", "outlook", "related",
                    "other languages", "help us", "useful links", "further information",
                    "support", "about this", "last updated", "source"]

SENTINEL_NEUTRAL = "__neutral__"

# The article ends here: everything after these is related links / page footer.
STOP_HEADINGS = ("also on nhs inform", "other health sites", "help us improve",
                 "useful links", "other languages", "related links")
STOP_TEXT_PREFIXES = ("source:", "last updated")
# Cross-reference lines that carry no medical content.
SKIP_TEXT_PREFIXES = ("the images on this page", "read more about", "find out more about", "see also",
                      "more information about", "learn more about")

_INTRO_SYMPTOM = re.compile(r"\b(symptoms?|signs)\b", re.I)


def classify_heading(text):
    t = text.lower().replace("\u2019", "'")   # curly apostrophe -> straight
    for field, keywords in SECTION_RULES:
        if any(k in t for k in keywords):
            return field
    if any(k in t for k in NEUTRAL_KEYWORDS):
        return SENTINEL_NEUTRAL
    return None


def clean_condition_name(name):
    """'About aplastic anaemia' -> 'Aplastic anaemia'."""
    name = re.sub(r"^(about|overview of|what is)\s+", "", name, flags=re.I).strip()
    return name[:1].upper() + name[1:] if name else name


def _clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)   # "pain , the" -> "pain, the"
    text = re.sub(r"\(\s+", "(", text)
    return text.strip(" -••")


def _main_content(soup):
    for sel in ["main", "article", "#main-content", ".main-content", "#content"]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup


def parse_condition_page(html, url, trace=None):
    """Extract a condition document from one page's HTML.

    If `trace` is a list, every heading and text item is appended to it with
    the field it was assigned to (used by inspect_page.py for debugging).
    """
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    name = _clean(h1.get_text(" ")) if h1 else urlparse(url).path.rstrip("/").split("/")[-1]
    name = clean_condition_name(name)

    main = _main_content(soup)
    for sel in NOISE_SELECTORS:
        for node in main.select(sel):
            node.decompose()

    fields = {"symptoms": [], "causes": [], "warnings": [], "recommendations": []}
    other = []
    current = None
    by_level = {}   # heading level -> field, so sub-headings inherit from parents
    seen_heading = False

    for el in main.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ("h2", "h3", "h4"):
            level = int(el.name[1])
            heading = _clean(el.get_text(" "))
            if heading.lower().startswith(STOP_HEADINGS):
                if trace is not None:
                    trace.append((el.name.upper(), heading, "STOP"))
                break
            seen_heading = True
            field = classify_heading(heading)
            if field is None:
                # e.g. "Types of spots" under "Symptoms of acne" -> symptoms
                parents = [by_level[l] for l in sorted(by_level) if l < level]
                field = parents[-1] if parents else None
            by_level = {l: f for l, f in by_level.items() if l < level}
            by_level[level] = field
            current = None if field == SENTINEL_NEUTRAL else field
            # Keep warning headings ("Phone 999 or go to A&E if:"): they carry the
            # urgency level that the list items below them do not repeat.
            if current == "warnings" and heading not in fields["warnings"]:
                fields["warnings"].append(heading)
            if trace is not None:
                trace.append((el.name.upper(), heading, current or "-"))
            continue
        # Skip paragraphs that live inside a list item (already captured by the li).
        if el.name == "p" and el.find_parent("li"):
            continue
        # Skip list items that contain nested lists (children are captured separately).
        if el.name == "li" and el.find(["ul", "ol"]):
            text = _clean(el.find(string=True, recursive=False) or "")
        else:
            text = _clean(el.get_text(" "))
        if len(text) < 3:
            continue
        low = text.lower()
        if low.startswith(STOP_TEXT_PREFIXES):
            if trace is not None:
                trace.append(("  " + el.name, text, "STOP"))
            break
        if low.startswith(SKIP_TEXT_PREFIXES):
            continue

        # Intro text before the first heading: once it starts talking about
        # symptoms ("The main symptom of X is..."), the following lines are symptoms.
        if not seen_heading and current is None and _INTRO_SYMPTOM.search(text):
            current = "symptoms"

        target = current
        if text.lower().startswith(WARNING_PREFIXES):
            target = "warnings"
        # Items inside a small callout/alert box inherit "warnings" from the box.
        # Large wrappers whose CSS class happens to contain "panel" etc. (some
        # pages put the whole article in one) are ignored.
        box = el.find_parent(class_=re.compile(r"callout|alert|panel|warning|urgent|emergency", re.I))
        if box is not None and not box.find(["h2"]) and len(box.get_text(" ")) < 1500:
            box_text = _clean(box.get_text(" ")).lower()
            if box_text.startswith(WARNING_PREFIXES) or "999" in box_text[:80]:
                target = "warnings"

        if target:
            if text not in fields[target]:
                fields[target].append(text)
        else:
            other.append(text)
        if trace is not None:
            trace.append(("  " + el.name, text, target or "overview"))

    category = urlparse(url).path.strip("/").split("/")
    return {
        "condition": name,
        "url": url,
        "category": category[1] if len(category) > 2 else None,
        **fields,
        "overview": other[:5],          # opening paragraphs, useful later as extra context
        "source": "NHS Inform",
        "scraped_at": datetime.now(timezone.utc),
    }


def scrape_condition(url):
    html = fetch(url)
    if html is None:
        return None
    return parse_condition_page(html, url)


def scrape_all(limit=None, on_progress=None, urls=None, skip=None, on_doc=None):
    """Scrape condition pages.

    urls     - pages to scrape (default: every link on the A-Z page)
    skip     - set of URLs already saved; they are not downloaded again
    on_doc   - called with each document as soon as it is parsed, so work is
               saved page by page and an interruption loses nothing
    Returns (documents, failed) where failed is a list of (url, reason).
    """
    links = urls if urls is not None else get_condition_links()
    log.info("Found %d condition links", len(links))
    if skip:
        links = [u for u in links if u not in skip]
        log.info("%d already saved, %d left to scrape", len(skip), len(links))
    if limit:
        links = links[:limit]

    docs, failed = [], []
    streak = 0   # consecutive failures
    for i, url in enumerate(links, 1):
        doc = scrape_condition(url)
        if doc is None:
            failed.append((url, last_error or "unknown"))
            streak += 1
            if streak >= 5:
                # Network drop or the site is blocking us: pause before going on.
                log.warning("%d failures in a row, pausing 60s...", streak)
                time.sleep(60)
                streak = 0
        else:
            streak = 0
            docs.append(doc)
            if on_doc:
                on_doc(doc)
        if on_progress:
            on_progress(i, len(links), url, doc is not None)
        time.sleep(config.REQUEST_DELAY_SECONDS)
    return docs, failed
