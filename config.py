"""Central settings. Override any value with an environment variable."""
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "medical_diagnosis")

CONDITIONS_COLLECTION = "conditions"
MODELS_COLLECTION = "models"

NHS_BASE_URL = "https://www.nhsinform.scot"
NHS_AZ_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"

# Be polite to the NHS server: wait between requests.
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3

# A normal browser User-Agent; the site rejects some default library agents.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}
