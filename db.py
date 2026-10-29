"""MongoDB connection helpers and collection setup."""
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, UpdateOne

import config

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[config.MONGO_DB]


def conditions():
    return get_db()[config.CONDITIONS_COLLECTION]


def models():
    return get_db()[config.MODELS_COLLECTION]


def init_indexes():
    """Create indexes once; safe to call repeatedly."""
    # Name is indexed for search but not unique: two pages can share a name.
    info = conditions().index_information()
    if info.get("condition_1", {}).get("unique"):
        conditions().drop_index("condition_1")
    conditions().create_index([("condition", ASCENDING)])
    conditions().create_index([("url", ASCENDING)], unique=True)
    models().create_index([("name", ASCENDING)], unique=True)


def upsert_conditions(docs):
    """Insert or update condition documents, keyed by URL.

    Returns (inserted, updated) counts.
    """
    if not docs:
        return 0, 0
    now = datetime.now(timezone.utc)
    ops = []
    for d in docs:
        d = dict(d)
        d["updated_at"] = now
        ops.append(
            UpdateOne(
                {"url": d["url"]},
                {"$set": d, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
        )
    result = conditions().bulk_write(ops, ordered=False)
    return result.upserted_count, result.modified_count
