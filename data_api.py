"""Data Handling API (Flask).

Endpoints
---------
GET  /health                     - API + MongoDB status
POST /scrape                     - scrape NHS Inform A-Z into MongoDB
                                   body (optional): {"limit": 20}
POST /scrape/one                 - scrape a single condition page
                                   body: {"url": "https://www.nhsinform.scot/..."}
GET  /conditions                 - list conditions (?q=text&page=1&per_page=50)
GET  /conditions/<name>          - one condition by exact name (case-insensitive)
GET  /stats                      - counts and empty-field report
GET  /export?format=json|csv     - download the whole collection
"""
import csv
import io
import logging
import re
import threading

from flask import Flask, Response, jsonify, request

import db
from scraper import nhs_scraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)

FIELDS = ["symptoms", "causes", "warnings", "recommendations"]

# Simple in-memory job state so a long scrape does not block the HTTP request.
job = {"running": False, "done": 0, "total": 0, "saved": 0, "failed": [], "error": None}
job_lock = threading.Lock()


def _public(doc):
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    for k in ("scraped_at", "created_at", "updated_at"):
        if k in doc and doc[k] is not None:
            doc[k] = doc[k].isoformat()
    return doc


def _run_scrape(limit):
    def progress(i, total, url, ok):
        job["done"], job["total"] = i, total

    try:
        docs, failed = nhs_scraper.scrape_all(limit=limit, on_progress=progress)
        inserted, updated = db.upsert_conditions(docs)
        job.update(saved=len(docs), inserted=inserted, updated=updated, failed=failed)
    except Exception as e:  # report the error through /scrape/status
        logging.exception("Scrape failed")
        job["error"] = str(e)
    finally:
        job["running"] = False


@app.get("/health")
def health():
    try:
        db.get_db().command("ping")
        mongo = "ok"
    except Exception as e:
        mongo = f"error: {e}"
    return jsonify(api="ok", mongodb=mongo)


@app.post("/scrape")
def scrape():
    body = request.get_json(silent=True) or {}
    limit = body.get("limit")
    with job_lock:
        if job["running"]:
            return jsonify(message="A scrape is already running", status=job), 409
        job.clear()
        job.update(running=True, done=0, total=0, saved=0, failed=[], error=None)
    threading.Thread(target=_run_scrape, args=(limit,), daemon=True).start()
    return jsonify(message="Scrape started", check="/scrape/status"), 202


@app.get("/scrape/status")
def scrape_status():
    return jsonify(job)


@app.post("/scrape/one")
def scrape_one():
    body = request.get_json(silent=True) or {}
    url = body.get("url")
    if not url or "nhsinform.scot/illnesses-and-conditions/" not in url:
        return jsonify(error="Provide an NHS Inform condition URL in 'url'"), 400
    doc = nhs_scraper.scrape_condition(url)
    if doc is None:
        return jsonify(error="Could not download that page"), 502
    db.upsert_conditions([doc])
    saved = db.conditions().find_one({"url": doc["url"]})
    return jsonify(_public(saved)), 201


@app.get("/conditions")
def list_conditions():
    q = request.args.get("q", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 50)), 1), 500)
    query = {}
    if q:
        rx = {"$regex": re.escape(q), "$options": "i"}
        query = {"$or": [{"condition": rx}, {"symptoms": rx}]}
    cursor = (db.conditions().find(query).sort("condition", 1)
              .skip((page - 1) * per_page).limit(per_page))
    return jsonify(
        total=db.conditions().count_documents(query),
        page=page,
        per_page=per_page,
        items=[_public(d) for d in cursor],
    )


@app.get("/conditions/<path:name>")
def get_condition(name):
    doc = db.conditions().find_one(
        {"condition": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
    )
    if not doc:
        return jsonify(error="Not found"), 404
    return jsonify(_public(doc))


@app.get("/stats")
def stats():
    col = db.conditions()
    total = col.count_documents({})
    empty = {f: col.count_documents({f: {"$size": 0}}) for f in FIELDS}
    by_category = list(col.aggregate([
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]))
    return jsonify(total_conditions=total, empty_fields=empty,
                   by_category={c["_id"]: c["count"] for c in by_category})


@app.get("/export")
def export():
    fmt = request.args.get("format", "json")
    docs = [_public(d) for d in db.conditions().find().sort("condition", 1)]
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["condition", "category", "url"] + FIELDS)
        for d in docs:
            w.writerow([d["condition"], d.get("category"), d["url"]] +
                       [" | ".join(d.get(f, [])) for f in FIELDS])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=conditions.csv"})
    return jsonify(docs)


if __name__ == "__main__":
    db.init_indexes()
    app.run(host="0.0.0.0", port=5001, debug=False)
