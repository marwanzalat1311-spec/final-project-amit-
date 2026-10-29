"""Preprocessing API (Flask, port 5002).

Endpoints
---------
GET  /health
POST /preprocess          run the whole pipeline
                          body (optional): {"source": "mongo"|"json", "n_train": 40,
                                            "n_val": 8, "n_test": 8, "seed": 42}
GET  /report              last preprocessing report (counts, exclusions, dataset sizes)
GET  /processed           processed conditions (?q=text&page=1&per_page=50)
GET  /processed/<name>    one processed condition
GET  /excluded            pages excluded and why
GET  /dataset/<split>     samples from train|val|test (?limit=20&offset=0)
POST /normalize           {"text": "..."} -> normalized text + tokens (same as at prediction)
"""
import json
import os
import re

from flask import Flask, jsonify, request

import db
from preprocessing import pipeline
from preprocessing.text import normalize_text, tokenize

app = Flask(__name__)
REPORT = os.path.join(pipeline.DATA_DIR, "preprocessing_report.json")


def _col():
    return db.get_db()[pipeline.PROCESSED_COLLECTION]


def _public(doc):
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _report():
    try:
        with open(REPORT, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@app.get("/health")
def health():
    try:
        db.get_db().command("ping")
        mongo = "ok"
    except Exception as e:
        mongo = f"error: {e}"
    return jsonify(api="ok", mongodb=mongo, processed=_report() is not None)


@app.post("/preprocess")
def preprocess():
    b = request.get_json(silent=True) or {}
    rep = pipeline.run(source=b.get("source", "mongo"),
                       n_train=int(b.get("n_train", 40)), n_val=int(b.get("n_val", 8)),
                       n_test=int(b.get("n_test", 8)), seed=int(b.get("seed", 42)))
    rep = dict(rep)
    rep.pop("excluded")          # long list: see GET /excluded
    return jsonify(rep), 201


@app.get("/report")
def report():
    rep = _report()
    return (jsonify(rep), 200) if rep else (jsonify(error="Run POST /preprocess first"), 404)


@app.get("/processed")
def processed():
    q = request.args.get("q", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 50)), 1), 500)
    query = {}
    if q:
        rx = {"$regex": re.escape(q), "$options": "i"}
        query = {"$or": [{"condition": rx}, {"symptom_phrases": rx}]}
    cur = _col().find(query).sort("condition", 1).skip((page - 1) * per_page).limit(per_page)
    return jsonify(total=_col().count_documents(query), page=page,
                   items=[_public(d) for d in cur])


@app.get("/processed/<path:name>")
def processed_one(name):
    doc = _col().find_one({"condition": {"$regex": f"^{re.escape(name)}$", "$options": "i"}})
    return (jsonify(_public(doc)), 200) if doc else (jsonify(error="Not found"), 404)


@app.get("/excluded")
def excluded():
    rep = _report()
    return jsonify(rep["excluded"] if rep else [])


@app.get("/dataset/<split>")
def dataset_split(split):
    if split not in ("train", "val", "test"):
        return jsonify(error="split must be train, val or test"), 400
    limit = min(int(request.args.get("limit", 20)), 1000)
    offset = int(request.args.get("offset", 0))
    path = os.path.join(pipeline.DATASET_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        return jsonify(error="Run POST /preprocess first"), 404
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return jsonify(split=split, offset=offset, items=rows)


@app.post("/normalize")
def normalize():
    text = (request.get_json(silent=True) or {}).get("text", "")
    return jsonify(text=text, normalized=normalize_text(text), tokens=tokenize(text))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
