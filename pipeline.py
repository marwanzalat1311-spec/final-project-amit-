"""End-to-end preprocessing pipeline used by both the CLI and the API.

raw conditions (MongoDB or data/conditions.json)
  -> clean + extract symptom phrases + metadata     (cleaner.py)
  -> MongoDB `processed_conditions` + data/processed_conditions.json
  -> synthetic patient samples, train/val/test       (dataset.py)
  -> data/dataset/{train,val,test}.jsonl + labels.json
  -> data/preprocessing_report.json                  (numbers for the report)
"""
import json
import os
from datetime import datetime, timezone

from . import cleaner, dataset

DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
PROCESSED_COLLECTION = "processed_conditions"


def load_raw(source="mongo", json_path=os.path.join(DATA_DIR, "conditions.json")):
    if source == "json":
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    import db
    return list(db.conditions().find({}, {"_id": 0}))


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(source="mongo", save_mongo=True, n_train=40, n_val=8, n_test=8, seed=42):
    raw = load_raw(source)
    kept, excluded = cleaner.process_all(raw)
    ds = dataset.build_dataset(kept, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "processed_conditions.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    for split, rows in ds.items():
        _write_jsonl(os.path.join(DATASET_DIR, f"{split}.jsonl"), rows)
    labels = sorted({r["condition"] for r in kept})
    with open(os.path.join(DATASET_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "params": {"n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": seed},
        "conditions": cleaner.summary(kept, excluded),
        "dataset": dataset.dataset_stats(ds),
        "excluded": excluded,
    }
    with open(os.path.join(DATA_DIR, "preprocessing_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if save_mongo:
        import db
        col = db.get_db()[PROCESSED_COLLECTION]
        col.delete_many({})
        if kept:
            col.insert_many([dict(r) for r in kept])
        col.create_index("condition")
    return report
