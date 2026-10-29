"""Model registry: save trained models to MongoDB (GridFS) and load them back.

`models` collection (schema from the project guide)
    name       String    model name/version, e.g. "tfidf_logreg_v1"
    type       String    "TF-IDF + LogisticRegression", "LSTM", "Transformer", ...
    gridfs_id  ObjectId  the serialized model file in GridFS (bucket "model_files")
    labels     [String]  condition labels, in the model's output order
    metrics    JSON      validation/test results (top-k, F1, emergency recall, ...)
    created    DateTime  when the model was saved
  (+ params, size_mb, local_path for convenience)

A local copy is also written to models/<name>.joblib so predictions work
without MongoDB and loading is fast.
"""
import io
import os
from datetime import datetime, timezone

import joblib

LOCAL_DIR = "models"
BUCKET = "model_files"


def _fs():
    import gridfs
    import db
    return gridfs.GridFS(db.get_db(), collection=BUCKET)


def _serialize(model):
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getvalue()


def save_model(model, name, model_type, labels, metrics, params=None, to_mongo=True):
    """Save locally and (optionally) to MongoDB. Re-saving a name replaces it."""
    data = _serialize(model)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_DIR, f"{name}.joblib")
    with open(local_path, "wb") as f:
        f.write(data)

    doc = {
        "name": name,
        "type": model_type,
        "labels": list(labels),
        "metrics": metrics,
        "params": params or {},
        "size_mb": round(len(data) / 1e6, 1),
        "local_path": local_path,
        "created": datetime.now(timezone.utc),
    }
    if to_mongo:
        import db
        fs = _fs()
        old = db.models().find_one({"name": name})
        if old and old.get("gridfs_id"):
            fs.delete(old["gridfs_id"])
        doc["gridfs_id"] = fs.put(data, filename=f"{name}.joblib", model_name=name)
        db.models().replace_one({"name": name}, doc, upsert=True)
    return doc


def list_models():
    import db
    return list(db.models().find({}, {"labels": 0}).sort("created", -1))


def get_model_doc(name=None):
    """Model metadata by name, or the most recently saved model."""
    import db
    if name:
        return db.models().find_one({"name": name})
    return db.models().find_one(sort=[("created", -1)])


def load_model(name=None, source="auto"):
    """Load a model object.

    source: "local" -> models/<name>.joblib
            "mongo" -> GridFS
            "auto"  -> local file if present, else GridFS
    """
    if source in ("auto", "local") and name:
        path = os.path.join(LOCAL_DIR, f"{name}.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        if source == "local":
            raise FileNotFoundError(path)
    doc = get_model_doc(name)
    if not doc:
        raise LookupError(f"No model named {name!r} in MongoDB" if name else "No models in MongoDB")
    data = _fs().get(doc["gridfs_id"]).read()
    return joblib.load(io.BytesIO(data))
