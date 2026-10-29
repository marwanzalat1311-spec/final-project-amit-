"""Load the Step 2 dataset into arrays every model can use."""
import json
import os

import numpy as np

DATASET_DIR = os.path.join("data", "dataset")
PROCESSED = os.path.join("data", "processed_conditions.json")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels():
    with open(os.path.join(DATASET_DIR, "labels.json"), encoding="utf-8") as f:
        return json.load(f)


def load_conditions():
    """condition name -> processed record (warnings, recommendations, metadata)."""
    with open(PROCESSED, encoding="utf-8") as f:
        return {r["condition"]: r for r in json.load(f)}


class Split:
    """One dataset split as aligned lists/arrays."""

    def __init__(self, rows, label_index):
        self.rows = rows
        self.text = [r["text_norm"] for r in rows]
        self.age = np.array([r["age"] for r in rows], dtype=float)
        self.gender = [r["gender"] for r in rows]
        self.y = np.array([label_index[r["label"]] for r in rows])
        self.emergency = np.array([bool(r["emergency"]) for r in rows])
        self.unseen = np.array([not r["seen_phrases"] for r in rows])

    def __len__(self):
        return len(self.rows)


def load_splits():
    labels = load_labels()
    idx = {l: i for i, l in enumerate(labels)}
    splits = {s: Split(read_jsonl(os.path.join(DATASET_DIR, f"{s}.jsonl")), idx)
              for s in ("train", "val", "test")}
    return labels, splits
