"""Turn a model's probabilities into the final answer for a user.

1. model probabilities for every condition
2. metadata adjustment  - gender-specific conditions are removed for the other
                          gender (e.g. ovarian cancer for a man); child-only
                          conditions are down-weighted for adults and vice versa
3. renormalise and rank - top-k conditions with probabilities
4. attach advice        - warnings and recommendations from processed_conditions
5. emergency alert      - if an emergency condition is in the top 3
"""
import json
import os

import numpy as np

from modeling import registry
from preprocessing.text import normalize_text

AGE_MISMATCH_FACTOR = 0.2
DISCLAIMER = ("This is an AI estimate based on NHS Inform content, not a diagnosis. "
              "If you are worried, contact your GP, phone 111, or phone 999 in an emergency.")


def load_condition_info():
    """condition -> processed record. Prefer MongoDB, fall back to the JSON file."""
    try:
        import db
        from preprocessing.pipeline import PROCESSED_COLLECTION
        docs = list(db.get_db()[PROCESSED_COLLECTION].find({}, {"_id": 0}))
        if docs:
            return {d["condition"]: d for d in docs}
    except Exception:
        pass
    with open(os.path.join("data", "processed_conditions.json"), encoding="utf-8") as f:
        return {d["condition"]: d for d in json.load(f)}


def advice_lines(lines, n):
    """Drop dangling lead-ins ("Your child:") unless they carry urgency ("Phone 999 if:")."""
    import re
    keep = [l for l in lines if not l.rstrip().endswith(":")
            or re.search(r"\b(999|111|A&E|GP|urgent)", l, re.I)]
    return keep[:n]


class Predictor:
    def __init__(self, model=None, model_name=None, info=None):
        self.model = model if model is not None else registry.load_model(model_name)
        self.model_name = model_name
        self.labels = list(self.model.labels)
        self.info = info if info is not None else load_condition_info()
        self._gender = np.array([self.info.get(l, {}).get("gender", "any") for l in self.labels])
        self._age = np.array([self.info.get(l, {}).get("age_group", "any") for l in self.labels])

    def adjust(self, p, age=None, gender=None):
        p = p.copy()
        if gender in ("female", "male"):
            other = "male" if gender == "female" else "female"
            p[self._gender == other] = 0.0
        if age is not None:
            if age >= 18:
                p[self._age == "child"] *= AGE_MISMATCH_FACTOR
            elif age < 16:
                p[self._age == "adult"] *= AGE_MISMATCH_FACTOR
        s = p.sum()
        return p / s if s > 0 else p

    def predict(self, text, age=None, gender=None, top_k=5, n_advice=5):
        norm = normalize_text(text)
        if not norm:
            raise ValueError("Please describe your symptoms.")
        g = gender if gender in ("female", "male") else "unknown"
        a = age if age is not None else -1
        p = self.model.predict_proba([norm], [a], [g])[0]
        p = self.adjust(p, age, gender)

        order = np.argsort(-p)[:top_k]
        results = []
        for rank, i in enumerate(order, 1):
            name = self.labels[i]
            rec = self.info.get(name, {})
            results.append({
                "rank": rank,
                "condition": name,
                "probability": round(float(p[i]), 4),
                "emergency": bool(rec.get("emergency", False)),
                "urgency": rec.get("urgency", "routine"),
                "warnings": advice_lines(rec.get("warnings", []), n_advice),
                "recommendations": advice_lines(rec.get("recommendations", []), n_advice),
                "url": rec.get("url"),
            })
        alert = any(r["emergency"] for r in results[:3])
        return {
            "input": {"text": text, "age": age, "gender": gender, "normalized": norm},
            "model": self.model_name,
            "predictions": results,
            "emergency_alert": alert,
            "emergency_message": ("One of the likely conditions can be a medical emergency. "
                                  "If symptoms are severe or sudden, phone 999 or go to A&E."
                                  if alert else None),
            "disclaimer": DISCLAIMER,
        }
