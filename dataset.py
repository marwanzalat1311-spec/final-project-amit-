"""Build a supervised training set from processed conditions.

Problem: NHS Inform gives ONE page per condition, but a classifier needs many
examples per label. So each condition's symptom phrases are combined into
synthetic "patient descriptions", e.g.

    phrases: ["fever", "sore throat", "swollen glands in the neck"]
    sample : "i've had fever, sore throat and swollen glands in the neck"
             age=24, gender=female, label="Tonsillitis"

Leakage control
---------------
For conditions with >= MIN_PHRASES_FOR_HOLDOUT phrases, ~25% of phrases are
held out: train samples never contain them, while every val/test sample
contains at least one. So validation/test measure generalisation to symptom
wordings the model has not seen, not memorisation. Conditions with few phrases
cannot hold any out; their val/test samples are flagged seen_phrases=True so the
report can show both numbers.
"""
import math
import random

from .text import normalize_text

TEMPLATES = [
    "{s}",
    "i have {s}",
    "i've got {s}",
    "i've had {s} for a few days",
    "for the past week i've had {s}",
    "symptoms: {s}",
    "i'm experiencing {s}",
    "my main problems are {s}",
    "since yesterday i have {s}",
    "i keep getting {s}",
    "it started with {s}",
    "{s}, getting worse",
    "patient reports {s}",
]
CHILD_TEMPLATES = [
    "my child has {s}",
    "my son has {s}",
    "my daughter has {s}",
    "my baby has {s}",
    "my kid has {s}",
]

AGE_RANGES = {"child": (0, 15), "adult": (18, 85), "any": (1, 85)}
MIN_PHRASES_FOR_HOLDOUT = 4
HOLDOUT_FRACTION = 0.25


def join_phrases(phrases):
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def sample_meta(rec, rng):
    lo, hi = AGE_RANGES[rec["age_group"]]
    age = rng.randint(lo, hi)
    gender = rec["gender"] if rec["gender"] != "any" else rng.choice(["female", "male"])
    return age, gender


def make_text(phrases, rec, age, rng):
    templates = TEMPLATES + (CHILD_TEMPLATES if age < 16 else [])
    return rng.choice(templates).format(s=join_phrases(phrases))


def split_phrases(phrases, rng):
    """Return (train_phrases, heldout_phrases)."""
    phrases = list(phrases)
    if len(phrases) < MIN_PHRASES_FOR_HOLDOUT:
        return phrases, []
    rng.shuffle(phrases)
    k = max(1, math.ceil(len(phrases) * HOLDOUT_FRACTION))
    return phrases[k:], phrases[:k]


def _pick(pool, k, rng):
    return rng.sample(pool, min(k, len(pool)))


def build_dataset(records, n_train=40, n_val=8, n_test=8, seed=42):
    """Return dict split -> list of samples."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}

    for rec in records:
        # Short phrases are what patients type; long sentences add context in train.
        phrases = rec["symptom_phrases"] or []
        sentences = [x[0].lower() + x[1:] for x in rec.get("symptom_sentences", []) if x]
        pool = phrases if len(phrases) >= 2 else phrases + sentences
        train_p, held_p = split_phrases(pool, rng)
        if not train_p:
            continue

        def sample(split, must_include=None):
            k = rng.randint(1, 4)
            chosen = _pick(train_p, k, rng)
            if must_include:
                chosen = _pick(must_include, rng.randint(1, min(2, len(must_include))), rng) + \
                         _pick(train_p, rng.randint(0, 2), rng)
            rng.shuffle(chosen)
            age, gender = sample_meta(rec, rng)
            text = make_text(chosen, rec, age, rng)
            return {
                "text": text,
                "text_norm": normalize_text(text),
                "age": age,
                "gender": gender,
                "label": rec["condition"],
                "category": rec.get("category"),
                "emergency": rec["emergency"],
                "split": split,
                "seen_phrases": not bool(must_include),
            }

        for _ in range(n_train):
            out["train"].append(sample("train"))
        # Sentences give extra train-only variety (never in val/test).
        for s in sentences[:5]:
            age, gender = sample_meta(rec, rng)
            out["train"].append({"text": s.lower(), "text_norm": normalize_text(s), "age": age,
                                 "gender": gender, "label": rec["condition"],
                                 "category": rec.get("category"), "emergency": rec["emergency"],
                                 "split": "train", "seen_phrases": True})
        for _ in range(n_val):
            out["val"].append(sample("val", held_p or None))
        for _ in range(n_test):
            out["test"].append(sample("test", held_p or None))

    for split in out.values():
        rng.shuffle(split)
    return out


def dataset_stats(ds):
    labels = {s["label"] for s in ds["train"]}
    return {
        "labels": len(labels),
        "samples": {k: len(v) for k, v in ds.items()},
        "test_unseen_phrase_share": round(
            sum(not s["seen_phrases"] for s in ds["test"]) / max(len(ds["test"]), 1), 3),
        "avg_words_train": round(sum(len(s["text"].split()) for s in ds["train"]) /
                                 max(len(ds["train"]), 1), 1),
    }
