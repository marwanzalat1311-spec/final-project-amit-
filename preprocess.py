"""Run Step 2 preprocessing from the command line.

    python preprocess.py                 # read raw conditions from MongoDB
    python preprocess.py --source json   # read data/conditions.json instead
    python preprocess.py --no-mongo      # don't write processed_conditions to MongoDB
"""
import argparse
import json

from preprocessing import pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mongo", "json"], default="mongo")
    p.add_argument("--no-mongo", action="store_true")
    p.add_argument("--n-train", type=int, default=40, help="train samples per condition")
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rep = pipeline.run(source=a.source, save_mongo=not a.no_mongo,
                       n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, seed=a.seed)
    c, d = rep["conditions"], rep["dataset"]
    print(f"Raw records        : {c['raw_records']}")
    print(f"Kept conditions    : {c['kept_conditions']}")
    print(f"Excluded           : {c['excluded']}  {c['excluded_reasons']}")
    print(f"Phrases/condition  : {c['phrases_per_condition']}")
    print(f"Age groups         : {c['age_group']}")
    print(f"Gender             : {c['gender']}")
    print(f"Emergency conds    : {c['emergency_conditions']}")
    print(f"Dataset samples    : {d['samples']}  (labels: {d['labels']})")
    print(f"Test w/ unseen phrases: {d['test_unseen_phrase_share']:.0%}")
    print("\nWrote data/processed_conditions.json, data/dataset/*.jsonl, data/preprocessing_report.json")


if __name__ == "__main__":
    main()
