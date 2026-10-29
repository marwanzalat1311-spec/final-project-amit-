"""Step 3: train the TF-IDF + Logistic Regression baseline.

    python train_baseline.py                 # train, evaluate, save to models/ and MongoDB
    python train_baseline.py --tune          # also try several alpha values on validation
    python train_baseline.py --no-mongo      # save only to models/

Model selection uses the VALIDATION set; the TEST set is evaluated once at
the end, so the reported test numbers are not tuned on.
"""
import argparse
import json
import os
import time

import numpy as np

from modeling.baseline import TfidfLogReg
from modeling.data import load_splits
from modeling.evaluation import evaluate, format_metrics
from modeling import registry


def fit_eval(alpha, labels, tr, va):
    t = time.time()
    m = TfidfLogReg(alpha=alpha).fit(tr.text, tr.age, tr.gender, tr.y, labels)
    r = evaluate(m.predict_proba(va.text, va.age, va.gender), va.y, va.emergency, va.unseen)
    return m, r, time.time() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true", help="grid-search alpha on validation")
    p.add_argument("--alpha", type=float, default=3e-5)
    p.add_argument("--name", default="tfidf_logreg_v1")
    p.add_argument("--no-mongo", action="store_true")
    a = p.parse_args()

    labels, S = load_splits()
    tr, va, te = S["train"], S["val"], S["test"]
    print(f"Train {len(tr)} | val {len(va)} | test {len(te)} | {len(labels)} conditions\n")

    experiments = []
    alphas = [1e-5, 3e-5, 1e-4] if a.tune else [a.alpha]
    best = None
    for alpha in alphas:
        print(f"Training alpha={alpha} ...", flush=True)
        m, r, secs = fit_eval(alpha, labels, tr, va)
        print(f"  {secs:.0f}s  val top1 {r['top1']:.3f}  top3 {r['top3']:.3f}  top5 {r['top5']:.3f}"
              f"  emergency@5 {r.get('emergency_recall@5', 0):.3f}")
        experiments.append({"alpha": alpha, "val": r, "seconds": round(secs)})
        # select on top-3 (what the user sees) with emergency recall as tie-breaker
        key = (r["top3"], r.get("emergency_recall@5", 0))
        if best is None or key > best[0]:
            best = (key, alpha, m, r)

    _, alpha, model, val_metrics = best
    # Smaller file, identical predictions (checked: same top-k to 4 decimals).
    model.clf.coef_ = model.clf.coef_.astype(np.float32)
    model.clf.intercept_ = model.clf.intercept_.astype(np.float32)

    T = model.calibrate(va.text, va.age, va.gender, va.y)
    print(f"Calibrated temperature on validation: T = {T:.3f}")

    test_metrics = evaluate(model.predict_proba(te.text, te.age, te.gender), te.y,
                            te.emergency, te.unseen)
    print(f"\nBest alpha = {alpha}")
    print("TEST results:")
    print(format_metrics(test_metrics))

    metrics = {"val": val_metrics, "test": test_metrics}
    doc = registry.save_model(model, a.name, model.model_type, labels, metrics,
                              params=model.params(), to_mongo=not a.no_mongo)
    print(f"\nSaved {doc['local_path']} ({doc['size_mb']} MB)"
          + ("" if a.no_mongo else f" and MongoDB models/{a.name} (GridFS id {doc['gridfs_id']})"))

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", f"{a.name}.json"), "w", encoding="utf-8") as f:
        json.dump({"name": a.name, "type": model.model_type, "params": model.params(),
                   "experiments": experiments, "selected_alpha": alpha, "metrics": metrics},
                  f, indent=2)
    print(f"Experiment log: reports/{a.name}.json")


if __name__ == "__main__":
    main()
