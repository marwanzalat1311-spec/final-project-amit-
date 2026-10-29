"""Step 4: train and compare deep learning models (RNN / LSTM / BiLSTM / GRU).

    python train_neural.py                       # RNN, LSTM, BiLSTM; keep the best
    python train_neural.py --arch bilstm         # only one architecture
    python train_neural.py --arch bilstm --subword   # best Step 4 model
    python train_neural.py --epochs 15 --no-mongo

Every architecture is trained with the same settings and compared on the
VALIDATION set. The best one is calibrated, evaluated ONCE on TEST, and saved
to models/ and MongoDB (GridFS) as e.g. "bilstm_v1".
"""
import argparse
import json
import os
import time

from modeling import registry
from modeling.data import load_splits
from modeling.evaluation import evaluate, format_metrics
from modeling.neural import SeqClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", nargs="+", default=["rnn", "lstm", "bilstm"],
                   choices=["rnn", "lstm", "bilstm", "gru", "bigru"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--subword", action="store_true", help="add fastText-style char n-gram embeddings")
    p.add_argument("--no-mongo", action="store_true")
    p.add_argument("--version", default="v1")
    a = p.parse_args()

    labels, S = load_splits()
    tr, va, te = S["train"], S["val"], S["test"]
    val = (va.text, va.age, va.gender, va.y)
    print(f"Train {len(tr)} | val {len(va)} | test {len(te)} | {len(labels)} conditions")

    results, best = [], None
    for arch in a.arch:
        print(f"\n=== {arch.upper()} ===")
        t = time.time()
        m = SeqClassifier(arch=arch, epochs=a.epochs, hidden=a.hidden, subword=a.subword)
        m.fit(tr.text, tr.age, tr.gender, tr.y, labels, val=val)
        r = evaluate(m.predict_proba(va.text, va.age, va.gender), va.y, va.emergency, va.unseen)
        secs = round(time.time() - t)
        print(f"  -> val top1 {r['top1']:.3f}  top3 {r['top3']:.3f}  top5 {r['top5']:.3f}  "
              f"emergency@5 {r.get('emergency_recall@5', 0):.3f}  ({secs}s, {len(m.history)} epochs)")
        results.append({"arch": arch + ("+subword" if a.subword else ""), "val": r, "seconds": secs, "history": m.history,
                        "params": m.params()})
        key = (r["top3"], r.get("emergency_recall@5", 0))
        if best is None or key > best[0]:
            best = (key, arch, m, r)

    print("\nValidation comparison:")
    print(f"  {'arch':<8} {'top1':>6} {'top3':>6} {'top5':>6} {'unseen5':>8} {'emerg@5':>8} {'time':>6}")
    for x in results:
        v = x["val"]
        print(f"  {x['arch']:<8} {v['top1']:6.3f} {v['top3']:6.3f} {v['top5']:6.3f} "
              f"{v.get('unseen_top5', 0):8.3f} {v.get('emergency_recall@5', 0):8.3f} {x['seconds']:5d}s")

    _, arch, model, val_metrics = best
    T = model.calibrate(va.text, va.age, va.gender, va.y)
    test = evaluate(model.predict_proba(te.text, te.age, te.gender), te.y, te.emergency, te.unseen)
    name = f"{arch}{'_subword' if a.subword else ''}_{a.version}"
    print(f"\nBest: {arch.upper()} (temperature {T:.3f}) -> TEST results:")
    print(format_metrics(test))

    doc = registry.save_model(model, name, model.model_type, labels,
                              {"val": val_metrics, "test": test}, params=model.params(),
                              to_mongo=not a.no_mongo)
    print(f"\nSaved {doc['local_path']} ({doc['size_mb']} MB)"
          + ("" if a.no_mongo else f" and MongoDB models/{name}"))

    os.makedirs("reports", exist_ok=True)
    log_name = f"neural_{'subword_' if a.subword else ''}{a.version}.json"
    with open(os.path.join("reports", log_name), "w", encoding="utf-8") as f:
        json.dump({"selected": name, "experiments": results, "test": test}, f, indent=2)
    print(f"Experiment log: reports/{log_name}")
    print(f"Try it: python predict.py \"...\" --model {name}")


if __name__ == "__main__":
    main()
