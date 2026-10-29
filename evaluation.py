"""Evaluation metrics shared by every model (baseline, LSTM, transformer).

All models produce a probability matrix P of shape (n_samples, n_labels) whose
columns follow the same `labels` list, so they are compared on equal terms.

Metrics
-------
top1/top3/top5     the true condition is among the k highest-ranked predictions
macro_f1           F1 of the top-1 prediction averaged over conditions (rare ones count equally)
mrr                mean reciprocal rank of the true condition (1 = always first)
nll                negative log-likelihood of the true condition (lower = better-calibrated)
mean_top1_prob     average probability given to the top prediction; compare with top1
                   accuracy - if they are close, the shown percentages are honest
emergency_recall@k for samples whose true condition is an emergency (heart attack,
                   sepsis, ...): share where it appears in the top k. Missing these is
                   the costly error, so this is reported separately.
unseen_*           the same metrics on test samples built from symptom phrases the
                   model never saw in training (the honest generalisation number).
"""
import numpy as np
from sklearn.metrics import f1_score


def ranks_of_truth(P, y):
    """1-based rank of the true label in each row of P.

    Ties are counted pessimistically: if the true condition has the same score
    as others, it is ranked after all of them. Otherwise a model that outputs
    many identical probabilities (e.g. all zeros) would get a perfect score.
    """
    true_scores = P[np.arange(len(y)), y]
    higher = (P > true_scores[:, None]).sum(axis=1)
    ties = (P == true_scores[:, None]).sum(axis=1) - 1
    return 1 + higher + ties


def topk_accuracy(ranks, k):
    return float((ranks <= k).mean()) if len(ranks) else 0.0


def evaluate(P, y, emergency_mask=None, unseen_mask=None, ks=(1, 3, 5)):
    """Return a dict of metrics.

    P              (n, L) probabilities
    y              (n,) integer true labels
    emergency_mask (n,) bool: true label is an emergency condition
    unseen_mask    (n,) bool: sample uses held-out phrases
    """
    y = np.asarray(y)
    ranks = ranks_of_truth(P, y)
    pred = P.argmax(axis=1)
    out = {f"top{k}": round(topk_accuracy(ranks, k), 4) for k in ks}
    out["macro_f1"] = round(float(f1_score(y, pred, average="macro", zero_division=0)), 4)
    out["mrr"] = round(float((1.0 / ranks).mean()), 4)
    # Calibration: are the probabilities trustworthy?
    out["nll"] = round(float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()), 4)
    out["mean_top1_prob"] = round(float(P.max(axis=1).mean()), 4)
    out["n"] = int(len(y))

    if emergency_mask is not None and np.any(emergency_mask):
        em = np.asarray(emergency_mask)
        for k in (1, 3, 5):
            out[f"emergency_recall@{k}"] = round(topk_accuracy(ranks[em], k), 4)
        out["n_emergency"] = int(em.sum())

    if unseen_mask is not None and np.any(unseen_mask):
        um = np.asarray(unseen_mask)
        for k in ks:
            out[f"unseen_top{k}"] = round(topk_accuracy(ranks[um], k), 4)
        out["n_unseen"] = int(um.sum())
    return out


def format_metrics(m):
    keys = ["top1", "top3", "top5", "macro_f1", "mrr", "nll", "mean_top1_prob",
            "emergency_recall@1", "emergency_recall@3", "emergency_recall@5",
            "unseen_top1", "unseen_top3", "unseen_top5"]
    return "\n".join(f"  {k:<20} {m[k]:.3f}" for k in keys if k in m)
