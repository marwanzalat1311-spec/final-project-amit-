"""Baseline: TF-IDF + Logistic Regression.

Features
  - word TF-IDF, 1-2 grams          ("chest pain", "sore throat")
  - character TF-IDF, 3-5 grams     robust to typos/variants ("diarhoea", "vomitting")
  - metadata: age bucket + gender   one-hot, so the model can use them

Classifier
  logistic regression (log-loss), one-vs-rest, trained with SGD so all CPU cores
  are used: ~70s instead of ~200s for the lbfgs solver at the same accuracy
  (see the experiment table in the README). Outputs one probability per
  condition, normalised to sum to 1 -> directly gives the ranked list.
"""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from preprocessing.text import normalize_text

AGE_BUCKETS = [(0, 4), (5, 15), (16, 39), (40, 64), (65, 200)]
GENDERS = ["female", "male"]


def meta_features(ages, genders, weight=1.0):
    rows = []
    for a, g in zip(ages, genders):
        v = [1.0 if lo <= a <= hi else 0.0 for lo, hi in AGE_BUCKETS]
        v += [1.0 if g == x else 0.0 for x in GENDERS]
        rows.append(v)
    return sparse.csr_matrix(np.array(rows) * weight)


def apply_temperature(P, T):
    """Sharpen (T<1) or soften (T>1) probabilities; argmax/ranking unchanged."""
    if T == 1.0:
        return P
    L = np.log(np.clip(P, 1e-12, None)) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def fit_temperature(P, y, grid=np.geomspace(0.05, 2.0, 40)):
    """Temperature minimising negative log-likelihood of the true labels."""
    idx = np.arange(len(y))
    nll = [-np.log(np.clip(apply_temperature(P, T)[idx, y], 1e-12, None)).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


class TfidfLogReg:
    model_type = "TF-IDF + LogisticRegression"

    def __init__(self, alpha=3e-5, solver="sgd", word_max=20000, char_max=30000,
                 meta_weight=1.0, epochs=30, C=5.0, seed=0):
        self.alpha, self.solver, self.C = alpha, solver, C
        self.meta_weight = meta_weight
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                    max_features=word_max)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                    min_df=2, max_features=char_max)
        if solver == "sgd":
            self.clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=epochs, tol=None,
                                     n_jobs=-1, random_state=seed)
        else:   # "lbfgs": exact multinomial logistic regression (slower)
            self.clf = LogisticRegression(C=C, max_iter=100, tol=1e-3)
        self.labels = None
        self.temperature = 1.0

    def _features(self, texts, ages, genders, fit=False):
        if fit:
            w = self.word.fit_transform(texts)
            c = self.char.fit_transform(texts)
        else:
            w = self.word.transform(texts)
            c = self.char.transform(texts)
        return sparse.hstack([w, c, meta_features(ages, genders, self.meta_weight)]).tocsr()

    def fit(self, texts, ages, genders, y, labels):
        self.labels = list(labels)
        X = self._features(texts, ages, genders, fit=True)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts, ages, genders):
        """Probabilities aligned with self.labels (columns for unseen labels = 0)."""
        X = self._features(texts, ages, genders)
        P = self.clf.predict_proba(X)
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        P = apply_temperature(P, getattr(self, "temperature", 1.0))
        full = np.zeros((P.shape[0], len(self.labels)))
        full[:, self.clf.classes_] = P
        return full

    def calibrate(self, texts, ages, genders, y):
        """Fit the temperature on validation data (does not change the ranking)."""
        self.temperature = 1.0
        P = self.predict_proba(texts, ages, genders)
        self.temperature = fit_temperature(P, y)
        return self.temperature

    def params(self):
        return {"solver": self.solver, "alpha": self.alpha, "C": self.C,
                "temperature": getattr(self, "temperature", 1.0),
                "meta_weight": self.meta_weight,
                "word_features": len(self.word.vocabulary_), "char_features": len(self.char.vocabulary_)}

    def predict_text(self, text, age, gender):
        """Single raw user text -> probability vector."""
        return self.predict_proba([normalize_text(text)], [age], [gender])[0]
