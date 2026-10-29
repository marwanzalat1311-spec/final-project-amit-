"""Deep learning models for Step 4: RNN, LSTM, BiLSTM (and GRU).

Architecture (same for every cell type, so the comparison is fair):

    tokens -> Embedding(128) -> dropout
           -> recurrent layer (RNN | LSTM | GRU, optionally bidirectional)
           -> attention pooling + max pooling over time
    age/gender one-hot -> Linear(16)
           -> concat -> dropout -> Linear -> one score per condition -> softmax

Subword option (subword=True, fastText-style): each word vector is
word_embedding + mean of hashed character 3-5-gram embeddings, so a word never
seen in training ("breathlessness") still gets a vector from its pieces
("breath", "less"). This addresses the main weakness of word-level RNNs here:
test samples are built from symptom wordings that were held out of training.

Training: AdamW, cross-entropy with label smoothing, early stopping on
validation top-3 accuracy, then temperature calibration on validation.

The wrapper class `SeqClassifier` has the same interface as the baseline
(fit / predict_proba / calibrate / params / labels), so evaluation,
the GridFS registry and the predictor work unchanged.
"""
import copy
import re
import time
import zlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.baseline import AGE_BUCKETS, GENDERS, apply_temperature, fit_temperature
from preprocessing.text import normalize_text

PAD, UNK = 0, 1
_WORD = re.compile(r"[a-z0-9]+")
CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
ARCH_NAMES = {"rnn": "RNN", "lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU", "bigru": "BiGRU"}


def words(text):
    # Stopwords are kept: word order and small words ("not", "after") matter to an RNN.
    return _WORD.findall(normalize_text(text))


N_BUCKETS = 50000
MAX_NGRAMS = 16


def char_ngrams(word, n_min=3, n_max=5):
    w = f"<{word}>"
    grams = [w[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)]
    # crc32 is stable across runs (Python's hash() is not); bucket 0 = padding
    return [1 + zlib.crc32(g.encode()) % (N_BUCKETS - 1) for g in grams][:MAX_NGRAMS]


def meta_matrix(ages, genders):
    rows = []
    for a, g in zip(ages, genders):
        v = [1.0 if lo <= a <= hi else 0.0 for lo, hi in AGE_BUCKETS]
        v += [1.0 if g == x else 0.0 for x in GENDERS]
        rows.append(v)
    return torch.tensor(rows, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self, vocab_size, n_classes, cell="lstm", bidirectional=True,
                 emb_dim=128, hidden=128, layers=1, dropout=0.3, n_meta=7, subword=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.sub = nn.Embedding(N_BUCKETS, emb_dim, padding_idx=0) if subword else None
        rnn_kwargs = dict(input_size=emb_dim, hidden_size=hidden, num_layers=layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        if cell == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"
        self.rnn = CELLS[cell](**rnn_kwargs)
        out_dim = hidden * (2 if bidirectional else 1)
        self.att = nn.Linear(out_dim, 1)
        self.meta = nn.Linear(n_meta, 16)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(out_dim * 2 + 16, n_classes)

    def embed(self, x, g=None):
        e = self.emb(x)
        if self.sub is not None and g is not None:        # g: (B, T, K) n-gram ids
            gm = (g > 0).unsqueeze(-1).float()
            e = e + (self.sub(g) * gm).sum(2) / gm.sum(2).clamp(min=1.0)
        return e

    def forward(self, x, meta, g=None):
        mask = x != PAD                                   # (B, T)
        if g is not None:                                 # unknown words still count
            mask = mask | (g[:, :, 0] > 0)
        h, _ = self.rnn(self.drop(self.embed(x, g)))      # (B, T, H)
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        att = torch.softmax(scores, dim=1).unsqueeze(-1)  # attention over words
        att_pool = (att * h).sum(1)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        z = torch.cat([att_pool, max_pool, F.relu(self.meta(meta))], dim=1)
        return self.out(self.drop(z))


class SeqClassifier:
    def __init__(self, arch="bilstm", emb_dim=128, hidden=128, layers=1, dropout=0.3,
                 max_len=48, max_vocab=20000, min_freq=1, lr=2e-3, batch_size=128,
                 epochs=25, patience=4, label_smoothing=0.1, subword=False, seed=0):
        self.arch = arch
        self.subword = subword
        self.cfg = dict(emb_dim=emb_dim, hidden=hidden, layers=layers, dropout=dropout)
        self.max_len, self.max_vocab, self.min_freq = max_len, max_vocab, min_freq
        self.lr, self.batch_size, self.epochs, self.patience = lr, batch_size, epochs, patience
        self.label_smoothing, self.seed = label_smoothing, seed
        self.labels, self.vocab, self.net = None, None, None
        self.temperature = 1.0
        self.history = []

    @property
    def model_type(self):
        name = ARCH_NAMES.get(self.arch, self.arch)
        return name + (" + subword" if getattr(self, "subword", False) else "")

    # ------------------------------------------------------------ vocabulary
    def build_vocab(self, texts):
        counts = Counter(w for t in texts for w in words(t))
        keep = [w for w, c in counts.most_common(self.max_vocab - 2) if c >= self.min_freq]
        self.vocab = {w: i + 2 for i, w in enumerate(keep)}

    def encode(self, texts):
        """-> (word ids (N, T), n-gram ids (N, T, K) or None)."""
        X = np.zeros((len(texts), self.max_len), dtype=np.int64)
        G = np.zeros((len(texts), self.max_len, MAX_NGRAMS), dtype=np.int64) if self.subword else None
        cache = {}
        for i, t in enumerate(texts):
            ws = words(t)[: self.max_len]
            X[i, : len(ws)] = [self.vocab.get(w, UNK) for w in ws]
            if G is not None:
                for j, w in enumerate(ws):
                    ids = cache.get(w) or cache.setdefault(w, char_ngrams(w))
                    G[i, j, : len(ids)] = ids
        return torch.from_numpy(X), (torch.from_numpy(G) if G is not None else None)

    def _new_net(self):
        cell = self.arch.replace("bi", "") if self.arch.startswith("bi") else self.arch
        return Net(len(self.vocab) + 2, len(self.labels), cell=cell,
                   bidirectional=self.arch.startswith("bi"),
                   subword=getattr(self, "subword", False), **self.cfg)

    # ------------------------------------------------------------ training
    def fit(self, texts, ages, genders, y, labels, val=None, log=print):
        """val = (texts, ages, genders, y) used for early stopping."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.labels = list(labels)
        self.build_vocab(texts)
        (X, G), M = self.encode(texts), meta_matrix(ages, genders)
        Y = torch.as_tensor(np.asarray(y))
        self.net = self._new_net()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        best, best_state, bad = -1.0, None, 0
        n = len(Y)
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                b = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[b], M[b], G[b] if G is not None else None), Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            row = {"epoch": epoch, "train_loss": round(total / n, 4), "seconds": round(time.time() - t0, 1)}
            if val is not None:
                P = self.predict_proba(*val[:3])
                ranks = 1 + (P > P[np.arange(len(val[3])), val[3]][:, None]).sum(1)
                row["val_top1"] = round(float((ranks <= 1).mean()), 4)
                row["val_top3"] = round(float((ranks <= 3).mean()), 4)
                sched.step(row["val_top3"])
                if row["val_top3"] > best:
                    best, best_state, bad = row["val_top3"], copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
            self.history.append(row)
            if log:
                log("  " + "  ".join(f"{k} {v}" for k, v in row.items()))
            if val is not None and bad >= self.patience:
                if log:
                    log(f"  early stop (no val top-3 gain for {self.patience} epochs)")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_proba(self, texts, ages, genders, batch=512):
        self.net.eval()
        (X, G), M = self.encode(texts), meta_matrix(ages, genders)
        out = []
        for i in range(0, len(X), batch):
            g = G[i: i + batch] if G is not None else None
            out.append(torch.softmax(self.net(X[i: i + batch], M[i: i + batch], g), dim=1).numpy())
        P = np.concatenate(out) if out else np.zeros((0, len(self.labels)))
        return apply_temperature(P, self.temperature)

    def calibrate(self, texts, ages, genders, y):
        self.temperature = 1.0
        self.temperature = fit_temperature(self.predict_proba(texts, ages, genders), np.asarray(y))
        return self.temperature

    def params(self):
        return {"arch": self.arch, "subword": getattr(self, "subword", False), **self.cfg, "max_len": self.max_len, "vocab_size": len(self.vocab),
                "lr": self.lr, "batch_size": self.batch_size, "label_smoothing": self.label_smoothing,
                "epochs_run": len(self.history), "temperature": self.temperature,
                "parameters": sum(p.numel() for p in self.net.parameters())}

    # ------------------------------------------------------------ pickling
    # Store weights as a plain state_dict so the saved file does not depend on
    # pickling torch module objects.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["net"] = None
        state["_weights"] = {k: v.cpu() for k, v in self.net.state_dict().items()} if self.net else None
        return state

    def __setstate__(self, state):
        weights = state.pop("_weights", None)
        self.__dict__.update(state)
        if weights is not None:
            self.net = self._new_net()
            self.net.load_state_dict(weights)
            self.net.eval()
