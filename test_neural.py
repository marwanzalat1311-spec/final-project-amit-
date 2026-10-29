"""Step 4: the sequence models learn, save/load, and plug into the predictor."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from modeling.neural import SeqClassifier, words
from modeling.predictor import Predictor
from modeling import registry
from tests.test_modeling import TEXTS, LABELS, Y, INFO


@pytest.mark.parametrize("arch", ["rnn", "lstm", "bilstm"])
def test_sequence_models_learn_toy_data(arch):
    m = SeqClassifier(arch=arch, epochs=40, hidden=32, emb_dim=32, lr=1e-2, dropout=0.0,
                      label_smoothing=0.0, batch_size=8)
    m.fit(TEXTS, [40] * 30, ["male"] * 30, Y, LABELS, log=None)
    P = m.predict_proba(["crushing chest pain left arm"], [50], ["male"])
    assert LABELS[int(P.argmax())] == "Heart attack"
    assert abs(P.sum() - 1) < 1e-5
    assert m.model_type in ("RNN", "LSTM", "BiLSTM")


def test_save_load_and_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "LOCAL_DIR", str(tmp_path))
    m = SeqClassifier(arch="lstm", epochs=30, hidden=32, emb_dim=32, lr=1e-2, batch_size=8)
    m.fit(TEXTS, [40] * 30, ["male"] * 30, Y, LABELS, log=None)
    registry.save_model(m, "toy_lstm", m.model_type, LABELS, {}, params=m.params(), to_mongo=False)
    m2 = registry.load_model("toy_lstm", source="local")
    x = (["sore throat and fever"], [20], ["female"])
    assert np.allclose(m.predict_proba(*x), m2.predict_proba(*x), atol=1e-6)
    res = Predictor(model=m2, model_name="toy_lstm", info=INFO).predict("sore throat fever", 20)
    assert res["predictions"][0]["condition"] == "Tonsillitis"


def test_words_keep_order_and_small_words():
    assert words("Not able to swallow after eating") == ["not", "able", "to", "swallow", "after", "eating"]


def test_subword_helps_unseen_words():
    """'rashes'/'itchiness' never appear in training; their character n-grams
    overlap 'rash'/'itchy', so the subword model should lean more to Eczema."""
    def eczema_prob(subword):
        probs = []
        for seed in range(3):
            m = SeqClassifier(arch="lstm", epochs=40, hidden=32, emb_dim=32, lr=1e-2, dropout=0.0,
                              label_smoothing=0.0, batch_size=8, subword=subword, seed=seed)
            m.fit(TEXTS, [40] * 30, ["male"] * 30, Y, LABELS, log=None)
            probs.append(m.predict_proba(["rashes and itchiness"], [40], ["male"])[0][2])
        return float(np.mean(probs)), m
    p_sub, m = eczema_prob(True)
    p_word, _ = eczema_prob(False)
    assert p_sub > p_word
    assert "subword" in m.model_type
