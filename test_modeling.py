"""Tests for Step 3: metrics, baseline model, registry, predictor."""
import numpy as np

from modeling.baseline import TfidfLogReg, apply_temperature, fit_temperature
from modeling.evaluation import evaluate, ranks_of_truth
from modeling.predictor import Predictor, advice_lines
from modeling import registry


def test_ranks_are_pessimistic_on_ties():
    P = np.array([[0.0, 0.0, 0.0, 1.0],     # true label 0 tied with 1 and 2
                  [0.7, 0.2, 0.1, 0.0]])    # true label 0 is first
    assert list(ranks_of_truth(P, np.array([0, 0]))) == [4, 1]


def test_metrics():
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.5, 0.1, 0.4]])
    m = evaluate(P, np.array([0, 2, 2]), emergency_mask=np.array([True, True, False]))
    assert m["top1"] == round(1 / 3, 4) and m["top3"] == 1.0
    assert m["emergency_recall@1"] == 0.5


def test_temperature_keeps_ranking():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(6), size=50)
    for T in (0.3, 2.0):
        assert (apply_temperature(P, T).argmax(1) == P.argmax(1)).all()
    y = P.argmax(1)                       # labels = argmax -> sharpening helps
    assert fit_temperature(P, y) < 1.0


TEXTS = ["chest pain arm sweating", "crushing chest pain left arm", "sore throat fever glands",
         "throat pain fever swollen glands", "itchy red rash spots", "red itchy spots on skin"] * 5
LABELS = ["Heart attack", "Tonsillitis", "Eczema"]
Y = np.array([0, 0, 1, 1, 2, 2] * 5)
INFO = {"Heart attack": {"emergency": True, "gender": "any", "age_group": "any",
                         "warnings": ["Phone 999 if:", "chest pain", "Your child:"],
                         "recommendations": ["rest"]},
        "Tonsillitis": {"gender": "any", "age_group": "any"},
        "Eczema": {"gender": "female", "age_group": "child"}}


def toy_model():
    m = TfidfLogReg(alpha=1e-3)
    m.char.min_df = 1
    return m.fit(TEXTS, [40] * 30, ["male"] * 30, Y, LABELS)


def test_baseline_learns_toy_data():
    m = toy_model()
    P = m.predict_proba(["pain in chest and arm"], [55], ["male"])
    assert LABELS[int(P.argmax())] == "Heart attack"
    assert abs(P.sum() - 1) < 1e-6


def test_registry_local_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "LOCAL_DIR", str(tmp_path))
    m = toy_model()
    registry.save_model(m, "toy", m.model_type, LABELS, {"top1": 1.0}, to_mongo=False)
    m2 = registry.load_model("toy", source="local")
    assert m2.labels == LABELS
    assert np.allclose(m.predict_proba(["sore throat"], [20], ["female"]),
                       m2.predict_proba(["sore throat"], [20], ["female"]))


def test_predictor_ranking_metadata_and_advice():
    pred = Predictor(model=toy_model(), model_name="toy", info=INFO)
    res = pred.predict("crushing chest pain down my arm", age=60, gender="male", top_k=3)
    top = res["predictions"][0]
    assert top["condition"] == "Heart attack" and res["emergency_alert"]
    assert top["warnings"] == ["Phone 999 if:", "chest pain"]          # "Your child:" dropped
    probs = [r["probability"] for r in res["predictions"]]
    assert probs == sorted(probs, reverse=True)
    eczema = next(r for r in res["predictions"] if r["condition"] == "Eczema")
    assert eczema["probability"] == 0.0                                 # female-only, male user


def test_advice_lines():
    assert advice_lines(["You may be advised to:", "rest", "Phone 111 if:"], 5) == ["rest", "Phone 111 if:"]
