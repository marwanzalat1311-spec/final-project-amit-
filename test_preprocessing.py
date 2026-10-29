"""Tests for Step 2 preprocessing, based on real NHS Inform lines."""
from preprocessing.cleaner import process_condition, to_phrase, infer_gender, infer_age_group
from preprocessing.dataset import build_dataset
from preprocessing.text import normalize_text, tokenize


def test_phrase_extraction():
    assert to_phrase("you’re wheezing") == "wheezing"
    assert to_phrase("has severe difficulty breathing") == "severe difficulty breathing"
    assert to_phrase("if you also feel feverish or unwell") == "feverish or unwell"
    assert to_phrase("itchy skin, or") == "itchy skin"
    # not symptoms
    assert to_phrase("Phone 111 if your GP practice is closed.") is None
    assert to_phrase("People may experience:") is None
    assert to_phrase("Chest pain and coughing up blood are not common symptoms of COPD.") is None
    assert to_phrase("Return to Symptoms") is None
    assert to_phrase("Symptoms of sepsis") is None


def test_normalize_maps_lay_terms():
    assert normalize_text("My tummy ache and throwing up") == "my abdominal pain and vomiting"
    assert "abdominal" in tokenize("tummy pain")
    assert "the" not in tokenize("the pain")


def test_metadata():
    assert infer_gender("Bacterial vaginosis", None) == "female"
    assert infer_gender("Breast cancer in men", None) == "male"
    assert infer_gender("Menopause", None) == "female"
    assert infer_gender("Asthma", None) == "any"
    assert infer_age_group("Fever in children", "x") == "child"
    assert infer_age_group("Fever in adults", "x") == "adult"


RAW = {
    "condition": "About heart attack",
    "url": "https://www.nhsinform.scot/illnesses-and-conditions/heart/heart-attack/",
    "category": "heart",
    "symptoms": [],
    "warnings": ["British Sign Language (BSL)", "You or someone else has symptoms like:",
                 "central chest pain", "unconsciousness", "rapid heart beat",
                 "Phone 999 immediately."],
    "causes": ["smoking"], "recommendations": ["take aspirin"],
}


def test_symptoms_recovered_from_warnings_and_emergency_flag():
    rec, reason = process_condition(RAW)
    assert reason is None
    assert rec["symptom_phrases"] == ["central chest pain", "unconsciousness", "rapid heart beat"]
    assert rec["emergency"] is True          # heart attack is an emergency condition
    assert rec["urgency"] == "emergency"     # page says phone 999
    assert "British Sign Language (BSL)" not in rec["warnings"]


def test_non_condition_pages_excluded():
    rec, reason = process_condition(dict(RAW, condition="Living with chronic pain"))
    assert reason == "not a condition page"


def test_dataset_holdout_has_no_leakage():
    rec, _ = process_condition(RAW)
    rec["symptom_phrases"] = ["fever", "rash", "headache", "stiff neck", "vomiting",
                              "light sensitivity", "cold hands", "confusion"]
    ds = build_dataset([rec], n_train=50, n_val=10, n_test=10, seed=1)
    train_text = " ".join(s["text"] for s in ds["train"])
    test_unseen = [s for s in ds["test"] if not s["seen_phrases"]]
    assert len(test_unseen) == 10
    # every held-out phrase in test never appears in train
    held = [p for p in rec["symptom_phrases"] if p not in train_text]
    assert held, "some phrases must be held out"
    assert all(any(h in s["text"] for h in held) for s in test_unseen)


def test_emergency_is_about_the_condition_not_red_flags():
    rec, _ = process_condition(dict(RAW, condition="Flu"))
    assert rec["urgency"] == "emergency"     # page lists 999 red flags...
    assert rec["emergency"] is False         # ...but flu itself is not an emergency


def test_self_help_guide_prompts_removed():
    from preprocessing.cleaner import is_junk
    assert is_junk("Complete this guide to assess your symptoms and find out more about where to get help.")
    assert not is_junk("drink plenty of fluids")
