"""Offline tests for the parser, using HTML shaped like an NHS Inform page."""
from scraper.nhs_scraper import parse_az_links, parse_condition_page

AZ_HTML = """
<html><body><main>
  <a href="/illnesses-and-conditions/a-to-z/">A to Z</a>
  <a href="/illnesses-and-conditions/skin-hair-and-nails/acne">Acne</a>
  <a href="/illnesses-and-conditions/skin-hair-and-nails/acne/">Acne again</a>
  <a href="https://www.nhsinform.scot/illnesses-and-conditions/heart-and-blood-vessels/conditions/angina/">Angina</a>
  <a href="/illnesses-and-conditions/">Category root</a>
  <a href="https://example.com/illnesses-and-conditions/x/y/">External</a>
</main></body></html>
"""

CONDITION_HTML = """
<html><body>
<nav><li>Home</li><li>Illnesses</li></nav>
<main>
  <h1>Acne</h1>
  <p>Acne is a common skin condition that affects most people at some point.</p>
  <h2>Symptoms of acne</h2>
  <ul><li>whiteheads</li><li>blackheads</li><li>painful red lumps</li></ul>
  <h2>What causes acne?</h2>
  <p>Acne is linked to changes in hormone levels during puberty.</p>
  <h2>Treating acne</h2>
  <ul><li>do not wash affected areas more than twice a day</li></ul>
  <div class="callout callout--attention">
    <h3>Speak to your GP if:</h3>
    <ul><li>you have moderate or severe acne</li></ul>
  </div>
  <div class="panel panel--emergency">
    <p>Phone 999 if you have swelling of the face and difficulty breathing.</p>
  </div>
</main>
<footer><p>Copyright NHS 24</p></footer>
</body></html>
"""


def test_az_links_filtered_and_deduplicated():
    links = parse_az_links(AZ_HTML)
    assert links == [
        "https://www.nhsinform.scot/illnesses-and-conditions/skin-hair-and-nails/acne/",
        "https://www.nhsinform.scot/illnesses-and-conditions/heart-and-blood-vessels/conditions/angina/",
    ]


def test_condition_sections():
    doc = parse_condition_page(
        CONDITION_HTML,
        "https://www.nhsinform.scot/illnesses-and-conditions/skin-hair-and-nails/acne/",
    )
    assert doc["condition"] == "Acne"
    assert doc["category"] == "skin-hair-and-nails"
    assert doc["symptoms"] == ["whiteheads", "blackheads", "painful red lumps"]
    assert doc["causes"] == ["Acne is linked to changes in hormone levels during puberty."]
    assert doc["recommendations"] == ["do not wash affected areas more than twice a day"]
    assert "you have moderate or severe acne" in doc["warnings"]
    assert any(w.startswith("Phone 999") for w in doc["warnings"])
    assert "Home" not in str(doc) and "Copyright" not in str(doc)


NESTED_HTML = """
<html><body><main>
  <h1>About acute cholecystitis</h1>
  <h2>Symptoms of acne</h2>
  <p>The main symptom is spots.</p>
  <h3>Types of spots</h3>
  <ul><li>whiteheads</li><li>papules</li></ul>
  <h3>Complications</h3>
  <p>Scarring can happen.</p>
  <h2>Help us improve</h2>
  <p>Was this page useful?</p>
</main></body></html>
"""


def test_subheadings_inherit_parent_section_and_name_cleaned():
    trace = []
    doc = parse_condition_page(NESTED_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/a/b/", trace)
    assert doc["condition"] == "Acute cholecystitis"
    assert doc["symptoms"] == ["The main symptom is spots.", "whiteheads", "papules"]
    assert "Scarring can happen." not in doc["symptoms"]
    assert trace[0][0] == "H2"


# Mirrors the real NHS Inform acute cholecystitis page structure.
INTRO_HTML = """
<html><body><main>
  <h1>Acute cholecystitis</h1>
  <p>Acute cholecystitis is swelling (inflammation) of the gallbladder.</p>
  <p>The main symptom of acute cholecystitis is a sudden sharp pain in the tummy.</p>
  <p>Unlike some others types of <a href="#">abdominal pain</a> , the pain is persistent.</p>
  <p>Some people may additional symptoms, such as:</p>
  <ul><li>a high temperature (fever)</li><li>nausea and vomiting</li></ul>
  <h2>When to seek medical advice</h2>
  <h3>Contact your GP practice urgently if:</h3>
  <ul><li>you develop sudden and severe abdominal pain</li></ul>
  <h2>Possible complications</h2>
  <p>Without treatment it can lead to gangrene.</p>
  <h2>Preventing acute cholecystitis</h2>
  <p>Eat a healthy, balanced diet.</p>
  <p>Read more about preventing gallstones .</p>
  <p>Source: NHS 24</p>
  <p>Last updated: 28 September 2026</p>
  <h3>Also on NHS inform</h3>
  <ul><li>Gallstones</li></ul>
  <h3>Other health sites</h3>
  <ul><li>NICE: cholecystectomy</li></ul>
</main></body></html>
"""


def test_intro_symptoms_and_footer_removed():
    doc = parse_condition_page(INTRO_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/x/acute-cholecystitis/")
    assert doc["symptoms"][0].startswith("The main symptom")
    assert "a high temperature (fever)" in doc["symptoms"]
    assert "nausea and vomiting" in doc["symptoms"]
    assert any("abdominal pain, the pain" in s for s in doc["symptoms"])   # spacing fixed
    assert doc["overview"] == ["Acute cholecystitis is swelling (inflammation) of the gallbladder.",
                               "Without treatment it can lead to gangrene."]
    assert doc["recommendations"] == ["Eat a healthy, balanced diet."]
    assert doc["warnings"] == ["When to seek medical advice", "Contact your GP practice urgently if:",
                               "you develop sudden and severe abdominal pain"]


# Whole article wrapped in one big "panel" (real Heart attack page layout).
WRAPPED_HTML = """
<html><body><main><div class="panel-wrapper">
  <h1>Heart attack</h1>
  <div class="callout callout--emergency"><p>Phone 999 if you or someone else has symptoms like:</p>
    <ul><li>central chest pain</li><li>unconsciousness</li></ul></div>
  <h2>Diagnosing a heart attack</h2>
  <ul><li>an electrocardiogram (ECG)</li></ul>
  <h2>Treating a heart attack</h2>
  <ul><li>oxygen therapy</li></ul>
  """ + "<p>" + "filler text " * 200 + "</p>" + """
</div></main></body></html>
"""


def test_big_wrapper_is_not_a_warning_box():
    doc = parse_condition_page(WRAPPED_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/h/heart-attack/")
    assert "central chest pain" in doc["warnings"]
    assert "an electrocardiogram (ECG)" not in doc["warnings"]
    assert "oxygen therapy" in doc["recommendations"]


# Self-help guide layout (Flu, UTI, Sore throat...): a JS widget with a hidden
# "Last Updated:" line sits above the real article.
SHG_HTML = """
<html><body>
<h1>Flu</h1>
<div class="selfcontainer"><div id="selfHelpGuideMain">
  <div class="self-help-guide"><p><a href="#">Return to Symptoms</a></p>
  <h2></h2><h4></h4><p>Last Updated:</p></div>
</div></div>
<p>Flu (influenza) is a common infectious viral illness.</p>
<h2>Flu symptoms</h2>
<ul><li>a sudden high temperature</li><li>sneezing</li></ul>
<h3>Phone 999 or go to A&amp;E if:</h3>
<ul><li>you have severe difficulty breathing</li></ul>
<p>Source: NHS 24</p>
</body></html>
"""


def test_self_help_guide_widget_ignored():
    doc = parse_condition_page(SHG_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/infections-and-poisoning/flu/")
    assert doc["symptoms"] == ["a sudden high temperature", "sneezing"]
    assert doc["warnings"] == ["Phone 999 or go to A&E if:", "you have severe difficulty breathing"]
    assert "Return to Symptoms" not in str(doc)
