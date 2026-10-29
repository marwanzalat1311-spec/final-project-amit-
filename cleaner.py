"""Clean raw scraped condition records into a structured, model-ready form.

Raw record (from the scraper)          Processed record
--------------------------------       -------------------------------------------
symptoms / warnings / causes / ...  -> symptom_phrases   short symptom phrases
                                       symptom_sentences longer symptom descriptions
                                       symptom_text      everything joined (for TF-IDF)
                                       causes / warnings / recommendations (cleaned)
                                       age_group, gender (metadata), emergency flag
"""
import math
import re

from .text import ascii_fold, normalize_text, tokenize

# ---------------------------------------------------------------- line filters

# Boilerplate seen on many pages (navigation, support lines, link text).
JUNK_PATTERNS = [
    r"^british sign language", r"^return to ", r"^back to ", r"^more about ",
    r"^read more", r"^find out more", r"^see also", r"^watch ", r"^video",
    r"macmillan", r"0808 \d", r"\bwebsite\b", r"\bleaflet\b", r"\bdownload\b",
    r"care of burns in scotland", r"^(further )?information (on|about|for)\b", r"\bself-help guide\b",
    r"cookies", r"^this video", r"\b(complete|use|try) (this|our|the) (self-help )?guide\b",
    r"\bassess your symptoms\b", r"^last updated", r"^source:", r"^print",
    r"^(about|symptoms of|causes of|treating|treatment of|preventing|diagnosing|"
    r"complications of|living with|recovering from) [\w' ()-]{0,60}$",   # in-page tab labels
]
_JUNK = [re.compile(p) for p in JUNK_PATTERNS]

# Lines that tell you what to DO rather than describe a symptom.
ACTION = re.compile(
    r"\b(phone|call|dial|999|111|gp|pharmacist|pharmacy|dentist|a&e|a and e|nhs|"
    r"doctor|midwife|health visitor|nurse|speak to|contact|go to|seek|appointment|"
    r"emergency department|hospital|ambulance|trust your instincts|advice|treatment)\b")

# Leading words that wrap a symptom: "you have a cough" -> "cough".
# Applied repeatedly, so "you or your child are feeling very dizzy" -> "dizzy".
LEAD = re.compile(
    r"^(?:if|when|while|you or your child|you or someone else|your child|your baby|the person|someone|"
    r"you|they|he|she|your|"
    r"have|has|had|are|is|were|was|get|gets|got|feel|feels|feeling|notice|notices|"
    r"develop|develops|experience|experiences|experiencing|start|starts|keep|keeps|"
    r"become|becomes|having|getting|suddenly|also|still|'ve|'re|'s|"
    r"a|an|the|any|some|new|or|and|but|with)\s+", re.I)
CONTRACTION = re.compile(r"^(you|they|he|she|it)'(re|ve|s|ll|d)\s+", re.I)

# Sentences ABOUT symptoms/staging/timing rather than a symptom itself.
META = re.compile(r"\b(symptoms? (of|usually|can|may|might|will|include|are|is|often|tend)|"
                  r"stages? of|types? of|days after|weeks after|months after|"
                  r"is diagnosed|diagnos\w*|can be caused|is caused|more likely|"
                  r"causes and risk|risk factors|ways of managing|managing \w+ symptoms)\b", re.I)

# Statements about what is NOT a symptom.
NEGATIVE = re.compile(r"\b(not (a |an )?(common |usual )?symptoms?|isn't a symptom|"
                      r"doesn't cause|does not cause|rarely causes?)\b", re.I)


def strip_lead(t):
    for _ in range(6):
        new = CONTRACTION.sub("", t)
        new = LEAD.sub("", new)
        if new == t:
            break
        t = new
    return t

MAX_PHRASE_WORDS = 12
MAX_SENTENCE_WORDS = 45


def is_junk(line):
    low = ascii_fold(line).lower().strip()
    return any(rx.search(low) for rx in _JUNK)


def clean_line(line):
    """Tidy a raw line without changing meaning (used for warnings/causes/recs)."""
    line = ascii_fold(line)
    line = re.sub(r"\s+", " ", line).strip(" ;,")
    return line


def to_phrase(line):
    """Turn a raw bullet into a short symptom phrase, or None if it is not one."""
    if is_junk(line):
        return None
    t = clean_line(line)
    if t.endswith(":"):                      # "People may experience:" lead-ins
        return None
    if ACTION.search(t.lower()) or NEGATIVE.search(t) or META.search(t):
        return None
    t = re.split(r"\s+-\s+|;|\. ", t)[0]     # keep the core before a dash/explanation
    t = t.lower().rstrip(". ,")
    t = re.sub(r"\bfeel(ing)? sick\b", "nausea", t)
    t = re.sub(r"\b(being|been) sick\b", "vomiting", t)
    t = strip_lead(t).strip()
    t = re.sub(r"\s*,?\s*(or|and)$", "", t)  # list endings: "... , or"
    words = t.split()
    if not (1 <= len(words) <= MAX_PHRASE_WORDS) or len(t) < 3:
        return None
    return t


def to_sentence(line):
    """A longer descriptive symptom sentence (from the symptoms section)."""
    if is_junk(line):
        return None
    t = clean_line(line)
    if t.endswith(":") or ACTION.search(t.lower()):
        return None
    n = len(t.split())
    if n <= MAX_PHRASE_WORDS or n > MAX_SENTENCE_WORDS:
        return None
    return t


def dedupe(items):
    seen, out = set(), []
    for it in items:
        key = normalize_text(it, synonyms=False)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


# ---------------------------------------------------------------- page filter

# Pages that are guides/support pages rather than a diagnosable condition.
NON_CONDITION = re.compile(
    r"^(living with|talking to|recovering from|rare conditions$|how to|types of|treatment for|"
    r"sunbed|trips and falls|soft tissue injury advice|complications of|managing |"
    r"information and support|support for|caring for)|"
    r"\band cancer$|^cancer-related|^body changes and cancer|^brain stem death$", re.I)

# ---------------------------------------------------------------- metadata

CHILD = re.compile(r"\b(children|child|babies|baby|infants?|toddlers?|newborns?|young people|"
                   r"childhood|bottom shuffling|teething|nappy|febrile seizures|croup|"
                   r"hand, foot and mouth)\b", re.I)
ADULT = re.compile(r"\b(adults?|in older people)\b", re.I)
FEMALE = re.compile(r"\b(women|woman|female|vagin\w*|vulva\w*|pelvic inflammatory|bartholin|ovar\w*|cervical cancer|cervix|"
                    r"uter\w*|womb|pregnan\w*|miscarriage|stillbirth|menopause|periods?|"
                    r"endometriosis|polycystic|ectopic|fibroids|thrush in women|pelvic organ prolapse|"
                    r"breastfeeding|mastitis|molar)\b", re.I)
MALE = re.compile(r"\b(men|male|testic\w*|prostat\w*|penile|penis|erectile|scrot\w*)\b", re.I)


def infer_age_group(name, url):
    text = f"{name} {url.replace('-', ' ')}"
    if CHILD.search(name) or "children-and-young-peoples" in url:
        return "child"
    if ADULT.search(text):
        return "adult"
    return "any"


def infer_gender(name, category):
    if category == "pregnancy-and-childbirth" or FEMALE.search(name):
        return "female"
    if MALE.search(name):
        return "male"
    return "any"


# Conditions that are medical emergencies in themselves (used for the
# "recall on emergency conditions" evaluation metric). Curated from the NHS
# Inform A-Z: conditions whose page tells you to phone 999 for the condition
# itself, not just for rare red-flag complications.
EMERGENCY_CONDITIONS = re.compile(
    r"\b(heart attack|cardiac arrest|stroke|transient ischaemic attack|sepsis|septic shock|"
    r"meningitis|anaphylaxis|pulmonary embolism|diabetic ketoacidosis|hypoglycaemia|"
    r"severe head injury|subarachnoid|brain haemorrhage|aortic aneurysm|aortic dissection|"
    r"ectopic pregnancy|appendicitis|acute pancreatitis|acute cholecystitis|"
    r"acute kidney injury|heatstroke|hypothermia|carbon monoxide poisoning|"
    r"testicular torsion|cauda equina|encephalitis|epiglottitis|"
    r"burns and scalds|acid and chemical burns|chest pain|coronary heart disease|"
    r"spinal injury|major trauma|asthma attack|croup|bronchiolitis|"
    r"kawasaki|toxic shock|necrotising fasciitis|pims)\b", re.I)


def urgency_level(warnings):
    """Highest advice level on the page: emergency > urgent > routine."""
    text = " ".join(warnings).lower()
    if re.search(r"\b999\b|\ba&e\b|emergency department", text):
        return "emergency"
    if re.search(r"\burgent|\b111\b|same day|immediately", text):
        return "urgent"
    return "routine"


def is_emergency(name):
    return bool(EMERGENCY_CONDITIONS.search(name))


# ---------------------------------------------------------------- main entry

def process_condition(raw):
    """Return (processed_record, reason_if_excluded)."""
    name = clean_line(raw["condition"])

    phrases, sentences = [], []
    for line in raw.get("symptoms", []):
        p = to_phrase(line)
        if p:
            phrases.append(p)
        else:
            s = to_sentence(line)
            if s:
                sentences.append(s)
    # Many NHS pages list symptoms only under "Phone 999 / speak to your GP if you have:"
    for line in raw.get("warnings", []):
        p = to_phrase(line)
        if p:
            phrases.append(p)

    phrases = dedupe(phrases)
    sentences = dedupe(sentences)

    def clean_list(lines):
        return dedupe([clean_line(l) for l in lines if not is_junk(l) and len(l.split()) > 1])

    warnings = clean_list(raw.get("warnings", []))
    rec = {
        "condition": name,
        "url": raw["url"],
        "category": raw.get("category"),
        "symptom_phrases": phrases,
        "symptom_sentences": sentences,
        "symptom_text": normalize_text(" . ".join(phrases + sentences)),
        "tokens": tokenize(" ".join(phrases + sentences)),
        "causes": clean_list(raw.get("causes", [])),
        "warnings": warnings,
        "recommendations": clean_list(raw.get("recommendations", [])),
        "age_group": infer_age_group(name, raw["url"]),
        "gender": infer_gender(name, raw.get("category")),
        "emergency": is_emergency(name),
        "urgency": urgency_level(warnings),
        "n_phrases": len(phrases),
    }

    if NON_CONDITION.search(name):
        return rec, "not a condition page"
    if len(phrases) + len(sentences) < 2:
        return rec, "too few symptoms"
    return rec, None


def process_all(raw_docs):
    """Process every raw record. Returns (kept, excluded) lists."""
    kept, excluded = [], []
    for raw in raw_docs:
        rec, reason = process_condition(raw)
        if reason:
            excluded.append({"condition": rec["condition"], "url": rec["url"], "reason": reason})
        else:
            kept.append(rec)
    return kept, excluded


def summary(kept, excluded):
    n = [r["n_phrases"] for r in kept] or [0]
    by = lambda key: {v: sum(1 for r in kept if r[key] == v) for v in sorted({r[key] for r in kept})}
    return {
        "raw_records": len(kept) + len(excluded),
        "kept_conditions": len(kept),
        "excluded": len(excluded),
        "excluded_reasons": {x: sum(1 for e in excluded if e["reason"] == x)
                             for x in {e["reason"] for e in excluded}},
        "phrases_per_condition": {"min": min(n), "median": sorted(n)[len(n) // 2],
                                  "mean": round(sum(n) / len(n), 1), "max": max(n)},
        "age_group": by("age_group"),
        "gender": by("gender"),
        "emergency_conditions": sum(r["emergency"] for r in kept),
        "urgency": by("urgency"),
    }
