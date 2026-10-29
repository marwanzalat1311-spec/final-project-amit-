"""Text normalisation shared by preprocessing, training and prediction.

The same normalize_text() must be applied to training data AND to the user's
free-text symptoms at prediction time, otherwise the model sees different text.
"""
import re
import unicodedata

# Everyday words patients use -> the wording NHS pages use.
SYNONYMS = {
    r"\b(tummy|belly|stomach) (pain|ache|cramps?)\b": "abdominal pain",
    r"\b(tummy|belly)\b": "abdomen",
    r"\bfeeling sick\b": "nausea",
    r"\bfeel sick\b": "nausea",
    r"\bbeing sick\b": "vomiting",
    r"\bthrowing up\b": "vomiting",
    r"\bthrow up\b": "vomiting",
    r"\bpuking\b": "vomiting",
    r"\bhigh temperature\b": "fever",
    r"\bfeverish\b": "fever",
    r"\bpoo\b": "stool",
    r"\bpee\b": "urine",
    r"\bweeing\b": "urinating",
    r"\bbreathless(ness)?\b": "shortness of breath",
    r"\bshort of breath\b": "shortness of breath",
    r"\btired(ness)?\b": "fatigue",
    r"\bexhausted\b": "fatigue",
    r"\bheadaches?\b": "headache",
    r"\bdizzy\b": "dizziness",
    r"\bitchy\b": "itching",
    r"\bitchiness\b": "itching",
}
_SYN = [(re.compile(k), v) for k, v in SYNONYMS.items()]

STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those there their they them you your yours you're youre i me my we our
have has had having do does did can could may might will would should shall must
very more most some any such so than too also just only not no nor own same other
about into over under again further then once here when where why how all both each few
up down out off above below between through during before after while until
like e.g eg etc usually often sometimes may might feel feeling get getting got
""".split())
# Keep words that change medical meaning.
STOPWORDS -= {"no", "not", "left", "right", "after", "before", "during"}

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def ascii_fold(text):
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-"))
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_text(text, synonyms=True):
    """Lowercase, fold accents/quotes, map lay terms, drop punctuation."""
    t = ascii_fold(text).lower()
    if synonyms:
        for rx, rep in _SYN:
            t = rx.sub(rep, t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text, remove_stopwords=True):
    tokens = _TOKEN.findall(normalize_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens
