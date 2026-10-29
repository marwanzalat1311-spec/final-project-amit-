# Medical Diagnosis AI

AI system that predicts probable conditions from free-text symptoms, trained on
data scraped from NHS Inform. Built step by step.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | **Data**: scrape NHS Inform A–Z → MongoDB, Data Handling API | ✅ |
| 2 | **Preprocessing**: clean, extract symptom phrases, metadata, training set, Preprocessing API | ✅ |
| 3 | **Baseline**: TF-IDF + Logistic Regression, metrics, GridFS model registry, predictor | ✅ |
| 4 | **Deep learning**: RNN / LSTM / BiLSTM (+ subword embeddings) | ✅ |
| 5 | Transformer: fine-tune BERT / BioBERT | next |
| 6 | Compare models (Top-K accuracy, emergency recall); save models to MongoDB GridFS | |
| 7 | Model API: predict + rank probabilities + attach warnings/recommendations | |
| 8 | Reports, architecture diagram, documentation | |

## Project layout

```
medical-diagnosis-ai/
├── config.py              settings (Mongo URI, delays, URLs)
├── db.py                  MongoDB connection, indexes, upserts
├── scraper/nhs_scraper.py A-Z link collector + condition page parser
├── scrape.py              command-line scraper
├── api/data_api.py        Data Handling API (Flask, port 5001)
├── preprocessing/
│   ├── text.py            normalize_text / tokenize (shared with prediction)
│   ├── cleaner.py         raw page -> symptom phrases, metadata, emergency flag
│   ├── dataset.py         synthetic patient samples + leakage-safe split
│   └── pipeline.py        runs everything, writes files + MongoDB
├── preprocess.py          command-line preprocessing
├── api/preprocess_api.py  Preprocessing API (Flask, port 5002)
├── inspect_page.py        debug: how one page is parsed
├── modeling/
│   ├── data.py            loads the dataset splits
│   ├── evaluation.py      top-k, macro-F1, MRR, calibration, emergency recall (all models)
│   ├── baseline.py        TF-IDF (word + char) + age/gender -> logistic regression
│   ├── neural.py          RNN / LSTM / BiLSTM / GRU (+ subword) with attention pooling
│   ├── registry.py        save/load models: models/ folder + MongoDB GridFS
│   └── predictor.py       ranking + age/gender adjustment + warnings/recommendations
├── train_baseline.py      Step 3 training script
├── train_neural.py        Step 4 training script
├── predict.py             try a model from the command line
├── reports/               experiment logs (JSON) for the report
├── tests/test_scraper.py  offline parser tests
└── requirements.txt
```

## Step 1 — get the data

### 1. Install

You need Python 3.10+ and MongoDB.

- MongoDB: install **MongoDB Community Server** (https://www.mongodb.com/try/download/community)
  or run it with Docker: `docker run -d -p 27017:27017 --name mongo mongo:7`
- Optional: **MongoDB Compass** to browse the data visually.

```bash
python -m venv venv
# Windows: venv\Scripts\activate     Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests           # should print "2 passed"
```

If MongoDB is not on `localhost:27017`, set `MONGO_URI` (e.g. a MongoDB Atlas connection string).

### 2. Scrape

Test on 10 conditions first, then run the full scrape (~1 second per page, so
a full A–Z run takes several minutes):

```bash
python scrape.py --limit 10
python scrape.py
```

Each page is saved to MongoDB and `data/conditions.json` as soon as it is
downloaded. If the run stops (network drop, Ctrl+C), run the same command again:
pages already saved are skipped. Pages that failed are listed in
`data/failed_urls.txt` with the reason; retry only those with:

```bash
python scrape.py --retry-failed
python scrape.py --retry-failed --delay 3   # slower, if the site keeps refusing
```

Use `python scrape.py --all` to re-download everything (e.g. after changing the parser).

At the end the script prints how many conditions have an empty field. A few
empty `causes` or `warnings` are normal (some NHS pages don't have those
sections). If almost everything is empty, the site layout has changed — open
a page, check its headings, and adjust `SECTION_RULES` in `scraper/nhs_scraper.py`.

### 3. Or use the Data Handling API

```bash
python -m api.data_api
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API and MongoDB status |
| POST | `/scrape` | start full scrape in the background; body `{"limit": 20}` optional |
| GET | `/scrape/status` | progress of the running scrape |
| POST | `/scrape/one` | scrape one page; body `{"url": "https://www.nhsinform.scot/illnesses-and-conditions/..."}` |
| GET | `/conditions?q=fever&page=1&per_page=50` | list / search conditions |
| GET | `/conditions/<name>` | one condition |
| GET | `/stats` | totals, empty-field counts, counts per category |
| GET | `/export?format=json` or `csv` | download the dataset |

Example:
```bash
curl -X POST http://localhost:5001/scrape -H "Content-Type: application/json" -d "{\"limit\": 10}"
curl http://localhost:5001/scrape/status
curl http://localhost:5001/conditions/Acne
```

## Database design

**`conditions`** collection

| Field | Type | Notes |
|---|---|---|
| condition | String | condition name (unique) |
| symptoms | Array\<String\> | list of symptoms |
| causes | Array\<String\> | list of causes |
| warnings | Array\<String\> | emergency / "see your GP" advice |
| recommendations | Array\<String\> | treatment and self-care advice |
| url | String | source page (unique) |
| category | String | NHS Inform category, e.g. `skin-hair-and-nails` |
| overview | Array\<String\> | intro paragraphs (extra text for training) |
| source, scraped_at, created_at, updated_at | String / Date | metadata |

**`models`** collection (filled in Step 6): `name`, `type`, `gridfs_id`, `labels`, `metrics`, `created`.

## How the parser decides which section text belongs to

It walks the page's headings in order and files the bullets and paragraphs
under each heading by keyword:

- **warnings** — "when to…", "urgent", "emergency", "999", "111", "speak to your GP"…
  (checked first, so "When to seek help for symptoms" counts as a warning).
  Callout boxes starting with "Phone 999", "Speak to your GP" etc. are always warnings.
- **symptoms** — "symptoms", "signs"
- **causes** — "causes", "risk factors", "triggers"
- **recommendations** — "treating", "self-care", "managing", "preventing", "living with"

Navigation, headers, footers and feedback blocks are stripped first.

## Step 2 — preprocessing

```bash
python preprocess.py              # reads MongoDB `conditions`
python preprocess.py --source json  # or data/conditions.json
```

What it does:

1. **Cleans lines**: removes navigation/boilerplate (BSL links, "Return to Symptoms",
   tab labels, Macmillan/helpline lines), folds curly quotes, fixes spacing.
2. **Extracts symptom phrases** from `symptoms` **and** `warnings`. Many NHS pages list
   the symptoms only under "Phone 999 / speak to your GP if you have:" (e.g. heart attack,
   COPD, sepsis). Action lines ("phone 111…"), lead-ins ("People may experience:"),
   negative statements ("…are not common symptoms of COPD") and meta sentences are dropped;
   wrappers are stripped ("you're wheezing" → "wheezing").
3. **Metadata** per condition: `age_group` (child / adult / any) and `gender`
   (female / male / any) from the condition name and category.
   `urgency` = highest advice level on the page (emergency = 999/A&E, urgent = GP urgently/111,
   routine). `emergency` = the condition itself is life-threatening (heart attack, stroke,
   sepsis, meningitis, …; curated list `EMERGENCY_CONDITIONS` in `cleaner.py`). Most pages list
   999 red flags, so `urgency` alone would mark flu as an emergency; `emergency` is what the
   "recall on emergency conditions" metric uses.
4. **Filters** pages that are not diagnosable conditions ("Living with…", "Talking to
   children about cancer", support pages) and pages with fewer than 2 symptom items.
5. **Builds the training set**. There is one NHS page per condition, so each condition's
   phrases are combined into synthetic patient descriptions with age and gender
   ("i've had fever, sore throat and swollen glands" + age 24, female).
   ~25% of each condition's phrases are **held out**: train never contains them and every
   val/test sample contains at least one, so test accuracy measures generalisation, not memory.

Outputs: MongoDB `processed_conditions`, `data/processed_conditions.json`,
`data/dataset/{train,val,test}.jsonl`, `data/dataset/labels.json`,
`data/preprocessing_report.json` (all numbers for the experimentation report).

### Preprocessing API

```bash
python -m api.preprocess_api      # http://localhost:5002
```

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/preprocess` | run the pipeline; body optional `{"source":"mongo","n_train":40,"seed":42}` |
| GET | `/report` | counts, exclusions, dataset sizes |
| GET | `/processed?q=wheezing` | search processed conditions |
| GET | `/processed/<name>` | one processed condition |
| GET | `/excluded` | excluded pages and why |
| GET | `/dataset/<train\|val\|test>?limit=20` | browse samples |
| POST | `/normalize` | `{"text":"tummy ache and throwing up"}` → normalized text + tokens |

## Step 3 — baseline: TF-IDF + Logistic Regression

```bash
pip install -r requirements.txt     # adds numpy, scipy, scikit-learn, joblib
python train_baseline.py            # ~1-2 min; add --tune to try 3 alpha values
python predict.py "sore throat, fever and swollen glands" --age 20 --gender female
python predict.py --interactive
```

**Model.** Features = word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams, robust to
typos) + one-hot age bucket and gender. Classifier = logistic regression (log-loss) trained
with SGD on all CPU cores. Probabilities are calibrated with **temperature scaling** fitted on
the validation set, so the percentages shown to users match how often the model is right.

**Selection.** Hyper-parameters are chosen on **validation**; **test** is evaluated once.

**Saving.** `models/<name>.joblib` + MongoDB: file in GridFS bucket `model_files`, metadata in
`models` (name, type, gridfs_id, labels, metrics, created, params, size_mb).
Weights are stored as float32: 41 MB instead of 82 MB, identical predictions.

**Prediction** (`modeling/predictor.py`): probabilities → gender-specific conditions removed for
the other gender, child-only/adult-only conditions ×0.2 for the wrong age → renormalised top-k →
warnings + recommendations attached from `processed_conditions` → emergency alert if an emergency
condition is in the top 3.

### Experiments (validation set, 365 conditions, earlier data snapshot)

| Variant | Train time | top-1 | top-5 | unseen top-5 | emergency@5 |
|---|---|---|---|---|---|
| LogisticRegression lbfgs, C=5 | 197 s | 0.535 | 0.688 | 0.670 | 0.716 |
| lbfgs, tol 1e-3, 100 iters | 93 s | 0.530 | 0.684 | 0.666 | 0.710 |
| lbfgs, half the features | 53 s | 0.513 | 0.667 | 0.648 | 0.699 |
| SGD modified-huber | 42 s | 0.496 | 0.524 | 0.499 | 0.506 |
| SGD log-loss α=3e-6 | 70 s | 0.552 | 0.681 | 0.663 | 0.705 |
| **SGD log-loss α=3e-5 (chosen)** | **69 s** | **0.546** | **0.692** | **0.674** | **0.739** |

Test set (chosen model): top-1 0.554, top-3 0.664, top-5 0.703, macro-F1 0.558, MRR 0.623,
emergency recall@5 0.659, unseen-phrase top-5 0.686. After calibration (T = 0.71) the mean
top-1 probability is 0.552 vs. top-1 accuracy 0.554 → well calibrated.

**Metric note.** Ranks are computed pessimistically on ties. With optimistic ties, modified-huber
(which outputs many identical zero probabilities) scored a fake top-5 of 0.998.

## Step 4 — deep learning: RNN / LSTM / BiLSTM

```bash
pip install -r requirements.txt            # adds torch (CPU is enough)
python train_neural.py                     # trains RNN, LSTM, BiLSTM; saves the best (~15-30 min CPU)
python train_neural.py --arch bilstm --subword   # the best Step 4 model
python predict.py "..." --model bilstm_subword_v1
```

**Architecture** (identical for every cell type, so the comparison is fair):
word embedding (128) → recurrent layer (RNN | LSTM, optionally bidirectional, hidden 128) →
attention pooling + max pooling over words → concat with age/gender (Linear 7→16) →
dropout → softmax over conditions.
Training: AdamW (lr 2e-3), cross-entropy with label smoothing 0.1, batch 128, up to 25 epochs,
learning rate halved on plateau, **early stopping on validation top-3** (patience 4), best
epoch restored, then temperature calibration on validation.

**Subword embeddings** (`--subword`, fastText idea): each word vector = word embedding + mean of
its hashed character 3-5-gram embeddings (50k buckets, crc32 hash). Test samples use symptom
wordings held out of training; a word-level model maps unseen words to `<unk>`, the subword model
still recognises their pieces ("breathlessness" → "breath", "less").

### Experiments (validation, 365 conditions, earlier data snapshot, 2 CPU threads)

| Model | top-1 | top-3 | top-5 | unseen top-5 | emergency@5 | time | size |
|---|---|---|---|---|---|---|---|
| RNN | 0.452 | 0.560 | 0.600 | 0.578 | 0.545 | 152 s | |
| LSTM | 0.476 | 0.594 | 0.636 | 0.615 | 0.585 | 206 s | |
| BiLSTM | 0.489 | 0.605 | 0.644 | 0.624 | 0.591 | 562 s | 3.4 MB |
| LSTM + subword | 0.488 | 0.607 | 0.650 | 0.630 | 0.636 | 637 s | |
| **BiLSTM + subword** | **0.497** | **0.611** | **0.653** | **0.634** | **0.642** | 618 s | 27 MB |
| *TF-IDF + LogReg (Step 3)* | *0.546* | *0.657* | *0.692* | *0.674* | *0.739* | *69 s* | *41 MB* |

Test (BiLSTM + subword): top-1 0.509, top-3 0.623, top-5 0.677, emergency recall@5 0.608,
unseen top-5 0.659.

**Findings**
- RNN < LSTM < BiLSTM on every metric: gating (LSTM) handles longer descriptions better than a
  plain RNN, and reading both directions adds a little more.
- Subword embeddings improve every metric, most of all emergency recall (+5 points), because
  unseen symptom wordings no longer become `<unk>`.
- The TF-IDF baseline is still ahead. With ~40 synthetic examples per condition, a linear model on
  word + character n-grams is very strong; the recurrent models have far more parameters to learn
  from the same data. Step 5 (pre-trained transformer) tests whether prior language knowledge
  closes this gap.
