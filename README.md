# MediMatch: NLP Medicine Substitute Recommender

If your pharmacy is out of **Panadol**, MediMatch lists other medicines with the same active ingredient (paracetamol). It also suggests close alternatives and can recommend medicines from a description of your symptoms.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```
Open http://127.0.0.1:5000

## Features
| Feature | How it works (NLP) |
|---|---|
| **Find substitute** | Parses composition text into salts, then returns medicines with the identical salt set (same strength first) |
| **Similar alternatives** | TF-IDF vectors over salts + uses; weighted cosine similarity (0.65 salts / 0.35 uses) |
| **Search by symptom** | Lay-term query expansion ("loose motions" → "diarrhea"), TF-IDF 1–2-grams over "Uses" |
| **Spelling correction** | RapidFuzz WRatio (Levenshtein-based) over brand stems and salt names |
| **Autocomplete** | Prefix match ranked by reviews, with fuzzy fallback |
| **Details** | Uses, side effects and review breakdown for each medicine |

## Project structure
```
app.py                 Flask server + REST API
recommender.py         NLP engine (run directly for a CLI demo)
data/Medicine_Details.csv   11,000 Medicine Details dataset (1mg, Kaggle)
data/local_brands.json      Pakistani brands (Panadol, Brufen, Risek, ...) missing from the dataset
templates/index.html   UI
static/css, static/js  Styling and frontend logic
```

## API
- `GET /api/suggest?q=pana`
- `GET /api/substitutes?q=Panadol`
- `GET /api/symptoms?q=headache and fever`
- `GET /api/medicine/<id>`

> This is an academic project, not medical advice.
