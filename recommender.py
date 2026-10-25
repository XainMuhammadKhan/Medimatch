"""
MediMatch - NLP-based medicine substitute & symptom recommender.

Pipeline
--------
1. Text normalisation  : lowercase, strip punctuation, collapse whitespace.
2. Composition parsing : "Paracetamol (500mg) + Caffeine (65mg)"  ->
                         salts = {paracetamol, caffeine}, strengths per salt.
3. Name resolution     : exact match -> prefix match -> fuzzy match (RapidFuzz,
                         token-set ratio / Levenshtein) for misspelled input.
4. Substitutes         : (a) EXACT  - identical salt set (same active ingredients)
                         (b) SIMILAR - TF-IDF cosine similarity over salts + uses.
5. Symptom search      : query expansion with a lay-term -> medical-term lexicon,
                         then TF-IDF (word 1-2 grams) cosine similarity on "Uses".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = Path(__file__).parent / "data"

# Spelling variants between regions (Pakistani/UK brands vs Indian dataset).
SALT_SYNONYMS = {
    "acetaminophen": "paracetamol",
    "amoxicillin": "amoxycillin",
    "cephradine": "cefradine",
    "chlorphenamine": "chlorpheniramine maleate",
    "chlorpheniramine": "chlorpheniramine maleate",
    "albuterol": "salbutamol",
    "acetylsalicylic acid": "aspirin",
}

# Lay-language -> clinical vocabulary used in the "Uses" column.
QUERY_EXPANSION = {
    "headache": "headache pain relief migraine",
    "head ache": "headache pain relief",
    "body ache": "pain relief muscle pain",
    "bodyache": "pain relief muscle pain",
    "ache": "pain relief",
    "fever": "fever",
    "temperature": "fever",
    "flu": "common cold flu fever",
    "cold": "common cold",
    "runny nose": "common cold allergic rhinitis",
    "blocked nose": "nasal congestion common cold",
    "sneezing": "allergic rhinitis allergy sneezing",
    "allergy": "allergic conditions allergic rhinitis",
    "itching": "itching skin allergy urticaria",
    "cough": "cough dry cough cough with mucus",
    "sore throat": "throat infection pharyngitis",
    "acidity": "acidity heartburn acid reflux peptic ulcer disease gastroesophageal reflux",
    "heartburn": "heartburn acid reflux gastroesophageal reflux disease",
    "gas": "flatulence bloating indigestion",
    "stomach ache": "abdominal pain stomach pain",
    "stomach pain": "abdominal pain stomach pain",
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "diarrhoea": "diarrhea",
    "vomiting": "nausea vomiting",
    "motion sickness": "motion sickness nausea vomiting",
    "constipation": "constipation",
    "bp": "hypertension high blood pressure",
    "blood pressure": "hypertension high blood pressure",
    "sugar": "diabetes type 2 diabetes mellitus",
    "diabetes": "type 2 diabetes mellitus",
    "cholesterol": "high cholesterol hyperlipidemia",
    "infection": "bacterial infections",
    "uti": "urinary tract infections",
    "asthma": "asthma bronchospasm",
    "breathing": "asthma chronic obstructive pulmonary disorder",
    "anxiety": "anxiety disorder panic disorder",
    "sleep": "insomnia",
    "insomnia": "insomnia sleeplessness",
    "depression": "depression",
    "period pain": "menstrual pain dysmenorrhea",
    "periods": "menstrual pain dysmenorrhea",
    "joint pain": "joint pain arthritis osteoarthritis rheumatoid arthritis",
    "back pain": "back pain muscle pain",
    "toothache": "dental pain toothache",
    "acne": "acne pimples",
    "worms": "worm infections",
    "pain": "pain relief",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) | {"treatment", "prevention", "of", "and", "tablet"}
# Words that, when they end a chunk, mean the next capitalised word continues the phrase.
_JOINERS = {"of", "and", "in", "the", "to", "with", "for", "due", "by", "caused",
            "associated", "or", "a", "an", "after", "during", "from", "on"}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9%\.\s/+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_salt(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip().lower()
    return SALT_SYNONYMS.get(name, name)


def parse_composition(comp: str) -> list[tuple[str, str]]:
    """'Paracetamol (500mg) + Caffeine (65mg)' -> [('paracetamol','500mg'), ('caffeine','65mg')]"""
    out = []
    for part in str(comp).split("+"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", part)
        name, strength = (m.group(1), m.group(2)) if m else (part, "")
        name = canonical_salt(name)
        if name:
            out.append((name, strength.replace(" ", "").lower()))
    return out


def split_phrases(text: str) -> list[str]:
    """The 1mg dataset concatenates items without separators:
    'Treatment of Fever Pain relief' -> ['Treatment of Fever', 'Pain relief']"""
    if not isinstance(text, str) or not text.strip():
        return []
    chunks = [c for c in re.split(r"\s+(?=[A-Z])|(?<=[a-z\)])(?=[A-Z])", text.strip()) if c.strip()]
    phrases: list[str] = []
    for ch in chunks:
        if phrases and phrases[-1].split()[-1].lower() in _JOINERS:
            phrases[-1] += " " + ch
        else:
            phrases.append(ch)
    # de-duplicate while preserving order
    seen, result = set(), []
    for p in phrases:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            result.append(p)
    return result


def expand_query(q: str) -> str:
    qn = normalise(q)
    extra = [v for k, v in QUERY_EXPANSION.items() if re.search(rf"\b{re.escape(k)}\b", qn)]
    return " ".join([qn] + extra)


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
@dataclass
class Resolution:
    kind: str            # "medicine" | "salt" | "none"
    index: int | None    # row index for medicine
    salt: str | None     # salt name for salt search
    matched: str | None  # display string actually matched
    score: float         # fuzzy score 0-100
    corrected: bool      # True when spelling correction was applied


class MedicineRecommender:
    def __init__(self, csv_path: Path = DATA_DIR / "Medicine_Details.csv",
                 local_brands: Path = DATA_DIR / "local_brands.json"):
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "Medicine Name": "name", "Composition": "composition", "Uses": "uses",
            "Side_effects": "side_effects", "Image URL": "image", "Manufacturer": "manufacturer",
            "Excellent Review %": "excellent", "Average Review %": "average", "Poor Review %": "poor",
        })
        df = df.dropna(subset=["name", "composition"]).drop_duplicates(subset=["name"])
        df["source"] = "1mg dataset"
        df = self._add_local_brands(df, local_brands)
        df = df.fillna({"uses": "", "side_effects": "", "image": "", "manufacturer": "",
                        "excellent": 0, "average": 0, "poor": 0}).reset_index(drop=True)

        parsed = df["composition"].map(parse_composition)
        df["salts"] = parsed.map(lambda p: sorted({s for s, _ in p}))
        df["salt_key"] = df["salts"].map(lambda s: "|".join(s))
        df["strength_key"] = parsed.map(lambda p: "|".join(sorted(f"{s}:{st}" for s, st in p)))
        df["name_norm"] = df["name"].map(normalise)
        df["score"] = df["excellent"] - df["poor"]  # simple review-based quality score
        self.df = df

        # ---- TF-IDF models -------------------------------------------------
        # Salt model: each salt is one token (so 'clavulanic acid' stays together).
        self.salt_vec = TfidfVectorizer(analyzer=lambda salts: salts)
        self.salt_mat = self.salt_vec.fit_transform(df["salts"])

        # Uses model: word uni+bi-grams over the indication text.
        self.uses_vec = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1, 2),
                                        sublinear_tf=True, min_df=1)
        self.uses_mat = self.uses_vec.fit_transform(df["uses"].map(normalise))

        # ---- lookup tables for name resolution ----------------------------
        self.names = df["name_norm"].tolist()
        # brand stem: 'panadol 500mg tablet' -> 'panadol'
        self.stems = df["name_norm"].map(lambda n: re.split(r"\s\d", n)[0].strip()).tolist()
        salt_counts = pd.Series([s for ss in df["salts"] for s in ss]).value_counts()
        self.salt_names = salt_counts.index.tolist()
        self.salt_counts = salt_counts.to_dict()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_local_brands(df: pd.DataFrame, path: Path) -> pd.DataFrame:
        if not path.exists():
            return df
        brands = json.loads(path.read_text(encoding="utf-8"))["brands"]
        key_of = lambda c: "|".join(sorted({s for s, _ in parse_composition(c)}))
        keys = df["composition"].map(key_of)
        rows = []
        for b in brands:
            if (df["name"].str.lower() == b["name"].lower()).any():
                continue
            k = key_of(b["composition"])
            donors = df[keys == k]
            donor = donors.sort_values("excellent", ascending=False).iloc[0] if len(donors) else None
            rows.append({
                "name": b["name"], "composition": b["composition"],
                "uses": donor["uses"] if donor is not None else b.get("uses", ""),
                "side_effects": donor["side_effects"] if donor is not None else b.get("side_effects", ""),
                "image": "", "manufacturer": b.get("manufacturer", ""),
                "excellent": 0, "average": 0, "poor": 0, "source": "Local brand (PK)",
            })
        return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df

    # ------------------------------------------------------------------ #
    def _row(self, i: int, **extra) -> dict:
        r = self.df.iloc[i]
        d = {
            "id": int(i), "name": r["name"], "composition": r["composition"],
            "salts": r["salts"], "manufacturer": r["manufacturer"], "image": r["image"],
            "uses": split_phrases(r["uses"]), "side_effects": split_phrases(r["side_effects"]),
            "reviews": {"excellent": int(r["excellent"]), "average": int(r["average"]),
                        "poor": int(r["poor"])},
            "source": r["source"],
        }
        d.update(extra)
        return d

    # ------------------------------------------------------------------ #
    def resolve(self, query: str) -> Resolution:
        q = normalise(query)
        if not q:
            return Resolution("none", None, None, None, 0, False)
        q_salt = canonical_salt(q)

        # 1. exact medicine name / exact salt
        if q in self.names:
            i = self.names.index(q)
            return Resolution("medicine", i, None, self.df.at[i, "name"], 100, False)
        if q_salt in self.salt_counts:
            return Resolution("salt", None, q_salt, q_salt.title(), 100, False)

        # 2. brand stem / prefix match -> best-reviewed variant
        cand = [i for i, (n, s) in enumerate(zip(self.names, self.stems)) if s == q or n.startswith(q + " ")]
        if cand:
            i = max(cand, key=lambda j: (self.stems[j] == q, self.df.at[j, "score"]))
            return Resolution("medicine", i, None, self.df.at[i, "name"], 100, False)

        # 3. fuzzy match (spelling correction) over names and salts
        best_name = process.extractOne(q, self.stems, scorer=fuzz.WRatio)
        best_salt = process.extractOne(q_salt, self.salt_names, scorer=fuzz.WRatio)
        if best_salt and (not best_name or best_salt[1] >= best_name[1]) and best_salt[1] >= 80:
            return Resolution("salt", None, best_salt[0], best_salt[0].title(), best_salt[1], True)
        if best_name and best_name[1] >= 75:
            stem = best_name[0]
            cand = [i for i, s in enumerate(self.stems) if s == stem]
            i = max(cand, key=lambda j: self.df.at[j, "score"])
            return Resolution("medicine", i, None, self.df.at[i, "name"], best_name[1], True)
        return Resolution("none", None, None, None, 0, False)

    # ------------------------------------------------------------------ #
    def suggest(self, query: str, limit: int = 8) -> list[dict]:
        q = normalise(query)
        if len(q) < 2:
            return []
        out, seen = [], set()
        # salts first if they prefix-match
        for s in self.salt_names:
            if s.startswith(q):
                out.append({"label": s.title(), "type": "salt",
                            "hint": f"{self.salt_counts[s]} medicines"})
                seen.add(s)
                if len(out) >= 3:
                    break
        prefix = [i for i, n in enumerate(self.names) if n.startswith(q)]
        prefix.sort(key=lambda i: (-self.df.at[i, "score"], len(self.names[i])))
        for i in prefix:
            if len(out) >= limit:
                break
            out.append({"label": self.df.at[i, "name"], "type": "medicine",
                        "hint": self.df.at[i, "composition"]})
        if len(out) < limit:  # fuzzy fill-in for typos
            for name, score, i in process.extract(q, self.names, scorer=fuzz.WRatio, limit=limit * 2):
                if len(out) >= limit:
                    break
                if score >= 70 and i not in prefix:
                    out.append({"label": self.df.at[i, "name"], "type": "medicine",
                                "hint": self.df.at[i, "composition"]})
        return out

    # ------------------------------------------------------------------ #
    def substitutes(self, query: str, limit: int = 24) -> dict:
        res = self.resolve(query)
        meta = {"query": query, "matched": res.matched, "corrected": res.corrected,
                "match_score": round(res.score, 1), "kind": res.kind}
        if res.kind == "none":
            return {**meta, "target": None, "exact": [], "similar": []}

        if res.kind == "salt":
            # medicines whose ONLY active ingredient is this salt
            mask = self.df["salt_key"] == res.salt
            idx = self.df[mask].sort_values("score", ascending=False).index[:limit]
            exact = [self._row(i, similarity=1.0, same_strength=None) for i in idx]
            # combos containing the salt
            combo = self.df[self.df["salts"].map(lambda s: res.salt in s) & ~mask]
            combo = combo.sort_values("score", ascending=False).index[:limit // 2]
            similar = [self._row(i, similarity=round(1 / len(self.df.at[i, "salts"]), 3),
                                 reason=f"Contains {res.salt.title()} + other ingredients") for i in combo]
            return {**meta, "target": {"name": res.salt.title(), "composition": res.salt.title(),
                                       "salts": [res.salt], "is_salt": True},
                    "exact": exact, "similar": similar}

        t = res.index
        target = self.df.iloc[t]
        # (a) exact substitutes - identical active ingredients
        same = self.df[(self.df["salt_key"] == target["salt_key"]) & (self.df.index != t)].copy()
        same["same_strength"] = same["strength_key"] == target["strength_key"]
        same = same.sort_values(["same_strength", "score"], ascending=False).head(limit)
        exact = [self._row(i, similarity=1.0, same_strength=bool(same.at[i, "same_strength"]))
                 for i in same.index]

        # (b) similar alternatives - TF-IDF cosine on salts (0.65) + uses (0.35)
        salt_sim = linear_kernel(self.salt_mat[t], self.salt_mat).ravel()
        uses_sim = linear_kernel(self.uses_mat[t], self.uses_mat).ravel()
        sim = 0.65 * salt_sim + 0.35 * uses_sim
        sim[self.df["salt_key"].values == target["salt_key"]] = -1  # exclude exact matches
        order = np.argsort(-sim)
        similar, seen_keys = [], set()
        for i in order:
            if sim[i] <= 0.15 or len(similar) >= limit // 2:
                break
            k = self.df.at[i, "salt_key"]
            if k in seen_keys:  # one representative per composition keeps results diverse
                continue
            seen_keys.add(k)
            shared = sorted(set(target["salts"]) & set(self.df.at[i, "salts"]))
            reason = (f"Shares {', '.join(s.title() for s in shared)}" if shared
                      else "Used for similar conditions")
            similar.append(self._row(int(i), similarity=round(float(sim[i]), 3), reason=reason))

        return {**meta, "target": self._row(t), "exact": exact, "similar": similar}

    # ------------------------------------------------------------------ #
    def by_symptom(self, query: str, limit: int = 15) -> dict:
        expanded = expand_query(query)
        qv = self.uses_vec.transform([expanded])
        sim = linear_kernel(qv, self.uses_mat).ravel()
        # small boost for well-reviewed medicines so popular, trusted options surface first
        quality = (self.df["score"].values + 100) / 200  # 0..1
        ranked = sim * (0.85 + 0.15 * quality)
        order = np.argsort(-ranked)
        results, seen_keys = [], set()
        for i in order:
            if sim[i] <= 0.05 or len(results) >= limit:
                break
            k = self.df.at[i, "salt_key"]
            if k in seen_keys:  # diversify by composition
                continue
            seen_keys.add(k)
            results.append(self._row(int(i), similarity=round(float(sim[i]), 3)))
        terms = [t for t in self.uses_vec.build_analyzer()(expanded)][:12]
        return {"query": query, "expanded_query": expanded, "terms": terms, "results": results}

    def detail(self, idx: int) -> dict | None:
        if 0 <= idx < len(self.df):
            return self._row(idx)
        return None

    def stats(self) -> dict:
        return {"medicines": int(len(self.df)), "salts": int(len(self.salt_names)),
                "manufacturers": int(self.df["manufacturer"].nunique())}


if __name__ == "__main__":
    rec = MedicineRecommender()
    print(rec.stats())
    for q in ["Panadol", "panadoll", "paracetamol", "Augmentin 625 Duo Tablet", "brufen"]:
        r = rec.substitutes(q)
        print(f"\n== {q} -> {r['matched']} (corrected={r['corrected']})")
        for m in r["exact"][:5]:
            print("   =", m["name"], "|", m["composition"])
        for m in r["similar"][:3]:
            print("   ~", m["name"], "|", m["composition"], m["similarity"])
    s = rec.by_symptom("headache and fever")
    print("\n== symptom:", s["expanded_query"])
    for m in s["results"][:6]:
        print("   ", m["name"], "|", m["composition"], m["similarity"])
