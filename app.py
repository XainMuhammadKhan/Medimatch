"""Flask web server for MediMatch. Run:  python app.py  -> http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request

from recommender import MedicineRecommender

app = Flask(__name__)
print("Loading dataset and building TF-IDF models...")
rec = MedicineRecommender()
print("Ready:", rec.stats())


@app.get("/")
def index():
    return render_template("index.html", stats=rec.stats())


@app.get("/api/suggest")
def api_suggest():
    return jsonify(rec.suggest(request.args.get("q", "")))


@app.get("/api/substitutes")
def api_substitutes():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Please enter a medicine name."}), 400
    return jsonify(rec.substitutes(q))


@app.get("/api/symptoms")
def api_symptoms():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Please describe your symptoms."}), 400
    return jsonify(rec.by_symptom(q))


@app.get("/api/medicine/<int:idx>")
def api_medicine(idx: int):
    d = rec.detail(idx)
    return (jsonify(d), 200) if d else (jsonify({"error": "Not found"}), 404)


@app.get("/api/stats")
def api_stats():
    return jsonify(rec.stats())


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
