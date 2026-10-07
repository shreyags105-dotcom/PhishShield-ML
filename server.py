"""server.py - website version of the demo.
Run:  python server.py     then open  http://127.0.0.1:5000
Needs saved models first (run `python app.py --retrain` once)."""
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from features import FEATURE_COLUMNS, extract_features
from models import MODEL_DIR, load_models

app = Flask(__name__)
DEMO_MODEL = "Random Forest"

# Numbers from our training run - edit if you retrain and the values change.
DATASET_INFO = {"total": 235370, "phishing": 100520, "safe": 134850}
METRICS = [
    {"model": "Decision Tree",       "accuracy": 0.9256, "precision": 0.9503, "recall": 0.8714, "f1": 0.9091},
    {"model": "Random Forest",       "accuracy": 0.9253, "precision": 0.9482, "recall": 0.8727, "f1": 0.9089},
    {"model": "Logistic Regression", "accuracy": 0.9101, "precision": 0.9507, "recall": 0.8326, "f1": 0.8878},
]

if not os.path.isdir(MODEL_DIR) or len(os.listdir(MODEL_DIR)) < 3:
    raise SystemExit("No saved models found. Run:  python app.py --retrain   (then type quit)  and try again.")
MODELS = load_models()


def explain(f):
    """Plain-English reasons shown under the verdict."""
    reasons = []
    reasons.append(("Uses HTTPS", False) if f["is_https"] else ("No HTTPS (plain HTTP)", True))
    if f["has_ip"]:
        reasons.append(("Raw IP address instead of a domain name", True))
    if f["has_at"]:
        reasons.append(("Contains '@' (can hide the real destination)", True))
    if f["has_keyword"]:
        reasons.append(("Contains suspicious keyword (login, verify, bank...)", True))
    if f["url_length"] > 75:
        reasons.append((f"Long URL ({f['url_length']} characters)", True))
    if f["num_dots"] > 3:
        reasons.append((f"Many dots ({f['num_dots']}) - possible deep subdomains", True))
    return [{"text": t, "risk": r} for t, r in reasons]


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS, info=DATASET_INFO, demo_model=DEMO_MODEL)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if len(url) > 2000:
        return jsonify({"error": "URL is too long (max 2000 characters)."}), 400

    feats = extract_features(url)
    row = pd.DataFrame([feats], columns=FEATURE_COLUMNS)   # same column order as training

    per_model = {}
    for name, model in MODELS.items():
        per_model[name] = round(float(model.predict_proba(row)[0][1]) * 100, 1)

    p_phish = per_model[DEMO_MODEL]
    is_phish = p_phish >= 50
    return jsonify({
        "url": url,
        "verdict": "PHISHING" if is_phish else "SAFE",
        "confidence": round(p_phish if is_phish else 100 - p_phish, 1),
        "phishing_probability": p_phish,
        "features": feats,
        "reasons": explain(feats),
        "models": per_model,
    })


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory("outputs", name)


if __name__ == "__main__":
    print("\n  Open this in your browser:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
