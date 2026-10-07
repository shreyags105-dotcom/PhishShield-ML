"""app.py  (Rouhin) - ties everything together + live judge demo.
Run:  python app.py            (train if needed, evaluate, start demo)
      python app.py --retrain  (force retraining)
"""
import os
import sys

import pandas as pd

from evaluation import run_evaluation
from features import FEATURE_COLUMNS, build_feature_df, extract_features
from models import MODEL_DIR, load_models, save_models, train_models

CSV_PATH = "data/phishing_urls.csv"
DEMO_MODEL = "Random Forest"


def predict_url_safety(url, model):
    """Return (label_text, phishing_probability_pct, features_dict)."""
    feats = extract_features(url)
    row = pd.DataFrame([feats], columns=FEATURE_COLUMNS)   # same column order as training
    proba = model.predict_proba(row)[0]
    p_phish = float(proba[1]) * 100
    if p_phish >= 50:
        return "🚨 PHISHING", p_phish, feats
    return "✅ SAFE", 100 - p_phish, feats


def demo_loop(model):
    print("\n" + "=" * 55)
    print("   PHISHING URL DETECTOR - type a URL (or 'quit')")
    print("=" * 55)
    while True:
        url = input("\nEnter URL > ").strip()
        if url.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not url:
            print("Please type a URL.")
            continue
        label, conf, feats = predict_url_safety(url, model)
        print(f"\nResult     : {label}")
        print(f"Confidence : {conf:.1f}%")
        print("Features   :")
        for k, v in feats.items():
            print(f"   {k:<12}: {v}")


def main():
    retrain = "--retrain" in sys.argv
    models_exist = os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) >= 3

    if retrain or not models_exist:
        print("Extracting features...")
        X, y = build_feature_df(CSV_PATH)
        print(f"Dataset: {len(X)} URLs ({int(y.sum())} phishing / {int((y == 0).sum())} safe)")
        models, X_test, y_test, preds = train_models(X, y)
        save_models(models)
        run_evaluation(models, y_test, preds, FEATURE_COLUMNS)
    else:
        print("Loading saved models (use --retrain to retrain)...")
        models = load_models()

    demo_loop(models[DEMO_MODEL])


if __name__ == "__main__":
    main()