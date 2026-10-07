"""models.py  (Riya) - split, train, tune, save/load the 3 models."""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "saved_models"


def get_models(tune=True, X_train=None, y_train=None):
    """Return the 3 classifiers. If tune=True, Random Forest / Decision Tree are grid-searched."""
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    if tune and X_train is not None:
        grids = {
            "Decision Tree": (DecisionTreeClassifier(random_state=42),
                              {"max_depth": [5, 10, 20, None]}),
            "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                              {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}),
        }
        for name, (est, grid) in grids.items():
            gs = GridSearchCV(est, grid, cv=3, scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"[tuning] {name}: best params = {gs.best_params_}")
            models[name] = gs.best_estimator_
    return models


def train_models(X, y, tune=True):
    """Split 70/30, fit all 3 models.
    Returns: models (dict), X_test, y_test, preds (dict name -> predictions)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    models = get_models(tune, X_train, y_train)
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        print(f"[trained] {name}")
    return models, X_test, y_test, preds


def save_models(models):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, name.replace(" ", "_").lower() + ".pkl"))
    print(f"[saved] models written to {MODEL_DIR}/")


def load_models():
    names = ["Decision Tree", "Random Forest", "Logistic Regression"]
    return {n: joblib.load(os.path.join(MODEL_DIR, n.replace(" ", "_").lower() + ".pkl"))
            for n in names}


if __name__ == "__main__":
    from features import build_feature_df
    X, y = build_feature_df("data/phishing_urls.csv")
    models, X_test, y_test, preds = train_models(X, y)
    save_models(models)