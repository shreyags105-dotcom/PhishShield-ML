"""features.py  (Shreya) - URL feature extraction.
Label convention used by the WHOLE project: 1 = phishing, 0 = safe.
"""
import re
from urllib.parse import urlparse

import pandas as pd

FEATURE_COLUMNS = ["url_length", "num_dots", "has_at", "is_https", "has_ip", "has_keyword"]

SUSPICIOUS_KEYWORDS = ["login", "verify", "bank", "update", "secure", "account",
                       "signin", "confirm", "password", "paypal", "webscr"]

IP_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _hostname(url):
    """Get hostname even if the URL has no http:// in front."""
    parsed = urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def extract_features(url):
    """Return the 6 required features for one URL as a dict."""
    url = str(url).strip()
    host = _hostname(url)
    lower = url.lower()
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "has_at": int("@" in url),
        "is_https": int(lower.startswith("https://")),
        "has_ip": int(bool(IP_REGEX.match(host))),
        "has_keyword": int(any(k in lower for k in SUSPICIOUS_KEYWORDS)),
    }


def _find_column(df, candidates):
    for c in df.columns:
        if c.strip().lower() in candidates:
            return c
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


def _encode_label(value, numeric_one_is_legit=False):
    """Map any common label style to 1 (phishing) / 0 (safe).
    Text labels (good/bad etc.) are mapped by meaning. Numeric labels follow
    numeric_one_is_legit (True for the PhiUSIIL dataset, where 1 = legitimate)."""
    v = str(value).strip().lower()
    if v in {"bad", "phishing", "malicious", "malware", "defacement"}:
        return 1
    if v in {"good", "legitimate", "benign", "safe"}:
        return 0
    if v in {"1", "1.0", "0", "0.0", "-1"}:
        is_one = v in {"1", "1.0"}
        return int(not is_one) if numeric_one_is_legit else int(is_one or v == "-1")
    raise ValueError(f"Unknown label value: {value!r} - edit _encode_label() in features.py")


def build_feature_df(csv_path):
    """Read CSV (url + label columns) -> (X, y) ready for training."""
    df = pd.read_csv(csv_path)
    url_col = _find_column(df, {"url", "urls", "link"})
    label_col = _find_column(df, {"label", "type", "class", "status", "result"})
    df = df.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col])

    # PhiUSIIL (UCI/Kaggle) uses label 1 = legitimate, 0 = phishing. Detect it automatically.
    one_is_legit = "URLSimilarityIndex" in df.columns
    if one_is_legit:
        print("[features] PhiUSIIL dataset detected -> label 1 treated as SAFE, 0 as PHISHING")

    X = pd.DataFrame([extract_features(u) for u in df[url_col]], columns=FEATURE_COLUMNS)
    y = df[label_col].map(lambda v: _encode_label(v, one_is_legit)).reset_index(drop=True)
    return X, y


if __name__ == "__main__":
    X, y = build_feature_df("data/phishing_urls.csv")
    print(X.head(10))
    print("\nShape:", X.shape)
    print("Missing values:\n", X.isnull().sum())
    print("\nLabel counts (1=phishing, 0=safe):\n", y.value_counts())