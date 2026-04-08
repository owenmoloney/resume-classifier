import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# NLTK stopwords for custom cleaning (in addition to TF-IDF's english stopwords)
import nltk
from nltk.corpus import stopwords


def _ensure_nltk_resources() -> None:
    """
    Ensure NLTK resources are available. If missing, attempt to download quietly.
    """
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - Lowercase
    - Remove URLs, punctuation, digits, special characters
    - Remove extra spaces
    - Remove stopwords using NLTK
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords with NLTK
    _ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in text.split() if t not in stop_words]

    return " ".join(tokens)


def preprocess_data(
    df: pd.DataFrame,
    text_col: str = "Resume_str",
    label_col: str = "Category",
    max_features: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, LabelEncoder]:
    """
    Clean text, apply TF-IDF vectorization, label-encode targets, and split into train/test.
    Returns (X_train, X_test, y_train, y_test, vectorizer, label_encoder).
    """
    print("[2/4] Cleaning text...")
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    print("[2/4] Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].values)

    print("[2/4] Splitting into train/test (80/20)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[text_col].values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("[2/4] Vectorizing with TF-IDF (fit on train only, max_features=1000)...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(
        f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, "
        f"Num classes: {len(label_encoder.classes_)}"
    )

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


if __name__ == "__main__":
    # Manual smoke test with a tiny DataFrame
    pd.set_option("display.width", 200)
    toy = pd.DataFrame(
        {
            "Resume_str": [
                "Experienced IT engineer with Python, cloud, and CI/CD.",
                "RN nurse with 5 years in ICU and patient care.",
                "Teacher specializing in math and curriculum design.",
                "Financial analyst skilled in valuation & modeling.",
                "Mechanical engineer with CAD and FEA experience.",
            ],
            "Category": [
                "Information-Technology",
                "Healthcare",
                "Teacher",
                "Finance",
                "Engineering",
            ],
        }
    )
    preprocess_data(toy)
