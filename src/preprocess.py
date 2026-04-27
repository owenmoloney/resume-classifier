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


_STOP_WORDS: set[str] | None = None


def _ensure_nltk_resources() -> None:
    """
    Ensure NLTK resources are available. If missing, attempt to download quietly.
    """
    try:
        _ = stopwords.words("english")
    except LookupError:
        # Avoid hanging on environments without network access.
        # We'll fall back to TF-IDF's built-in english stopwords.
        return


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

    # Remove stopwords with NLTK (cached for performance)
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _ensure_nltk_resources()
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            _STOP_WORDS = set()
    tokens = [t for t in text.split() if t not in _STOP_WORDS]

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


def preprocess_two_domains_shared_vectorizer(
    resume_df: pd.DataFrame,
    posting_df: pd.DataFrame,
    resume_text_col: str = "Resume_str",
    posting_text_col: str = "Posting_str",
    label_col: str = "Category",
    max_features: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
    TfidfVectorizer,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Train/test split + label encoding for resumes and postings separately, but use ONE shared TF-IDF
    vectorizer fitted on the union of *training* texts (resumes + postings).

    This enables:
    - Two separate classifiers (resume/posting) that operate in the same feature space
    - Cross-domain cosine similarity (posting vector vs resume vectors)

    Returns:
    - Xr_train, Xr_test, yr_train, yr_test, resume_label_encoder
    - Xp_train, Xp_test, yp_train, yp_test, posting_label_encoder
    - shared_vectorizer
    - cleaned_resume_df, cleaned_posting_df (with cleaned text columns)
    """
    print("[2/6] Cleaning resume and posting text...")
    resume_df = resume_df.copy()
    posting_df = posting_df.copy()
    resume_df[resume_text_col] = resume_df[resume_text_col].astype(str).apply(clean_text)
    posting_df[posting_text_col] = posting_df[posting_text_col].astype(str).apply(clean_text)

    print("[2/6] Encoding labels (separately for each domain)...")
    resume_le = LabelEncoder()
    posting_le = LabelEncoder()
    yr = resume_le.fit_transform(resume_df[label_col].astype(str).values)
    yp = posting_le.fit_transform(posting_df[label_col].astype(str).values)

    print("[2/6] Splitting resumes and postings into train/test...")
    r_train_text, r_test_text, yr_train, yr_test = train_test_split(
        resume_df[resume_text_col].values,
        yr,
        test_size=test_size,
        random_state=random_state,
        stratify=yr,
    )
    p_train_text, p_test_text, yp_train, yp_test = train_test_split(
        posting_df[posting_text_col].values,
        yp,
        test_size=test_size,
        random_state=random_state,
        stratify=yp,
    )

    print("[2/6] Vectorizing with shared TF-IDF (fit on BOTH train splits only)...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    combined_train_text = np.concatenate([r_train_text, p_train_text])
    vectorizer.fit(combined_train_text)

    Xr_train = vectorizer.transform(r_train_text)
    Xr_test = vectorizer.transform(r_test_text)
    Xp_train = vectorizer.transform(p_train_text)
    Xp_test = vectorizer.transform(p_test_text)

    print(
        "Resume train/test shapes: "
        f"{Xr_train.shape}/{Xr_test.shape}, classes={len(resume_le.classes_)}"
    )
    print(
        "Posting train/test shapes: "
        f"{Xp_train.shape}/{Xp_test.shape}, classes={len(posting_le.classes_)}"
    )

    return (
        Xr_train,
        Xr_test,
        yr_train,
        yr_test,
        resume_le,
        Xp_train,
        Xp_test,
        yp_train,
        yp_test,
        posting_le,
        vectorizer,
        resume_df,
        posting_df,
    )