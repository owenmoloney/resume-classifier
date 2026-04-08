import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from .preprocess import clean_text


def _read_job_postings(postings_dir: str) -> List[Tuple[str, str]]:
    """
    Read all .txt files from postings_dir.
    Returns list of (stem, text).
    """
    if not os.path.isdir(postings_dir):
        return []
    postings: List[Tuple[str, str]] = []
    for name in os.listdir(postings_dir):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(postings_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            postings.append((os.path.splitext(name)[0], text))
        except Exception:
            continue
    return postings


def _predict_posting_category(
    posting_text: str,
    vectorizer: TfidfVectorizer,
    model: MultinomialNB,
    label_encoder: LabelEncoder,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Clean text, transform using fitted vectorizer, and compute predict_proba.
    Returns: (predicted_label, proba_vector, posting_vector)
    """
    cleaned = clean_text(posting_text)
    p_vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(p_vec)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label, proba, p_vec


def _rank_resumes_by_similarity(
    posting_vec,
    resume_matrix,
    resume_df: pd.DataFrame,
    target_category: str,
    top_k: int = 10,
) -> List[Tuple[int, float, str, str]]:
    """
    Rank resumes that match target_category by cosine similarity to posting_vec.
    Returns list of tuples: (row_index, score, category, snippet)
    """
    # Filter by true category
    mask = (resume_df["Category"] == target_category).values
    if mask.sum() == 0:
        # fallback: use all resumes
        mask = np.ones(resume_df.shape[0], dtype=bool)

    sub_matrix = resume_matrix[mask]
    sub_df = resume_df[mask].reset_index()

    # Cosine similarity between posting_vec and all candidate resumes
    sims = cosine_similarity(posting_vec, sub_matrix).ravel()
    top_idx = np.argsort(-sims)[:top_k]

    results: List[Tuple[int, float, str, str]] = []
    for i in top_idx:
        row = sub_df.iloc[i]
        snippet = str(row["Resume_str"])[:200].replace("\n", " ")
        results.append((int(row["index"]), float(sims[i]), str(row["Category"]), snippet))
    return results


def rank_postings(
    df_all: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model: MultinomialNB,
    label_encoder: LabelEncoder,
    postings_dir: str = "job_postings",
    results_dir: str = "results",
    top_k: int = 10,
) -> None:
    """
    Given trained artifacts and full resume DataFrame, rank resumes for each posting.
    - Transforms all resumes with the fitted vectorizer (no fitting here)
    - Reads postings from postings_dir and classifies them via predict_proba
    - Screens resumes by predicted category, ranks by cosine similarity
    - Writes per-posting rankings to results/ranking_<stem>.txt
    """
    os.makedirs(results_dir, exist_ok=True)

    postings = _read_job_postings(postings_dir)
    if not postings:
        print(f"No postings found in '{postings_dir}'. Skipping ranking.")
        return

    # Transform all resumes using the fitted vectorizer (deployment transform)
    all_resume_vecs = vectorizer.transform(df_all["Resume_str"].astype(str).tolist())

    for stem, raw_text in postings:
        print(f"Ranking resumes for posting: {stem}")
        pred_label, proba, p_vec = _predict_posting_category(raw_text, vectorizer, model, label_encoder)

        ranked = _rank_resumes_by_similarity(
            posting_vec=p_vec,
            resume_matrix=all_resume_vecs,
            resume_df=df_all[["Resume_str", "Category"]],
            target_category=pred_label,
            top_k=top_k,
        )

        out_path = os.path.join(results_dir, f"ranking_{stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Posting: {stem}\n")
            f.write(f"Predicted category: {pred_label}\n")
            # Show top-5 class probabilities for transparency
            top5_idx = np.argsort(-proba)[:5]
            top5 = [(label_encoder.inverse_transform([int(i)])[0], float(proba[i])) for i in top5_idx]
            f.write(f"Top-5 class probabilities: {top5}\n\n")
            f.write("Top candidates:\n")
            for rank, (row_index, score, category, snippet) in enumerate(ranked, start=1):
                f.write(f"{rank}. idx={row_index}, score={score:.4f}, category={category}\n")
                f.write(f"   snippet: {snippet}\n")
        print(f"Wrote: {out_path}")
