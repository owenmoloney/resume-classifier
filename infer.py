import argparse
import glob
import os
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from src.preprocess import clean_text


def _read_text_arg(text: Optional[str], text_file: Optional[str]) -> str:
    if text and text_file:
        raise SystemExit("Provide only one of --text or --text-file.")
    if text_file:
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if text:
        return text
    raise SystemExit("You must provide --text or --text-file.")


def _load_artifacts(model_dir: str) -> Tuple[TfidfVectorizer, MultinomialNB, LabelEncoder, Optional[MultinomialNB], Optional[LabelEncoder]]:
    """
    Load saved artifacts from results/models/.

    Returns:
    - vectorizer
    - resume_model, resume_label_encoder
    - posting_model, posting_label_encoder (optional; may be missing if you trained resume-only)
    """
    vec = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    resume_model = joblib.load(os.path.join(model_dir, "resume_model_nb.joblib"))
    resume_le = joblib.load(os.path.join(model_dir, "resume_label_encoder.joblib"))

    posting_model_path = os.path.join(model_dir, "posting_model_nb.joblib")
    posting_le_path = os.path.join(model_dir, "posting_label_encoder.joblib")
    posting_model = joblib.load(posting_model_path) if os.path.isfile(posting_model_path) else None
    posting_le = joblib.load(posting_le_path) if os.path.isfile(posting_le_path) else None
    return vec, resume_model, resume_le, posting_model, posting_le


def _topk_proba(
    proba: np.ndarray,
    label_encoder: LabelEncoder,
    k: int = 5,
) -> List[Tuple[str, float]]:
    idx = np.argsort(-proba)[:k]
    return [(str(label_encoder.inverse_transform([int(i)])[0]), float(proba[int(i)])) for i in idx]


def _find_resume_csv(data_root: str = "data") -> str:
    matches = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    preferred_names = {"resume.csv", "updatedresumedataset.csv"}
    preferred = [m for m in matches if os.path.basename(m).lower() in preferred_names]
    csv_path = preferred[0] if preferred else (matches[0] if matches else None)
    if not csv_path:
        raise FileNotFoundError(f"No CSV found under '{data_root}/'. Run training once or place the resume CSV there.")
    return csv_path


def _load_resume_df(resume_csv: Optional[str]) -> pd.DataFrame:
    path = resume_csv or _find_resume_csv()
    df = pd.read_csv(path)
    if "Resume_str" not in df.columns or "Category" not in df.columns:
        raise ValueError(f"Resume CSV must contain columns Resume_str and Category. Found: {list(df.columns)}")
    df = df[["Resume_str", "Category"]].dropna()
    return df


def cmd_predict_resume(args: argparse.Namespace) -> None:
    vec, resume_model, resume_le, _posting_model, _posting_le = _load_artifacts(args.model_dir)
    raw = _read_text_arg(args.text, args.text_file)
    cleaned = clean_text(raw)
    x = vec.transform([cleaned])
    proba = resume_model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(resume_le.inverse_transform([pred_idx])[0])
    print(f"predicted_category={pred_label}")
    print(f"top{args.top_k}_proba={_topk_proba(proba, resume_le, k=args.top_k)}")


def cmd_predict_posting(args: argparse.Namespace) -> None:
    vec, _resume_model, _resume_le, posting_model, posting_le = _load_artifacts(args.model_dir)
    if posting_model is None or posting_le is None:
        raise SystemExit(
            "Posting model artifacts not found. Train with the labeled job-postings dataset enabled so "
            "results/models/posting_model_nb.joblib and posting_label_encoder.joblib are created."
        )
    raw = _read_text_arg(args.text, args.text_file)
    cleaned = clean_text(raw)
    x = vec.transform([cleaned])
    proba = posting_model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(posting_le.inverse_transform([pred_idx])[0])
    print(f"predicted_category={pred_label}")
    print(f"top{args.top_k}_proba={_topk_proba(proba, posting_le, k=args.top_k)}")


def cmd_match_posting(args: argparse.Namespace) -> None:
    vec, resume_model, resume_le, posting_model, posting_le = _load_artifacts(args.model_dir)

    raw = _read_text_arg(args.text, args.text_file)
    cleaned_posting = clean_text(raw)
    posting_vec = vec.transform([cleaned_posting])

    # Prefer the posting classifier (domain-matched) when present; otherwise fall back to resume model.
    cat_model = posting_model if posting_model is not None else resume_model
    cat_le = posting_le if posting_le is not None else resume_le
    proba = cat_model.predict_proba(posting_vec)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(cat_le.inverse_transform([pred_idx])[0])

    df = _load_resume_df(args.resume_csv)
    cleaned_resumes = df["Resume_str"].astype(str).apply(clean_text).tolist()
    resume_matrix = vec.transform(cleaned_resumes)

    mask = (df["Category"].astype(str) == pred_label).values
    if mask.sum() == 0:
        # If the predicted category doesn't exist in the resume set, rank across all resumes.
        mask = np.ones(df.shape[0], dtype=bool)

    sub_matrix = resume_matrix[mask]
    sub_df = df[mask].reset_index()

    sims = cosine_similarity(posting_vec, sub_matrix).ravel()
    top_idx = np.argsort(-sims)[: args.top_k]

    print(f"predicted_posting_category={pred_label}")
    print(f"top{args.proba_k}_proba={_topk_proba(proba, cat_le, k=args.proba_k)}")
    print("matches=[")
    for i in top_idx:
        row = sub_df.iloc[int(i)]
        orig_idx = int(row["index"])
        score = float(sims[int(i)])
        snippet = str(row["Resume_str"])[:200].replace("\n", " ")
        cat = str(row["Category"])
        print(f"  {{'resume_index': {orig_idx}, 'score': {score:.4f}, 'category': '{cat}', 'snippet': {snippet!r}}},")
    print("]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference utilities for the resume/job-posting pipeline.")
    p.add_argument("--model-dir", default=os.path.join("results", "models"), help="Directory containing .joblib artifacts.")

    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("predict-resume", help="Predict category for a new resume text.")
    pr.add_argument("--text", default=None)
    pr.add_argument("--text-file", default=None)
    pr.add_argument("--top-k", type=int, default=5)
    pr.set_defaults(func=cmd_predict_resume)

    pp = sub.add_parser("predict-posting", help="Predict category for a new job posting text.")
    pp.add_argument("--text", default=None)
    pp.add_argument("--text-file", default=None)
    pp.add_argument("--top-k", type=int, default=5)
    pp.set_defaults(func=cmd_predict_posting)

    mp = sub.add_parser("match-posting", help="Predict posting category and return top resumes by cosine similarity.")
    mp.add_argument("--text", default=None)
    mp.add_argument("--text-file", default=None)
    mp.add_argument("--resume-csv", default=None, help="Path to resume CSV (must have Resume_str, Category).")
    mp.add_argument("--top-k", type=int, default=5, help="How many resumes to return.")
    mp.add_argument("--proba-k", type=int, default=5, help="How many class probabilities to print.")
    mp.set_defaults(func=cmd_match_posting)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

