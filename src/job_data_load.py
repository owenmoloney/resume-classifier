import os
import zipfile
from typing import Optional, Tuple

import pandas as pd


def _extract_zip(zip_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def _find_first_csv(root_dir: str) -> Optional[str]:
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".csv"):
                return os.path.join(base, name)
    return None


def _pick_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    lower_to_orig = {str(c).strip().lower(): c for c in cols}

    text_candidates = [
        "description",
        "job_description",
        "jobdescription",
        "text",
        "content",
        "requirements",
        "responsibilities",
        "job_posting",
        "posting",
    ]
    label_candidates = [
        "category",
        "job_category",
        "industry",
        "function",
        "department",
        "sector",
        "job_type",
        "title",
        "job_title",
    ]

    text_col = next((lower_to_orig[c] for c in text_candidates if c in lower_to_orig), None)
    label_col = next((lower_to_orig[c] for c in label_candidates if c in lower_to_orig), None)

    if not text_col or not label_col:
        raise ValueError(
            "Could not infer text/label columns for job postings dataset.\n"
            f"Columns present: {cols}\n\n"
            "Expected a text-like column (e.g. description/job_description/text) and a label-like column "
            "(e.g. category/industry/function/title)."
        )
    return str(text_col), str(label_col)


def load_job_postings_from_local_zip(
    zip_path: str = "data/job_postings/archive.zip",
    extract_dir: str = "data/job_postings/extracted",
) -> pd.DataFrame:
    """
    Load a labeled job-postings dataset from a local Kaggle-style archive.zip.

    Returns a DataFrame with two columns:
    - Posting_str: posting text
    - Category: label (string)

    Notes:
    - This loader uses heuristics to infer which columns are text vs label.
    - If inference fails, it raises with the dataset's columns so you can choose explicitly.
    """
    if not os.path.isfile(zip_path):
        # Backward-compatible fallback: some users store Kaggle archives under job_postings/
        alt_zip_path = os.path.join("job_postings", "archive.zip")
        if os.path.isfile(alt_zip_path):
            zip_path = alt_zip_path
            extract_dir = os.path.join("job_postings", "extracted_labeled_dataset")
        else:
            raise FileNotFoundError(
                f"Job postings archive not found at '{zip_path}'.\n"
                f"Also checked '{alt_zip_path}'.\n"
                "Place your downloaded Kaggle 'archive.zip' under one of those paths."
            )

    if not os.path.isdir(extract_dir) or not _find_first_csv(extract_dir):
        print(f"Extracting job postings dataset from {zip_path} ...")
        _extract_zip(zip_path, extract_dir)

    csv_path = _find_first_csv(extract_dir)
    if not csv_path:
        raise FileNotFoundError(
            f"No CSV found after extracting '{zip_path}' into '{extract_dir}'."
        )

    print(f"Loading job postings CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    text_col, label_col = _pick_columns(df_raw)
    df = df_raw[[text_col, label_col]].rename(columns={text_col: "Posting_str", label_col: "Category"})
    df = df.dropna(subset=["Posting_str", "Category"])
    df["Posting_str"] = df["Posting_str"].astype(str)
    df["Category"] = df["Category"].astype(str)

    print(f"Loaded {len(df)} job postings across {df['Category'].nunique()} categories")
    return df

