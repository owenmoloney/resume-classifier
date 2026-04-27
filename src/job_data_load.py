import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _find_first_excel(root_dir: str) -> Optional[str]:
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".xlsx") or lower.endswith(".xls"):
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


def _normalize_industry(series: pd.Series) -> pd.Series:
    """
    Normalize messy 'industry' strings from LinkedIn exports:
    - collapse internal whitespace/newlines
    - strip
    - if multiple industries are present, keep the first (single-label pipeline)
    """
    s = series.copy()
    s = s.astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.split(",").str[0].str.strip()
    # Treat literal "nan" (from astype(str) on NaN) as missing
    s = s.replace({"nan": ""})
    return s


_RESUME_CATEGORIES_24 = {
    "HR",
    "Designer",
    "Information-Technology",
    "Teacher",
    "Advocate",
    "Business-Development",
    "Healthcare",
    "Fitness",
    "Agriculture",
    "BPO",
    "Sales",
    "Consultant",
    "Digital-Media",
    "Automobile",
    "Chef",
    "Finance",
    "Apparel",
    "Engineering",
    "Accountant",
    "Construction",
    "Public-Relations",
    "Banking",
    "Arts",
    "Aviation",
}


def _load_industry_to_resume_mapping(mapping_csv_path: str) -> Dict[str, str]:
    """
    Load manual industry→resume-category mapping.

    Expected columns:
    - industry_name
    - resume_category
    """
    m = pd.read_csv(mapping_csv_path)
    required = {"industry_name", "resume_category"}
    if not required.issubset(set(map(str, m.columns))):
        raise ValueError(
            f"Mapping file '{mapping_csv_path}' must contain columns {sorted(required)}. "
            f"Found: {list(m.columns)}"
        )

    # Treat empty cells as empty strings (pandas reads blanks as NaN by default).
    m["industry_name"] = m["industry_name"].fillna("").astype(str).str.strip().replace({"nan": ""})
    m["resume_category"] = m["resume_category"].fillna("").astype(str).str.strip().replace({"nan": ""})
    m = m[(m["industry_name"] != "") & (m["resume_category"] != "")]

    invalid = sorted(set(m["resume_category"]) - _RESUME_CATEGORIES_24)
    if invalid:
        raise ValueError(
            "Mapping file contains resume_category values not in the 24 resume categories.\n"
            f"Invalid values: {invalid}\n"
            f"Allowed values: {sorted(_RESUME_CATEGORIES_24)}"
        )

    return dict(zip(m["industry_name"].tolist(), m["resume_category"].tolist()))


def load_job_postings_from_new_dataset(
    base_dir: str = "data/job_postings/new_dataset",
    mapping_csv_path: str = "data/job_postings/industry_to_resume_category.csv",
    min_examples_per_class: int = 10,
) -> pd.DataFrame:
    """
    Load job postings from the multi-table dataset layout:
    - postings.csv (job_id, description, ...)
    - jobs/job_industries.csv (job_id, industry_id)
    - mappings/industries.csv (industry_id, industry_name)

    Then map industry_name → one of the 24 resume categories via a manual mapping CSV.

    Returns DataFrame with:
    - Posting_str
    - Category  (one of the resume categories)
    """
    root = Path(base_dir)
    postings_path = root / "postings.csv"
    job_inds_path = root / "jobs" / "job_industries.csv"
    inds_path = root / "mappings" / "industries.csv"

    if not (postings_path.is_file() and job_inds_path.is_file() and inds_path.is_file()):
        raise FileNotFoundError(
            "New job-postings dataset not found in expected layout under "
            f"'{root.as_posix()}'. Expected files:\n"
            f"- {postings_path.as_posix()}\n"
            f"- {job_inds_path.as_posix()}\n"
            f"- {inds_path.as_posix()}"
        )

    mapping = _load_industry_to_resume_mapping(mapping_csv_path)
    if not mapping:
        raise ValueError(
            f"Industry→resume mapping file '{mapping_csv_path}' has no mappings filled in yet. "
            "Fill the 'resume_category' column for at least some industries and rerun."
        )

    print(f"Loading new job postings dataset from: {root.as_posix()}")
    print("Preparing mapped job_id→Category lookup...")

    # Step 1: industry_id -> industry_name, then industry_name -> resume_category
    inds = pd.read_csv(inds_path, usecols=["industry_id", "industry_name"])
    inds = inds.dropna(subset=["industry_id", "industry_name"])
    inds["industry_name"] = inds["industry_name"].astype(str).str.strip()
    inds = inds[inds["industry_name"].isin(set(mapping.keys()))]
    inds["Category"] = inds["industry_name"].map(mapping)
    inds = inds.dropna(subset=["Category"])
    inds["Category"] = inds["Category"].astype(str).str.strip()
    inds = inds[inds["Category"] != ""]
    inds = inds[["industry_id", "Category"]].drop_duplicates()

    # Step 2: job_id -> industry_id -> Category (many-to-one or one-to-one)
    job_inds = pd.read_csv(job_inds_path, usecols=["job_id", "industry_id"])
    job_to_cat = job_inds.merge(inds, on="industry_id", how="inner")[["job_id", "Category"]].drop_duplicates()
    job_id_set = set(job_to_cat["job_id"].tolist())
    print(f"Mapped industries cover {len(job_id_set)} job_ids. Streaming postings.csv...")

    # Step 3: stream postings.csv and keep only job_ids we have a mapped category for
    chunks = []
    for chunk in pd.read_csv(postings_path, usecols=["job_id", "description"], chunksize=50_000):
        sub = chunk[chunk["job_id"].isin(job_id_set)]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        raise ValueError(
            "After applying the industry→resume mapping, no postings remained. "
            "Fill more mappings in 'industry_to_resume_category.csv' and rerun."
        )
    postings = pd.concat(chunks, ignore_index=True)

    df = postings.merge(job_to_cat, on="job_id", how="inner")
    df = df.dropna(subset=["description", "Category"])
    df["Posting_str"] = df["description"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[(df["Posting_str"] != "") & (df["Category"] != "")]
    df = df[["Posting_str", "Category"]]

    if min_examples_per_class > 1:
        counts = df["Category"].value_counts()
        keep = counts[counts >= min_examples_per_class].index
        before = len(df)
        df = df[df["Category"].isin(keep)].copy()
        dropped = before - len(df)
        if dropped:
            print(
                f"Dropped {dropped} postings from rare mapped categories "
                f"(< {min_examples_per_class} examples). Remaining categories: {df['Category'].nunique()}"
            )

    print(f"Loaded {len(df)} job postings across {df['Category'].nunique()} mapped categories")
    return df


def load_job_postings_from_local_zip(
    zip_path: str = "data/job_postings/archive.zip",
    extract_dir: str = "data/job_postings/extracted",
    min_examples_per_class: int = 10,
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
    base_dir = Path(__file__).resolve().parents[1]  # .../resume-classifier
    # Prefer the new multi-table dataset + manual mapping when present.
    new_dataset_dir = base_dir / "data" / "job_postings" / "new_dataset"
    mapping_csv = base_dir / "data" / "job_postings" / "industry_to_resume_category.csv"
    if new_dataset_dir.is_dir() and (new_dataset_dir / "postings.csv").is_file():
        return load_job_postings_from_new_dataset(
            base_dir=new_dataset_dir.as_posix(),
            mapping_csv_path=mapping_csv.as_posix(),
            min_examples_per_class=min_examples_per_class,
        )

    # Resolve relative paths robustly no matter where the script is launched from.
    zip_candidates = [
        Path(zip_path),
        base_dir / zip_path,
        Path("job_postings") / "archive.zip",
        base_dir / "job_postings" / "archive.zip",
        base_dir / "data" / "job_postings" / "archive.zip",
    ]

    zip_found = next((p for p in zip_candidates if p.is_file()), None)
    if not zip_found:
        checked = "\n".join(f"- {p.as_posix()}" for p in zip_candidates)
        raise FileNotFoundError(
            "No labeled job-postings archive.zip found. Checked:\n"
            f"{checked}\n\n"
            "Place your downloaded Kaggle 'archive.zip' under `resume-classifier/job_postings/` "
            "or `resume-classifier/data/job_postings/` (or pass zip_path explicitly)."
        )

    zip_path = zip_found.as_posix()

    extract_dir_path = Path(extract_dir)
    if not extract_dir_path.is_absolute():
        # Keep extracted data colocated with the resume-classifier project by default.
        extract_dir_path = base_dir / extract_dir_path
    extract_dir = extract_dir_path.as_posix()

    if not os.path.isdir(extract_dir) or not (_find_first_csv(extract_dir) or _find_first_excel(extract_dir)):
        print(f"Extracting job postings dataset from {zip_path} ...")
        _extract_zip(zip_path, extract_dir)

    csv_path = _find_first_csv(extract_dir)
    excel_path = _find_first_excel(extract_dir) if not csv_path else None
    if not csv_path and not excel_path:
        raise FileNotFoundError(
            f"No CSV/XLSX found after extracting '{zip_path}' into '{extract_dir}'."
        )

    if csv_path:
        print(f"Loading job postings CSV: {csv_path}")
        df_raw = pd.read_csv(csv_path)
    else:
        print(f"Loading job postings Excel: {excel_path}")
        try:
            df_raw = pd.read_excel(excel_path)  # type: ignore[call-arg]
        except ImportError as e:
            raise ImportError(
                "Reading .xlsx requires an Excel engine (usually 'openpyxl'). "
                "Install it with `pip install openpyxl` and rerun."
            ) from e

    # Prefer using 'industry' as the label when present (matches our pipeline intent).
    cols = list(df_raw.columns)
    lower_to_orig = {str(c).strip().lower(): c for c in cols}
    industry_col = lower_to_orig.get("industry")

    text_col = None
    if industry_col:
        # Use an explicit text-like column if available; otherwise synthesize from metadata.
        for candidate in [
            "description",
            "job_description",
            "jobdescription",
            "text",
            "content",
            "requirements",
            "responsibilities",
            "job_posting",
            "posting",
        ]:
            if candidate in lower_to_orig:
                text_col = lower_to_orig[candidate]
                break

        if text_col:
            df = df_raw[[text_col, industry_col]].rename(columns={text_col: "Posting_str", industry_col: "Category"})
        else:
            text_parts_candidates = [
                "job_title",
                "job_function",
                "employment_type",
                "seniority_level",
                "company_name",
                "location",
            ]
            text_cols = [lower_to_orig[c] for c in text_parts_candidates if c in lower_to_orig]
            if not text_cols:
                raise ValueError(
                    "Job postings dataset has 'industry' but no usable text columns to build Posting_str.\n"
                    f"Columns present: {cols}"
                )

            df = df_raw[text_cols + [industry_col]].copy()
            df["Posting_str"] = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
            df["Category"] = _normalize_industry(df[industry_col])
            df = df[["Posting_str", "Category"]]
    else:
        # Generic fallback for other labeled datasets.
        text_col, label_col = _pick_columns(df_raw)
        df = df_raw[[text_col, label_col]].rename(columns={text_col: "Posting_str", label_col: "Category"})

    df = df.dropna(subset=["Posting_str", "Category"])
    df["Posting_str"] = df["Posting_str"].astype(str)
    df["Category"] = df["Category"].astype(str)
    df["Category"] = df["Category"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["Category"] != ""]

    if min_examples_per_class > 1:
        counts = df["Category"].value_counts()
        keep = counts[counts >= min_examples_per_class].index
        before = len(df)
        df = df[df["Category"].isin(keep)].copy()
        after = len(df)
        dropped = before - after
        if dropped:
            print(
                f"Dropped {dropped} postings from rare categories "
                f"(< {min_examples_per_class} examples). Remaining categories: {df['Category'].nunique()}"
            )

    print(f"Loaded {len(df)} job postings across {df['Category'].nunique()} categories")
    return df

