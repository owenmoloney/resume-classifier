import os
from dotenv import load_dotenv

# MUST load .env and set credentials before importing kaggle
load_dotenv()
_kaggle_username = os.getenv("KAGGLE_USERNAME")
_kaggle_key = os.getenv("KAGGLE_KEY")
if _kaggle_username:
    os.environ["KAGGLE_USERNAME"] = _kaggle_username
if _kaggle_key:
    os.environ["KAGGLE_KEY"] = _kaggle_key

# Now import kaggle AFTER credentials are set
import kaggle
import pandas as pd
import glob

def load_and_filter_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the resume dataset from disk if already present; otherwise download it via Kaggle.

    Set force_download=True to re-download and re-unzip even if a CSV exists locally.
    """
    def _has_resume_columns(path: str) -> bool:
        try:
            cols = set(pd.read_csv(path, nrows=0).columns.astype(str))
            return {"Resume_str", "Category"}.issubset(cols)
        except Exception:
            return False

    matches = glob.glob("data/**/*.csv", recursive=True)
    resume_csvs = [m for m in matches if _has_resume_columns(m)]

    if resume_csvs and not force_download:
        print("Found existing resume CSV under data/ (skipping Kaggle download).")
    else:
        print("Authenticating with Kaggle...")
        kaggle.api.authenticate()

        print("Downloading dataset...")
        kaggle.api.dataset_download_files(
            "snehaanbhawal/resume-dataset",
            path="data/",
            unzip=True,
            force=force_download,
        )
    # Inspect downloaded contents and locate CSV recursively
    try:
        print("Contents of data/:", os.listdir("data"))
    except FileNotFoundError:
        print("data/ directory not found; creating it now.")
        os.makedirs("data", exist_ok=True)
        print("Contents of data/:", os.listdir("data"))

    matches = glob.glob("data/**/*.csv", recursive=True)
    preferred_names = {"resume.csv", "updatedresumedataset.csv"}
    preferred = [m for m in matches if os.path.basename(m).lower() in preferred_names]

    csv_path = None
    for candidate in (preferred + matches):
        if candidate and _has_resume_columns(candidate):
            csv_path = candidate
            break

    if not csv_path:
        raise FileNotFoundError(
            "No resume CSV found under data/ containing columns 'Resume_str' and 'Category'. "
            f"CSV candidates found: {len(matches)}"
        )

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} resumes")

    # Diagnose available categories before filtering
    print("Available categories (top 30):")
    try:
        print(df["Category"].value_counts().head(30))
    except Exception:
        print("Warning: 'Category' column not found in the dataset.")

    # Normalize categories to improve matching with selected targets
    def normalize_cat(s: str) -> str:
        return (str(s).strip().lower()
                .replace("-", " ")
                .replace("_", " "))

    # Desired categories (raw) and normalized variants
    selected_raw = [
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
    ]
    selected_norm = [normalize_cat(s) for s in selected_raw]

    # Create a normalized helper column and filter
    if "Category" not in df.columns:
        raise ValueError("Expected 'Category' column not found in dataset.")
    df["__cat_norm__"] = df["Category"].apply(normalize_cat)
    df_filtered = df[df["__cat_norm__"].isin(selected_norm)]

    if df_filtered.empty:
        print("No exact normalized matches for the selected categories.")
        print("Here are the top categories present in the dataset:")
        print(df["Category"].value_counts().head(30))
        raise ValueError(
            "Selected categories not present in dataset. "
            "Update the category list or provide a mapping from dataset categories "
            "to the five target groups."
        )

    # Keep only required columns and drop nulls
    df = df_filtered[["Resume_str", "Category"]].dropna()
    print(f"Filtered to {len(df)} resumes across {df['Category'].nunique()} categories")
    print(f"Category counts after filter:\n{df['Category'].value_counts()}")
    
    return df
