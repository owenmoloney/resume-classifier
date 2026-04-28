import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
 
from src.data_load import load_and_filter_dataset
from src.preprocess import clean_text
 
 
@dataclass(frozen=True)
class ExperimentResult:
    name: str
    f1_macro_mean: float
    f1_macro_std: float
    acc_mean: float
    acc_std: float
 
 
def _as_markdown_table(rows: Iterable[ExperimentResult]) -> str:
    header = "| Model | CV macro-F1 (mean ± std) | CV accuracy (mean ± std) |"
    sep = "|---|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.f1_macro_mean:.4f} ± {r.f1_macro_std:.4f} | {r.acc_mean:.4f} ± {r.acc_std:.4f} |"
        )
    return "\n".join(lines)
 
 
def _build_text_features(
    df: pd.DataFrame,
    text_col: str = "Resume_str",
    label_col: str = "Category",
) -> Tuple[np.ndarray, np.ndarray]:
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns {text_col!r} and {label_col!r}. Found: {list(df.columns)}")
 
    texts = df[text_col].astype(str).fillna("").tolist()
    y = df[label_col].astype(str).fillna("").to_numpy()
    x = np.array([clean_text(t) for t in texts], dtype=object)
    return x, y
 
 
def _cv_eval(
    name: str,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int,
    ngram_range: Tuple[int, int],
    folds: int,
    random_state: int,
) -> ExperimentResult:
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                ),
            ),
            ("clf", model),
        ]
    )
 
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipe,
        x,
        y,
        cv=cv,
        scoring={"f1_macro": "f1_macro", "acc": "accuracy"},
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
 
    return ExperimentResult(
        name=name,
        f1_macro_mean=float(np.mean(scores["test_f1_macro"])),
        f1_macro_std=float(np.std(scores["test_f1_macro"], ddof=1)),
        acc_mean=float(np.mean(scores["test_acc"])),
        acc_std=float(np.std(scores["test_acc"], ddof=1)),
    )
 
 
def run_experiments(
    *,
    folds: int = 5,
    random_state: int = 42,
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 1),
) -> List[ExperimentResult]:
    df = load_and_filter_dataset()
    x, y = _build_text_features(df)
 
    results: List[ExperimentResult] = []
 
    # Baselines (common, defensible text classifiers)
    results.append(
        _cv_eval(
            "LogisticRegression",
            LogisticRegression(max_iter=3000, n_jobs=-1),
            x,
            y,
            max_features=max_features,
            ngram_range=ngram_range,
            folds=folds,
            random_state=random_state,
        )
    )
    results.append(
        _cv_eval(
            "LinearSVC",
            LinearSVC(),
            x,
            y,
            max_features=max_features,
            ngram_range=ngram_range,
            folds=folds,
            random_state=random_state,
        )
    )
 
    # Tiny tune: NB alpha
    for a in (0.1, 0.5, 1.0, 2.0):
        results.append(
            _cv_eval(
                f"MultinomialNB (alpha={a})",
                MultinomialNB(alpha=float(a)),
                x,
                y,
                max_features=max_features,
                ngram_range=ngram_range,
                folds=folds,
                random_state=random_state,
            )
        )
 
    # Sort by macro-F1, then accuracy
    results = sorted(results, key=lambda r: (r.f1_macro_mean, r.acc_mean), reverse=True)
    return results
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Model comparison via cross-validation (resume dataset).")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=1000)
    ap.add_argument("--bigrams", action="store_true", help="Use unigram+bigrams (ngram_range=(1,2)).")
    args = ap.parse_args()
 
    ngram_range = (1, 2) if args.bigrams else (1, 1)
    results = run_experiments(
        folds=args.folds,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_range=ngram_range,
    )
 
    print("\n## Cross-validated model comparison")
    print(f"- folds={args.folds}, random_state={args.random_state}, max_features={args.max_features}, ngram_range={ngram_range}")
    print()
    print(_as_markdown_table(results))
    print()
    best = results[0]
    print(f"Best by macro-F1: {best.name} (macro-F1={best.f1_macro_mean:.4f}, acc={best.acc_mean:.4f})")
 
 
if __name__ == "__main__":
    main()
