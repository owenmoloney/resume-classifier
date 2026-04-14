import os

from src.preprocess import preprocess_data
from src.model import train_model, predict
from src.evaluate import evaluate_and_save
from src.rank import rank_postings



def main() -> None:
    print("Starting Resume Classifier pipeline...")

    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1) Load
    # Import here to ensure env vars in data_load are applied before kaggle is imported
    from src.data_load import load_and_filter_dataset
    df = load_and_filter_dataset()

    # 2) Preprocess
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(df)

    # 3) Train
    model = train_model(X_train, y_train)

    # 4) Evaluate
    y_pred = predict(model, X_test)
    evaluate_and_save(y_test, y_pred, label_encoder, output_dir="results")

    # 5) Ranking job postings (if any exist under job_postings/)
    rank_postings(
        df_all=df,
        vectorizer=vectorizer,
        model=model,
        label_encoder=label_encoder,
        postings_dir="job_postings",
        results_dir="results",
        top_k=10,
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
