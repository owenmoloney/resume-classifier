import os

import joblib

from src.preprocess import preprocess_data, preprocess_two_domains_shared_vectorizer
from src.model import train_model, predict
from src.evaluate import evaluate_and_save
from src.rank import rank_postings


def main() -> None:
    print("Starting Resume + Job Posting pipeline...")

    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("results", "models"), exist_ok=True)
    model_dir = os.path.join("results", "models")

    # 1) Load
    # Import here to ensure env vars in data_load are applied before kaggle is imported
    from src.data_load import load_and_filter_dataset
    df = load_and_filter_dataset()

    # Optional: load labeled job postings dataset if provided locally.
    postings_df = None
    try:
        from src.job_data_load import load_job_postings_from_local_zip

        postings_df = load_job_postings_from_local_zip()
    except FileNotFoundError:
        print(
            "No labeled job-postings dataset found (skipping posting model training). "
            "Expected a CSV inside `data/job_postings/archive.zip` or `job_postings/archive.zip`."
        )
    except Exception as e:
        print(f"Job-postings dataset found but failed to load: {e}")

    if postings_df is None:
        # Resume-only fallback (original behavior)
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(df)
        model = train_model(X_train, y_train)
        y_pred = predict(model, X_test)
        evaluate_and_save(y_test, y_pred, label_encoder, output_dir="results", filename="confusion_matrix_resumes.png")

        # Persist artifacts so inference can run without retraining.
        joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(model, os.path.join(model_dir, "resume_model_nb.joblib"))
        joblib.dump(label_encoder, os.path.join(model_dir, "resume_label_encoder.joblib"))
        print(f"Saved resume model/vectorizer to {model_dir}")

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
        return

    # 2) Preprocess with shared vectorizer for cross-domain robustness
    (
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
        shared_vectorizer,
        cleaned_resume_df,
        cleaned_posting_df,
    ) = preprocess_two_domains_shared_vectorizer(
        resume_df=df,
        posting_df=postings_df,
        resume_text_col="Resume_str",
        posting_text_col="Posting_str",
        label_col="Category",
        max_features=1000,
        test_size=0.2,
        random_state=42,
    )

    # 3) Train two models
    resume_model = train_model(Xr_train, yr_train)
    posting_model = train_model(Xp_train, yp_train)

    # 4) Evaluate both
    y_resume_pred = predict(resume_model, Xr_test)
    evaluate_and_save(
        yr_test,
        y_resume_pred,
        resume_le,
        output_dir="results",
        filename="confusion_matrix_resumes.png",
    )

    y_posting_pred = predict(posting_model, Xp_test)
    evaluate_and_save(
        yp_test,
        y_posting_pred,
        posting_le,
        output_dir="results",
        filename="confusion_matrix_postings.png",
    )

    # 5) Persist artifacts for "new unseen resumes/postings" across runs
    joblib.dump(shared_vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(resume_model, os.path.join(model_dir, "resume_model_nb.joblib"))
    joblib.dump(resume_le, os.path.join(model_dir, "resume_label_encoder.joblib"))
    joblib.dump(posting_model, os.path.join(model_dir, "posting_model_nb.joblib"))
    joblib.dump(posting_le, os.path.join(model_dir, "posting_label_encoder.joblib"))
    print(f"Saved models/vectorizer to {model_dir}")

    # 6) Ranking job postings from job_postings/ using the posting classifier for category.
    rank_postings(
        df_all=cleaned_resume_df,
        vectorizer=shared_vectorizer,
        model=resume_model,
        label_encoder=resume_le,
        postings_dir="job_postings",
        results_dir="results",
        top_k=10,
        posting_model=posting_model,
        posting_label_encoder=posting_le,
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
