from typing import Optional

import numpy as np
from sklearn.naive_bayes import MultinomialNB


def train_model(X_train, y_train, alpha: float = 1.0) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes classifier.
    """
    print("[3/4] Training Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("[3/4] Training complete.")
    return model


def predict(model: MultinomialNB, X_test) -> np.ndarray:
    """
    Predict labels for test features.
    """
    print("[3/4] Generating predictions on test set...")
    return model.predict(X_test)
