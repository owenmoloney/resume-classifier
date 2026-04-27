import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: str = "results",
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Print accuracy and a full classification report.
    Save a confusion matrix heatmap to results/confusion_matrix.png.
    """
    print("[4/4] Evaluating model...")
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    target_names: List[str] = list(label_encoder.classes_)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("Classification Report:")
    print(report)

    # Confusion matrix (can get extremely large for many classes)
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    n_classes = len(target_names)
    # Heuristic: annotated heatmaps become unusably slow/noisy past ~40 classes.
    if n_classes > 40:
        print(
            f"Skipping annotated confusion matrix for {n_classes} classes (too large). "
            "Saving a non-annotated heatmap without tick labels instead."
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
        plt.title(f"Confusion Matrix ({n_classes} classes)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved to: {out_path}")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {out_path}")


if __name__ == "__main__":
    # Manual smoke test for shape only (not executed in normal run)
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["A", "B", "C"])
    y_t = np.array([0, 1, 2, 1, 0])
    y_p = np.array([0, 1, 1, 1, 0])
    evaluate_and_save(y_t, y_p, le)
