"""
Multiclass Classification on PathMNIST
=======================================

This script compares three classic multiclass classification approaches on the
PathMNIST dataset (colorectal histology patches, 9 tissue classes):

    1. SGDClassifier  - linear model, trained with stochastic gradient descent.
                        Assigns one weight per pixel per class; the decision is
                        just a weighted sum of pixel intensities. Fast, but
                        blind to spatial/shape patterns.

    2. SVC (RBF)      - kernel SVM. Scikit-Learn auto-applies One-vs-One (OvO)
                        for multiclass problems. Can capture non-linear
                        boundaries, but expensive on high-dimensional raw
                        pixels, so PCA is used to speed it up.

    3. RandomForest    - ensemble of decision trees (bagging). Naturally
                        handles multiclass and non-linear feature interactions
                        without needing OvR/OvO.

For each model we report accuracy, a full classification report, a confusion
matrix, and cross-validation scores. At the end, we run an error analysis
(confusion matrix heatmaps + visual inspection of the most-confused class
pair) for the best-performing model, following the same methodology as the
"Error Analysis" section in Hands-On Machine Learning (Aurélien Géron).

Dataset: https://medmnist.com/  (PathMNIST)
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import medmnist
from medmnist import INFO

from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

# ============================================================
# Configuration
# ============================================================

DATA_FLAG = "pathmnist"
CV_FOLDS = 3
RANDOM_STATE = 42

# Speed/accuracy trade-off knobs.
# Increase these for better accuracy at the cost of runtime.
TRAIN_SAMPLE_SIZE = 10_000   # subsample of the ~90,000 training images
PCA_COMPONENTS = 500          # dimensionality reduction for the SVM pipeline
RUN_CROSS_VALIDATION = True

CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}
N_CLASSES = len(CLASS_NAMES)
CLASS_LABELS = list(CLASS_NAMES.values())


# ============================================================
# Data loading & preparation
# ============================================================

def load_splits():
    info = INFO[DATA_FLAG]
    data_class = getattr(medmnist, info["python_class"])

    train_dataset = data_class(split="train", download=True)
    val_dataset = data_class(split="val", download=True)
    test_dataset = data_class(split="test", download=True)

    return train_dataset, val_dataset, test_dataset


def prepare_dataset(dataset):
    """Flatten each image to a 1D float32 vector and collect integer labels."""
    X, y = [], []
    for img, label in dataset:
        X.append(np.asarray(img, dtype=np.float32).reshape(-1))
        y.append(int(label[0]))
    return np.asarray(X), np.asarray(y)


def print_dataset_info(X, y, split_name):
    print("=" * 60)
    print(split_name.upper())
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(Counter(y))
    print()


def subsample(X, y, n_samples, random_state=RANDOM_STATE):
    """Randomly subsample the dataset to speed up training."""
    if n_samples >= len(X):
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], y[idx]


# ============================================================
# Model evaluation helper
# ============================================================

def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test,
                    run_cv=True):
    """Fit a pipeline, print metrics, and return predictions + accuracy."""
    print("\n" + "#" * 60)
    print(f"# {name}")
    print("#" * 60)

    start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Training time: {fit_time:.1f}s")
    print(f"Test accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=CLASS_LABELS, digits=4
    ))

    conf_mx = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(conf_mx)

    if run_cv:
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=CV_FOLDS, scoring="accuracy", n_jobs=-1,
        )
        print(f"\nMean CV accuracy: {cv_scores.mean():.4f}  (folds: {cv_scores})")

    return {
        "name": name,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "accuracy": acc,
        "confusion_matrix": conf_mx,
    }


# ============================================================
# Error analysis (confusion matrix heatmaps + example images)
# ============================================================

def plot_confusion_matrices(conf_mx, model_name):
    """Plot raw and row-normalized (errors-only) confusion matrices."""

    # Raw counts
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        conf_mx, annot=True, fmt="d", cmap="gray_r",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} - Confusion Matrix (raw counts)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_raw_{model_name}.png", dpi=150)
    plt.show()

    # Row-normalized, diagonal zeroed -> highlights error rates only
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        norm_conf_mx, annot=True, fmt=".2f", cmap="gray_r",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} - Confusion Matrix (errors only, row-normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_errors_{model_name}.png", dpi=150)
    plt.show()


def find_worst_confused_pair(conf_mx):
    """Return the class pair (i, j) with the highest combined off-diagonal
    confusion (i predicted as j, plus j predicted as i)."""
    errors_only = conf_mx.copy().astype(float)
    np.fill_diagonal(errors_only, 0)

    worst_pair, worst_value = None, -1
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            combined = errors_only[i, j] + errors_only[j, i]
            if combined > worst_value:
                worst_value = combined
                worst_pair = (i, j)
    return worst_pair, int(worst_value)


def plot_digits(images, images_per_row=5, title=""):
    """Render a grid of flattened 28x28x3 PathMNIST images."""
    n_images = len(images)
    if n_images == 0:
        plt.title(f"{title}\n(no samples)")
        plt.axis("off")
        return

    n_rows = int(np.ceil(n_images / images_per_row))
    grid = np.ones((n_rows * 28, images_per_row * 28, 3), dtype=np.uint8) * 255

    for idx, img_flat in enumerate(images[: n_rows * images_per_row]):
        img = img_flat.reshape(28, 28, 3).astype(np.uint8)
        r, c = divmod(idx, images_per_row)
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28, :] = img

    plt.imshow(grid)
    plt.title(title)
    plt.axis("off")


def plot_worst_pair_examples(X_test, y_test, y_pred, cl_a, cl_b, model_name):
    """
    Show 4 blocks of example images for the most-confused class pair:
    correctly classified A, A misclassified as B, B misclassified as A,
    correctly classified B. Mirrors the 3-vs-5 error inspection in
    Hands-On ML, adapted to PathMNIST's RGB tissue images.
    """
    X_aa = X_test[(y_test == cl_a) & (y_pred == cl_a)]
    X_ab = X_test[(y_test == cl_a) & (y_pred == cl_b)]
    X_ba = X_test[(y_test == cl_b) & (y_pred == cl_a)]
    X_bb = X_test[(y_test == cl_b) & (y_pred == cl_b)]

    plt.figure(figsize=(9, 9))

    plt.subplot(221)
    plot_digits(X_aa[:25],
                title=f"True: {CLASS_NAMES[cl_a]}\nPred: {CLASS_NAMES[cl_a]} (correct)")

    plt.subplot(222)
    plot_digits(X_ab[:25],
                title=f"True: {CLASS_NAMES[cl_a]}\nPred: {CLASS_NAMES[cl_b]} (WRONG)")

    plt.subplot(223)
    plot_digits(X_ba[:25],
                title=f"True: {CLASS_NAMES[cl_b]}\nPred: {CLASS_NAMES[cl_a]} (WRONG)")

    plt.subplot(224)
    plot_digits(X_bb[:25],
                title=f"True: {CLASS_NAMES[cl_b]}\nPred: {CLASS_NAMES[cl_b]} (correct)")

    plt.tight_layout()
    plt.savefig(f"error_analysis_worst_pair_{model_name}.png", dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    train_dataset, val_dataset, test_dataset = load_splits()

    X_train, y_train = prepare_dataset(train_dataset)
    X_val, y_val = prepare_dataset(val_dataset)
    X_test, y_test = prepare_dataset(test_dataset)

    print_dataset_info(X_train, y_train, "Train (full)")
    print_dataset_info(X_val, y_val, "Validation")
    print_dataset_info(X_test, y_test, "Test")

    X_train, y_train = subsample(X_train, y_train, TRAIN_SAMPLE_SIZE)
    print_dataset_info(X_train, y_train, f"Train (subsampled to {TRAIN_SAMPLE_SIZE})")

    # --------------------------------------------------------
    # Model 1: SGDClassifier (linear model)
    # --------------------------------------------------------
    # A linear model: score = sum(weight_i * pixel_i) + bias, per class.
    # Trained with stochastic gradient descent (one sample/mini-batch at a
    # time), which makes it very fast even on large datasets. It has no
    # notion of shape or spatial structure -- it only weighs raw pixel
    # intensities -- so it tends to struggle when classes differ mainly in
    # spatial arrangement rather than overall pixel intensity.
    sgd_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDClassifier(random_state=RANDOM_STATE, max_iter=1000, tol=1e-3)),
    ])
    sgd_result = evaluate_model(
        "SGDClassifier (linear)", sgd_pipeline,
        X_train, y_train, X_test, y_test, run_cv=RUN_CROSS_VALIDATION,
    )

    # --------------------------------------------------------
    # Model 2: SVC with RBF kernel
    # --------------------------------------------------------
    # SVC is a binary classifier; Scikit-Learn automatically applies the
    # One-vs-One (OvO) strategy for multiclass problems, training one
    # classifier per pair of classes. The RBF kernel can capture non-linear
    # decision boundaries, but is expensive on high-dimensional inputs, so
    # PCA is used to reduce the raw pixel space before training.
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ("svm", SVC(random_state=RANDOM_STATE)),
    ])
    svm_result = evaluate_model(
        "SVC (RBF kernel + PCA)", svm_pipeline,
        X_train, y_train, X_test, y_test, run_cv=RUN_CROSS_VALIDATION,
    )

    # --------------------------------------------------------
    # Model 3: Random Forest
    # --------------------------------------------------------
    # An ensemble of decision trees (bagging). Natively supports multiclass
    # classification -- no OvR/OvO needed -- and can model non-linear
    # feature interactions without any dimensionality reduction.
    rf_pipeline = Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    rf_result = evaluate_model(
        "RandomForestClassifier", rf_pipeline,
        X_train, y_train, X_test, y_test, run_cv=RUN_CROSS_VALIDATION,
    )

    # --------------------------------------------------------
    # Model comparison summary
    # --------------------------------------------------------
    results = [sgd_result, svm_result, rf_result]
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for r in sorted(results, key=lambda r: r["accuracy"], reverse=True):
        print(f"{r['name']:35s}  accuracy = {r['accuracy']:.4f}")

    # --------------------------------------------------------
    # Error analysis for the best model
    # --------------------------------------------------------
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nRunning error analysis for the best model: {best['name']}")

    model_tag = best["name"].split()[0]  # e.g. "SGDClassifier", "SVC", "RandomForestClassifier"
    plot_confusion_matrices(best["confusion_matrix"], model_tag)

    cl_a, cl_b = find_worst_confused_pair(best["confusion_matrix"])[0]
    worst_value = find_worst_confused_pair(best["confusion_matrix"])[1]
    print(f"Biggest mutual confusion pair: "
          f"{CLASS_NAMES[cl_a]} (class {cl_a}) <-> {CLASS_NAMES[cl_b]} (class {cl_b}) "
          f"-> {worst_value} total confused samples")

    plot_worst_pair_examples(
        X_test, y_test, best["y_pred"], cl_a, cl_b, model_tag
    )


if __name__ == "__main__":
    main()