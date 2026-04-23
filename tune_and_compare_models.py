import os

import cv2
import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.applications.resnet50 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svm import SVM as CustomSVM


# Custom SVM hyperparameters (gradient-descent SVM in svm.py)
CUSTOM_SVM_LR = 0.00001
CUSTOM_SVM_LAMBDA = 0.6
CUSTOM_SVM_ITERS = 1000

DATA_DIR = "./Art"
CLASSES = ["RealArt", "AIArtData"]
FEATURE_PATH = "extracted_features/X_features.npy"
LABEL_PATH = "extracted_features/y_labels.npy"
CNN_MODEL_PATH = "saved_models/ai_art_classifier_resnet.keras"
OUTPUT_DIR = "reports"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = "saved_models"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def collect_paths_and_labels(data_dir):
    """Same iteration order as feature_extraction.py (only rows that load as images)."""
    paths = []
    y_list = []
    for label, class_label in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_label)
        if not os.path.exists(class_path):
            continue
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            paths.append(img_path)
            y_list.append(label)
    return np.array(paths), np.array(y_list)


def load_data():
    if not (os.path.exists(FEATURE_PATH) and os.path.exists(LABEL_PATH)):
        raise FileNotFoundError(
            "Feature files not found. Run: python feature_extraction.py first."
        )
    X = np.load(FEATURE_PATH)
    y = np.load(LABEL_PATH)
    paths, y_paths = collect_paths_and_labels(DATA_DIR)
    if len(paths) != len(y) or not np.array_equal(y, y_paths):
        raise ValueError(
            "Feature .npy rows do not match live image scan. "
            "Re-run: python feature_extraction.py"
        )
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        paths[test_idx],
    )


def evaluate_model(name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)

    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    else:
        y_score = estimator.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm.ax_.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{name}.png"), dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"roc_curve_{name}.png"), dpi=160)
    plt.close()

    return metrics


def preprocess_image_bgr(img_bgr):
    """Match app.py: square center crop, 224, ResNet50 preprocess."""
    height, width = img_bgr.shape[:2]
    min_dim = min(height, width)
    start_x = (width // 2) - (min_dim // 2)
    start_y = (height // 2) - (min_dim // 2)
    cropped = img_bgr[start_y : start_y + min_dim, start_x : start_x + min_dim]
    resized = cv2.resize(cropped, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb, axis=0).astype(np.float32)
    return preprocess_input(batch)


def evaluate_cnn(name, model, paths_test, y_test):
    """
    y_test: 0 = RealArt, 1 = AIArtData (same as feature_extraction).
    Keras model + app.py: score > 0.5 => Real Art. So y_pred_feat: Real=0 when score>0.5.
    ROC for positive class AIArt (y=1): use P(AI) = 1 - P(Real score).
    """
    y_scores_real = []
    for p in paths_test:
        img = cv2.imread(p)
        if img is None:
            y_scores_real.append(0.5)
            continue
        x = preprocess_image_bgr(img)
        y_scores_real.append(float(model.predict(x, verbose=0)[0][0]))
    y_scores_real = np.array(y_scores_real, dtype=np.float64)
    y_score_ai = 1.0 - y_scores_real
    y_pred = np.where(y_scores_real > 0.5, 0, 1)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score_ai),
    }

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm.ax_.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{name}.png"), dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_score_ai)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"roc_curve_{name}.png"), dpi=160)
    plt.close()

    return metrics


class _ScaledCustomSVM:
    """StandardScaler + svm.SVM, same API as sklearn for evaluate_model."""

    def __init__(self, scaler, inner):
        self._scaler = scaler
        self._inner = inner

    def predict(self, X):
        return self._inner.predict(self._scaler.transform(X))

    def decision_function(self, X):
        return self._inner.decision_function(self._scaler.transform(X))


def tune_and_train(X_train, y_train):
    results = {}

    svm_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
    )
    svm_grid = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }
    svm_search = GridSearchCV(
        svm_pipe, svm_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    svm_search.fit(X_train, y_train)
    results["svm"] = svm_search

    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    results["random_forest"] = rf_search

    return results


def write_tuning_report(search_results):
    lines = ["# Hyperparameter Tuning Summary", ""]
    for name, search in search_results.items():
        lines.append(f"## {name}")
        lines.append(f"- Best CV F1: {search.best_score_:.4f}")
        lines.append(f"- Best params: {search.best_params_}")
        lines.append("")

    lines.append("## custom_svm")
    lines.append(
        "- Implementation in `svm.py` (gradient-descent SVM), hyperparameters: "
        f"learning_rate={CUSTOM_SVM_LR}, lambda_param={CUSTOM_SVM_LAMBDA}, "
        f"iterations={CUSTOM_SVM_ITERS}, with `StandardScaler` on the same train split."
    )
    lines.append("")

    lines.append("## cnn")
    lines.append(
        "- Not tuned in this script; train with `train_cnn.py` (ResNet50 + head). "
        "Evaluated here on the **same** stratified test image set as SVM / Random Forest."
    )
    lines.append("")

    with open(os.path.join(OUTPUT_DIR, "tuning_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    X_train, X_test, y_train, y_test, paths_test = load_data()
    searches = tune_and_train(X_train, y_train)
    write_tuning_report(searches)

    metrics_rows = []
    for name, search in searches.items():
        best_estimator = search.best_estimator_
        metrics_rows.append(evaluate_model(name, best_estimator, X_test, y_test))
        joblib.dump(best_estimator, os.path.join(MODEL_DIR, f"{name}_best.joblib"))

    scaler_custom = StandardScaler()
    X_train_custom = scaler_custom.fit_transform(X_train)
    custom_inner = CustomSVM(
        learning_rate=CUSTOM_SVM_LR,
        lambda_param=CUSTOM_SVM_LAMBDA,
        iterations=CUSTOM_SVM_ITERS,
    )
    custom_inner.fit(X_train_custom, y_train)
    custom_wrapped = _ScaledCustomSVM(scaler_custom, custom_inner)
    metrics_rows.append(
        evaluate_model("custom_svm", custom_wrapped, X_test, y_test)
    )
    joblib.dump(
        {"scaler": scaler_custom, "svm": custom_inner},
        os.path.join(MODEL_DIR, "custom_svm_best.joblib"),
    )

    if os.path.isfile(CNN_MODEL_PATH):
        model = keras.models.load_model(CNN_MODEL_PATH)
        metrics_rows.append(
            evaluate_cnn("cnn", model, paths_test, y_test)
        )
    else:
        print(
            f"Skipping CNN: {CNN_MODEL_PATH} not found. Train with: python train_cnn.py"
        )

    df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="f1", hue="model", palette="Set3", legend=False)
    plt.title(
        "Model Comparison by F1 Score (sklearn SVM, custom SVM, Random Forest, CNN)"
    )
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison_f1.png"), dpi=160)
    plt.close()

    print("Tuning + comparison complete. See reports/model_comparison.csv.")


if __name__ == "__main__":
    main()
