import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


DATA_DIR = "./Art"
FEATURE_PATH = "extracted_features/X_features.npy"
LABEL_PATH = "extracted_features/y_labels.npy"
OUTPUT_DIR = "reports"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CLASSES = ["RealArt", "AIArtData"]


def ensure_output_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def collect_dataset_stats(data_dir):
    stats = {}
    unreadable = {}
    size_samples = {}

    for class_name in CLASSES:
        class_path = os.path.join(data_dir, class_name)
        image_files = []
        bad = 0
        sizes = []

        if not os.path.exists(class_path):
            stats[class_name] = {"total_files": 0, "readable_files": 0}
            unreadable[class_name] = 0
            size_samples[class_name] = sizes
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_files.append(filename)
            path = os.path.join(class_path, filename)
            img = cv2.imread(path)
            if img is None:
                bad += 1
                continue
            h, w = img.shape[:2]
            sizes.append((w, h))

        stats[class_name] = {
            "total_files": len(image_files),
            "readable_files": len(image_files) - bad,
        }
        unreadable[class_name] = bad
        size_samples[class_name] = sizes

    return stats, unreadable, size_samples


def plot_class_counts(stats):
    labels = list(stats.keys())
    values = [stats[k]["readable_files"] for k in labels]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=values, hue=labels, palette="Set2", legend=False)
    plt.title("Class Balance (Readable Images)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_balance.png"), dpi=160)
    plt.close()


def plot_top_image_sizes(size_samples):
    counter = Counter()
    for class_sizes in size_samples.values():
        counter.update(class_sizes)

    common = counter.most_common(10)
    if not common:
        return

    labels = [f"{w}x{h}" for (w, h), _ in common]
    values = [c for _, c in common]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=labels, y=values, hue=labels, palette="viridis", legend=False)
    plt.title("Top 10 Image Resolutions")
    plt.ylabel("Frequency")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "image_size_distribution.png"), dpi=160)
    plt.close()


def load_features():
    if not (os.path.exists(FEATURE_PATH) and os.path.exists(LABEL_PATH)):
        raise FileNotFoundError(
            "Feature files not found. Run: python feature_extraction.py first."
        )
    X = np.load(FEATURE_PATH)
    y = np.load(LABEL_PATH)
    return X, y


def plot_feature_variance(X):
    variances = np.var(X, axis=0)
    plt.figure(figsize=(8, 4))
    sns.histplot(variances, bins=30, kde=True)
    plt.title("Feature Variance Distribution")
    plt.xlabel("Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_variance_hist.png"), dpi=160)
    plt.close()


def plot_correlation_heatmap(X):
    # Correlation on a subset to keep the chart readable.
    subset = X[:, : min(30, X.shape[1])]
    corr = np.corrcoef(subset, rowvar=False)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (first 30 features)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_correlation_heatmap.png"), dpi=160)
    plt.close()


def plot_pca_tsne(X, y):
    labels = np.where(y == 0, CLASSES[0], CLASSES[1])

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=pca_points[:, 0], y=pca_points[:, 1], hue=labels, s=20, alpha=0.75)
    plt.title("PCA (2D) of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_2d.png"), dpi=160)
    plt.close()

    # t-SNE can be slow on huge datasets, so subsample if needed.
    sample_limit = min(800, X.shape[0])
    idx = np.random.default_rng(42).choice(X.shape[0], size=sample_limit, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    labels_sub = np.where(y_sub == 0, CLASSES[0], CLASSES[1])
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    tsne_points = tsne.fit_transform(X_sub)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=tsne_points[:, 0], y=tsne_points[:, 1], hue=labels_sub, s=20, alpha=0.75)
    plt.title("t-SNE (2D) of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tsne_2d.png"), dpi=160)
    plt.close()


def write_summary(stats, unreadable, X, y):
    lines = []
    lines.append("# EDA Summary\n")
    lines.append("## Dataset Overview")
    for cls in CLASSES:
        lines.append(
            f"- {cls}: total_files={stats[cls]['total_files']}, "
            f"readable_files={stats[cls]['readable_files']}, "
            f"unreadable={unreadable[cls]}"
        )

    lines.append("\n## Feature Stats")
    lines.append(f"- Number of instances: {X.shape[0]}")
    lines.append(f"- Number of features: {X.shape[1]}")
    lines.append(f"- Missing values in features: {int(np.isnan(X).sum())}")
    lines.append(f"- Class 0 count: {int(np.sum(y == 0))}")
    lines.append(f"- Class 1 count: {int(np.sum(y == 1))}")

    with open(os.path.join(OUTPUT_DIR, "eda_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    stats, unreadable, size_samples = collect_dataset_stats(DATA_DIR)
    plot_class_counts(stats)
    plot_top_image_sizes(size_samples)

    X, y = load_features()
    plot_feature_variance(X)
    plot_correlation_heatmap(X)
    plot_pca_tsne(X, y)
    write_summary(stats, unreadable, X, y)

    print("EDA complete. Outputs saved to reports/ and reports/figures/.")


if __name__ == "__main__":
    main()
