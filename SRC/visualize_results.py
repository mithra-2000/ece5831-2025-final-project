

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ensure_dirs():
    os.makedirs("../outputs/visualizations/plots", exist_ok=True)


def load_metrics():
    print("📥 Loading baseline metrics...")
    baseline_preds = np.load("../outputs/metrics/preds.npy")
    baseline_labels = np.load("../outputs/metrics/labels.npy")

    print("📥 Loading hybrid metrics...")
    hybrid_preds = np.load("../outputs/metrics/hybrid_preds.npy")
    hybrid_labels = np.load("../outputs/metrics/hybrid_labels.npy")

    return baseline_preds, baseline_labels, hybrid_preds, hybrid_labels


def plot_confusion_matrix(labels, preds, save_path):
    cm = sns.heatmap(
        confusion_matrix(labels, preds),
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Hybrid Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved confusion matrix: {save_path}")


# Need to import here for safety
from sklearn.metrics import confusion_matrix


def plot_feature_distribution(save_path):
    print("📥 Loading hybrid feature vectors...")
    X = np.load("../outputs/models/hybrid_features.npy")  # (60000, ~670)

    mean_vec = np.mean(X, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(mean_vec)
    plt.title("Hybrid Feature Mean Activation Distribution")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📊 Saved feature distribution: {save_path}")


def plot_accuracy_comparison(baseline_preds, baseline_labels,
                             hybrid_preds, hybrid_labels,
                             save_path):

    baseline_acc = (baseline_preds == baseline_labels).mean()
    hybrid_acc = (hybrid_preds == hybrid_labels).mean()

    plt.figure(figsize=(4, 4))
    plt.bar(["Baseline CNN", "Hybrid Model"], [baseline_acc, hybrid_acc], color=["gray", "green"])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Saved accuracy comparison: {save_path}")


def run_visualizations():
    ensure_dirs()

    baseline_preds, baseline_labels, hybrid_preds, hybrid_labels = load_metrics()

    # Confusion Matrix
    cm_path = "../outputs/visualizations/plots/hybrid_confusion_matrix.png"
    plot_confusion_matrix(hybrid_labels, hybrid_preds, cm_path)

    # Feature Distribution
    dist_path = "../outputs/visualizations/plots/hybrid_feature_distribution.png"
    plot_feature_distribution(dist_path)

    # Accuracy Comparison
    acc_path = "../outputs/visualizations/plots/accuracy_comparison.png"
    plot_accuracy_comparison(
        baseline_preds, baseline_labels,
        hybrid_preds, hybrid_labels,
        acc_path
    )

    print("\n🎉 All visualization plots saved!")


if __name__ == "__main__":
    run_visualizations()
