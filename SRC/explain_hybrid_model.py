
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os



def ensure_dirs():
    os.makedirs("../outputs/visualizations/gradients", exist_ok=True)


def load_hybrid_data():
    print("📥 Loading hybrid model...")
    hybrid = joblib.load("../outputs/models/hybrid_model.pkl")
    clf = hybrid["classifier"]
    scaler = hybrid["scaler"]

    print("📥 Loading hybrid feature vectors...")
    X = np.load("../outputs/models/hybrid_features.npy")

    return clf, scaler, X


# ---------------------------------------------------------
# RandomForest Feature Importance
# ---------------------------------------------------------
def plot_feature_importance(clf, save_path):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    top_n = 30  # show top 30 important features

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), importances[idx][:top_n])
    plt.title("Top 30 Hybrid Feature Importances (RandomForest)")
    plt.xlabel("Feature Index (CNN + Handcrafted)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved RandomForest feature importance plot: {save_path}")



def shap_explain(clf, X, save_bar, save_scatter):
    print("🔍 Computing SHAP values... (TreeSHAP)")

    # Use a subset for SHAP speed
    X_sample = X[:500]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    print("📊 Generating SHAP bar plot...")
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.savefig(save_bar)
    plt.close()

    print("📊 Generating SHAP scatter plot...")
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )
    plt.savefig(save_scatter)
    plt.close()

    print("🎯 SHAP plots saved.")



def run_explain_hybrid():
    ensure_dirs()

    clf, scaler, X = load_hybrid_data()


    rf_path = "../outputs/visualizations/gradients/hybrid_feature_importance.png"
    plot_feature_importance(clf, rf_path)

    shap_bar = "../outputs/visualizations/gradients/hybrid_shap_summary_bar.png"
    shap_scatter = "../outputs/visualizations/gradients/hybrid_shap_summary_scatter.png"

    shap_explain(clf, X, shap_bar, shap_scatter)



if __name__ == "__main__":
    run_explain_hybrid()
