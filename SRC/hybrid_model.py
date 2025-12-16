

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from train_utils import get_device, load_saved_tensor, ensure_dirs
from handcrafted_features import extract_handcrafted
from torchvision import models



def build_resnet18_embedder():
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Identity()

    return model



def extract_hybrid_features(model, raw_data, device):
    hybrid_features = []
    labels = []

    model.eval()

    for img, label in tqdm(raw_data, desc="Extracting hybrid features"):
        # img shape: (28, 28) → make 4D tensor
        img_tensor = img.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)

        # Resize to 224×224 (ResNet requirement)
        img_resized = F.interpolate(
            img_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize same as baseline CNN
        img_resized = (img_resized - 0.5) / 0.5

        # ----- CNN EMBEDDING -----
        with torch.no_grad():
            embedding = model(img_resized).cpu().numpy().flatten()  # shape: (512,)

        # ----- HANDCRAFTED FEATURES -----
        handcrafted = extract_handcrafted(img.numpy())  # (HOG + GLCM)

        # ----- CONCATENATE -----
        hybrid_vec = np.concatenate([embedding, handcrafted], axis=0)
        hybrid_features.append(hybrid_vec)
        labels.append(label)

    return np.array(hybrid_features), np.array(labels)



def train_hybrid():
    ensure_dirs()
    device = get_device()

    print("\n📥 Loading raw Fashion-MNIST tensor dataset...")
    raw_obj = load_saved_tensor("../data/fashionmnist_train_tensor.pt")
    raw_data = raw_obj["data"]

    print("📥 Loading CNN embedder...")
    embedder = build_resnet18_embedder().to(device)

    # Load baseline CNN weights (excluding final FC layer)
    baseline_path = "../outputs/models/baseline_cnn.pth"
    state = torch.load(baseline_path, map_location=device)
    # Drop fc.* since embedder has Identity FC
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    embedder.load_state_dict(state, strict=False)

    print("🔎 Extracting hybrid feature vectors...")
    X, y = extract_hybrid_features(embedder, raw_data, device)

    print(f"📦 Hybrid feature vector shape: {X.shape}")

    # Save for SHAP and visualizations
    np.save("../outputs/models/hybrid_features.npy", X)

    # Split train/test for RandomForest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🌲 Training RandomForest hybrid classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    acc = clf.score(X_test_scaled, y_test)
    print(f"\n🎯 Hybrid Model Accuracy: {acc:.4f}")

    # Save everything
    save_path = "../outputs/models/hybrid_model.pkl"
    joblib.dump(
        {
            "classifier": clf,
            "scaler": scaler,
        },
        save_path
    )

    print(f"\n💾 Hybrid model saved at: {save_path}")

    return acc


if __name__ == "__main__":
    train_hybrid()
