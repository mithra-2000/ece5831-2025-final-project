
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import os

from handcrafted_features import extract_handcrafted
from train_utils import get_device, ensure_dirs



def build_resnet18_embedder():
    model = models.resnet18(weights=None)

    # Modify to accept grayscale images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Remove classifier → output = 512-d embedding
    model.fc = nn.Identity()

    return model



def load_test_raw():
    transform = transforms.Compose([
        transforms.ToTensor(),   # raw 28×28 in [0,1]
    ])

    return datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )



def extract_features(model, testset, scaler, device):
    X = []
    y_true = []

    model.eval()

    for img, label in tqdm(testset, desc="Extracting hybrid test features"):
        # img: (1,28,28)
        img_tensor = img.unsqueeze(0).to(device)  # (1,1,28,28)

        # Resize to 224×224 for CNN
        img_resized = F.interpolate(
            img_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        # Normalize as in baseline CNN
        img_resized = (img_resized - 0.5) / 0.5

        with torch.no_grad():
            embedding = model(img_resized).cpu().numpy().flatten()  # 512-dim

        # Handcrafted features
        handcrafted = extract_handcrafted(img.squeeze().numpy())

        # Final hybrid vector
        hybrid_vec = np.concatenate([embedding, handcrafted])
        X.append(hybrid_vec)
        y_true.append(label)

    X = np.array(X)
    X_scaled = scaler.transform(X)

    return X_scaled, np.array(y_true)



def evaluate_hybrid():
    ensure_dirs()
    device = get_device()

    print("\n📥 Loading hybrid model...")
    hybrid = joblib.load("../outputs/models/hybrid_model.pkl")
    clf = hybrid["classifier"]
    scaler = hybrid["scaler"]

    print("📥 Loading CNN embedder...")
    embedder = build_resnet18_embedder().to(device)

    baseline_path = "../outputs/models/baseline_cnn.pth"
    state = torch.load(baseline_path, map_location=device)

    # Remove FC layer weights
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    embedder.load_state_dict(state, strict=False)

    print("📥 Loading Fashion-MNIST test set...")
    testset = load_test_raw()

    print("\n🔎 Extracting hybrid test features...")
    X_test, y_test = extract_features(embedder, testset, scaler, device)

    print("\n🌲 Predicting with Hybrid RandomForest...")
    y_pred = clf.predict(X_test)

    # Save for confusion matrix & SHAP
    np.save("../outputs/metrics/hybrid_preds.npy", y_pred)
    np.save("../outputs/metrics/hybrid_labels.npy", y_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    acc = (y_pred == y_test).mean()
    print(f"\n🎯 Hybrid Model Test Accuracy: {acc:.4f}")

    return acc



if __name__ == "__main__":
    evaluate_hybrid()
