
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from train_utils import get_device, ensure_dirs
import numpy as np
import os


def build_resnet18():
    model = models.resnet18(weights=None)

    # Modify input channel from 3 → 1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final classification layer for 10 classes
    model.fc = nn.Linear(512, 10)

    return model



def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # SAME as baseline training
    ])

    testset = datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=transform
    )

    return torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



def evaluate_baseline():
    ensure_dirs()
    device = get_device()

    print("\n📥 Loading baseline model...")
    model_path = "../outputs/models/baseline_cnn.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "baseline_cnn.pth not found. Run baseline_cnn.py first!"
        )

    model = build_resnet18().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.eval()

    test_loader = get_test_loader()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    print("🔎 Running evaluation on test set...\n")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"🎯 Baseline Model Accuracy: {acc:.4f}")

    # Save predictions for hybrid analysis later
    np.save("../outputs/metrics/preds.npy", np.array(all_preds))
    np.save("../outputs/metrics/labels.npy", np.array(all_labels))

    print("\n💾 Saved predictions:")
    print(" - outputs/metrics/preds.npy")
    print(" - outputs/metrics/labels.npy")

    return acc


if __name__ == "__main__":
    evaluate_baseline()
