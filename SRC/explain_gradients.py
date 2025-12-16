
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

from train_utils import get_device, ensure_dirs



def build_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model



def get_single_test_sample():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    testset = datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=transform
    )

    img, label = testset[0]  # pick first image
    return img.unsqueeze(0), label  # shape: (1,1,224,224)



def compute_saliency(model, image, device):
    image = image.to(device)
    image.requires_grad = True

    output = model(image)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    saliency = image.grad.data.abs().squeeze().cpu().numpy()
    return saliency, pred_class.item()



def compute_integrated_gradients(model, image, device, steps=50):
    baseline = torch.zeros_like(image).to(device)
    image = image.to(device)

    scaled_inputs = [
        baseline + (float(i) / steps) * (image - baseline)
        for i in range(steps + 1)
    ]

    grads = []

    for scaled in scaled_inputs:
        scaled.requires_grad = True
        output = model(scaled)
        pred_class = output.argmax(dim=1)

        model.zero_grad()
        output[0, pred_class].backward()
        grads.append(scaled.grad.data.cpu().numpy())

    avg_grads = np.mean(grads, axis=0)
    integrated_gradients = (image.cpu().numpy() - baseline.cpu().numpy()) * avg_grads
    return integrated_gradients.squeeze(), pred_class.item()



def save_heatmap(img, title, save_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def run_explainability():
    ensure_dirs()
    device = get_device()

    # Load model
    print("\n📥 Loading baseline CNN...")
    model_path = "../outputs/models/baseline_cnn.pth"
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load sample image
    img, label = get_single_test_sample()

    # ---------------- SALIENCY ----------------
    print("🔍 Computing Saliency Map...")
    saliency, pred_class_s = compute_saliency(model, img, device)
    save_heatmap(
        saliency,
        f"Saliency Map (Pred: {pred_class_s}, True: {label})",
        "../outputs/visualizations/gradients/saliency_map.png"
    )

    # ---------------- INTEGRATED GRADIENTS ----------------
    print("🔍 Computing Integrated Gradients...")
    ig, pred_class_ig = compute_integrated_gradients(model, img, device)
    save_heatmap(
        ig,
        f"Integrated Gradients (Pred: {pred_class_ig}, True: {label})",
        "../outputs/visualizations/gradients/integrated_gradients.png"
    )

    print("\n🎯 Explainability visualizations saved in:")
    print("../outputs/visualizations/gradients/")



if __name__ == "__main__":
    run_explainability()
