
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import os

from train_utils import get_device, ensure_dirs


def build_resnet18():
    model = models.resnet18(weights=None)

    # Modify first conv (1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final FC to output 10 classes
    model.fc = nn.Linear(512, 10)

    return model



def load_test_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=transform
    )

    img, label = testset[1]   
    return img.unsqueeze(0), label



activations = None
gradients = None

def save_activation(module, input, output):
    global activations
    activations = output.detach()

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()


# ------------------------------------------------------------
# Generate Grad-CAM Heatmap
# ------------------------------------------------------------
def generate_gradcam(model, img, device):
    global activations, gradients

    img = img.to(device)

    # Forward pass
    output = model(img)
    pred_class = output.argmax(dim=1).item()

    # Backprop to get gradients
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Mean over channels
    pooled_gradients = gradients.mean(dim=[0, 2, 3])

    # Weight the channels by gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Compute heatmap
    heatmap = activations.mean(dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= heatmap.max() + 1e-8

    return heatmap, pred_class



def overlay_heatmap(heatmap, img_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.savefig(img_path)
    plt.close()



def run_gradcam():
    ensure_dirs()
    device = get_device()

    print("\n📥 Loading baseline CNN...")
    model = build_resnet18().to(device)

    state = torch.load("../outputs/models/baseline_cnn.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Hook into the last convolutional layer (layer4[-1])
    target_layer = model.layer4[-1]

    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)

    # Load image
    print("🖼 Loading test sample...")
    img, label = load_test_image()

    # Grad-CAM
    print("🔥 Generating Grad-CAM...")
    heatmap, pred_class = generate_gradcam(model, img, device)

    save_path = "../outputs/visualizations/gradcam/gradcam_result.png"
    overlay_heatmap(heatmap, save_path)

    print(f"\n🎯 Grad-CAM saved at: {save_path}")
    print(f"Predicted Class: {pred_class}, True Label: {label}")



if __name__ == "__main__":
    run_gradcam()
