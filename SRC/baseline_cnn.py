import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.data as data



def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Apple Metal (mps) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU found, using CPU")
        return torch.device("cpu")



def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  
    ])

    train = datasets.FashionMNIST(
        root="../data", train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=transform
    )

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def build_resnet18(pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)


    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

 
    model.fc = nn.Linear(512, 10)

    return model



def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"🔍 Test Accuracy: {acc:.4f}")
    return acc


def train_baseline(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    pretrained: bool = False,
):
    device = get_device()
    train_loader, test_loader = get_dataloaders(batch_size)

    model = build_resnet18(pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n🚀 Starting Training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| Loss={epoch_loss:.4f} "
            f"| Train Acc={epoch_acc:.4f}"
        )

   
    save_dir = "../outputs/models"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "baseline_cnn.pth")
    torch.save(model.state_dict(), save_path)

    print(f"\n💾 Model saved successfully at: {save_path}")

    # Final evaluation
    evaluate(model, test_loader, device)

    return model



if __name__ == "__main__":
    # You can tweak epochs/lr if needed
    train_baseline(epochs=10, batch_size=64, lr=1e-3, pretrained=False)
