
import os
import torch
from torchvision import datasets, transforms



def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Apple Metal (mps) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU (no GPU detected)")
        return torch.device("cpu")



def ensure_dirs():
    os.makedirs("../outputs/models", exist_ok=True)
    os.makedirs("../outputs/visualizations/gradcam", exist_ok=True)
    os.makedirs("../outputs/visualizations/gradients", exist_ok=True)
    os.makedirs("../outputs/visualizations/plots", exist_ok=True)
    os.makedirs("../outputs/metrics", exist_ok=True)


def get_dataloaders(batch_size=64, resize_for_model=224, train=True, return_dataset=False):
    """
    Loads Fashion-MNIST with resizing and normalization.
    Resize is needed because ResNet18 expects 224×224 inputs.

    Args:
        batch_size: minibatch size
        resize_for_model: resized dimensions for CNN
        train: whether to load train or test split
        return_dataset: return dataset object instead of DataLoader

    Returns:
        dataset or DataLoader
    """

    transform = transforms.Compose([
        transforms.Resize((resize_for_model, resize_for_model)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # MUST MATCH baseline CNN
    ])

    dataset = datasets.FashionMNIST(
        root="../data",
        train=train,
        download=True,
        transform=transform,
    )

    if return_dataset:
        return dataset

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader



def load_saved_tensor(tensor_path="../data/fashionmnist_train_tensor.pt"):
  
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(
            f"{tensor_path} not found. Run save_fashionmnist_tensor.py first."
        )

    return torch.load(tensor_path)
