

import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os


def save_fashionmnist():
    os.makedirs("../data", exist_ok=True)

    print("📥 Loading raw Fashion-MNIST training set...")

    transform = transforms.Compose([
        transforms.ToTensor(),   # output in [0, 1]
    ])

    trainset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    data_list = []

    print("💾 Saving raw image tensors...")
    for img, label in tqdm(trainset):
     
        img = img.squeeze(0)  # -> (28, 28)
        data_list.append((img, label))

    save_path = "../data/fashionmnist_train_tensor.pt"

   
    torch.save(
        {
            "data": data_list,
            "length": len(data_list),
            "description": "Raw Fashion-MNIST images (28×28), unnormalized, tensor format",
        },
        save_path
    )

    print(f"\n✅ Saved raw tensor dataset at: {save_path}")
    print(f"📦 Total samples saved: {len(data_list)}")



if __name__ == "__main__":
    save_fashionmnist()
