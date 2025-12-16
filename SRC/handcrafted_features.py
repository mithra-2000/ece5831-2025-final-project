
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops

try:
    import torch
except ImportError:
    torch = None



def _to_numpy_gray(img):
    """
    Accepts:
      - numpy array of shape (H, W) or (1, H, W)
      - torch tensor of shape (H, W) or (1, H, W)

    Returns:
      - numpy array of shape (H, W), float32 in [0, 1]
    """
    if torch is not None and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.squeeze(img)  # remove channel dimension if present

    # Normalize to [0,1] if not already
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    return img



def extract_hog(img):
    """
    Computes HOG descriptor for a single grayscale image.

    Args:
        img: numpy array or torch tensor, shape (H, W) or (1, H, W)

    Returns:
        1D numpy array of HOG features.
    """
    img = _to_numpy_gray(img)

    # skimage expects 2D array
    features = hog(
        img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features



def extract_glcm(img):
    """
    Computes basic GLCM texture features: contrast, energy, homogeneity.

    Args:
        img: numpy array or torch tensor, shape (H, W) or (1, H, W)

    Returns:
        numpy array of shape (3,) -> [contrast, energy, homogeneity]
    """
    img = _to_numpy_gray(img)

    # GLCM expects uint8 levels
    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0],
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    return np.array([contrast, energy, homogeneity], dtype=np.float32)



def extract_handcrafted(img):
    """
    Combines HOG + GLCM into a single feature vector.

    Args:
        img: numpy array or torch tensor, shape (H, W) or (1, H, W)

    Returns:
        1D numpy array: [HOG features..., contrast, energy, homogeneity]
    """
    hog_feat = extract_hog(img)
    glcm_feat = extract_glcm(img)
    return np.concatenate([hog_feat.astype(np.float32), glcm_feat], axis=0)


if __name__ == "__main__":
    if torch is None:
        print("Torch not installed, skipping self-test.")
    else:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        ds = datasets.FashionMNIST(
            root="../data", train=True, download=True, transform=transform
        )

        img, label = ds[0]  # img: torch.Size([1, 28, 28])

        feats = extract_handcrafted(img)
        print("Label:", label)
        print("Feature vector length:", feats.shape[0])
        print("First 10 features:", feats[:10])
