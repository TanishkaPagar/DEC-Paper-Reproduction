"""
STL-10 dataset construction, paper-style HOG features (Section 4.1).

Uses all 13,000 labeled images (5,000 train + 8,000 test splits combined,
as in the paper). Extracts HOG descriptors per image, then reduces to
exactly 1,428 dimensions via PCA (the paper's exact HOG cell/block
parameters were never published, only in unreleased Caffe code, so PCA
gives us a principled, documented way to hit the specified dimensionality).
"""

import numpy as np
import torch
from torchvision.datasets import STL10
from torchvision import transforms
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

N_CLUSTERS = 10
TARGET_DIM = 1428
SEED = 42


def extract_hog(img_array):
    """img_array: (96, 96, 3) uint8 -> 1D HOG descriptor."""
    gray = rgb2gray(img_array)
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")
    return feat


def load_stl10(data_home="data"):
    print("Loading STL-10 (train + test labeled splits, 13,000 images)...",
          flush=True)
    train = STL10(root=data_home, split="train", download=True)
    test = STL10(root=data_home, split="test", download=True)

    imgs = np.concatenate([train.data, test.data])   # (13000, 3, 96, 96)
    imgs = imgs.transpose(0, 2, 3, 1)                 # -> (13000, 96, 96, 3)
    y = np.concatenate([train.labels, test.labels])

    print(f"Extracting HOG features for {len(imgs)} images "
          f"(this takes a few minutes on CPU)...", flush=True)
    feats = []
    for i, img in enumerate(imgs):
        feats.append(extract_hog(img))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(imgs)} done", flush=True)
    feats = np.array(feats, dtype=np.float32)
    print(f"Raw HOG dimension: {feats.shape[1]}", flush=True)

    # Reduce to exactly TARGET_DIM via PCA
    pca = PCA(n_components=TARGET_DIM, random_state=SEED)
    x = pca.fit_transform(feats).astype(np.float32)
    print(f"Reduced to {x.shape[1]} dims via PCA "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})",
          flush=True)

    # Scale: (1/d)*||x||^2 ~= 1 per sample, matching MNIST/REUTERS normalization
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms * np.sqrt(TARGET_DIM)

    print(f"STL-10 ready: x={x.shape}, y={y.shape}, "
          f"classes={np.bincount(y)}", flush=True)
    return torch.from_numpy(x).float(), y
