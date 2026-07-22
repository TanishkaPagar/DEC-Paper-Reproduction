"""
Baseline comparison for the paper's Table 2:
  - k-means on raw input features (no deep learning)
  - k-means on the DEC autoencoder embedding (= AE + k-means)
  - DEC without backprop (frozen encoder, centroids-only)
for both MNIST and REUTERS-10k.

Reuses saved weights; run from repo root:
    python -m experiments.baselines --ckpt_dir /content/drive/MyDrive/dec_checkpoints
"""

import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import StackedAutoencoder
from src.dec import DEC, kl_loss
from src.metrics import cluster_accuracy, nmi
from src.train import load_mnist
from src.data_reuters import load_reuters10k, N_FEATURES


def kmeans_raw(x, y, k):
    """k-means directly on raw input features."""
    pred = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(x.numpy())
    return cluster_accuracy(y, pred), nmi(y, pred)


def dec_no_backprop(encoder, x, y, k, device, batch_size=256,
                    tol=0.001, max_iters=100):
    """DEC with the encoder frozen — only centroids update."""
    dataset = TensorDataset(x, torch.zeros(len(x)))
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    dec = DEC(encoder, n_clusters=k).to(device)
    dec.init_centroids(eval_loader, device)
    for p in dec.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.SGD([dec.centroids], lr=0.01, momentum=0.9)

    prev = None
    pred = None
    for it in range(max_iters):
        dec.eval()
        q = []
        with torch.no_grad():
            for xb, _ in eval_loader:
                q.append(dec(xb.to(device)).cpu())
        q = torch.cat(q)
        p_target = DEC.target_distribution(q)
        pred = q.argmax(1).numpy()
        if prev is not None and np.mean(pred != prev) < tol:
            break
        prev = pred
        dec.train()
        perm = torch.randperm(len(x))
        for s in range(0, len(x), batch_size):
            idx = perm[s:s + batch_size]
            qb = dec(x[idx].to(device))
            loss = kl_loss(p_target[idx].to(device), qb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return cluster_accuracy(y, pred), nmi(y, pred)


def run(name, x, y, k, sae_path, device):
    print(f"\n========== {name} ==========", flush=True)

    acc, nm = kmeans_raw(x, y, k)
    print(f"k-means (raw)        ACC={acc:.4f}  NMI={nm:.4f}", flush=True)

    input_dim = x.shape[1]
    sae = StackedAutoencoder(input_dim=input_dim).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))

    acc, nm = dec_no_backprop(sae.encoder, x, y, k, device)
    print(f"DEC w/o backprop     ACC={acc:.4f}  NMI={nm:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=".")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNIST
    x, y = load_mnist()
    run("MNIST", x, y, 10, f"{args.ckpt_dir}/sae_pretrained.pth", device)

    # REUTERS-10k
    xr, yr = load_reuters10k()
    run("REUTERS-10k", xr, yr, 4, f"{args.ckpt_dir}/sae_reuters.pth", device)


if __name__ == "__main__":
    main()