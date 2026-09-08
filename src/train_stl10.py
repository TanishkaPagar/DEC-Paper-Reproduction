"""
DEC on STL-10 (paper Section 4/5, Table 2).
Same two-phase pipeline as MNIST/REUTERS, with input_dim=1428, k=10.

Run:  python -m src.train_stl10
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import StackedAutoencoder, pretrain_layerwise, finetune
from src.dec import DEC, kl_loss
from src.metrics import cluster_accuracy, nmi
from src.data_stl10 import load_stl10, TARGET_DIM, N_CLUSTERS


def train_dec_stl10(device="cuda" if torch.cuda.is_available() else "cpu",
                    batch_size=256, tol=0.001, max_iters=200, ckpt_dir=".",
                    seed=None):
    device = torch.device(device)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    x, y = load_stl10()
    dataset = TensorDataset(x, torch.zeros(len(x)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    sae = StackedAutoencoder(input_dim=TARGET_DIM).to(device)
    pretrain_layerwise(sae, loader, device)
    finetune(sae, loader, device)
    torch.save(sae.state_dict(), os.path.join(ckpt_dir, "sae_stl10.pth"))

    dec = DEC(sae.encoder, n_clusters=N_CLUSTERS).to(device)
    init_pred = dec.init_centroids(eval_loader, device)
    print(f"AE + k-means  ACC={cluster_accuracy(y, init_pred):.4f}  "
          f"NMI={nmi(y, init_pred):.4f}")

    optimizer = torch.optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    prev_pred = init_pred.copy()

    for it in range(max_iters):
        dec.eval()
        q_full = []
        with torch.no_grad():
            for xb, _ in eval_loader:
                q_full.append(dec(xb.to(device)).cpu())
        q_full = torch.cat(q_full)
        p_full = DEC.target_distribution(q_full)
        pred = q_full.argmax(dim=1).numpy()

        acc = cluster_accuracy(y, pred)
        delta = np.mean(pred != prev_pred)
        print(f"iter {it:3d}  ACC={acc:.4f}  NMI={nmi(y, pred):.4f}  "
              f"changed={delta:.4%}", flush=True)
        if it > 0 and delta < tol:
            print("Converged: assignment change below 0.1%. Stopping.")
            break
        prev_pred = pred

        dec.train()
        perm = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx].to(device)
            pb = p_full[idx].to(device)
            qb = dec(xb)
            loss = kl_loss(pb, qb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_stl10.pth"))
    print(f"\nFinal DEC (STL-10)  ACC={cluster_accuracy(y, pred):.4f}  "
          f"NMI={nmi(y, pred):.4f}")
    return dec, pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=".")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train_dec_stl10(ckpt_dir=args.ckpt_dir, seed=args.seed)
