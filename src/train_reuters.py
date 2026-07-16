"""
DEC on REUTERS-10k (paper Section 4/5, Table 2).
Same two-phase pipeline as MNIST, with input_dim=2000 and k=4.
Uses the Adam (fast) schedule, which outperformed the paper schedule
on MNIST in this implementation.

Run:                          python -m src.train_reuters
Fixed-seed reproducible run:  python -m src.train_reuters --seed 42
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import StackedAutoencoder, pretrain_layerwise, finetune
from src.dec import DEC, kl_loss
from src.metrics import cluster_accuracy, nmi
from src.data_reuters import load_reuters10k, N_FEATURES


def train_dec_reuters(device="cuda" if torch.cuda.is_available() else "cpu",
                      batch_size=256, tol=0.001, max_iters=200,
                      n_clusters=4, ckpt_dir=".", seed=None):
    device = torch.device(device)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    # ---------- Data ----------
    x, y = load_reuters10k()
    dataset = TensorDataset(x, torch.zeros(len(x)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # ---------- Phase 1: autoencoder (2000-500-500-2000-10) ----------
    sae = StackedAutoencoder(input_dim=N_FEATURES).to(device)
    pretrain_layerwise(sae, loader, device)
    finetune(sae, loader, device)
    torch.save(sae.state_dict(), os.path.join(ckpt_dir, "sae_reuters.pth"))

    # ---------- Baseline: AE + k-means ----------
    dec = DEC(sae.encoder, n_clusters=n_clusters).to(device)
    init_pred = dec.init_centroids(eval_loader, device)
    print(f"AE + k-means  ACC={cluster_accuracy(y, init_pred):.4f}  "
          f"NMI={nmi(y, init_pred):.4f}")

    # ---------- Phase 2: KL clustering ----------
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

    torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_reuters.pth"))
    print(f"\nFinal DEC (REUTERS-10k)  ACC={cluster_accuracy(y, pred):.4f}  "
          f"NMI={nmi(y, pred):.4f}")
    return dec, pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=".")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: unseeded)")
    args = parser.parse_args()
    train_dec_reuters(ckpt_dir=args.ckpt_dir, seed=args.seed)