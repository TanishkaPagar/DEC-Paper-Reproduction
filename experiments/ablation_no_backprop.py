"""
Ablation: DEC without backprop (Table 2, last row of the paper).

The encoder f_theta is FROZEN after pretraining; only the cluster
centroids are updated during the KL phase. This tests whether DEC's
gain comes from joint feature learning or merely from moving centroids.

Run from repo root:  python -m experiments.ablation_no_backprop
Requires: sae_pretrained.pth in the repo root.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import StackedAutoencoder
from src.dec import DEC, kl_loss
from src.metrics import cluster_accuracy, nmi
from src.train import load_mnist


def main(batch_size=256, tol=0.001, max_iters=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = load_mnist()
    dataset = TensorDataset(x, torch.zeros(len(x)))
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # Load the SAME pretrained autoencoder used by full DEC
    sae = StackedAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_pretrained.pth", map_location=device))

    dec = DEC(sae.encoder).to(device)
    init_pred = dec.init_centroids(eval_loader, device)
    print(f"AE + k-means  ACC={cluster_accuracy(y, init_pred):.4f}  "
          f"NMI={nmi(y, init_pred):.4f}")

    # ---- THE ABLATION: freeze the encoder ----
    for param in dec.encoder.parameters():
        param.requires_grad = False

    # Optimizer sees ONLY the centroids now
    optimizer = torch.optim.SGD([dec.centroids], lr=0.01, momentum=0.9)

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
              f"changed={delta:.4%}")
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

    print(f"\nDEC w/o backprop  ACC={cluster_accuracy(y, pred):.4f}  "
          f"NMI={nmi(y, pred):.4f}")


if __name__ == "__main__":
    main()