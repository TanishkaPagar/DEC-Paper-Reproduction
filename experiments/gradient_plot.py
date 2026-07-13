"""
Figure 4 reproduction: magnitude of the gradient dL/dz_i vs. soft
assignment q_ij to the most likely cluster, at the START of the KL phase.

Shows that high-confidence points contribute larger gradients —
the paper's justification for its self-training target distribution.

Run from repo root:  python -m experiments.gradient_plot
Requires: sae_pretrained.pth in the repo root.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import StackedAutoencoder
from src.dec import DEC, kl_loss
from src.train import load_mnist

N_PLOT = 10000  # points to show in the scatter


def main(batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = load_mnist()
    dataset = TensorDataset(x, torch.zeros(len(x)))
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Pretrained encoder + fresh k-means init = "start of KL phase"
    sae = StackedAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_pretrained.pth", map_location=device))
    dec = DEC(sae.encoder).to(device)
    print("Running k-means initialization ...")
    dec.init_centroids(eval_loader, device)

    # Pass 1: q over the full dataset (no grad) -> target distribution p
    dec.eval()
    q_full = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            q_full.append(dec(xb.to(device)).cpu())
    q_full = torch.cat(q_full)
    p_full = DEC.target_distribution(q_full)

    # Pass 2: per-point gradient of the KL loss w.r.t. the embedding z
    grad_norms, q_max = [], []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        pb = p_full[start:start + batch_size].to(device)

        z = dec.encoder(xb)
        z.retain_grad()                 # keep per-sample gradients on z
        qb = dec.soft_assign(z)
        loss = kl_loss(pb, qb) * qb.shape[0]  # undo batch mean -> sum
        loss.backward()

        grad_norms.append(z.grad.norm(dim=1).detach().cpu().numpy())
        q_max.append(qb.max(dim=1).values.detach().cpu().numpy())
        dec.zero_grad()
    grad_norms = np.concatenate(grad_norms)
    q_max = np.concatenate(q_max)

    # Subsample for a readable scatter
    rng = np.random.RandomState(42)
    idx = rng.choice(len(q_max), N_PLOT, replace=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(q_max[idx], grad_norms[idx], s=3, alpha=0.3)
    plt.xlabel("Soft assignment to most likely cluster  $q_{ij}$", fontsize=12)
    plt.ylabel(r"Gradient magnitude  $\|\partial L / \partial z_i\|$", fontsize=12)
    plt.title("Gradient contribution vs. assignment confidence\n"
              "(start of KL phase — Figure 4 reproduction)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig("results/figures/gradient_vs_confidence.png",
                dpi=150, bbox_inches="tight")
    print("Saved: results/figures/gradient_vs_confidence.png")


if __name__ == "__main__":
    main()