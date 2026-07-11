"""
Figure 5 reproduction: t-SNE visualization of the embedded space,
before DEC (autoencoder embedding) vs after DEC (KL-optimized embedding).

Run from repo root AFTER training:  python -m experiments.visualize_tsne
Requires: sae_pretrained.pth and dec_final.pth in the repo root.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.autoencoder import StackedAutoencoder
from src.dec import DEC
from src.train import load_mnist

N_POINTS = 10000  # t-SNE on all 70k would take very long; 10k is plenty


def get_embeddings(encoder, x, device, batch=1024):
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(encoder(x[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = load_mnist()

    # random subsample for t-SNE speed
    rng = np.random.RandomState(42)
    idx = rng.choice(len(x), N_POINTS, replace=False)
    x_sub, y_sub = x[idx], y[idx]

    # ---- BEFORE: autoencoder embedding ----
    sae = StackedAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_pretrained.pth", map_location=device))
    z_before = get_embeddings(sae.encoder, x_sub, device)

    # ---- AFTER: DEC-optimized embedding ----
    sae2 = StackedAutoencoder().to(device)
    dec = DEC(sae2.encoder).to(device)
    dec.load_state_dict(torch.load("dec_final.pth", map_location=device))
    z_after = get_embeddings(dec.encoder, x_sub, device)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, z, title in [
        (axes[0], z_before, "Before DEC (autoencoder embedding)"),
        (axes[1], z_after, "After DEC (KL-optimized embedding)"),
    ]:
        print(f"Running t-SNE: {title} ...")
        z2d = TSNE(n_components=2, random_state=42, init="pca").fit_transform(z)
        sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=y_sub, cmap="tab10", s=3)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=axes, ticks=range(10), fraction=0.025)
    cbar.set_label("True digit")
    plt.savefig("results/figures/tsne_before_after_dec.png",
                dpi=150, bbox_inches="tight")
    print("Saved: results/figures/tsne_before_after_dec.png")


if __name__ == "__main__":
    main()