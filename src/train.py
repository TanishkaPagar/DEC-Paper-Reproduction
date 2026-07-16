"""
Full DEC training pipeline on MNIST.
Phase 1: stacked denoising autoencoder pretraining (autoencoder.py)
Phase 2: KL-divergence clustering optimization (dec.py)

Run (fast schedule, ~1 hr on T4):    python -m src.train
Run (paper-faithful, ~4-6 hrs):      python -m src.train --schedule paper

On Colab, point checkpoints at Google Drive so progress survives
session resets:
    python -m src.train --schedule paper --ckpt_dir /content/drive/MyDrive/dec_checkpoints
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from src.autoencoder import (StackedAutoencoder, pretrain_layerwise, finetune,
                             pretrain_layerwise_paper, finetune_paper)
from src.dec import DEC, kl_loss
from src.metrics import cluster_accuracy, nmi


def load_mnist():
    """
    Full MNIST: train + test = 70,000 images (as in the paper).
    Flattened to 784 and scaled so features have roughly unit variance
    (paper Section 4.1: divide by 0.02 after 0-1 scaling... we use the
    standard equivalent: multiply normalized pixels by 2).
    """
    train = datasets.MNIST(root="data", train=True, download=True)
    test = datasets.MNIST(root="data", train=False, download=True)

    x = torch.cat([train.data, test.data]).float().view(-1, 784) / 255.0
    x = x * 2.0  # approximate the paper's rescaling trick
    y = torch.cat([train.targets, test.targets]).numpy()
    return x, y


def train_dec(device="cuda" if torch.cuda.is_available() else "cpu",
              batch_size=256, tol=0.001, max_iters=100, update_interval=1,
              schedule="fast", ckpt_dir="."):
    device = torch.device(device)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Training schedule: {schedule}")
    print(f"Checkpoint directory: {ckpt_dir}")

    # ---------- Data ----------
    x, y = load_mnist()
    dataset = TensorDataset(x, torch.zeros(len(x)))  # labels unused in training
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # ---------- Phase 1: autoencoder ----------
    sae = StackedAutoencoder().to(device)
    if schedule == "paper":
        pretrain_layerwise_paper(sae, loader, device, ckpt_dir=ckpt_dir)
        finetune_paper(sae, loader, device, ckpt_dir=ckpt_dir)
    else:
        pretrain_layerwise(sae, loader, device)
        finetune(sae, loader, device)
    torch.save(sae.state_dict(), os.path.join(ckpt_dir, "sae_pretrained.pth"))

    # ---------- Baseline: AE + k-means (Table 3 comparison) ----------
    dec = DEC(sae.encoder).to(device)
    init_pred = dec.init_centroids(eval_loader, device)
    print(f"AE + k-means  ACC={cluster_accuracy(y, init_pred):.4f}  "
          f"NMI={nmi(y, init_pred):.4f}")

    # ---------- Phase 2: KL clustering ----------
    optimizer = torch.optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    prev_pred = init_pred.copy()

    for it in range(max_iters):
        # -- Update target distribution P over the FULL dataset --
        dec.eval()
        q_full = []
        with torch.no_grad():
            for xb, _ in eval_loader:
                q_full.append(dec(xb.to(device)).cpu())
        q_full = torch.cat(q_full)
        p_full = DEC.target_distribution(q_full)
        pred = q_full.argmax(dim=1).numpy()

        # -- Metrics + stopping check --
        acc = cluster_accuracy(y, pred)
        delta = np.mean(pred != prev_pred)
        print(f"iter {it:3d}  ACC={acc:.4f}  NMI={nmi(y, pred):.4f}  "
              f"changed={delta:.4%}", flush=True)
        if it > 0 and delta < tol:
            print("Converged: assignment change below 0.1%. Stopping.")
            break
        prev_pred = pred

        # -- One epoch of SGD on KL(P || Q) --
        dec.train()
        perm = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx].to(device)
            pb = p_full[idx].to(device)   # fixed targets for this epoch
            qb = dec(xb)
            loss = kl_loss(pb, qb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---------- Final results ----------
    torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_final.pth"))
    print(f"\nFinal DEC  ACC={cluster_accuracy(y, pred):.4f}  "
          f"NMI={nmi(y, pred):.4f}")
    return dec, pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=["fast", "paper"], default="fast",
                        help="'fast' = Adam reduced schedule, "
                             "'paper' = 50k iters/layer + 100k finetune (Section 5.1)")
    parser.add_argument("--ckpt_dir", default=".",
                        help="Directory for checkpoints/weights "
                             "(point at Google Drive on Colab)")
    args = parser.parse_args()
    train_dec(schedule=args.schedule, ckpt_dir=args.ckpt_dir)