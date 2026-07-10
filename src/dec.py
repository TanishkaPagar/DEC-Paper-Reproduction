"""
Deep Embedded Clustering (DEC) — Xie, Girshick, Farhadi, ICML 2016.
Section 3.1: soft assignment (Eq. 1), target distribution (Eq. 3),
KL divergence clustering loss (Eq. 2).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class DEC(nn.Module):
    def __init__(self, encoder, n_clusters=10, embedding_dim=10, alpha=1.0):
        """
        Args:
            encoder: the pretrained encoder from the autoencoder (f_theta)
            n_clusters: number of clusters k
            alpha: degrees of freedom of Student's t (paper uses 1)
        """
        super().__init__()
        self.encoder = encoder
        self.alpha = alpha
        # Cluster centroids are LEARNABLE parameters, updated by SGD
        # together with the encoder weights.
        self.centroids = nn.Parameter(torch.zeros(n_clusters, embedding_dim))

    def init_centroids(self, data_loader, device):
        """Run k-means once on the embedded data to initialize centroids."""
        self.eval()
        embeddings = []
        with torch.no_grad():
            for x, _ in data_loader:
                embeddings.append(self.encoder(x.to(device)).cpu().numpy())
        embeddings = np.concatenate(embeddings)

        kmeans = KMeans(n_clusters=self.centroids.shape[0], n_init=20)
        pred = kmeans.fit_predict(embeddings)
        self.centroids.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )
        return pred  # initial cluster assignments

    def soft_assign(self, z):
        """
        Equation 1: Student's t-distribution similarity between
        embedded points z (batch, d) and centroids (k, d).
        Returns q of shape (batch, k), rows sum to 1.
        """
        # squared distance between every point and every centroid
        dist_sq = torch.sum((z.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def forward(self, x):
        z = self.encoder(x)
        return self.soft_assign(z)

    @staticmethod
    def target_distribution(q):
        """
        Equation 3: sharpen q into the self-training target p.
        p_ij = (q_ij^2 / f_j) / normalizer,  f_j = soft cluster frequency.
        """
        f = q.sum(dim=0)                    # soft cluster sizes
        p = (q ** 2) / f
        p = p / p.sum(dim=1, keepdim=True)
        return p


def kl_loss(p, q):
    """Equation 2: KL(P || Q), averaged over the batch."""
    return torch.sum(p * torch.log(p / (q + 1e-10) + 1e-10)) / p.shape[0]