"""
Evaluation metrics for clustering, as used in the DEC paper (Xie et al., 2016).

ACC (unsupervised clustering accuracy) — Equation 1 of the paper:
finds the best one-to-one mapping between cluster IDs and true labels
using the Hungarian algorithm, then computes accuracy under that mapping.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def cluster_accuracy(y_true, y_pred):
    """
    Compute unsupervised clustering accuracy (ACC).

    Args:
        y_true: array of true labels, shape (n_samples,)
        y_pred: array of predicted cluster IDs, shape (n_samples,)

    Returns:
        accuracy: float in [0, 1]
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_true.shape == y_pred.shape

    n_clusters = max(y_pred.max(), y_true.max()) + 1

    # Build a "vote" matrix: count[i, j] = how many points
    # in cluster i actually have true label j
    count = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(y_true.shape[0]):
        count[y_pred[i], y_true[i]] += 1

    # Hungarian algorithm finds the mapping that maximizes total matches.
    # It minimizes cost, so we negate the count matrix.
    row_ind, col_ind = linear_sum_assignment(-count)

    n_correct = count[row_ind, col_ind].sum()
    return n_correct / y_true.shape[0]


def nmi(y_true, y_pred):
    """Normalized Mutual Information between true labels and clusters."""
    return normalized_mutual_info_score(y_true, y_pred)