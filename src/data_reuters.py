"""
REUTERS-10k dataset construction (paper Section 4.1).

From the RCV1-v2 corpus: keep documents labeled with exactly ONE of the
four root categories (CCAT=corporate, ECAT=economics, GCAT=government,
MCAT=markets), restrict features to the 2000 highest-frequency word stems,
and sample 10,000 documents.

Each sample is scaled so that (1/d)*||x||^2 is approximately 1, matching
the normalization used for MNIST.

NOTE: fetch_rcv1 downloads ~700MB on first call - run on Colab, not locally.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_rcv1

ROOT_CATEGORIES = ["CCAT", "ECAT", "GCAT", "MCAT"]
N_SAMPLES = 10000
N_FEATURES = 2000
SEED = 42


def load_reuters10k(data_home="data"):
    print("Fetching RCV1 (first call downloads ~700MB)...", flush=True)
    rcv1 = fetch_rcv1(data_home=data_home)

    # --- Filter: documents with exactly one root category ---
    root_idx = [list(rcv1.target_names).index(c) for c in ROOT_CATEGORIES]
    root_labels = rcv1.target[:, root_idx].toarray()   # (n_docs, 4) of 0/1
    single_root = root_labels.sum(axis=1) == 1
    print(f"Documents with exactly one root category: {single_root.sum()}",
          flush=True)

    x_all = rcv1.data[single_root]                     # sparse tf-idf
    y_all = root_labels[single_root].argmax(axis=1)    # 0..3

    # --- Sample 10,000 documents ---
    rng = np.random.RandomState(SEED)
    idx = rng.choice(x_all.shape[0], N_SAMPLES, replace=False)
    x = x_all[idx]
    y = y_all[idx]

    # --- Keep the 2000 most frequent features (by document frequency) ---
    doc_freq = np.asarray((x > 0).sum(axis=0)).ravel()
    top_feats = np.argsort(doc_freq)[::-1][:N_FEATURES]
    x = x[:, top_feats].toarray().astype(np.float32)

    # --- Scale: (1/d)*||x||^2 ~= 1 per sample ---
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms * np.sqrt(N_FEATURES)

    print(f"REUTERS-10k ready: x={x.shape}, y={y.shape}, "
          f"classes={np.bincount(y)}", flush=True)
    return torch.from_numpy(x), y