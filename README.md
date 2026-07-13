# DEC Paper Reproduction — Unsupervised Deep Embedding for Clustering Analysis

PyTorch reproduction of **"Unsupervised Deep Embedding for Clustering Analysis"**
by Junyuan Xie, Ross Girshick, Ali Farhadi (ICML 2016).
[Paper link](https://proceedings.mlr.press/v48/xieb16.pdf)

## What is DEC?

Classical clustering (e.g., k-means) operates directly on raw features, where
distances are often meaningless — two images of the same digit can be far apart
in pixel space. DEC instead **learns a feature space and cluster centroids
simultaneously**, using a two-phase approach:

1. **Initialization:** A stacked denoising autoencoder (784–500–500–2000–10)
   is pretrained layer-wise and finetuned end-to-end. The decoder is then
   discarded, and k-means on the embedded data provides initial centroids.
2. **Clustering:** Soft assignments between embedded points and centroids are
   computed with a Student's t-distribution (Eq. 1). A sharpened **target
   distribution** (Eq. 3) is derived from the model's own confident
   predictions, and both the encoder weights and the centroids are jointly
   optimized by minimizing **KL(P‖Q)** (Eq. 2) — a form of self-training.
   Training stops when < 0.1% of points change cluster between iterations.

## Results (MNIST, 70,000 images)

| Method | ACC (paper) | ACC (this repro) | NMI (this repro) |
|---|---|---|---|
| AE + k-means | 81.84% | 77.00% | 0.7387 |
| DEC w/o backprop | 79.82% | 75.82% | 0.7161 |
| **DEC** | **84.30%** | **81.61%** | **0.8340** |

The paper's core claim reproduces clearly: the KL-divergence clustering phase
improves accuracy by **+4.6 points** over the autoencoder + k-means baseline,
with NMI rising from 0.739 to 0.834. A second independent training run scored
82.28% ACC / 0.850 NMI, indicating the result is stable across random
initializations.

### Embedded space visualization (Figure 5 reproduction)

![t-SNE before and after DEC](results/figures/tsne_before_after_dec.png)

t-SNE projection of the 10-d embedded space (10,000 sampled points, colored by
true digit). The KL-divergence phase visibly compacts and separates clusters
compared to the raw autoencoder embedding. The main remaining confusion is the
overlapping 4/9 region — a known hard case that also limits the original paper.

### Gradient contribution vs. confidence (Figure 4 reproduction)

![Gradient magnitude vs soft assignment](results/figures/gradient_vs_confidence.png)

Per-point gradient magnitude ‖∂L/∂z‖ against soft assignment confidence q,
measured at the start of the KL phase. Gradient contribution grows with
confidence — confident points dominate the learning signal, validating the
paper's self-training formulation of the target distribution.

## Reproduction insights

- **First run failed informatively.** With short SGD pretraining
  (15 epochs/layer, 30 finetune epochs), the autoencoder was undertrained
  (finetune loss still falling at cutoff) and results collapsed to
  ACC ≈ 64%. DEC could only add +1.4 points — confirming that **DEC refines
  an embedding but cannot rescue a poor one**. Switching to Adam and training
  longer (25 epochs/layer, 60 finetune epochs) recovered the expected behavior.
- **Remaining gap (~2.7% vs paper) is explained by pretraining budget.** The
  original work pretrains for 50,000 iterations per layer and 100,000
  finetuning iterations; we use a reduced schedule to fit free-tier GPU
  constraints (~1 hour total on a Colab T4).
- **Ablation confirms joint optimization is essential.** With the encoder
  frozen (no backprop into f_θ), centroid-only optimization slightly
  *degrades* performance over iterations (77.1% → 75.8%), while full DEC
  gains +4.6 points. The improvement comes from the feature space
  reorganizing itself, not from centroid movement alone — the paper's
  central claim, reproduced.

## Deviations from the paper

| Aspect | Paper | This repro | Reason |
|---|---|---|---|
| Framework | Caffe | PyTorch | Modern standard |
| Pretraining optimizer | SGD (lr 0.1, momentum 0.9) | Adam (lr 1e-3) | Faster convergence under a small epoch budget |
| Pretraining length | 50k iters/layer + 100k finetune | 25 epochs/layer + 60 finetune | Compute constraints |
| Datasets | MNIST, STL-10, REUTERS | MNIST | STL-10 requires a dated HOG pipeline; full REUTERS is memory-prohibitive |

## Repository structure

```
src/
  autoencoder.py             # Stacked denoising autoencoder: layer-wise pretrain + finetune
  dec.py                     # DEC model: soft assignment (Eq.1), target distribution (Eq.3), KL loss (Eq.2)
  metrics.py                 # Unsupervised clustering accuracy (Hungarian algorithm), NMI
  train.py                   # Full pipeline: pretrain -> k-means init -> KL optimization
experiments/
  visualize_tsne.py          # Figure 5: t-SNE of embedded space before vs after DEC
  ablation_no_backprop.py    # Table 2 ablation: frozen encoder, centroid-only updates
  gradient_plot.py           # Figure 4: gradient magnitude vs assignment confidence
  02_dec_mnist_colab.ipynb   # Colab notebook with full training logs
results/
  figures/                   # Generated figures
```

## How to run

```bash
pip install -r requirements.txt

# Full pipeline: pretraining + DEC (~1 hour on a Colab T4 GPU)
python -m src.train

# After training (uses the saved sae_pretrained.pth / dec_final.pth):
python -m experiments.ablation_no_backprop   # Table 2 ablation
python -m experiments.visualize_tsne         # Figure 5 visualization
python -m experiments.gradient_plot          # Figure 4 gradient analysis
```

Trained weights (`sae_pretrained.pth`, `dec_final.pth`) are saved to the
repository root and are required by the ablation and visualization scripts.

## Reference

Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for
Clustering Analysis.* ICML 2016.

---

*Reproduced by [Tanishka Pagar](https://github.com/TanishkaPagar) as part of a
research internship at LABTECH.*