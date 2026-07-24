Experiment logs
===============

Console output from every reported experiment. Numbers here match the
results tables in the repository README exactly.

00_mnist_first_attempt_undertrained.txt   First attempt (SGD, short schedule) -> ACC 63.92%
01_mnist_adam_schedule.txt                Adam schedule, headline MNIST run -> ACC 81.61%
02_mnist_paper_schedule.txt               Paper-faithful schedule (50k/layer + 100k) -> ACC 80.05%
03_mnist_adam_second_run_stability.txt    Independent repeat -> ACC 82.28% (stability check)
04_ablation_no_backprop_mnist.txt         Standalone frozen-encoder ablation -> ACC 75.82%
05_baselines_both_datasets.txt            k-means (raw) + DEC w/o backprop, MNIST and REUTERS
06_reuters10k.txt                         REUTERS-10k full pipeline -> ACC 71.26%

Long training logs are abridged here for readability; the complete
unabridged output for every run is preserved with cell outputs in the
Colab notebooks under experiments/.

Trained weights (.pth) for reproducing these numbers without retraining
are attached to the GitHub Release (see the main README).
