# ✍️ Homework 2 — CNNs & Autoencoders

**Files:** [`hw/hw2/`](../../hw/hw2/) | **Report:** [`main2.pdf`](../../hw/hw2/main2.pdf)

## 📋 Overview

Assignment 2 moves into **convolutional neural networks and unsupervised representation learning**: learnable convolution kernels, YOLO-style object detection losses, and contractive/denoising autoencoders for anomaly detection.

## 📑 Questions Covered

1. **Learnable convolution masks** — a single convolution layer with a trainable mask `W_final = W * M`, implemented from scratch via `F.unfold` + `torch.matmul` (`Ml_hw2_q1.ipynb`)
2. **Bottleneck blocks** — ResNet-style `BottleneckBlock` (1×1 → 3×3 → 1×1) with parameter-count analysis across different filter configurations (`ml_hw2_q2.ipynb`)
3. **Dice loss for YOLO** — a PyTorch implementation of a *Soft Dice* bounding-box loss (`YOLODiceLoss`) with classification, confidence, and overlap analysis (`ML-HW2-Q3.ipynb`)
4. **Contractive Denoising Autoencoder (CAE)** — an anomaly-detection autoencoder with an analytical Jacobian-based contractive penalty (`MLHW2Q4.py` + test script)
5. **Conceptual questions** — CAE/DAE objective function analysis, reconstruction-vs-representation trade-offs

## 📂 Folder Contents

| Path | Description |
|------|-------------|
| `IML4042_HW2.pdf` | Original problem statement |
| `IML4042_HW2_Sol.pdf` | Official solution |
| `main.pdf` / `main2.pdf` | My typeset LaTeX report (Persian) |
| `Notebooks/` | Interactive notebooks for Q1, Q2 & Q3 |
| `python codes/MLHW2Q4.py` | Q4 — Contractive Denoising Autoencoder implementation |
| `python codes/MLHW2Q4_test.py` | Q4 — test: trains on synthetic sinusoid data for anomaly detection |
| `latex code and files/` | Full LaTeX source |

## 🧪 Key Code

### Q4 — Contractive Denoising Autoencoder (`MLHW2Q4.py`)

```python
class ContractiveDenoisingAE(nn.Module):
    """
    Single-layer Contractive (Denoising) Autoencoder with Sigmoid encoder.

    Contractive penalty (computed analytically, no full Jacobian):
        ||J_f(x)||_F^2 = sum_i (h_i (1 - h_i))^2 * ||W_i||_2^2
    """
```

The companion `MLHW2Q4_test.py` trains the CAE on 16 synthetic 20-D samples (sinusoid + Gaussian noise), then uses **reconstruction error as an anomaly score** to flag a purely-noisy sample.

### Q3 — YOLO Dice Loss (`ML-HW2-Q3.ipynb`)

```python
class YOLODiceLoss(nn.Module):
    def __init__(self, S=7, B=2, num_classes=20, lambda_noobj=0.5, eps=1e-6):
        ...
    # L = 1 - (2I + eps) / (A_gt + A_pred + eps)   (soft Dice on bounding boxes)
```

## 🛠 Dependencies

- `torch`
- `numpy`
- `matplotlib`

> 🔗 View the HW3 writeup → [`docs/hw3/README.md`](../hw3/README.md)