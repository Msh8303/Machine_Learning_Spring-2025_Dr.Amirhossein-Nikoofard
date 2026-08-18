# ✍️ Homework 4 — Dimensionality Reduction & Generative Models

**Files:** [`hw/hw4/`](../../hw/hw4/) | **Report:** [`main 4.pdf`](../../hw/hw4/main%204.pdf)

## 📋 Overview

Assignment 4 explores **unsupervised representation learning** from two complementary angles:

1. **Dimensionality reduction** — comparing linear (PCA) and manifold (Isomap) embeddings of CIFAR-10 for image classification
2. **Deep generative modeling** — a **Conditional DCGAN** trained on medical imaging data

## 📑 Questions Covered

1. **PCA vs Isomap on CIFAR-10** (`ML_Hw4_Q2.ipynb`) — flattening & standardizing the 3-channel images, then evaluating:
   - Fixed-parameter PCA and Isomap embeddings (2D & higher dims) with visualization
   - K-Nearest-Neighbors classification accuracy on the embedded features
   - Dimensionality-reduction trade-offs (variance explained vs. manifold structure)
2. **Conditional DCGAN for medical imaging** (`Final_CDCGAN.ipynb` / `final_cdcgan.py`) — training a **conditional Deep Convolutional GAN** on **BreastMNIST** (MedMNIST):
   - Conditional generator/discriminator with class-embedding conditioning
   - Full training loop with loss curves, generated-sample visualization
   - Classification evaluation (confusion matrix, classification report) of the learned representations

## 📂 Folder Contents

| Path | Description |
|------|-------------|
| `IML4042_HW4.pdf` | Original problem statement |
| `main.pdf` / `main 4.pdf` | My typeset LaTeX report (Persian) |
| `Notebooks/ML_Hw4_Q2.ipynb` | Q2 — PCA / Isomap / KNN on CIFAR-10 |
| `Notebooks/Final_CDCGAN.ipynb` | Q1 — Conditional DCGAN on BreastMNIST |
| `python codes/ml_hw4_q2.py` | Q2 — standalone script |
| `python codes/final_cdcgan.py` | Q1 — standalone script (from the Colab notebook) |
| `view.php` | Actually a **ZIP archive** containing the original Q1 report PDF (download artifact from the course platform) |
| `latex code and files/` | Full LaTeX source |

## 🧪 Key Code

### Q1 — Conditional DCGAN (`final_cdcgan.py`)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import BreastMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- Conditioning via class embeddings injected into both generator and discriminator
- Reproducibility: global seed set to `42` (`set_seed`)
- Evaluation with `torchinfo`, `seaborn` confusion matrices and `sklearn` classification reports

### Q2 — Dimensionality Reduction (`ml_hw4_q2.py`)

```python
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Flatten → StandardScaler → PCA / Isomap → KNN
```

## 🛠 Dependencies

- `torch`, `torchvision`, `torchinfo`
- `medmnist`
- `keras` / `tensorflow` (CIFAR-10 loading)
- `scikit-learn` (PCA, Isomap, KNN, metrics)
- `matplotlib`, `seaborn`, `pandas`

> 🔗 View the HW5 writeup → [`docs/hw5/README.md`](../hw5/README.md)