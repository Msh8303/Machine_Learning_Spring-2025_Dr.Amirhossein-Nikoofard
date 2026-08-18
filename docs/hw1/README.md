# ✍️ Homework 1 — ML Foundations & Backpropagation

**Files:** [`hw/hw1/`](../../hw/hw1/) | **Report:** [`main.pdf`](../../hw/hw1/main.pdf)

## 📋 Overview

The first assignment covers the **mathematical foundations of neural networks**: forward/backward propagation, gradient-based optimization, and a first hands-on classification task.

## 📑 Questions Covered

1. **Neural network intuition** — conceptual questions on perceptrons, activation functions, and network capacity
2. **Backpropagation** — deriving and implementing gradient computation by hand and in code
3. **Optimization** — training dynamics with different neuron counts (1–4 hidden neurons) and the effect of architecture on capacity
4. **ADAM optimizer** — implementing the backward pass and ADAM updates
5. **"Habitable Zone" classification** — a binary classification problem on a two-moon (concentric circles) dataset, solved with logistic regression and compared against a small neural network

## 📂 Folder Contents

| Path | Description |
|------|-------------|
| `IML4042_HW1.pdf` | Original problem statement |
| `IML4042_HW1_Sol.pdf` | Official solution |
| `main.pdf` | My typeset LaTeX report (in Persian) |
| `latex code and files/` | Full LaTeX source (`.tex`, fonts, images, bibliography) |
| `python codes/MlHw1-Q5.py` | Q5 — "Habitable Zone" dataset generation & classification |
| `python codes/backward_and_adam_update.py` | Q4 — manual backpropagation + ADAM update implementation |

## 🧪 Key Code

### Q5 — Habitable Zone Classification (`MlHw1-Q5.py`)
```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate the "Habitable Zone" dataset
X, y = make_circles(n_samples=600, noise=0.05, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Q4 — Backward Pass + ADAM Update (`backward_and_adam_update.py`)
```python
import numpy as np

def backward_and_adam_update(X, A1, W2, dZ2, W1, b1, m, t, m_t, v_t):
    """Manual backpropagation through a 2-layer ReLU net with ADAM updates."""
    # ... full derivation and update rules inside
```

## 🛠 Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn`

> 🔗 View the HW2 writeup → [`docs/hw2/README.md`](../hw2/README.md)