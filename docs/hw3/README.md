# ✍️ Homework 3 — RNNs, LSTMs & Sequence Modeling

**Files:** [`hw/hw3/`](../../hw/hw3/) | **Report:** [`main3.pdf`](../../hw/hw3/main3.pdf)

## 📋 Overview

Assignment 3 dives into **recurrent neural networks**: the math behind gradient problems, deriving backpropagation-through-time (BPTT), and implementing RNN/LSTM variants — from scratch in NumPy and professionally in PyTorch.

## 📑 Questions Covered

1. **Vanishing/exploding gradients** — mathematical analysis of gradient decay in deep RNNs and the role of gate-based architectures (LSTM/GRU)
2. **BPTT derivation** — deriving the exact gradient of the error w.r.t. the gate weight matrices and the hidden state at time `t`
3. **Coupled Input-Forget Gate LSTM (CIFG-LSTM)** — a full NumPy implementation, executed on a synthetic time series, including an *optimized* batched variant (`ML-Hw3-Q3.py` / notebook)
4. **Vectorized (batched) Vanilla RNN** — `BatchedVanillaRNN` implemented with combined input-hidden weights, plus softmax + cross-entropy loss and a parity-check synthetic task, validated against a reference implementation (`ML-Hw3-Q4.py`)
5. **Professional BiLSTM** — a robust, training-ready `RobustBiLSTM` in PyTorch: bidirectional, stacked layers, dropout regularization, `pack_padded_sequence` for variable-length inputs, plus a balanced synthetic sequence-classification benchmark (`hw3_q5_py.py`)

## 📂 Folder Contents

| Path | Description |
|------|-------------|
| `IML4042_HW3.pdf` | Original problem statement |
| `IML4042_HW3_Sol .pdf` | Official solution |
| `main.pdf` / `main3.pdf` | My typeset LaTeX report (Persian) |
| `Notebooks/` | Interactive notebooks for Q3, Q4 & Q5 |
| `python codes/` | Standalone `.py` versions of each solution |
| `latex code and files/` | Full LaTeX source |

## 🧪 Key Code

### Q5 — Robust BiLSTM (`hw3_q5_py.py`)

```python
class RobustBiLSTM(nn.Module):
    """
    - Capture both past and future contextual information (Bidirectional)
    - Learn hierarchical temporal representations (Stacked LSTM)
    - Reduce overfitting using Dropout (Regularization)
    - Avoid unnecessary computation on padding tokens
    - Handle variable-length sequences efficiently
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 num_layers, num_classes, dropout=0.5):
        ...
```

The dataset builder (`generate_balanced_symmetric_dataset`) creates 10,000 balanced variable-length sequences, split into train/val, with class-distribution visualization before training.

### Q4 — Batched Vanilla RNN (`ML-Hw3-Q4.py`)

```python
class BatchedVanillaRNN:
    def __init__(self, features, hidden_size, output_size):
        # W_combined: (features + hidden_size, hidden_size) — single matmul per step
```

## 🛠 Dependencies

- `numpy`
- `torch` (Q5)
- `scikit-learn` (train/test split)

> 🔗 View the HW4 writeup → [`docs/hw4/README.md`](../hw4/README.md)