# ✍️ Homework 5 — Transformers & MiniGPT (JAX)

**Files:** [`hw/hw5/`](../../hw/hw5/) | **Report:** [`main5.pdf`](../../hw/hw5/main5.pdf)

## 📋 Overview

Assignment 5 is a guided, bottom-up journey through **Transformer architectures and autoregressive language modeling**, implemented in **JAX**. You build a **MiniGPT** step by step — from tokenization all the way to text generation with modern sampling strategies — through five staged notebooks themed after Iranian tech companies (Snapp!, Digikala, Tapsi, Quera, Divar).

## 📑 The 5 Staged Notebooks

| # | Notebook | Theme | What You Build |
|---|----------|-------|----------------|
| 1 | [`01_intro_and_tokenization_snapp`](../../hw/hw5/Notebooks/01_intro_and_tokenization_snapp/) | **Snapp!** | Setup + tokenization: byte-pair / subword tokenization, token-id mapping, vocabulary construction | 
| 2 | [`02_self_attention_digikala`](../../hw/hw5/Notebooks/02_self_attention_digikala/) | **Digikala** | Scaled dot-product attention, causal masking, self-attention from scratch |
| 3 | [`03_multihead_and_block_tapsi`](../../hw/hw5/Notebooks/03_multihead_and_block_tapsi/) | **Tapsi** | Multi-head attention, LayerNorm, residual connections — the full Transformer decoder block |
| 4 | [`04_training_minigpt_quera`](../../hw/hw5/Notebooks/04_training_minigpt_quera/) | **Quera** | Assembling & training **MiniGPT** (config, parameter counts, warmup scheduling, training curves) |
| 5 | [`05_generation_and_sampling_divar`](../../hw/hw5/Notebooks/05_generation_and_sampling_divar/) | **Divar** | Text generation & sampling: temperature, top-k, top-p (nucleus) — diversity vs. quality trade-offs |

## 📂 Folder Contents

| Path | Description |
|------|-------------|
| `صورت سوالات و راهنمای حل تمرین.pdf` | Original problem statement & solution guide (Persian) |
| `main.pdf` / `main5.pdf` | My typeset LaTeX report (Persian) |
| `Notebooks/` | The 5 staged JAX notebooks (each with its own `requirements.txt` + images) |
| `Notebooks/04_training_minigpt_quera/minigpt_ckpt.pkl` | Pre-trained MiniGPT checkpoint for generation |
| `transformers-jax-lab.zip` | The original lab bundle (statements + starter code) |
| `latex code and files/` | Full LaTeX source |

## 🧪 What the Lab Covers

- **Tokenization** — subword tokenization & vocabulary lookup for text preprocessing
- **Attention** — `softmax(QKᵀ/√d)V` with causal masking, implemented in pure JAX
- **Multi-head attention & blocks** — head splitting, LayerNorm, residuals, the decoder block stack
- **Training** — MiniGPT config, parameter counting, AdamW with **warmup + cosine decay**, loss curves
- **Generation** — greedy → temperature sampling → **top-k** → **top-p** (nucleus), with diversity-vs-quality visualization of the next-token distribution, PCA/velocity analysis of embeddings, and more

## 🛠 Dependencies

- `jax`, `jaxlib`, `flax`
- `numpy`, `matplotlib`
- `tqdm`, `pickle` (checkpoint)

> Every staged notebook ships with its own `requirements.txt` under `Notebooks/*/`.

> 🔙 Back to the HW-index → [`docs/`](../)

> 🔗 View the Final Exam writeup → [`docs/final-exam/README.md`](../final-exam/README.md)