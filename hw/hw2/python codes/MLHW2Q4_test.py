
"""
Test script for Contractive Denoising Autoencoder (CAE).

This script:
  1. Imports the ContractiveDenoisingAE model from mlhw2q4.py.
  2. Generates a tiny synthetic dataset:
       - 16 samples
       - each sample is a 20-dimensional vector (sinusoid + Gaussian noise)
  3. Trains the model for 50 epochs on this dataset.
  4. Generates a purely noisy "anomalous" sample (no sinusoidal structure).
  5. Computes reconstruction errors for:
       - training samples (normal)
       - the anomalous sample
     and uses them as anomaly scores.
  6. Logs detailed training information.
  7. Produces simple plots:
       - training loss vs. epoch
       - reconstruction error distribution for normal vs anomalous sample.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import model from the same directory (file: mlhw2q4.py)
from MLHW2Q4 import ContractiveDenoisingAE


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_sinusoid_dataset(
    num_samples: int = 16,
    input_dim: int = 20,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """
    Generate a tiny dataset of sinusoidal signals with Gaussian noise.

    Each sample:
      x(t) = sin(2π f t + φ) + ε
      where:
        - frequency f is sampled from a small range (e.g., [0.8, 1.2])
        - phase φ is sampled from [0, 2π]
        - ε ~ N(0, noise_std^2)

    The time axis is normalized to [0, 1].

    Returns:
        torch.Tensor of shape (num_samples, input_dim)
    """
    t = np.linspace(0.0, 1.0, input_dim, endpoint=False)
    data = []

    for _ in range(num_samples):
        # Random frequency and phase to create slight variations
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0.0, 2.0 * math.pi)

        clean_signal = np.sin(2.0 * math.pi * freq * t + phase)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=input_dim)
        noisy_signal = clean_signal + noise
        data.append(noisy_signal.astype(np.float32))

    data = np.stack(data, axis=0)  # shape: (num_samples, input_dim)
    return torch.from_numpy(data)


def generate_pure_noise_sample(
    input_dim: int = 20,
    noise_std: float = 1.0,
) -> torch.Tensor:
    """
    Generate a single purely noisy sample, i.e., an anomaly.

    This sample has no sinusoidal structure; it is just Gaussian noise.
    """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=input_dim).astype(
        np.float32
    )
    return torch.from_numpy(noise).unsqueeze(0)  # shape: (1, input_dim)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Simple training loop.

    Assumes model.loss(x) returns:
        total_loss, recon_loss, contractive_loss

    Logs per-epoch statistics and returns history for plotting.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "total_loss": [],
        "recon_loss": [],
        "contractive_loss": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_contractive = 0.0
        num_batches = 0

        for batch in train_loader:
            x_batch = batch[0].to(device)

            optimizer.zero_grad()

            # total_loss, recon_loss, contractive_loss
            total_loss, recon_loss, contractive_loss = model.loss(x_batch)

            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_recon += recon_loss.item()
            running_contractive += contractive_loss.item()
            num_batches += 1

        avg_total = running_total / num_batches
        avg_recon = running_recon / num_batches
        avg_contractive = running_contractive / num_batches

        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["contractive_loss"].append(avg_contractive)

        print(
            f"[Epoch {epoch:03d}] "
            f"total_loss={avg_total:.6f} "
            f"recon_loss={avg_recon:.6f} "
            f"contractive_loss={avg_contractive:.6f}"
        )

    return history


def compute_reconstruction_errors(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute reconstruction errors (MSE per sample) for a given batch x.

    We turn off noise during testing by calling encode with add_noise=False.

    Returns:
        Tensor of shape (num_samples,) containing per-sample MSE errors.
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        # Encode without noise for clean reconstruction
        h, _, _ = model.encode(x, add_noise=False)
        x_recon = model.decode(h)

        # Mean squared error per sample
        errors = torch.mean((x_recon - x) ** 2, dim=1)

    return errors.cpu()


def plot_training_history(history, output_dir: str = "./results"):
    """
    Plot training loss curves and save them as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["total_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["total_loss"], label="Total Loss")
    plt.plot(epochs, history["recon_loss"], label="Reconstruction Loss")
    plt.plot(epochs, history["contractive_loss"], label="Contractive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_losses.png"), dpi=150)
    plt.close()


def plot_reconstruction_errors(
    normal_errors: torch.Tensor,
    anomaly_error: float,
    output_dir: str = "./results",
):
    """
    Plot reconstruction errors for normal samples and the anomalous sample.

    Produces:
      - a bar plot comparing normal errors vs anomaly error.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_normal = normal_errors.shape[0]
    x_axis = np.arange(num_normal)

    plt.figure(figsize=(8, 5))
    plt.bar(
        x_axis,
        normal_errors.numpy(),
        color="blue",
        alpha=0.6,
        label="Normal samples",
    )

    # Plot anomaly error at an index beyond the normal ones
    anomaly_index = num_normal + 1
    plt.bar(
        [anomaly_index],
        [anomaly_error],
        color="red",
        alpha=0.8,
        label="Anomalous sample",
    )

    plt.xlabel("Sample index")
    plt.ylabel("Reconstruction MSE")
    plt.title("Reconstruction Error (Anomaly Score)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_errors.png"), dpi=150)
    plt.close()


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    input_dim = 20
    hidden_dim = 10  # you can change this
    batch_size = 4
    num_epochs = 50

    # -----------------------------
    # Generate training data
    # -----------------------------
    print("Generating sinusoidal training dataset ...")
    train_data = generate_sinusoid_dataset(
        num_samples=16,
        input_dim=input_dim,
        noise_std=0.1,
    )

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # -----------------------------
    # Initialize model
    # -----------------------------
    print("Initializing ContractiveDenoisingAE model ...")
    model = ContractiveDenoisingAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        noise_std=0.1,   # should match training noise
        lambda_c=1e-3    # contractive penalty coefficient
    )

    # -----------------------------
    # Train model
    # -----------------------------
    print("Starting training ...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=1e-3,
        device=device,
    )

    # -----------------------------
    # Compute reconstruction errors for training samples
    # -----------------------------
    print("Computing reconstruction errors for training (normal) samples ...")
    normal_errors = compute_reconstruction_errors(
        model=model,
        x=train_data,
        device=device,
    )

    print("Per-sample reconstruction MSE (normal samples):")
    for idx, err in enumerate(normal_errors):
        print(f"  Sample {idx:02d}: MSE = {err.item():.6f}")

    avg_normal_err = normal_errors.mean().item()
    print(f"Average reconstruction error (normal): {avg_normal_err:.6f}")

    # -----------------------------
    # Generate anomalous sample and compute its error
    # -----------------------------
    print("Generating anomalous sample (pure noise) ...")
    anomaly_sample = generate_pure_noise_sample(
        input_dim=input_dim,
        noise_std=1.0,
    )

    anomaly_error = compute_reconstruction_errors(
        model=model,
        x=anomaly_sample,
        device=device,
    )[0].item()

    print(f"Anomalous sample reconstruction MSE (Anomaly Score): {anomaly_error:.6f}")

    # Compare anomaly error with normal average
    ratio = anomaly_error / (avg_normal_err + 1e-12)
    print(
        f"Anomaly error is {ratio:.2f} times the average normal reconstruction error."
    )

    # -----------------------------
    # Plot results
    # -----------------------------
    output_dir = "./results"
    print(f"Saving plots and logs to: {output_dir}")
    plot_training_history(history, output_dir=output_dir)
    plot_reconstruction_errors(
        normal_errors=normal_errors,
        anomaly_error=anomaly_error,
        output_dir=output_dir,
    )

    print("Done. Check the 'results' directory for generated plots.")


if __name__ == "__main__":
    main()
