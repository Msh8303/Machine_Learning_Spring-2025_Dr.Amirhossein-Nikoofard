# mlhw2q4.py
import torch
import torch.nn as nn


class ContractiveDenoisingAE(nn.Module):
    """
    Single-layer Contractive (Denoising) Autoencoder with Sigmoid encoder.

    Encoder:
        h = sigmoid(W x_noisy + b)

    Decoder:
        x_hat = decoder(h)   (here: a simple linear layer)

    Contractive penalty:
        For encoder f_theta(x) = h = sigma(Wx + b),
        the Jacobian J_f(x) has entries:
            J_ij = ∂h_i / ∂x_j = sigma'(u_i) * W_ij
        where u = Wx + b and sigma'(u_i) = h_i (1 - h_i).

        Thus:
            ||J_f(x)||_F^2 = sum_i (sigma'(u_i))^2 * ||W_i||_2^2
                            = sum_i (h_i (1 - h_i))^2 * ||W_i||_2^2

        We compute this analytically without forming the full Jacobian.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        noise_std=0.1,
        lambda_c=1e-2
    ):
        """
        Args:
            input_dim (int): Dimensionality of input x.
            hidden_dim (int): Number of hidden units in encoder.
            noise_std (float): Standard deviation of Gaussian noise
                               for denoising behavior.
            lambda_c (float): Coefficient for contractive penalty term.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.lambda_c = lambda_c

        # Encoder: linear layer + Sigmoid activation
        self.encoder_linear = nn.Linear(input_dim, hidden_dim)
        self.encoder_act = nn.Sigmoid()

        # Decoder: here a simple linear layer
        self.decoder = nn.Linear(hidden_dim, input_dim)

    # ------------------------------------------------------------
    # Noise injection (for denoising behavior)
    # ------------------------------------------------------------
    def add_noise(self, x):
        """
        Add Gaussian noise to the input x.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Noisy input x_noisy.
        """
        if self.noise_std <= 0.0:
            return x
        noise = self.noise_std * torch.randn_like(x)
        return x + noise

    # ------------------------------------------------------------
    # Encoder / Decoder
    # ------------------------------------------------------------
    def encode(self, x, add_noise=True):
        """
        Encode input x into latent representation h.

        Args:
            x (Tensor): Input tensor [batch_size, input_dim].
            add_noise (bool): If True, apply Gaussian noise before encoding.

        Returns:
            h (Tensor): Latent representation [batch_size, hidden_dim].
            x_noisy (Tensor): Possibly noised input used by encoder.
            u (Tensor): Pre-activation u = Wx + b [batch_size, hidden_dim].
        """
        if add_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        u = self.encoder_linear(x_noisy)  # pre-activation Wx + b
        h = self.encoder_act(u)           # h = sigmoid(u)
        return h, x_noisy, u

    def decode(self, h):
        """
        Decode latent representation h into reconstructed input x_hat.

        Args:
            h (Tensor): Latent tensor [batch_size, hidden_dim].

        Returns:
            x_hat (Tensor): Reconstructed input [batch_size, input_dim].
        """
        x_hat = self.decoder(h)
        return x_hat

    # ------------------------------------------------------------
    # Analytic contractive penalty (Jacobian Frobenius norm)
    # ------------------------------------------------------------
    def contractive_penalty(self, h):
        """
        Compute the analytic contractive penalty:

            ||J_f(x)||_F^2 = sum_i (h_i (1 - h_i))^2 * ||W_i||_2^2

        where W_i is the i-th row of the encoder weight matrix W.

        Args:
            h (Tensor): Latent representation [batch_size, hidden_dim].

        Returns:
            Tensor (scalar): Mean Jacobian Frobenius norm over batch.
        """
        # Encoder weights W have shape [hidden_dim, input_dim]
        W = self.encoder_linear.weight  # [H, D]

        # For each hidden unit i, compute ||W_i||_2^2
        # row_norm_squared: [H]
        row_norm_squared = torch.sum(W ** 2, dim=1)

        # h: [B, H]
        dh = h * (1.0 - h)      # sigma'(u) = h (1 - h)
        dh2 = dh ** 2           # [B, H]

        # penalty_per_unit[b, i] = (h_i(1 - h_i))^2 * ||W_i||_2^2
        penalty_per_unit = dh2 * row_norm_squared  # [B, H]

        # sum over hidden units and average over batch
        penalty_per_sample = torch.sum(penalty_per_unit, dim=1)  # [B]
        penalty = torch.mean(penalty_per_sample)                 # scalar

        return penalty

    # ------------------------------------------------------------
    # Loss function: reconstruction + contractive penalty
    # ------------------------------------------------------------
    def loss(self, x):
        """
        Compute the total loss for a batch:

            L = L_reconstruction + lambda_c * ||J_f(x)||_F^2

        where the reconstruction loss is MSE between x and x_hat.

        Args:
            x (Tensor): Input batch [batch_size, input_dim].

        Returns:
            total_loss (Tensor): Scalar loss.
            recon_loss (Tensor): Reconstruction MSE.
            contr_penalty (Tensor): Contractive penalty term.
        """
        # Encode with noise (denoising behavior)
        h, x_noisy, u = self.encode(x, add_noise=True)

        # Decode
        x_hat = self.decode(h)

        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x_hat - x) ** 2)

        # Contractive penalty (Jacobian Frobenius norm)
        contr_penalty = self.contractive_penalty(h)

        # Total loss
        total_loss = recon_loss + self.lambda_c * contr_penalty

        return total_loss, recon_loss, contr_penalty
