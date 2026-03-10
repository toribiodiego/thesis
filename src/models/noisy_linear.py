"""
Noisy linear layer for exploration in Rainbow DQN.

Implements factorised Gaussian noise from Fortunato et al. (2018),
"Noisy Networks for Exploration" (ICLR 2018). Drop-in replacement
for nn.Linear that adds learnable noise to weights and biases,
enabling state-dependent exploration without epsilon-greedy.

Key equations (Section 3, Eq. 10-11):
    eps^w_{i,j} = f(eps_i) * f(eps_j)   (weight noise, factorised)
    eps^b_j     = f(eps_j)               (bias noise)
    f(x)        = sgn(x) * sqrt(|x|)    (noise transform)
    y = (mu^w + sigma^w * eps^w) x + (mu^b + sigma^b * eps^b)

Factorised noise uses p+q noise variables instead of p*q+q,
reducing memory and computation for large layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Linear layer with factorised Gaussian noise for exploration.

    Replaces nn.Linear as a drop-in module. During training, the
    output includes learned noise scaled by sigma parameters. During
    eval, only the mean (mu) parameters are used for deterministic
    action selection.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        sigma_0: Initial noise magnitude. Default 0.5 per the paper.

    Attributes:
        weight_mu: Learnable mean weights (out_features, in_features).
        weight_sigma: Learnable noise scale for weights.
        bias_mu: Learnable mean bias (out_features,).
        bias_sigma: Learnable noise scale for bias.
    """

    def __init__(
        self, in_features: int, out_features: int, sigma_0: float = 0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # Learnable parameters: mu (mean) and sigma (noise magnitude)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers: not parameters, but need to be on the right device.
        # Factorised noise stores p+q vectors instead of p*q+q values.
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))

        self._initialize_parameters()
        self.reset_noise()

    def _initialize_parameters(self):
        """Initialize mu and sigma per Section 3.2 of the paper."""
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        sigma_init = self.sigma_0 / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Noise transform: f(x) = sgn(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample factorised noise vectors."""
        eps_in = self._f(torch.randn(self.in_features, device=self.eps_in.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.eps_out.device))
        self.eps_in.copy_(eps_in)
        self.eps_out.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional noise.

        During training: y = (mu^w + sigma^w * eps^w) x + (mu^b + sigma^b * eps^b)
        During eval: y = mu^w x + mu^b  (deterministic)

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        if self.training:
            # Factorised noise: weight noise is outer product of
            # transformed input and output noise vectors
            weight_eps = self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
            bias_eps = self.eps_out

            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
