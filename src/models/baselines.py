"""
Classical baseline models: MLP, Random Kitchen Sinks, Attention, MPS.
All parameter-matched to the PQC for fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn


class HybridMLP(nn.Module):
    """
    MLP baseline replacing the PQC.
    Uses a bottleneck architecture to control parameter count.

    Args:
        sbert_dim: SBERT embedding dimension
        n_qubits: Input/output dimension (matches PQC)
        hidden_dim: Bottleneck width (controls param count)
    """
    def __init__(self, sbert_dim=384, n_qubits=16, hidden_dim=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))
        self.mlp = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits, bias=False),
            nn.Tanh(),
        )

    def forward(self, x_sbert):
        h = torch.tanh(self.projection(x_sbert)) * self.scale
        return self.mlp(h)

    def forward_single(self, x_sbert):
        return self.forward(x_sbert.unsqueeze(0)).squeeze(0)

    def get_representations(self, x_sbert):
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_scaled = torch.tanh(h_proj) * self.scale
            h_out = self.mlp(h_scaled)
        return {'projected': h_proj, 'output': h_out}

    def circuit_param_count(self):
        return sum(p.numel() for p in self.mlp.parameters())

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (f"HybridMLP(n_qubits={self.n_qubits}, "
                f"circuit_params={self.circuit_param_count()}, "
                f"total_params={self.total_param_count()})")


class HybridRKS(nn.Module):
    """
    Random Kitchen Sinks (Random Fourier Features) baseline.
    Projects input through fixed random features, then linear output.
    Tests whether random non-linearity suffices.
    """
    def __init__(self, sbert_dim=384, n_qubits=16, n_random_features=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))

        # Fixed random projection (NOT trainable)
        self.register_buffer('random_weights',
                             torch.randn(n_qubits, n_random_features) * 0.5)
        self.register_buffer('random_bias',
                             torch.rand(n_random_features) * 2 * np.pi)

        # Trainable output layer
        self.output_layer = nn.Linear(n_random_features, n_qubits, bias=False)

    def forward(self, x_sbert):
        h = torch.tanh(self.projection(x_sbert)) * self.scale
        # Random Fourier features: cos(Wx + b)
        z = torch.cos(h @ self.random_weights + self.random_bias)
        return torch.tanh(self.output_layer(z))

    def forward_single(self, x_sbert):
        return self.forward(x_sbert.unsqueeze(0)).squeeze(0)

    def get_representations(self, x_sbert):
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_out = self.forward(x_sbert)
        return {'projected': h_proj, 'output': h_out}

    def circuit_param_count(self):
        return self.output_layer.weight.numel()

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (f"HybridRKS(n_qubits={self.n_qubits}, "
                f"circuit_params={self.circuit_param_count()}, "
                f"total_params={self.total_param_count()})")


class HybridAttention(nn.Module):
    """
    Single-head self-attention baseline.
    Treats the n_qubits dimensions as a sequence of 1-dim tokens.
    """
    def __init__(self, sbert_dim=384, n_qubits=16, head_dim=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.head_dim = head_dim
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))

        # Q, K, V projections (no bias to reduce params)
        self.W_q = nn.Linear(1, head_dim, bias=False)
        self.W_k = nn.Linear(1, head_dim, bias=False)
        self.W_v = nn.Linear(1, head_dim, bias=False)
        self.W_o = nn.Linear(head_dim, 1, bias=False)

    def forward(self, x_sbert):
        h = torch.tanh(self.projection(x_sbert)) * self.scale
        # Reshape to (batch, n_qubits, 1) — treat dims as sequence
        h = h.unsqueeze(-1)  # (batch, n_qubits, 1)

        Q = self.W_q(h)  # (batch, n_qubits, head_dim)
        K = self.W_k(h)
        V = self.W_v(h)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)  # (batch, n_qubits, head_dim)
        out = self.W_o(context).squeeze(-1)  # (batch, n_qubits)
        return torch.tanh(out)

    def forward_single(self, x_sbert):
        return self.forward(x_sbert.unsqueeze(0)).squeeze(0)

    def get_representations(self, x_sbert):
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_out = self.forward(x_sbert)
        return {'projected': h_proj, 'output': h_out}

    def circuit_param_count(self):
        return sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith('W_'))

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (f"HybridAttention(n_qubits={self.n_qubits}, "
                f"circuit_params={self.circuit_param_count()}, "
                f"total_params={self.total_param_count()})")
