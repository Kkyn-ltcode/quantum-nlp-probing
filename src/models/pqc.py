"""
Configurable PQC Model for the QNLP pipeline.

Supports variable qubit count, layer depth, and entanglement topology.
Used for both the main experiments and ablation studies.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class HybridPQC(nn.Module):
    """
    Hybrid SBERT → Projection → PQC model.

    Args:
        sbert_dim: SBERT embedding dimension (384 for MiniLM)
        n_qubits: Number of qubits (4, 8, 12, 16, 20)
        n_layers: Number of variational layers
        entanglement: CNOT topology ('linear', 'circular', 'full', 'none')
    """
    def __init__(self, sbert_dim=384, n_qubits=16, n_layers=3,
                 entanglement='linear'):
        super().__init__()
        self.sbert_dim = sbert_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement

        # Trainable projection: SBERT → qubit angles
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)

        # Trainable scale for angle encoding
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))

        # PQC trainable weights
        self.weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits)
        )

        # Build the quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _get_entanglement_pairs(self):
        """Return list of (control, target) pairs for CNOT gates."""
        n = self.n_qubits
        if self.entanglement == 'none':
            return []
        elif self.entanglement == 'linear':
            return [(i, i+1) for i in range(n-1)]
        elif self.entanglement == 'circular':
            pairs = [(i, i+1) for i in range(n-1)]
            pairs.append((n-1, 0))
            return pairs
        elif self.entanglement == 'full':
            return [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            raise ValueError(f"Unknown entanglement: {self.entanglement}")

    def _build_circuit(self):
        """Build the QNode with current configuration."""
        entanglement_pairs = self._get_entanglement_pairs()
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for ctrl, tgt in entanglement_pairs:
                    qml.CNOT(wires=[ctrl, tgt])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x_sbert):
        """Forward pass for a batch. x_sbert: (batch, sbert_dim)."""
        h = torch.tanh(self.projection(x_sbert)) * self.scale
        batch_out = []
        for i in range(h.shape[0]):
            out = self.circuit(h[i], self.weights)
            batch_out.append(torch.stack(out))
        return torch.stack(batch_out).float()

    def forward_single(self, x_sbert):
        """Forward pass for a single sample. x_sbert: (sbert_dim,)."""
        h = torch.tanh(self.projection(x_sbert)) * self.scale
        out = self.circuit(h, self.weights)
        return torch.stack(out).float()

    def get_representations(self, x_sbert):
        """Extract intermediate representations (no grad)."""
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_scaled = torch.tanh(h_proj) * self.scale
            batch_out = []
            for i in range(h_scaled.shape[0]):
                out = self.circuit(h_scaled[i], self.weights)
                batch_out.append(torch.stack(out))
            h_pqc = torch.stack(batch_out).float()
        return {'projected': h_proj, 'output': h_pqc}

    def circuit_param_count(self):
        """Count only the PQC circuit parameters (not projection)."""
        return self.n_layers * self.n_qubits

    def total_param_count(self):
        """Count all trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (f"HybridPQC(qubits={self.n_qubits}, layers={self.n_layers}, "
                f"entanglement='{self.entanglement}', "
                f"circuit_params={self.circuit_param_count()}, "
                f"total_params={self.total_param_count()})")
