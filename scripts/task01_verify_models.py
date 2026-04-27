"""
Task 01: Verify the Production Models
=======================================
Run this to verify all models work correctly before experiments.

    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python scripts/task01_verify_models.py
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.models.pqc import HybridPQC
from src.models.baselines import HybridMLP, HybridRKS, HybridAttention

torch.manual_seed(42)

print("=" * 60)
print("  TASK 01: Model Verification")
print("=" * 60)

SBERT_DIM = 384
N_QUBITS = 16
dummy_input = torch.randn(4, SBERT_DIM)  # batch of 4

# ── Build all models ──
models = {
    'PQC (linear)': HybridPQC(SBERT_DIM, N_QUBITS, n_layers=3, entanglement='linear'),
    'PQC (circular)': HybridPQC(SBERT_DIM, N_QUBITS, n_layers=3, entanglement='circular'),
    'PQC (none)': HybridPQC(SBERT_DIM, N_QUBITS, n_layers=3, entanglement='none'),
    'MLP': HybridMLP(SBERT_DIM, N_QUBITS, hidden_dim=4),
    'RKS': HybridRKS(SBERT_DIM, N_QUBITS, n_random_features=32),
    'Attention': HybridAttention(SBERT_DIM, N_QUBITS, head_dim=4),
}

print(f"\n  Testing {len(models)} models with {N_QUBITS} qubits...\n")
print(f"  {'Model':<20s}  {'Circuit':>8s}  {'Total':>8s}  {'Output':>12s}  {'Grad':>5s}")
print(f"  {'─'*58}")

all_pass = True
for name, model in models.items():
    try:
        # Forward pass
        out = model(dummy_input)
        assert out.shape == (4, N_QUBITS), f"Bad shape: {out.shape}"

        # Gradient check
        loss = out.sum()
        loss.backward()
        has_grad = all(p.grad is not None for p in model.parameters()
                       if p.requires_grad)

        # Representations
        reps = model.get_representations(dummy_input)
        assert 'projected' in reps and 'output' in reps

        model.zero_grad()

        shape_str = str(tuple(out.shape))
        print(f"  {name:<20s}  {model.circuit_param_count():>8d}  "
              f"{model.total_param_count():>8d}  "
              f"{shape_str:>12s}  {'✓' if has_grad else '✗':>5s}")

    except Exception as e:
        print(f"  {name:<20s}  FAILED: {e}")
        all_pass = False

# ── Ablation: qubit scaling ──
print(f"\n\n  Qubit Scaling Test (PQC):")
print(f"  {'Qubits':>8s}  {'Circuit':>8s}  {'Total':>8s}  {'Hilbert':>10s}")
print(f"  {'─'*40}")

for nq in [4, 8, 12, 16]:
    model = HybridPQC(SBERT_DIM, nq, n_layers=3, entanglement='linear')
    hilbert = 2 ** nq
    print(f"  {nq:>8d}  {model.circuit_param_count():>8d}  "
          f"{model.total_param_count():>8d}  {hilbert:>10d}")

# ── Summary ──
print(f"\n\n{'=' * 60}")
print(f"  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
print(f"{'=' * 60}")


# ── YOUR TASK ──
print(f"""
  ┌─────────────────────────────────────────────────┐
  │  YOUR TASK                                       │
  │                                                  │
  │  1. Run this script and verify all tests pass    │
  │  2. Answer the questions below                   │
  │  3. Share the output with me                     │
  └─────────────────────────────────────────────────┘
""")

answers = {
    "Q1: How many circuit params does the 16-qubit PQC (linear) "
    "have? How does this compare to the MLP's circuit params? "
    "Is this a fairer comparison than Lesson 5?":
        "the 16-qubit PQC (linear) has 48 circuit params, the MLP has 128 circuit params, almost 3 times more than PQC (linear) but the total parameters of those two quite similar (6193 and 6273) so i'd say its quite fair, at least fairer than Lesson 5.",

    "Q2: Look at the Hilbert space dimension for 16 qubits "
    "(65,536). The MLP operates in R^16. Why is this massive "
    "difference in representational capacity important for "
    "our argument?":
        "PQC do not scale linearly like classical models, they scale exponentially so with only 16 qubits they have about 6000 parameters to control a continuous 65000 dimensional space while a classical MLP will need 65000 X 65000 parameters to operate a hidden layers of 65000 neurons. this will demonstrate that quantum models can punch drastically above their weight class.",

    "Q3: What is the 'none' entanglement PQC? Why do we need "
    "it? (Hint: think ablation study — what happens if we "
    "remove all CNOTs?)":
        "the 'none' models remove all two-qubits gates, leaving only independent, single qubit rotations. if the pqc linear outperformed the pqc none, then we will have mathematical proof that the multi-body correlations enabled by the entanglement are doing the heavy lifting the learning task. if pqc none performs just as well as the entangled one, then the dataset's features are independent enough that the quantum correlations aren't actually helping and single qubit superposition is sufficient.",
}

for q, a in answers.items():
    print(f"  {q}")
    print(f"  → {a}\n")
