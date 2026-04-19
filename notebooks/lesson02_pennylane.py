"""
Lesson 02: PennyLane Quantum Circuits
======================================
HOMEWORK: Work through each exercise and fill in the TODOs.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson02_pennylane.py

What this script does:
    1. Builds basic quantum circuits and inspects qubit states
    2. Shows how rotation gates change quantum states
    3. Implements angle encoding (classical → quantum)
    4. Builds a full Parameterized Quantum Circuit (PQC)
    5. Trains the PQC on a toy binary classification task

Your goal: Understand how a PQC works before we plug it into our
SBERT → PQC pipeline.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# EXERCISE 1: Your first quantum circuit
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 1: Basic quantum circuit")
print("=" * 60)

# Create a 1-qubit device (simulator)
dev1 = qml.device("default.qubit", wires=1)

# A quantum circuit that does NOTHING (identity)
@qml.qnode(dev1)
def circuit_identity():
    """Circuit with no gates. Qubit stays in |0⟩."""
    return qml.state()

# A circuit that flips the qubit to |1⟩
@qml.qnode(dev1)
def circuit_flip():
    """Apply a NOT gate (Pauli-X). Flips |0⟩ to |1⟩."""
    qml.PauliX(wires=0)
    return qml.state()

# A circuit that creates superposition |+⟩
@qml.qnode(dev1)
def circuit_hadamard():
    """Apply Hadamard gate. Creates equal superposition."""
    qml.Hadamard(wires=0)
    return qml.state()

print("\n  Circuit 1 (identity):  ", circuit_identity())
print("  Circuit 2 (flip):     ", circuit_flip())
print("  Circuit 3 (Hadamard): ", circuit_hadamard())

print("\n  Interpretation:")
print("    |0⟩ = [1, 0]  → 100% chance of measuring 0")
print("    |1⟩ = [0, 1]  → 100% chance of measuring 1")
print("    |+⟩ = [0.707, 0.707] → 50/50 chance")


# ═══════════════════════════════════════════════════════════
# EXERCISE 2: Rotation gates and the Bloch sphere
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 2: Rotation gates")
print("=" * 60)

dev2 = qml.device("default.qubit", wires=1)

@qml.qnode(dev2)
def circuit_ry(theta):
    """Single RY rotation. theta=0 → |0⟩, theta=π → |1⟩."""
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

# Sweep theta from 0 to 2π and measure PauliZ expectation
thetas = np.linspace(0, 2 * np.pi, 50)
expectations = [circuit_ry(t) for t in thetas]

# Plot the result
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thetas, expectations, 'b-', linewidth=2)
ax.set_xlabel('θ (radians)', fontsize=12)
ax.set_ylabel('⟨Z⟩ (expectation value)', fontsize=12)
ax.set_title('RY(θ) gate: How rotation angle affects measurement', fontsize=14)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=np.pi, color='red', linestyle='--', alpha=0.5, label='θ=π (|1⟩)')
ax.axvline(x=np.pi/2, color='green', linestyle='--', alpha=0.5, label='θ=π/2 (|+⟩)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "ry_sweep.png", dpi=150)
print(f"\n  Plot saved to: {output_dir / 'ry_sweep.png'}")

print(f"\n  Key observations:")
print(f"    θ = 0:   ⟨Z⟩ = {circuit_ry(0.0):.3f}  (qubit in |0⟩)")
print(f"    θ = π/2: ⟨Z⟩ = {circuit_ry(np.pi/2):.3f}  (qubit in |+⟩)")
print(f"    θ = π:   ⟨Z⟩ = {circuit_ry(np.pi):.3f}  (qubit in |1⟩)")


# ═══════════════════════════════════════════════════════════
# EXERCISE 3: Angle encoding — Classical data → Quantum
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 3: Angle encoding")
print("=" * 60)

n_qubits = 4
dev3 = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev3)
def angle_encoding_circuit(inputs):
    """
    Encode a 4-dimensional classical vector into 4 qubits.
    Each input value becomes a RY rotation angle.
    """
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Test with different input vectors
test_vectors = [
    np.array([0.0, 0.0, 0.0, 0.0]),           # All zeros
    np.array([np.pi, np.pi, np.pi, np.pi]),     # All π
    np.array([0.5, 1.0, 1.5, 2.0]),            # Increasing
    np.array([np.pi/4, np.pi/2, 3*np.pi/4, np.pi]),  # Quarter turns
]

print("\n  Input Vector              →  Measurement Output")
print("  " + "-" * 56)
for v in test_vectors:
    output = angle_encoding_circuit(v)
    v_str = "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
    o_str = "[" + ", ".join(f"{x:.3f}" for x in output) + "]"
    print(f"  {v_str:28s} →  {o_str}")

print("""
  Observation:
    - Input [0,0,0,0] → Output [1,1,1,1]  (all qubits in |0⟩)
    - Input [π,π,π,π] → Output [-1,-1,-1,-1]  (all qubits in |1⟩)
    - The output is cos(input) for each qubit!
    - This is because ⟨Z⟩ after RY(θ) = cos(θ)
""")


# ═══════════════════════════════════════════════════════════
# EXERCISE 4: Full Parameterized Quantum Circuit (PQC)
# ═══════════════════════════════════════════════════════════

print(f"{'=' * 60}")
print("  EXERCISE 4: Full PQC")
print("=" * 60)

n_qubits = 4
n_layers = 2
dev4 = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev4, interface="torch")
def pqc(inputs, weights):
    """
    A parameterized quantum circuit:
      1. Angle-encode the input data
      2. Apply trainable rotation + entanglement layers
      3. Measure PauliZ on all qubits

    Args:
        inputs: shape (n_qubits,) — compressed sentence embedding
        weights: shape (n_layers, n_qubits) — trainable parameters
    Returns:
        list of n_qubits expectation values
    """
    # Step 1: Data encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Step 2: Trainable layers
    for layer in range(n_layers):
        # Parameterized rotations
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)

        # Entangling CNOT ladder
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Close the ring (last qubit → first)
        qml.CNOT(wires=[n_qubits - 1, 0])

    # Step 3: Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# Initialize random weights and a sample input
torch.manual_seed(42)
weights = torch.randn(n_layers, n_qubits, requires_grad=True)
sample_input = torch.tensor([0.5, 1.0, 1.5, 2.0])

# Run the circuit
output = pqc(sample_input, weights)
print(f"\n  Input:   {sample_input.tolist()}")
print(f"  Weights shape: ({n_layers}, {n_qubits})")
print(f"  Output:  {[f'{x:.4f}' for x in output]}")

# Draw the circuit
print(f"\n  Circuit diagram:")
print(qml.draw(pqc)(sample_input, weights))

# Save the circuit figure
fig_circuit, ax_circuit = qml.draw_mpl(pqc)(sample_input, weights)
fig_circuit.savefig(output_dir / "pqc_circuit.png", dpi=150, bbox_inches='tight')
print(f"\n  Circuit diagram saved to: {output_dir / 'pqc_circuit.png'}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 5: Train the PQC on a toy task
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 5: Training a PQC")
print("=" * 60)

# Toy binary classification:
#   Class 0: inputs with small angles (close to 0)
#   Class 1: inputs with large angles (close to π)

np.random.seed(42)
torch.manual_seed(42)

# Generate toy dataset
n_samples = 40
class_0 = np.random.uniform(0, np.pi/3, size=(n_samples // 2, n_qubits))
class_1 = np.random.uniform(2*np.pi/3, np.pi, size=(n_samples // 2, n_qubits))

X = np.vstack([class_0, class_1])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Shuffle
perm = np.random.permutation(n_samples)
X = torch.tensor(X[perm], dtype=torch.float32)
y = torch.tensor(y[perm], dtype=torch.float32)

# Split into train/test
X_train, X_test = X[:30], X[30:]
y_train, y_test = y[:30], y[30:]


# Define the model
class QuantumClassifier(nn.Module):
    """
    A quantum classifier:
      Input (4 features) → PQC → first qubit expectation → sigmoid → prediction
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.5)

    def forward(self, x):
        """Run each sample through the PQC and return the first qubit's value."""
        predictions = []
        for sample in x:
            # Get expectation values from PQC
            output = pqc(sample, self.weights)
            # Use first qubit as prediction (map from [-1,1] to [0,1])
            pred = (output[0] + 1) / 2
            predictions.append(pred)
        return torch.stack(predictions).float()  # ensure float32 for BCELoss


model = QuantumClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
loss_fn = nn.BCELoss()

# Training loop
print(f"\n  Training quantum classifier...")
print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>9}  {'Test Acc':>8}")
print(f"  {'─'*35}")

losses = []
train_accs = []
test_accs = []

for epoch in range(25):
    # Forward pass
    preds_train = model(X_train)
    loss = loss_fn(preds_train, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracies
    with torch.no_grad():
        train_acc = ((preds_train > 0.5).float() == y_train).float().mean()
        preds_test = model(X_test)
        test_acc = ((preds_test > 0.5).float() == y_test).float().mean()

    losses.append(loss.item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())

    if epoch % 5 == 0 or epoch == 24:
        print(f"  {epoch:5d}  {loss.item():8.4f}  {train_acc.item():9.1%}  {test_acc.item():8.1%}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, 'b-', linewidth=2, label='Train')
ax2.plot(test_accs, 'r--', linewidth=2, label='Test')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Classification Accuracy', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(output_dir / "pqc_training.png", dpi=150)
print(f"\n  Training plot saved to: {output_dir / 'pqc_training.png'}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 6: Inspect the trained circuit
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 6: Understanding what the PQC learned")
print("=" * 60)

with torch.no_grad():
    print("\n  Trained weights:")
    for layer in range(n_layers):
        w = model.weights[layer].numpy()
        print(f"    Layer {layer}: [{', '.join(f'{x:.3f}' for x in w)}]")

    print("\n  Sample predictions:")
    print(f"    {'Input':42s} {'Pred':>6} {'True':>5} {'Correct':>8}")
    print(f"    {'─'*63}")
    for i in range(min(10, len(X_test))):
        pred = model(X_test[i:i+1]).item()
        true = y_test[i].item()
        correct = "✓" if (pred > 0.5) == true else "✗"
        inp = "[" + ", ".join(f"{x:.2f}" for x in X_test[i]) + "]"
        print(f"    {inp:42s} {pred:6.3f} {true:5.0f} {correct:>8}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

# TODO: Answer these questions by replacing "..." with your answers.

answers = {
    "Q1: In Exercise 2, what is ⟨Z⟩ when θ = π/2? "
    "Why is it 0 and not 0.5?":
        "...",

    "Q2: In Exercise 3, the output of angle encoding is cos(θ) "
    "for each qubit. If our input features are in range [-1, 1], "
    "should we scale them to [0, π] first? Why?":
        "...",

    "Q3: How many trainable parameters does our PQC have? "
    "(count from the weights shape)":
        "...",

    "Q4: In Exercise 5, why do we use only the FIRST qubit's "
    "expectation value for classification? Could we use all 4?":
        "...",

    "Q5: What role do the CNOT gates play? What would happen "
    "if we removed all CNOT gates from the circuit?":
        "...",

    "Q6: Look at the training curve. Did the PQC converge? "
    "What is the final test accuracy?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")


print("\n\nDone! Review the saved figures in results/figures/")
print("Share this output with me when ready for review.")
