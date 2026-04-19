# Lesson 2: Your First Quantum Circuits in PennyLane

## Prerequisites
- ✅ Lesson 1 completed (you understand what DisCoCat diagrams are)
- ✅ `qnlp` conda environment working
- ✅ PennyLane installed

## What You'll Learn

By the end of this lesson, you will:
1. Understand what a qubit is and how quantum states work
2. Build quantum circuits with parameterized gates
3. Measure quantum circuits and get expectation values
4. Encode classical vectors into quantum circuits (angle encoding)
5. Train a quantum circuit with gradient descent

> [!IMPORTANT]
> **Connection to our research:** In our pipeline, sentence embeddings (768-dim SBERT vectors) get compressed to ~8 dimensions, then fed into a Parameterized Quantum Circuit. This lesson teaches you what that PQC actually IS.

---

## Part 1: What Is a Qubit?

### Classical bit vs Quantum bit

A classical bit is either `0` or `1`. A qubit can be in a **superposition** of both:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

where `α` and `β` are complex numbers satisfying `|α|² + |β|² = 1`.

- `|α|²` = probability of measuring `0`
- `|β|²` = probability of measuring `1`

### Key states to remember

| State | Vector | What it means |
|:------|:-------|:--------------|
| `\|0⟩` | `[1, 0]` | "Definitely 0" — the starting state |
| `\|1⟩` | `[0, 1]` | "Definitely 1" |
| `\|+⟩` | `[1/√2, 1/√2]` | "50/50 chance of 0 or 1" |

In PennyLane, every qubit starts in state `|0⟩`. We apply **gates** to rotate it.

### The Bloch Sphere

Think of a qubit as a point on a sphere:
- North pole = `|0⟩`
- South pole = `|1⟩`
- Equator = equal superposition (like `|+⟩`)

Gates **rotate** the point around different axes (X, Y, Z).

---

## Part 2: Quantum Gates

### Single-qubit rotation gates

These are the workhorses of PQCs. Each takes an **angle parameter** θ:

| Gate | What it does | PennyLane |
|:-----|:-------------|:----------|
| `RX(θ)` | Rotates around X-axis by angle θ | `qml.RX(theta, wires=0)` |
| `RY(θ)` | Rotates around Y-axis by angle θ | `qml.RY(theta, wires=0)` |
| `RZ(θ)` | Rotates around Z-axis by angle θ | `qml.RZ(theta, wires=0)` |

If θ = 0, the gate does nothing. If θ = π, it's a full flip.

Example:
```python
qml.RY(np.pi, wires=0)  # Flips |0⟩ to |1⟩
qml.RY(np.pi/2, wires=0)  # Creates |+⟩ from |0⟩
```

### Two-qubit entangling gate: CNOT

```python
qml.CNOT(wires=[0, 1])  # If qubit 0 is |1⟩, flip qubit 1
```

CNOT creates **entanglement** — correlations between qubits that have no classical analogue. This is what gives quantum circuits their power to model complex relationships.

### Why these gates matter for us

In our PQC, the rotation angles θ come from two sources:
1. **Data encoding:** The compressed SBERT embedding values become RY angles
2. **Trainable parameters:** Additional rotation angles that the circuit learns

---

## Part 3: Measurement

After applying gates, we **measure** the qubits. PennyLane offers several options:

```python
# Option 1: Expectation value of Pauli-Z
# Returns a number between -1 and +1
return qml.expval(qml.PauliZ(0))

# Option 2: Probability distribution
# Returns [prob_of_0, prob_of_1]
return qml.probs(wires=0)

# Option 3: State vector (for debugging, not for real quantum hardware)
return qml.state()
```

For our research, we use `qml.expval(qml.PauliZ(wires=i))` — one expectation value per qubit. With 4 qubits, we get a 4-dimensional output vector.

> [!TIP]
> **`expval(PauliZ)` explained simply:** It returns +1 if the qubit is in state `|0⟩`, -1 if in state `|1⟩`, and something in between for superpositions. Think of it as "how much is this qubit pointing north vs south on the Bloch sphere?"

---

## Part 4: Building a PQC

A Parameterized Quantum Circuit has this structure:

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐
│  Data        │   │  Trainable   │   │  Trainable   │   │         │
│  Encoding    │ → │  Layer 1     │ → │  Layer 2     │ → │ Measure │
│  (RY gates)  │   │  (RY + CNOT) │   │  (RY + CNOT) │   │         │
└─────────────┘   └──────────────┘   └──────────────┘   └─────────┘
```

### Step-by-step circuit construction:

```python
import pennylane as qml
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, weights):
    # 1. DATA ENCODING: embed classical data as rotation angles
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # 2. TRAINABLE LAYER: parameterized rotations + entanglement
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # 3. MEASUREMENT: read out expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

### What does `@qml.qnode(dev)` mean?

It tells PennyLane: "This function defines a quantum circuit. Run it on this device. And make it **differentiable** so we can compute gradients."

---

## Part 5: Angle Encoding

How do we get a classical vector into a quantum circuit?

**Angle encoding:** Each dimension of the input vector becomes a rotation angle.

```
Input vector: [0.3, 1.2, -0.5, 0.8]
                ↓     ↓      ↓     ↓
Circuit:     RY(0.3) RY(1.2) RY(-0.5) RY(0.8)
              q0      q1       q2       q3
```

This means: **the number of qubits = the dimensionality of our compressed embedding.**

In our pipeline:
- SBERT gives us 768 dimensions
- We compress to 8 dimensions (trainable linear projection)
- We use 8 qubits, one RY gate per dimension

> [!WARNING]
> Angle encoding means we can only handle as many features as we have qubits. On a simulator with 8 qubits, that's fine. On real quantum hardware, 8 qubits is trivially small. This is a key limitation we discuss in the paper.

---

## Part 6: Gradients and Training

The magic of PennyLane is that quantum circuits are **differentiable**. We can compute:

```
∂(circuit output) / ∂(trainable weight)
```

This means we can train quantum circuits with standard PyTorch optimizers:

```python
import torch

weights = torch.randn(n_qubits, requires_grad=True)
optimizer = torch.optim.Adam([weights], lr=0.1)

for step in range(100):
    inputs = torch.tensor([0.3, 1.2, -0.5, 0.8])
    output = circuit(inputs, weights)
    loss = some_loss_function(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

PennyLane uses the **parameter-shift rule** under the hood — a quantum-native way to compute exact gradients without backpropagation.

---

## Homework

Run the script at `notebooks/lesson02_pennylane.py`. It has 5 exercises:

1. **Build a basic circuit** and observe how gates change qubit states
2. **Visualize** circuit outputs on the Bloch sphere
3. **Build an angle encoding** circuit and see how input data maps to measurements
4. **Build a full PQC** with data encoding + trainable layers + CNOT entanglement
5. **Train the PQC** on a toy classification task

Answer the comprehension questions at the end.

> [!TIP]
> **Time estimate:** ~3-4 hours. Take your time with the visualizations — understanding what happens to qubit states as you change angles is crucial.
