"""
Lesson 03: The Hybrid SBERT → PQC Pipeline
=============================================
HOMEWORK: Work through each exercise and fill in the TODOs.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson03_hybrid_pipeline.py

What this script does:
    1. Generates SBERT embeddings and explores the embedding space
    2. Builds a trainable linear projection (768 → 8)
    3. Wires SBERT + Projection + PQC into a single model
    4. Trains on a sentence similarity task
    5. Extracts intermediate representations for future probing

Your goal: Build the actual hybrid pipeline from our paper and understand
how information flows through each component.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# EXERCISE 1: Explore SBERT embeddings
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 1: SBERT Sentence Embeddings")
print("=" * 60)

# Load a lightweight SBERT model
print("\n  Loading SBERT model (all-MiniLM-L6-v2)...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')
try:
    sbert_dim = sbert.get_embedding_dimension()
except AttributeError:
    sbert_dim = sbert.get_sentence_embedding_dimension()
print(f"  Loaded. Embedding dimension: {sbert_dim}")

# Our test sentences (same as Lesson 1!)
sentences = [
    "dogs chase cats",               # 0: simple active
    "cats are chased by dogs",       # 1: passive (same meaning as 0)
    "big dogs chase small cats",     # 2: active with adjectives
    "dogs that chase cats run",      # 3: relative clause
    "dogs run",                      # 4: intransitive
    "birds fly south",               # 5: unrelated sentence
    "the weather is nice today",     # 6: completely unrelated
]

# Encode all sentences
print("\n  Encoding sentences...")
embeddings = sbert.encode(sentences, convert_to_numpy=True)
print(f"  Embeddings shape: {embeddings.shape}")

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(embeddings)

print("\n  Pairwise Cosine Similarity:")
print(f"  {'':35s}", end="")
for j in range(len(sentences)):
    print(f"  [{j}]", end="")
print()
for i in range(len(sentences)):
    label = sentences[i][:33].ljust(35)
    print(f"  {label}", end="")
    for j in range(len(sentences)):
        sim = sim_matrix[i, j]
        print(f" {sim:.2f}", end="")
    print()

# Save similarity heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
ax.set_xticks(range(len(sentences)))
ax.set_yticks(range(len(sentences)))
short_labels = [s[:25] + "..." if len(s) > 25 else s for s in sentences]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='black' if sim_matrix[i,j] > 0.3 else 'gray')
plt.colorbar(im, ax=ax, label='Cosine Similarity')
ax.set_title('SBERT Embedding Similarity Matrix', fontsize=14)
fig.tight_layout()
fig.savefig(output_dir / "sbert_similarity.png", dpi=150)
print(f"\n  Heatmap saved to: {output_dir / 'sbert_similarity.png'}")

# Key observation
print("""
  Key observations:
    - "dogs chase cats" and "cats are chased by dogs" should have
      HIGH similarity (same meaning, different syntax)
    - "dogs chase cats" and "the weather is nice" should have
      LOW similarity (completely unrelated)
    - SBERT captures SEMANTICS well. But does it capture SYNTAX?
      That's what our research investigates.
""")


# ═══════════════════════════════════════════════════════════
# EXERCISE 2: Trainable Linear Projection
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 2: Trainable Projection (768 → 8)")
print("=" * 60)

n_qubits = 8  # Our target dimensionality

# Convert SBERT embeddings to torch tensors
X_sbert = torch.tensor(embeddings, dtype=torch.float32)

# Build a simple projection layer
projection = nn.Linear(sbert_dim, n_qubits, bias=False)

# Project the embeddings
with torch.no_grad():
    X_proj = projection(X_sbert)  # shape: (7, 8)

print(f"\n  Before projection: {X_sbert.shape}")
print(f"  After projection:  {X_proj.shape}")

# Compare similarities before and after projection
sim_before = cosine_similarity(X_sbert.numpy())
sim_after = cosine_similarity(X_proj.numpy())

print(f"\n  Similarity preservation (before → after projection):")
print(f"  {'Sentence pair':45s} {'Before':>7} {'After':>7} {'Δ':>7}")
print(f"  {'─'*68}")
pairs = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (3, 4)]
for i, j in pairs:
    s1 = sentences[i][:20]
    s2 = sentences[j][:20]
    before = sim_before[i, j]
    after = sim_after[i, j]
    delta = after - before
    print(f"  {s1} ↔ {s2:20s}  {before:+.3f}  {after:+.3f}  {delta:+.3f}")

print("""
  Note: The random projection DISTORTS similarities.
  After training, the projection will learn to preserve
  the information that matters for the task.
""")


# ═══════════════════════════════════════════════════════════
# EXERCISE 3: Build the Full Hybrid Model
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 3: Full Hybrid Model")
print("=" * 60)

n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """
    PQC with angle encoding and trainable layers.
    Same architecture as Lesson 2, but with 8 qubits.
    """
    # Data encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Trainable variational layers
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
        # Ring of CNOTs for entanglement
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HybridQNLPModel(nn.Module):
    """
    The full hybrid pipeline:
        SBERT (frozen) → Linear Projection → Scaling → PQC → Output

    This is THE model from our paper.
    """
    def __init__(self, sbert_dim, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits

        # Trainable projection: 768 → n_qubits
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)

        # Scale projected features to [0, π] for angle encoding
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))

        # PQC trainable weights
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.3
        )

    def forward(self, x_sbert):
        """
        Args:
            x_sbert: (batch, 768) — SBERT embeddings (pre-computed, frozen)
        Returns:
            h_pqc: (batch, n_qubits) — PQC output representations
        Also stores intermediate representations for probing.
        """
        # Step 1: Project
        h_proj = self.projection(x_sbert)  # (batch, n_qubits)

        # Step 2: Scale to angle range
        h_scaled = torch.tanh(h_proj) * self.scale  # (batch, n_qubits)

        # Step 3: Run through PQC
        batch_outputs = []
        for i in range(h_scaled.shape[0]):
            qc_out = quantum_circuit(h_scaled[i], self.q_weights)
            batch_outputs.append(torch.stack(qc_out))
        h_pqc = torch.stack(batch_outputs)  # (batch, n_qubits)

        return h_pqc

    def get_intermediate_representations(self, x_sbert):
        """Extract representations at each stage for probing."""
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_scaled = torch.tanh(h_proj) * self.scale

            batch_outputs = []
            for i in range(h_scaled.shape[0]):
                qc_out = quantum_circuit(h_scaled[i], self.q_weights)
                batch_outputs.append(torch.stack(qc_out))
            h_pqc = torch.stack(batch_outputs)

        return {
            'sbert': x_sbert,        # (batch, 768)
            'projected': h_proj,      # (batch, 8)
            'scaled': h_scaled,       # (batch, 8)
            'pqc': h_pqc,             # (batch, 8)
        }


# Instantiate the model
model = HybridQNLPModel(sbert_dim, n_qubits, n_layers)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model created:")
print(f"    Projection: {sbert_dim} → {n_qubits} = {sbert_dim * n_qubits} params")
print(f"    Scale: 1 param")
print(f"    PQC weights: {n_layers} × {n_qubits} = {n_layers * n_qubits} params")
print(f"    Total trainable: {trainable_params}")

# Quick forward pass test
with torch.no_grad():
    test_output = model(X_sbert[:2])
    print(f"\n  Test forward pass:")
    print(f"    Input:  {X_sbert[:2].shape}")
    print(f"    Output: {test_output.shape}")
    print(f"    Values: {test_output[0].numpy().round(3)}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 4: Train on Sentence Similarity
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 4: Training the Hybrid Model")
print("=" * 60)

# Create a sentence pair dataset
# Paraphrase pairs (label = 1) and non-paraphrase pairs (label = 0)
train_pairs = [
    # Paraphrases (active/passive, same meaning)
    ("dogs chase cats", "cats are chased by dogs", 1.0),
    ("the boy kicked the ball", "the ball was kicked by the boy", 1.0),
    ("she wrote a letter", "a letter was written by her", 1.0),
    ("the cat sat on the mat", "the mat had a cat sitting on it", 1.0),
    ("he ate the apple", "the apple was eaten by him", 1.0),
    ("the teacher praised the student", "the student was praised by the teacher", 1.0),
    ("the dog bit the man", "the man was bitten by the dog", 1.0),
    ("rain falls from the sky", "from the sky rain falls", 1.0),

    # Non-paraphrases (different meaning)
    ("dogs chase cats", "birds fly south", 0.0),
    ("the boy kicked the ball", "the weather is nice today", 0.0),
    ("she wrote a letter", "fish swim in the ocean", 0.0),
    ("the cat sat on the mat", "cars drive on roads", 0.0),
    ("he ate the apple", "the sun is very bright", 0.0),
    ("the teacher praised the student", "the river flows to the sea", 0.0),
    ("the dog bit the man", "flowers bloom in spring", 0.0),
    ("rain falls from the sky", "computers process data", 0.0),
]

# Encode all unique sentences with SBERT
all_sents = list(set(
    [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
))
print(f"\n  Dataset: {len(train_pairs)} pairs, {len(all_sents)} unique sentences")

all_embeds = sbert.encode(all_sents, convert_to_numpy=True)
sent2embed = {s: torch.tensor(e, dtype=torch.float32) for s, e in zip(all_sents, all_embeds)}

# Prepare training data
X_a = torch.stack([sent2embed[p[0]] for p in train_pairs])
X_b = torch.stack([sent2embed[p[1]] for p in train_pairs])
y = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)

print(f"  X_a shape: {X_a.shape}")
print(f"  X_b shape: {X_b.shape}")
print(f"  Labels: {y.tolist()}")

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print(f"\n  Training...")
print(f"  {'Epoch':>5}  {'Loss':>8}  {'Accuracy':>9}")
print(f"  {'─'*26}")

losses = []
accs = []

for epoch in range(30):
    # Forward pass: encode both sentences through the hybrid model
    h_a = model(X_a)  # (16, 8)
    h_b = model(X_b)  # (16, 8)

    # Compute cosine similarity between each pair
    cos_sim = F.cosine_similarity(h_a, h_b, dim=1)  # (16,)

    # Map from [-1, 1] to [0, 1]
    pred = ((cos_sim + 1) / 2).float()  # ensure float32

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy(pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy
    acc = ((pred > 0.5).float() == y).float().mean()
    losses.append(loss.item())
    accs.append(acc.item())

    if epoch % 5 == 0 or epoch == 29:
        print(f"  {epoch:5d}  {loss.item():8.4f}  {acc.item():9.1%}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)

ax2.plot(accs, 'g-', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy'); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(output_dir / "hybrid_training.png", dpi=150)
print(f"\n  Training plot saved to: {output_dir / 'hybrid_training.png'}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 5: Extract Intermediate Representations
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 5: Extracting Representations for Probing")
print("=" * 60)

# Use our original 7 analysis sentences
X_analysis = torch.tensor(embeddings, dtype=torch.float32)  # (7, 768)

# Get representations at every stage
reps = model.get_intermediate_representations(X_analysis)

print(f"\n  Representations extracted:")
for name, tensor in reps.items():
    print(f"    {name:12s}: shape {tuple(tensor.shape)}")

# Compare similarity matrices at each stage
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
stage_names = ['sbert', 'projected', 'pqc']
stage_titles = [
    f'SBERT (dim={sbert_dim})',
    f'After Projection (dim={n_qubits})',
    f'After PQC (dim={n_qubits})'
]

for ax, name, title in zip(axes, stage_names, stage_titles):
    rep = reps[name].numpy()
    sim = cosine_similarity(rep)
    im = ax.imshow(sim, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(sentences)))
    ax.set_yticks(range(len(sentences)))
    ax.set_xticklabels([str(i) for i in range(len(sentences))], fontsize=9)
    ax.set_yticklabels([s[:18] for s in sentences], fontsize=8)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            ax.text(j, i, f'{sim[i,j]:.1f}', ha='center', va='center',
                    fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('How Similarity Structure Changes Through the Pipeline',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(output_dir / "representation_comparison.png", dpi=150)
print(f"  Comparison plot saved to: {output_dir / 'representation_comparison.png'}")

# Save representations for future probing (Lesson 4)
save_path = Path("results/representations.pt")
torch.save({
    'sentences': sentences,
    'sbert': reps['sbert'],
    'projected': reps['projected'],
    'pqc': reps['pqc'],
}, save_path)
print(f"  Representations saved to: {save_path}")

# Print the actual values at each stage for one sentence
print(f"\n  Detailed view for: \"{sentences[0]}\"")
print(f"    SBERT (first 10 of {sbert_dim}): "
      f"{reps['sbert'][0, :10].numpy().round(3)}")
print(f"    Projected (all {n_qubits}):      "
      f"{reps['projected'][0].numpy().round(3)}")
print(f"    After PQC (all {n_qubits}):      "
      f"{reps['pqc'][0].numpy().round(3)}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the SBERT similarity matrix. What is the cosine "
    "similarity between 'dogs chase cats' and 'cats are chased by dogs'? "
    "Is it close to 1.0? What does this tell you about SBERT?":
        "...",

    "Q2: How many trainable parameters does the projection layer have? "
    "How does this compare to the PQC's trainable parameters?":
        "...",

    "Q3: Why do we apply tanh() before scaling? What would happen if "
    "we fed raw projected values (which could be very large) as angles?":
        "...",

    "Q4: Look at the 3-panel similarity comparison plot. Does the PQC "
    "change the similarity structure compared to the projection? "
    "How?":
        "...",

    "Q5: In the training loop, we compute cosine similarity between "
    "PQC outputs. Why cosine similarity and not Euclidean distance?":
        "...",

    "Q6: After training, do the PQC representations for 'dogs chase cats' "
    "and 'cats are chased by dogs' become MORE or LESS similar compared "
    "to the SBERT representations? Why might this be?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print("\n\nDone! Review the saved figures in results/figures/")
print("Share this output with me for review.")
print("\nNext: Lesson 4 will use the saved representations for CKA probing!")
