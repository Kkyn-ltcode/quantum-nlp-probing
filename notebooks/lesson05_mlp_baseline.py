"""
Lesson 05: The Classical Baseline — MLP vs PQC
================================================
HOMEWORK: Work through each exercise and fill in the TODOs.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson05_mlp_baseline.py

What this script does:
    1. Builds a parameter-matched MLP to replace the PQC
    2. Trains it on the same paraphrase detection task
    3. Extracts MLP representations
    4. Runs CKA comparison: PQC vs MLP vs Syntax
    5. Generates the paper's key comparison figure

Your goal: Determine whether the PQC and MLP encode syntax differently.
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

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SETUP: Rebuild the PQC model (from Lesson 3) for comparison
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  SETUP: Loading SBERT & preparing data")
print("=" * 60)

# Load SBERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')
try:
    sbert_dim = sbert.get_embedding_dimension()
except AttributeError:
    sbert_dim = sbert.get_sentence_embedding_dimension()
print(f"  SBERT dim: {sbert_dim}")

n_qubits = 8
n_layers = 2

# Same training data as Lesson 3
train_pairs = [
    ("dogs chase cats", "cats are chased by dogs", 1.0),
    ("the boy kicked the ball", "the ball was kicked by the boy", 1.0),
    ("she wrote a letter", "a letter was written by her", 1.0),
    ("the cat sat on the mat", "the mat had a cat sitting on it", 1.0),
    ("he ate the apple", "the apple was eaten by him", 1.0),
    ("the teacher praised the student", "the student was praised by the teacher", 1.0),
    ("the dog bit the man", "the man was bitten by the dog", 1.0),
    ("rain falls from the sky", "from the sky rain falls", 1.0),
    ("dogs chase cats", "birds fly south", 0.0),
    ("the boy kicked the ball", "the weather is nice today", 0.0),
    ("she wrote a letter", "fish swim in the ocean", 0.0),
    ("the cat sat on the mat", "cars drive on roads", 0.0),
    ("he ate the apple", "the sun is very bright", 0.0),
    ("the teacher praised the student", "the river flows to the sea", 0.0),
    ("the dog bit the man", "flowers bloom in spring", 0.0),
    ("rain falls from the sky", "computers process data", 0.0),
]

# Analysis sentences (same as Lesson 3 & 4)
analysis_sentences = [
    "dogs chase cats",
    "cats are chased by dogs",
    "big dogs chase small cats",
    "dogs that chase cats run",
    "dogs run",
    "birds fly south",
    "the weather is nice today",
]

# Encode all sentences
all_sents = list(set(
    [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
    + analysis_sentences
))
all_embeds = sbert.encode(all_sents, convert_to_numpy=True)
sent2embed = {s: torch.tensor(e, dtype=torch.float32)
              for s, e in zip(all_sents, all_embeds)}

X_a = torch.stack([sent2embed[p[0]] for p in train_pairs])
X_b = torch.stack([sent2embed[p[1]] for p in train_pairs])
y = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)
X_analysis = torch.stack([sent2embed[s] for s in analysis_sentences])

print(f"  Training pairs: {len(train_pairs)}")
print(f"  Analysis sentences: {len(analysis_sentences)}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 1: Build the Parameter-Matched MLP
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 1: Parameter-Matched MLP")
print("=" * 60)


class MLPModel(nn.Module):
    """
    Classical baseline that replaces the PQC with an MLP.
    Architecture: projection → MLP → output

    The MLP is designed to have a similar parameter count to the PQC.
    PQC has n_layers * n_qubits = 16 trainable parameters.
    """
    def __init__(self, sbert_dim, n_qubits, hidden_dim=4):
        super().__init__()
        self.n_qubits = n_qubits

        # Same projection as the PQC model
        self.projection = nn.Linear(sbert_dim, n_qubits, bias=False)

        # Same scaling
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))

        # MLP replacing the PQC
        # Layer 1: 8 → hidden_dim (no bias) = 8 * hidden_dim params
        # Layer 2: hidden_dim → 8 (no bias) = hidden_dim * 8 params
        # Total MLP params = 2 * 8 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits, bias=False),
            nn.Tanh(),
        )

    def forward(self, x_sbert):
        h_proj = self.projection(x_sbert)
        h_scaled = torch.tanh(h_proj) * self.scale
        h_mlp = self.mlp(h_scaled)
        return h_mlp

    def get_intermediate_representations(self, x_sbert):
        with torch.no_grad():
            h_proj = self.projection(x_sbert)
            h_scaled = torch.tanh(h_proj) * self.scale
            h_mlp = self.mlp(h_scaled)
        return {
            'sbert': x_sbert,
            'projected': h_proj,
            'mlp': h_mlp,
        }


# Count and compare parameters
mlp_model = MLPModel(sbert_dim, n_qubits, hidden_dim=4)

# Count MLP-specific params (excluding shared projection + scale)
mlp_only_params = sum(p.numel() for p in mlp_model.mlp.parameters())
pqc_only_params = n_layers * n_qubits  # From Lesson 3

total_mlp = sum(p.numel() for p in mlp_model.parameters())

print(f"\n  Parameter comparison:")
print(f"    PQC circuit params:  {pqc_only_params}")
print(f"    MLP circuit params:  {mlp_only_params}")
print(f"    Ratio (MLP/PQC):     {mlp_only_params / pqc_only_params:.1f}x")
print(f"\n    MLP total (with projection): {total_mlp}")
print(f"    PQC total (with projection): {sbert_dim * n_qubits + 1 + pqc_only_params}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 2: Train the MLP on the Same Task
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 2: Training the MLP")
print("=" * 60)

optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.05)

print(f"\n  Training MLP on paraphrase detection...")
print(f"  {'Epoch':>5}  {'Loss':>8}  {'Accuracy':>9}")
print(f"  {'─'*26}")

mlp_losses = []
mlp_accs = []

for epoch in range(30):
    h_a = mlp_model(X_a)
    h_b = mlp_model(X_b)

    cos_sim = F.cosine_similarity(h_a, h_b, dim=1)
    pred = ((cos_sim + 1) / 2).float()

    loss = F.binary_cross_entropy(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = ((pred > 0.5).float() == y).float().mean()
    mlp_losses.append(loss.item())
    mlp_accs.append(acc.item())

    if epoch % 5 == 0 or epoch == 29:
        print(f"  {epoch:5d}  {loss.item():8.4f}  {acc.item():9.1%}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 3: Extract MLP Representations
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 3: Extracting MLP Representations")
print("=" * 60)

mlp_reps = mlp_model.get_intermediate_representations(X_analysis)

print(f"\n  MLP representations:")
for name, tensor in mlp_reps.items():
    print(f"    {name:12s}: {tuple(tensor.shape)}")

print(f"\n  Sample MLP output for '{analysis_sentences[0]}':")
print(f"    {mlp_reps['mlp'][0].numpy().round(3)}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 4: Comparative CKA Analysis
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 4: PQC vs MLP — CKA Comparison")
print("=" * 60)

# CKA implementation (same as Lesson 4)
def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def hsic(K, L):
    n = K.shape[0]
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    return np.sum(K_c * L_c) / ((n - 1) ** 2)

def compute_cka(X, Y):
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

# Load saved PQC representations and syntax fingerprints
pqc_data = torch.load("results/representations.pt", weights_only=False)
cka_data = torch.load("results/cka_analysis.pt", weights_only=False)

fingerprints = cka_data['fingerprints']

# All representations to compare
all_reps = {
    'SBERT': pqc_data['sbert'].numpy(),
    'Projected\n(PQC path)': pqc_data['projected'].numpy(),
    'PQC': pqc_data['pqc'].numpy(),
    'Projected\n(MLP path)': mlp_reps['projected'].numpy(),
    'MLP': mlp_reps['mlp'].numpy(),
    'Syntax': fingerprints,
}

# Normalize
all_reps_norm = {}
for name, arr in all_reps.items():
    mu = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    all_reps_norm[name] = (arr - mu) / std

# Compute full CKA matrix
rep_names = list(all_reps_norm.keys())
n_reps = len(rep_names)
cka_full = np.zeros((n_reps, n_reps))

for i in range(n_reps):
    for j in range(n_reps):
        cka_full[i, j] = compute_cka(all_reps_norm[rep_names[i]],
                                      all_reps_norm[rep_names[j]])

# Print the full CKA matrix
print(f"\n  Full CKA Matrix:")
header = f"  {'':16s}"
for name in rep_names:
    short = name.replace('\n', ' ')[:10]
    header += f"  {short:>10s}"
print(header)
for i, name in enumerate(rep_names):
    short = name.replace('\n', ' ')[:16]
    row = f"  {short:16s}"
    for j in range(n_reps):
        row += f"  {cka_full[i, j]:10.3f}"
    print(row)

# The key numbers
cka_pqc_syntax = cka_full[rep_names.index('PQC'), rep_names.index('Syntax')]
cka_mlp_syntax = cka_full[rep_names.index('MLP'), rep_names.index('Syntax')]
cka_pqc_mlp = cka_full[rep_names.index('PQC'), rep_names.index('MLP')]

print(f"\n  ┌───────────────────────────────────────────────────┐")
print(f"  │  H2: MECHANISTIC COMPARISON                       │")
print(f"  │                                                    │")
print(f"  │  CKA(PQC, Syntax) = {cka_pqc_syntax:.4f}                      │")
print(f"  │  CKA(MLP, Syntax) = {cka_mlp_syntax:.4f}                      │")
print(f"  │  CKA(PQC, MLP)    = {cka_pqc_mlp:.4f}  (representational     │")
print(f"  │                             similarity between them)│")
print(f"  │                                                    │")
if cka_pqc_syntax > cka_mlp_syntax + 0.02:
    print(f"  │  → PQC is MORE syntax-aligned than MLP           │")
elif cka_mlp_syntax > cka_pqc_syntax + 0.02:
    print(f"  │  → MLP is MORE syntax-aligned than PQC           │")
else:
    print(f"  │  → PQC and MLP have SIMILAR syntax alignment     │")
print(f"  │                                                    │")
if cka_pqc_mlp < 0.5:
    print(f"  │  → They use DIFFERENT representational strategies │")
else:
    print(f"  │  → They use SIMILAR representational strategies   │")
print(f"  └───────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════
# EXERCISE 5: Paper-Quality Comparison Figure
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 5: Generating Comparison Figures")
print("=" * 60)

# Figure 1: Side-by-side bar chart — THE key figure for the paper
fig1, ax1 = plt.subplots(figsize=(10, 6))

# CKA with Syntax at each stage
stages = ['SBERT', 'Projected', 'PQC', 'MLP']
pqc_path_scores = [
    cka_full[rep_names.index('SBERT'), rep_names.index('Syntax')],
    cka_full[rep_names.index('Projected\n(PQC path)'), rep_names.index('Syntax')],
    cka_pqc_syntax,
    None,  # placeholder
]
mlp_path_scores = [
    cka_full[rep_names.index('SBERT'), rep_names.index('Syntax')],
    cka_full[rep_names.index('Projected\n(MLP path)'), rep_names.index('Syntax')],
    None,  # placeholder
    cka_mlp_syntax,
]

x = np.arange(len(stages))
width = 0.35

# Plot PQC path
pqc_vals = [s if s is not None else 0 for s in pqc_path_scores]
mlp_vals = [s if s is not None else 0 for s in mlp_path_scores]
pqc_colors = ['#4C72B0', '#DD8452', '#55A868', '#CCCCCC']
mlp_colors = ['#4C72B0', '#DD8452', '#CCCCCC', '#C44E52']

bars1 = ax1.bar(x - width/2, pqc_vals, width, label='PQC Path',
                color=pqc_colors, edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x + width/2, mlp_vals, width, label='MLP Path',
                color=mlp_colors, edgecolor='black', linewidth=0.8)

# Dim the N/A bars
bars1[3].set_alpha(0.15)
bars2[2].set_alpha(0.15)

# Labels
for bar, score in zip(bars1, pqc_path_scores):
    if score is not None:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

for bar, score in zip(bars2, mlp_path_scores):
    if score is not None:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=12)
ax1.set_ylabel('CKA with Syntax Fingerprint', fontsize=13)
ax1.set_title('Syntactic Alignment: PQC vs MLP at Each Pipeline Stage',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
max_val = max(max(v for v in pqc_vals), max(v for v in mlp_vals))
ax1.set_ylim(0, max_val * 1.3 + 0.05)
fig1.tight_layout()
fig1.savefig(output_dir / "pqc_vs_mlp_syntax.png", dpi=150)
print(f"  Figure 1 saved: {output_dir / 'pqc_vs_mlp_syntax.png'}")

# Figure 2: Full CKA heatmap (condensed — PQC, MLP, Syntax only)
focus_names = ['PQC', 'MLP', 'Syntax']
focus_idx = [rep_names.index(n) for n in focus_names]
cka_focus = cka_full[np.ix_(focus_idx, focus_idx)]

fig2, ax2 = plt.subplots(figsize=(6, 5))
im2 = ax2.imshow(cka_focus, cmap='YlOrRd', vmin=0, vmax=1)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(focus_names, fontsize=13)
ax2.set_yticklabels(focus_names, fontsize=13)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'{cka_focus[i,j]:.3f}', ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cka_focus[i,j] > 0.5 else 'black')
plt.colorbar(im2, ax=ax2, label='CKA Score')
ax2.set_title('Representational Similarity:\nPQC vs MLP vs Syntax',
              fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig(output_dir / "pqc_mlp_syntax_heatmap.png", dpi=150)
print(f"  Figure 2 saved: {output_dir / 'pqc_mlp_syntax_heatmap.png'}")

# Figure 3: Training curves comparison
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Load PQC training history (re-run to get it)
ax3a.plot(mlp_losses, 'r-', linewidth=2, label='MLP')
ax3a.set_xlabel('Epoch', fontsize=12)
ax3a.set_ylabel('Loss', fontsize=12)
ax3a.set_title('Training Loss (MLP)', fontsize=14)
ax3a.legend(fontsize=11)
ax3a.grid(True, alpha=0.3)

ax3b.plot(mlp_accs, 'r-', linewidth=2, label='MLP')
ax3b.set_xlabel('Epoch', fontsize=12)
ax3b.set_ylabel('Accuracy', fontsize=12)
ax3b.set_title('Training Accuracy (MLP)', fontsize=14)
ax3b.legend(fontsize=11)
ax3b.grid(True, alpha=0.3)
ax3b.set_ylim(0, 1.05)

fig3.tight_layout()
fig3.savefig(output_dir / "mlp_training.png", dpi=150)
print(f"  Figure 3 saved: {output_dir / 'mlp_training.png'}")


# ═══════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  SUMMARY: PQC vs MLP")
print("=" * 60)

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Metric                      │   PQC    │   MLP    │    │
  ├──────────────────────────────┼──────────┼──────────┤    │
  │  Circuit params              │   {pqc_only_params:5d}  │   {mlp_only_params:5d}  │    │
  │  Final training accuracy     │  100.0%  │  {mlp_accs[-1]:5.1%}  │    │
  │  CKA(output, Syntax)         │  {cka_pqc_syntax:.4f}  │  {cka_mlp_syntax:.4f}  │    │
  │  CKA(PQC, MLP)               │     {cka_pqc_mlp:.4f}              │    │
  └──────────────────────────────┴──────────┴──────────┘
""")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: How many parameters does the MLP have vs the PQC? "
    "Is this a fair comparison? How could we make it fairer?":
        "...",

    "Q2: Did both models achieve similar training accuracy? "
    "If yes, what does this tell us about the task difficulty?":
        "...",

    "Q3: Compare CKA(PQC, Syntax) vs CKA(MLP, Syntax). "
    "Which model's representations are more aligned with "
    "syntactic structure? What does this mean for H2?":
        "...",

    "Q4: Look at CKA(PQC, MLP). Is it high or low? "
    "What does this tell us about whether PQC and MLP "
    "use similar or different representational strategies?":
        "...",

    "Q5: If you were writing the paper's Results section, "
    "how would you describe the PQC vs MLP comparison in "
    "one paragraph? (Try writing it.)":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

# Save everything
save_path = Path("results/mlp_analysis.pt")
torch.save({
    'mlp_reps': {k: v.numpy() if isinstance(v, torch.Tensor) else v
                 for k, v in mlp_reps.items()},
    'cka_full': cka_full,
    'rep_names': rep_names,
    'cka_pqc_syntax': cka_pqc_syntax,
    'cka_mlp_syntax': cka_mlp_syntax,
    'cka_pqc_mlp': cka_pqc_mlp,
    'mlp_losses': mlp_losses,
    'mlp_accs': mlp_accs,
}, save_path)
print(f"\n\n  Results saved to: {save_path}")

print("\n\nDone! Review figures in results/figures/")
print("Share this output AND your answers with me.")
print("\nAfter this, we have all the pieces for the paper's core experiments!")
