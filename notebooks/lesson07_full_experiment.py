"""
Lesson 07: Full Experiment — Paper-Scale Results
=================================================
Trains PQC + MLP on the Tier 1 dataset with multiple seeds,
runs CKA analysis with permutation tests, and produces paper results.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson07_full_experiment.py

WARNING: Takes ~30-60 minutes (5 seeds × 2 models × training + parsing).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
N_EPOCHS = 50
N_QUBITS = 8
N_LAYERS = 2
LR = 0.05
N_PERMUTATIONS = 500

# ═══════════════════════════════════════════════════════════
# PART 1: Load Dataset & SBERT Encode
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 1: Loading dataset & encoding with SBERT")
print("=" * 60)

dataset_path = PROJECT_ROOT / "data" / "syntactic_dataset.pt"
if not dataset_path.exists():
    print(f"  ⚠ {dataset_path} not found! Run lesson06_dataset.py first.")
    exit(1)

ds = torch.load(dataset_path, weights_only=False)
sentences = ds['sentences']
fingerprints = ds['fingerprints']
train_pairs = ds['train_pairs']
test_pairs = ds['test_pairs']
constructions = ds['constructions']
metadata = ds['metadata']

print(f"  Sentences: {len(sentences)}")
print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

sbert = SentenceTransformer('all-MiniLM-L6-v2')
try:
    sbert_dim = sbert.get_embedding_dimension()
except AttributeError:
    sbert_dim = sbert.get_sentence_embedding_dimension()

print(f"  Encoding {len(sentences)} sentences with SBERT...")
sbert_embeddings = sbert.encode(sentences, convert_to_numpy=False,
                                 show_progress_bar=True)
sbert_embeddings = torch.tensor(np.array(sbert_embeddings), dtype=torch.float32)
print(f"  SBERT embeddings: {sbert_embeddings.shape}")

# Prepare pair tensors
def make_pair_tensors(pairs, embeddings):
    X_a = torch.stack([embeddings[p[0]] for p in pairs])
    X_b = torch.stack([embeddings[p[1]] for p in pairs])
    y = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    return X_a, X_b, y

X_train_a, X_train_b, y_train = make_pair_tensors(train_pairs, sbert_embeddings)
X_test_a, X_test_b, y_test = make_pair_tensors(test_pairs, sbert_embeddings)
print(f"  Train: {X_train_a.shape[0]} pairs, Test: {X_test_a.shape[0]} pairs")


# ═══════════════════════════════════════════════════════════
# PART 2: Model Definitions
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  PART 2: Model Definitions")
print("=" * 60)

# PQC
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class PQCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sbert_dim, N_QUBITS, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS))

    def forward(self, x):
        h = torch.tanh(self.projection(x)) * self.scale
        batch_out = []
        for i in range(h.shape[0]):
            out = quantum_circuit(h[i], self.weights)
            batch_out.append(torch.stack(out))
        return torch.stack(batch_out).float()

    def get_representations(self, x):
        with torch.no_grad():
            h_proj = self.projection(x)
            h_scaled = torch.tanh(h_proj) * self.scale
            batch_out = []
            for i in range(h_scaled.shape[0]):
                out = quantum_circuit(h_scaled[i], self.weights)
                batch_out.append(torch.stack(out))
            h_pqc = torch.stack(batch_out).float()
        return h_proj, h_pqc

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sbert_dim, N_QUBITS, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))
        self.mlp = nn.Sequential(
            nn.Linear(N_QUBITS, 4, bias=False), nn.Tanh(),
            nn.Linear(4, N_QUBITS, bias=False), nn.Tanh(),
        )

    def forward(self, x):
        h = torch.tanh(self.projection(x)) * self.scale
        return self.mlp(h)

    def get_representations(self, x):
        with torch.no_grad():
            h_proj = self.projection(x)
            h_scaled = torch.tanh(h_proj) * self.scale
            h_mlp = self.mlp(h_scaled)
        return h_proj, h_mlp

pqc_params = sum(p.numel() for p in PQCModel().parameters())
mlp_params = sum(p.numel() for p in MLPModel().parameters())
print(f"  PQC total params: {pqc_params}")
print(f"  MLP total params: {mlp_params}")


# ═══════════════════════════════════════════════════════════
# PART 3: Training Loop
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  PART 3: Training ({N_SEEDS} seeds × 2 models)")
print("=" * 60)

def train_model(model, X_a, X_b, y, n_epochs=N_EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        h_a, h_b = model(X_a), model(X_b)
        cos_sim = F.cosine_similarity(h_a, h_b, dim=1)
        pred = ((cos_sim + 1) / 2).float()
        loss = F.binary_cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Final accuracy
    with torch.no_grad():
        h_a, h_b = model(X_a), model(X_b)
        cos_sim = F.cosine_similarity(h_a, h_b, dim=1)
        pred = ((cos_sim + 1) / 2).float()
        train_acc = ((pred > 0.5).float() == y).float().mean().item()
        # Test
        h_ta, h_tb = model(X_test_a), model(X_test_b)
        cos_sim_t = F.cosine_similarity(h_ta, h_tb, dim=1)
        pred_t = ((cos_sim_t + 1) / 2).float()
        test_acc = ((pred_t > 0.5).float() == y_test).float().mean().item()
    return train_acc, test_acc

# CKA utilities
def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def hsic(K, L):
    n = K.shape[0]
    return np.sum(center_kernel(K) * center_kernel(L)) / ((n - 1) ** 2)

def compute_cka(X, Y):
    K, L = X @ X.T, Y @ Y.T
    hxy, hxx, hyy = hsic(K, L), hsic(K, K), hsic(L, L)
    if hxx < 1e-10 or hyy < 1e-10:
        return 0.0
    return hxy / np.sqrt(hxx * hyy)

# Normalize fingerprints once
fp = fingerprints.copy()
fp = (fp - fp.mean(0, keepdims=True)) / (fp.std(0, keepdims=True) + 1e-8)

# Storage for results across seeds
results = {'pqc': [], 'mlp': []}

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  ── Seed {seed+1}/{N_SEEDS} ──")

    for model_type in ['pqc', 'mlp']:
        t0 = time.time()
        model = PQCModel() if model_type == 'pqc' else MLPModel()
        train_acc, test_acc = train_model(model, X_train_a, X_train_b, y_train)

        # Extract representations
        h_proj, h_out = model.get_representations(sbert_embeddings)
        h_proj_np = h_proj.numpy()
        h_out_np = h_out.numpy()

        # Normalize
        def norm(x):
            return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)

        cka_sbert = compute_cka(norm(sbert_embeddings.numpy()), fp)
        cka_proj = compute_cka(norm(h_proj_np), fp)
        cka_out = compute_cka(norm(h_out_np), fp)

        elapsed = time.time() - t0
        print(f"    {model_type.upper():3s}: train={train_acc:.1%} test={test_acc:.1%} "
              f"CKA(out,syn)={cka_out:.4f}  [{elapsed:.0f}s]")

        results[model_type].append({
            'seed': seed,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cka_sbert_syntax': cka_sbert,
            'cka_proj_syntax': cka_proj,
            'cka_out_syntax': cka_out,
            'h_out': h_out_np,
        })


# ═══════════════════════════════════════════════════════════
# PART 4: Aggregate Results
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 4: Results Summary")
print("=" * 60)

def stats(vals):
    return np.mean(vals), np.std(vals)

pqc_cka = [r['cka_out_syntax'] for r in results['pqc']]
mlp_cka = [r['cka_out_syntax'] for r in results['mlp']]
pqc_test = [r['test_acc'] for r in results['pqc']]
mlp_test = [r['test_acc'] for r in results['mlp']]
sbert_cka = results['pqc'][0]['cka_sbert_syntax']  # same for all seeds
proj_cka_pqc = [r['cka_proj_syntax'] for r in results['pqc']]
proj_cka_mlp = [r['cka_proj_syntax'] for r in results['mlp']]

print(f"\n  {'Metric':<30s}  {'PQC':>16s}  {'MLP':>16s}")
print(f"  {'─'*66}")
print(f"  {'Test accuracy':<30s}  {stats(pqc_test)[0]:>6.1%} ± {stats(pqc_test)[1]:.1%}  "
      f"  {stats(mlp_test)[0]:>6.1%} ± {stats(mlp_test)[1]:.1%}")
print(f"  {'CKA(SBERT, Syntax)':<30s}  {sbert_cka:>16.4f}  {sbert_cka:>16.4f}")
print(f"  {'CKA(Projected, Syntax)':<30s}  "
      f"{stats(proj_cka_pqc)[0]:>6.4f} ± {stats(proj_cka_pqc)[1]:.4f}  "
      f"  {stats(proj_cka_mlp)[0]:>6.4f} ± {stats(proj_cka_mlp)[1]:.4f}")
print(f"  {'CKA(Output, Syntax)':<30s}  "
      f"{stats(pqc_cka)[0]:>6.4f} ± {stats(pqc_cka)[1]:.4f}  "
      f"  {stats(mlp_cka)[0]:>6.4f} ± {stats(mlp_cka)[1]:.4f}")

# PQC vs MLP CKA
pqc_vs_mlp = []
for i in range(N_SEEDS):
    h_pqc = results['pqc'][i]['h_out']
    h_mlp = results['mlp'][i]['h_out']
    def norm(x):
        return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
    pqc_vs_mlp.append(compute_cka(norm(h_pqc), norm(h_mlp)))

print(f"  {'CKA(PQC, MLP)':<30s}  {stats(pqc_vs_mlp)[0]:>6.4f} ± {stats(pqc_vs_mlp)[1]:.4f}")


# ═══════════════════════════════════════════════════════════
# PART 5: Permutation Test
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print(f"  PART 5: Permutation Test ({N_PERMUTATIONS} shuffles)")
print("=" * 60)

# Use seed 0 representations for permutation test
h_pqc_0 = results['pqc'][0]['h_out']
h_mlp_0 = results['mlp'][0]['h_out']

def norm(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)

real_cka_pqc = compute_cka(norm(h_pqc_0), fp)
real_cka_mlp = compute_cka(norm(h_mlp_0), fp)

null_pqc = []
null_mlp = []
for perm in range(N_PERMUTATIONS):
    perm_idx = np.random.permutation(len(fp))
    fp_shuffled = fp[perm_idx]
    null_pqc.append(compute_cka(norm(h_pqc_0), fp_shuffled))
    null_mlp.append(compute_cka(norm(h_mlp_0), fp_shuffled))

p_pqc = np.mean(np.array(null_pqc) >= real_cka_pqc)
p_mlp = np.mean(np.array(null_mlp) >= real_cka_mlp)

print(f"\n  CKA(PQC, Syntax) = {real_cka_pqc:.4f}, p = {p_pqc:.4f} "
      f"{'***' if p_pqc < 0.001 else '**' if p_pqc < 0.01 else '*' if p_pqc < 0.05 else 'n.s.'}")
print(f"  CKA(MLP, Syntax) = {real_cka_mlp:.4f}, p = {p_mlp:.4f} "
      f"{'***' if p_mlp < 0.001 else '**' if p_mlp < 0.01 else '*' if p_mlp < 0.05 else 'n.s.'}")
print(f"\n  Null distribution: PQC mean={np.mean(null_pqc):.4f}, MLP mean={np.mean(null_mlp):.4f}")


# ═══════════════════════════════════════════════════════════
# PART 6: Paper Figures
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 6: Generating Paper Figures")
print("=" * 60)

# Figure 1: Main bar chart with error bars
fig1, ax1 = plt.subplots(figsize=(10, 6))
stages = ['SBERT', 'Projected', 'Output']
pqc_means = [sbert_cka, stats(proj_cka_pqc)[0], stats(pqc_cka)[0]]
pqc_stds = [0, stats(proj_cka_pqc)[1], stats(pqc_cka)[1]]
mlp_means = [sbert_cka, stats(proj_cka_mlp)[0], stats(mlp_cka)[0]]
mlp_stds = [0, stats(proj_cka_mlp)[1], stats(mlp_cka)[1]]

x = np.arange(len(stages))
w = 0.35
bars1 = ax1.bar(x - w/2, pqc_means, w, yerr=pqc_stds, label='PQC',
                color='#55A868', edgecolor='black', capsize=5)
bars2 = ax1.bar(x + w/2, mlp_means, w, yerr=mlp_stds, label='MLP',
                color='#C44E52', edgecolor='black', capsize=5)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=13)
ax1.set_ylabel('CKA with Syntax Fingerprint', fontsize=13)
ax1.set_title(f'Syntactic Alignment at Each Pipeline Stage\n(mean ± std over {N_SEEDS} seeds)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig(output_dir / "paper_main_result.png", dpi=150)
print(f"  Figure 1: {output_dir / 'paper_main_result.png'}")

# Figure 2: Permutation test distribution
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
ax2a.hist(null_pqc, bins=30, alpha=0.7, color='#55A868', edgecolor='black')
ax2a.axvline(real_cka_pqc, color='red', linewidth=2, linestyle='--',
             label=f'Real CKA={real_cka_pqc:.4f}\np={p_pqc:.4f}')
ax2a.set_title('PQC: Permutation Test', fontsize=14, fontweight='bold')
ax2a.set_xlabel('CKA(shuffled Syntax)')
ax2a.legend(fontsize=11)
ax2a.grid(alpha=0.3)

ax2b.hist(null_mlp, bins=30, alpha=0.7, color='#C44E52', edgecolor='black')
ax2b.axvline(real_cka_mlp, color='red', linewidth=2, linestyle='--',
             label=f'Real CKA={real_cka_mlp:.4f}\np={p_mlp:.4f}')
ax2b.set_title('MLP: Permutation Test', fontsize=14, fontweight='bold')
ax2b.set_xlabel('CKA(shuffled Syntax)')
ax2b.legend(fontsize=11)
ax2b.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(output_dir / "paper_permutation_test.png", dpi=150)
print(f"  Figure 2: {output_dir / 'paper_permutation_test.png'}")

# Figure 3: Per-seed CKA scatter
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(range(N_SEEDS), pqc_cka, s=100, c='#55A868', label='PQC',
            edgecolors='black', zorder=3)
ax3.scatter(range(N_SEEDS), mlp_cka, s=100, c='#C44E52', label='MLP',
            edgecolors='black', zorder=3)
ax3.axhline(stats(pqc_cka)[0], color='#55A868', linestyle='--', alpha=0.5)
ax3.axhline(stats(mlp_cka)[0], color='#C44E52', linestyle='--', alpha=0.5)
ax3.set_xlabel('Random Seed', fontsize=13)
ax3.set_ylabel('CKA(Output, Syntax)', fontsize=13)
ax3.set_title('CKA Stability Across Seeds', fontsize=14, fontweight='bold')
ax3.legend(fontsize=12)
ax3.set_xticks(range(N_SEEDS))
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig(output_dir / "paper_seed_stability.png", dpi=150)
print(f"  Figure 3: {output_dir / 'paper_seed_stability.png'}")


# ═══════════════════════════════════════════════════════════
# PART 7: Save All Results
# ═══════════════════════════════════════════════════════════
save_path = PROJECT_ROOT / "results" / "full_experiment.pt"
torch.save({
    'results': results,
    'pqc_cka_scores': pqc_cka,
    'mlp_cka_scores': mlp_cka,
    'pqc_vs_mlp_cka': pqc_vs_mlp,
    'p_value_pqc': p_pqc,
    'p_value_mlp': p_mlp,
    'null_pqc': null_pqc,
    'null_mlp': null_mlp,
    'n_seeds': N_SEEDS,
    'n_permutations': N_PERMUTATIONS,
}, save_path)
print(f"\n  Results saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the per-seed scatter plot. Is CKA(PQC, Syntax) "
    "consistently higher than CKA(MLP, Syntax) across ALL seeds, "
    "or does it flip sometimes?":
        "...",

    "Q2: What are the p-values from the permutation test? "
    "Is the syntactic alignment statistically significant "
    "for PQC? For MLP?":
        "...",

    "Q3: Compare test accuracy between PQC and MLP. If they're "
    "similar, what does that confirm about our study design?":
        "...",

    "Q4: Look at the information flow: SBERT → Projected → Output. "
    "Does CKA with syntax INCREASE or DECREASE at each stage? "
    "Is the pattern the same for PQC and MLP?":
        "...",

    "Q5: Based on all the results, write 2-3 sentences for the "
    "paper's Abstract summarizing the key finding.":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print(f"\n\nDone! This is the paper's core experiment.")
print(f"Review figures in results/figures/ and share your answers.")
