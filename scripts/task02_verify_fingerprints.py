"""
Task 02: Verify Graph Kernel Fingerprints
==========================================
Tests the WL graph kernel on real DisCoCat diagrams and compares
against the old count-based fingerprints.

    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python scripts/task02_verify_fingerprints.py

Requires: BobcatParser + bobcat model
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.fingerprint.graph_kernel import (
    WLFingerprint, diagram_to_graph, compute_wl_kernel_matrix
)
from src.analysis.cka import compute_cka, permutation_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# PART 1: Parse Test Sentences
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 1: Parsing test sentences")
print("=" * 60)

from lambeq import BobcatParser
parser = BobcatParser(model_name_or_path='bobcat', verbose='suppress')

# Sentences grouped by construction type
test_sentences = {
    'active': [
        "dogs chase cats",
        "the tall doctor examined the young patient",
        "the brave farmer praised the clever student",
        "the kind teacher helped the shy nurse",
        "cats watch dogs",
    ],
    'passive': [
        "cats are chased by dogs",
        "the young patient was examined by the tall doctor",
        "the clever student was praised by the brave farmer",
        "the shy nurse was helped by the kind teacher",
        "dogs are watched by cats",
    ],
    'relative': [
        "the doctor that helped the nurse examined the patient",
        "the farmer that praised the student watched the cat",
        "the teacher that called the driver helped the nurse",
    ],
    'cleft': [
        "it was the doctor who examined the patient",
        "it was the farmer who praised the student",
        "it was the teacher who helped the nurse",
    ],
}

all_sentences = []
all_labels = []  # construction type index
constructions = list(test_sentences.keys())

for ctype, sents in test_sentences.items():
    for s in sents:
        all_sentences.append(s)
        all_labels.append(constructions.index(ctype))

# Parse all
print(f"  Parsing {len(all_sentences)} sentences...")
diagrams = []
valid_indices = []

for i, sent in enumerate(all_sentences):
    try:
        diag = parser.sentence2diagram(sent)
        if diag is not None:
            diagrams.append(diag)
            valid_indices.append(i)
            print(f"    ✓ [{constructions[all_labels[i]]:8s}] \"{sent}\"")
        else:
            print(f"    ✗ [{constructions[all_labels[i]]:8s}] \"{sent}\" (null)")
    except Exception as e:
        print(f"    ✗ [{constructions[all_labels[i]]:8s}] \"{sent}\" ({e})")

valid_labels = [all_labels[i] for i in valid_indices]
valid_sents = [all_sentences[i] for i in valid_indices]
print(f"\n  Parsed: {len(diagrams)}/{len(all_sentences)}")


# ═══════════════════════════════════════════════════════════
# PART 2: Inspect Graph Structure
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 2: Graph Structure")
print("=" * 60)

print(f"\n  Comparing active vs passive graph structure:\n")

for i, (sent, diag) in enumerate(zip(valid_sents[:4], diagrams[:4])):
    nodes, adj = diagram_to_graph(diag)
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  [{constructions[valid_labels[i]]:8s}] \"{sent}\"")
    print(f"    Nodes ({len(nodes)}): {nodes}")
    print(f"    Edges: {n_edges}")
    print()


# ═══════════════════════════════════════════════════════════
# PART 3: WL Fingerprint Extraction
# ═══════════════════════════════════════════════════════════
print(f"{'=' * 60}")
print("  PART 3: WL Fingerprint Extraction")
print("=" * 60)

wl = WLFingerprint(n_iterations=3, max_features=128)
X_wl = wl.fit_transform(diagrams)

print(f"\n  WL feature matrix: {X_wl.shape}")
print(f"  Non-zero features: {np.count_nonzero(X_wl.sum(axis=0))}/{X_wl.shape[1]}")

# Show a few feature names
feature_names = wl.feature_names()
print(f"\n  Sample WL features (first 10):")
for i, fname in enumerate(feature_names[:10]):
    print(f"    [{i:3d}] {fname[:70]}")


# ═══════════════════════════════════════════════════════════
# PART 4: WL vs Count-Based Comparison
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 4: WL vs Count-Based — Discriminative Power")
print("=" * 60)

# Build the old count-based fingerprint for comparison
def count_fingerprint(diagram):
    boxes = diagram.boxes
    n_words = sum(1 for b in boxes if type(b).__name__ == "Word")
    n_cups = sum(1 for b in boxes if type(b).__name__ == "Cup")
    n_caps = sum(1 for b in boxes if type(b).__name__ == "Cap")
    n_swaps = sum(1 for b in boxes if type(b).__name__ == "Swap")
    total = len(boxes)
    cups_ratio = n_cups / max(n_words, 1)
    return np.array([n_words, n_cups, n_caps, n_swaps, total, cups_ratio])

X_count = np.stack([count_fingerprint(d) for d in diagrams])

# Cosine similarity matrices
from numpy.linalg import norm

def cosine_sim_matrix(X):
    norms = norm(X, axis=1, keepdims=True) + 1e-8
    X_n = X / norms
    return X_n @ X_n.T

cos_wl = cosine_sim_matrix(X_wl)
cos_count = cosine_sim_matrix(X_count)

# Compare active[0] vs passive[0]
print(f"\n  Cosine similarity: 'dogs chase cats' vs 'cats are chased by dogs'")
# Find indices
idx_active0 = valid_sents.index("dogs chase cats") if "dogs chase cats" in valid_sents else 0
idx_passive0 = valid_sents.index("cats are chased by dogs") if "cats are chased by dogs" in valid_sents else 1
print(f"    Count-based: {cos_count[idx_active0, idx_passive0]:.4f}")
print(f"    WL kernel:   {cos_wl[idx_active0, idx_passive0]:.4f}")

# Average within-group vs between-group similarity
print(f"\n  Within-group vs Between-group similarity:")
labels_arr = np.array(valid_labels)

for method_name, cos_mat in [("Count-based", cos_count), ("WL kernel", cos_wl)]:
    within_sims = []
    between_sims = []
    for i in range(len(cos_mat)):
        for j in range(i + 1, len(cos_mat)):
            if labels_arr[i] == labels_arr[j]:
                within_sims.append(cos_mat[i, j])
            else:
                between_sims.append(cos_mat[i, j])

    within_mean = np.mean(within_sims) if within_sims else 0
    between_mean = np.mean(between_sims) if between_sims else 0
    gap = within_mean - between_mean

    print(f"    {method_name:12s}: within={within_mean:.3f}  "
          f"between={between_mean:.3f}  gap={gap:+.3f}")


# ═══════════════════════════════════════════════════════════
# PART 5: CKA with Construction Labels
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 5: CKA — Does fingerprint discriminate constructions?")
print("=" * 60)

# One-hot construction labels
n = len(valid_labels)
n_types = len(constructions)
onehot = np.zeros((n, n_types))
for i, label in enumerate(valid_labels):
    onehot[i, label] = 1.0

# Normalize features
def normalize(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

cka_wl = compute_cka(normalize(X_wl), onehot)
cka_count = compute_cka(normalize(X_count), onehot)

print(f"\n  CKA(Fingerprint, ConstructionType):")
print(f"    Count-based: {cka_count:.4f}")
print(f"    WL kernel:   {cka_wl:.4f}")
print(f"    Improvement: {cka_wl/max(cka_count, 1e-8):.1f}×")

# Permutation test for WL
real_cka, p_val, null_dist = permutation_test(
    normalize(X_wl), onehot, n_permutations=500
)
print(f"\n  Permutation test (WL): CKA={real_cka:.4f}, p={p_val:.4f} "
      f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}")


# ═══════════════════════════════════════════════════════════
# PART 6: Visualization
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 6: Figures")
print("=" * 60)

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# Figure 1: Similarity matrices side by side
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels_short = [f"{constructions[l][0].upper()}{i}"
                for i, l in enumerate(valid_labels)]

im1 = ax1.imshow(cos_count, cmap='RdYlBu_r', vmin=0, vmax=1)
ax1.set_title('Count-Based Fingerprints', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(labels_short)))
ax1.set_yticks(range(len(labels_short)))
ax1.set_xticklabels(labels_short, rotation=45, fontsize=8)
ax1.set_yticklabels(labels_short, fontsize=8)

im2 = ax2.imshow(cos_wl, cmap='RdYlBu_r', vmin=0, vmax=1)
ax2.set_title('WL Graph Kernel', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(labels_short)))
ax2.set_yticks(range(len(labels_short)))
ax2.set_xticklabels(labels_short, rotation=45, fontsize=8)
ax2.set_yticklabels(labels_short, fontsize=8)
fig1.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label='Cosine Similarity')

fig1.suptitle('Syntactic Fingerprint Comparison:\nCount-Based vs WL Graph Kernel',
              fontsize=16, fontweight='bold')
fig1.tight_layout()
fig1.savefig(output_dir / "task02_fingerprint_comparison.png", dpi=150)
print(f"  Figure 1: {output_dir / 'task02_fingerprint_comparison.png'}")

# Figure 2: CKA bar chart
fig2, ax = plt.subplots(figsize=(8, 5))
methods = ['Count-Based', 'WL Kernel']
cka_vals = [cka_count, cka_wl]
bars = ax.bar(methods, cka_vals, color=['#C44E52', '#55A868'], edgecolor='black')
ax.set_ylabel('CKA with Construction Type', fontsize=13)
ax.set_title('Fingerprint Discriminative Power', fontsize=14, fontweight='bold')
for bar, val in zip(bars, cka_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig(output_dir / "task02_cka_comparison.png", dpi=150)
print(f"  Figure 2: {output_dir / 'task02_cka_comparison.png'}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the graph structure output (Part 2). How many "
    "nodes does 'dogs chase cats' have vs 'cats are chased by dogs'? "
    "Why are they different?":
        "...",

    "Q2: Compare the cosine similarity between active/passive for "
    "count-based vs WL kernel. Which method better separates "
    "them? What does this tell you about topological features?":
        "...",

    "Q3: Look at the within-group vs between-group gap. Which "
    "method has a larger gap? Why does a larger gap mean a "
    "better fingerprint for our paper?":
        "...",

    "Q4: The WL kernel uses 'iterative neighborhood hashing'. "
    "In simple terms, what does this mean? Why is it more "
    "powerful than just counting box types?":
        "...",

    "Q5: If CKA(WL, ConstructionType) is high and statistically "
    "significant (p < 0.05), what does this prove about our "
    "fingerprint design? Is it ready for the paper?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print(f"\n\nDone! Review figures in results/figures/")
print(f"Share your output and answers with me.")
