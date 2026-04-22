"""
Lesson 04: Syntax Fingerprints & CKA Probing
==============================================
HOMEWORK: Work through each exercise and fill in the TODOs.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson04_probing.py

What this script does:
    1. Parses sentences with BobcatParser and extracts structural fingerprints
    2. Implements CKA (Centered Kernel Alignment) from scratch
    3. Loads Lesson 3's saved representations
    4. Computes CKA between every pipeline stage and syntax fingerprints
    5. Generates paper-quality figures

Your goal: Run the core analysis from our paper and interpret the results.
"""

import numpy as np
import torch
from pathlib import Path
from lambeq import BobcatParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# Resolve project root from script location so paths work from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# EXERCISE 1: Extract Structural Fingerprints
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 1: DisCoCat Structural Fingerprints")
print("=" * 60)

# The same sentences from Lesson 3
sentences = [
    "dogs chase cats",               # 0: simple active SVO
    "cats are chased by dogs",       # 1: passive (same meaning, different syntax!)
    "big dogs chase small cats",     # 2: active with adjectives
    "dogs that chase cats run",      # 3: relative clause
    "dogs run",                      # 4: simple intransitive SV
    "birds fly south",               # 5: unrelated SV + adverb
    "the weather is nice today",     # 6: copular construction
]

# Parse all sentences
print("\n  Loading BobcatParser...")
parser = BobcatParser(model_name_or_path='bobcat', verbose='suppress')
print("  Parsing sentences...")

diagrams = []
for sent in sentences:
    d = parser.sentence2diagram(sent)
    diagrams.append(d)
    print(f"    ✓ \"{sent}\"")


def extract_fingerprint(diagram, max_type_atoms=10):
    """
    Extract a structural fingerprint vector from a DisCoCat diagram.

    This captures ONLY grammatical structure — no semantic content.
    Two sentences with the same grammar produce identical fingerprints.

    Returns:
        np.array of shape (fingerprint_dim,)
    """
    boxes = diagram.boxes

    # Count box types
    n_words = 0
    n_cups = 0
    n_caps = 0
    n_swaps = 0
    word_type_complexities = []

    for box in boxes:
        box_type = type(box).__name__
        if box_type == "Word":
            n_words += 1
            # Count atomic types in this word's cod (output type)
            cod_str = str(box.cod)
            # Count atomic symbols (n, s, etc.)
            atoms = [c for c in cod_str.replace('.r', '').replace('.l', '')
                     .replace('(', '').replace(')', '')
                     .split(' @ ') if c.strip()]
            word_type_complexities.append(len(atoms))
        elif box_type == "Cup":
            n_cups += 1
        elif box_type == "Cap":
            n_caps += 1
        elif box_type == "Swap":
            n_swaps += 1

    # Compute aggregate features
    total_boxes = len(boxes)
    n_other = total_boxes - n_words - n_cups - n_caps - n_swaps

    avg_type_complexity = (np.mean(word_type_complexities)
                          if word_type_complexities else 0)
    max_type_complexity = (max(word_type_complexities)
                          if word_type_complexities else 0)

    # Type atom histogram: count occurrences of 'n', 's', and other types
    full_cod_str = ' '.join(str(box.cod) for box in boxes
                           if type(box).__name__ == "Word")
    n_count = full_cod_str.count('n')
    s_count = full_cod_str.count('s')

    # Diagram-level features
    cod_str = str(diagram.cod)
    dom_str = str(diagram.dom)

    # Cups-to-words ratio (measure of syntactic complexity)
    cups_ratio = n_cups / max(n_words, 1)

    # Build the fingerprint vector
    fingerprint = np.array([
        n_words,                    # 0: word count
        n_cups,                     # 1: cup count (syntactic bonds)
        n_caps,                     # 2: cap count
        n_swaps,                    # 3: swap count
        total_boxes,                # 4: total box count
        n_other,                    # 5: other box count
        avg_type_complexity,        # 6: avg atoms per word type
        max_type_complexity,        # 7: max atoms per word type
        n_count,                    # 8: number of noun-type atoms
        s_count,                    # 9: number of sentence-type atoms
        cups_ratio,                 # 10: cups/words ratio
        n_words - n_cups,           # 11: words minus cups (always 1 for valid sentence)
        max_type_complexity - avg_type_complexity,  # 12: type complexity spread
    ], dtype=np.float64)

    return fingerprint


# Extract fingerprints for all sentences
print("\n  Extracting structural fingerprints...")
fingerprints = []
for i, (sent, diag) in enumerate(zip(sentences, diagrams)):
    fp = extract_fingerprint(diag)
    fingerprints.append(fp)
    print(f"\n    [{i}] \"{sent}\"")
    print(f"        words={fp[0]:.0f}  cups={fp[1]:.0f}  "
          f"total_boxes={fp[4]:.0f}  avg_complexity={fp[6]:.2f}  "
          f"cups_ratio={fp[10]:.2f}")

fingerprints = np.stack(fingerprints)
print(f"\n  Fingerprint matrix shape: {fingerprints.shape}")

# Verify: same-grammar sentences should have similar fingerprints
from sklearn.metrics.pairwise import cosine_similarity

fp_sim = cosine_similarity(fingerprints)
print(f"\n  Fingerprint cosine similarity:")
print(f"    'dogs chase cats' vs 'cats are chased by dogs': "
      f"{fp_sim[0, 1]:.3f}")
print(f"    'dogs chase cats' vs 'big dogs chase small cats': "
      f"{fp_sim[0, 2]:.3f}")
print(f"    'dogs chase cats' vs 'dogs run': "
      f"{fp_sim[0, 4]:.3f}")

print("""
  Key check:
    - Active vs Passive should be LOW (different syntax, same meaning)
    - Similar structures should be HIGH
    - This is the OPPOSITE of SBERT similarity (which is meaning-based)
""")


# ═══════════════════════════════════════════════════════════
# EXERCISE 2: Implement CKA from Scratch
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  EXERCISE 2: CKA (Centered Kernel Alignment)")
print("=" * 60)


def center_kernel(K):
    """Center a kernel matrix by removing row and column means."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    return H @ K @ H


def hsic(K, L):
    """
    Hilbert-Schmidt Independence Criterion.
    Measures statistical dependence between two kernel matrices.
    """
    n = K.shape[0]
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    return np.sum(K_c * L_c) / ((n - 1) ** 2)


def compute_cka(X, Y):
    """
    Compute CKA (Centered Kernel Alignment) between representations X and Y.

    Args:
        X: np.array of shape (n_samples, d1)
        Y: np.array of shape (n_samples, d2)

    Returns:
        float: CKA score in [0, 1]

    CKA = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    """
    # Linear kernel
    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


# Quick sanity check
print("\n  CKA sanity checks:")
X_rand1 = np.random.randn(7, 10)
X_rand2 = np.random.randn(7, 10)

print(f"    CKA(X, X)      = {compute_cka(X_rand1, X_rand1):.4f}  (should be 1.0)")
print(f"    CKA(X, random)  = {compute_cka(X_rand1, X_rand2):.4f}  (should be ~0)")
print(f"    CKA(X, 2*X)    = {compute_cka(X_rand1, 2*X_rand1):.4f}  (should be 1.0 — scale invariant)")

# Test with rotated version (CKA is NOT invariant to arbitrary linear transforms,
# but IS invariant to orthogonal transforms and isotropic scaling)
Q, _ = np.linalg.qr(np.random.randn(10, 10))  # Random orthogonal matrix
X_rotated = X_rand1 @ Q
print(f"    CKA(X, X@Q)    = {compute_cka(X_rand1, X_rotated):.4f}  (should be 1.0 — rotation invariant)")


# ═══════════════════════════════════════════════════════════
# EXERCISE 3: Load Representations & Run CKA Analysis
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 3: The Core Probing Analysis")
print("=" * 60)

# Load saved representations from Lesson 3
rep_path = PROJECT_ROOT / "results" / "representations.pt"
if not rep_path.exists():
    print("\n  ⚠ results/representations.pt not found!")
    print("    Please run lesson03_hybrid_pipeline.py first.")
    exit(1)

saved = torch.load(rep_path, weights_only=False)
print(f"\n  Loaded representations from: {rep_path}")
print(f"    Sentences: {saved['sentences']}")

# Convert all to numpy
reps = {
    'SBERT': saved['sbert'].numpy(),
    'Projected': saved['projected'].numpy(),
    'PQC': saved['pqc'].numpy(),
    'Syntax': fingerprints,  # Our structural fingerprints from Exercise 1
}

print(f"\n  Representation shapes:")
for name, arr in reps.items():
    print(f"    {name:12s}: {arr.shape}")

# Normalize each representation (zero mean, unit variance per feature)
reps_norm = {}
for name, arr in reps.items():
    mu = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    reps_norm[name] = (arr - mu) / std

# Compute CKA between ALL pairs
rep_names = list(reps_norm.keys())
n_reps = len(rep_names)
cka_matrix = np.zeros((n_reps, n_reps))

print(f"\n  Computing CKA matrix...")
for i in range(n_reps):
    for j in range(n_reps):
        cka_matrix[i, j] = compute_cka(reps_norm[rep_names[i]],
                                        reps_norm[rep_names[j]])

# Print the CKA matrix
print(f"\n  CKA Matrix:")
print(f"  {'':12s}", end="")
for name in rep_names:
    print(f"  {name:>10s}", end="")
print()
for i, name in enumerate(rep_names):
    print(f"  {name:12s}", end="")
    for j in range(n_reps):
        score = cka_matrix[i, j]
        print(f"  {score:10.3f}", end="")
    print()

# THE KEY COMPARISON
cka_proj_syntax = cka_matrix[rep_names.index('Projected'),
                              rep_names.index('Syntax')]
cka_pqc_syntax = cka_matrix[rep_names.index('PQC'),
                             rep_names.index('Syntax')]
cka_sbert_syntax = cka_matrix[rep_names.index('SBERT'),
                               rep_names.index('Syntax')]

print(f"\n  ┌─────────────────────────────────────────────┐")
print(f"  │  THE KEY COMPARISON                          │")
print(f"  │                                              │")
print(f"  │  CKA(SBERT, Syntax)     = {cka_sbert_syntax:.4f}            │")
print(f"  │  CKA(Projected, Syntax) = {cka_proj_syntax:.4f}            │")
print(f"  │  CKA(PQC, Syntax)       = {cka_pqc_syntax:.4f}            │")
print(f"  │                                              │")
if cka_pqc_syntax > cka_proj_syntax:
    print(f"  │  → PQC INCREASED syntactic alignment! ↑     │")
elif cka_pqc_syntax < cka_proj_syntax:
    print(f"  │  → PQC DECREASED syntactic alignment ↓      │")
else:
    print(f"  │  → PQC did NOT change syntactic alignment    │")
print(f"  └─────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════
# EXERCISE 4: Paper-Quality Visualizations
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 4: Generating Figures")
print("=" * 60)

# Figure 1: CKA Heatmap
fig1, ax1 = plt.subplots(figsize=(8, 6))
im = ax1.imshow(cka_matrix, cmap='YlOrRd', vmin=0, vmax=1)
ax1.set_xticks(range(n_reps))
ax1.set_yticks(range(n_reps))
ax1.set_xticklabels(rep_names, fontsize=12, rotation=45, ha='right')
ax1.set_yticklabels(rep_names, fontsize=12)
for i in range(n_reps):
    for j in range(n_reps):
        ax1.text(j, i, f'{cka_matrix[i,j]:.3f}',
                 ha='center', va='center', fontsize=11,
                 color='white' if cka_matrix[i,j] > 0.6 else 'black')
plt.colorbar(im, ax=ax1, label='CKA Score')
ax1.set_title('Representational Similarity (CKA)\nBetween Pipeline Stages and Syntax',
              fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig(output_dir / "cka_heatmap.png", dpi=150)
print(f"  Figure 1 saved: {output_dir / 'cka_heatmap.png'}")

# Figure 2: Bar chart — CKA with Syntax at each stage
fig2, ax2 = plt.subplots(figsize=(8, 5))
stages = ['SBERT', 'Projected', 'PQC']
syntax_cka_scores = [cka_matrix[rep_names.index(s), rep_names.index('Syntax')]
                     for s in stages]
colors = ['#4C72B0', '#DD8452', '#55A868']
bars = ax2.bar(stages, syntax_cka_scores, color=colors, width=0.6, edgecolor='black')
ax2.set_ylabel('CKA with Syntax Fingerprint', fontsize=13)
ax2.set_title('Syntactic Alignment at Each Pipeline Stage', fontsize=14,
              fontweight='bold')
ax2.set_ylim(0, max(syntax_cka_scores) * 1.3 + 0.05)
ax2.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, syntax_cka_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=12,
             fontweight='bold')
# Add arrow annotation for the key comparison
if len(syntax_cka_scores) >= 3:
    ax2.annotate('', xy=(2, syntax_cka_scores[2]),
                 xytext=(1, syntax_cka_scores[1]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
fig2.tight_layout()
fig2.savefig(output_dir / "cka_syntax_stages.png", dpi=150)
print(f"  Figure 2 saved: {output_dir / 'cka_syntax_stages.png'}")

# Figure 3: Fingerprint similarity vs SBERT similarity
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# SBERT similarities
sbert_sim = cosine_similarity(reps['SBERT'])
im3a = ax3a.imshow(sbert_sim, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
ax3a.set_title('SBERT Similarity\n(captures MEANING)', fontsize=12,
               fontweight='bold')
ax3a.set_xticks(range(len(sentences)))
ax3a.set_yticks(range(len(sentences)))
short = [s[:18] for s in sentences]
ax3a.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax3a.set_yticklabels(short, fontsize=8)
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax3a.text(j, i, f'{sbert_sim[i,j]:.2f}', ha='center', va='center',
                  fontsize=7)
plt.colorbar(im3a, ax=ax3a, fraction=0.046)

# Syntax fingerprint similarities
fp_sim_plot = cosine_similarity(reps['Syntax'])
im3b = ax3b.imshow(fp_sim_plot, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
ax3b.set_title('Syntax Fingerprint Similarity\n(captures GRAMMAR)', fontsize=12,
               fontweight='bold')
ax3b.set_xticks(range(len(sentences)))
ax3b.set_yticks(range(len(sentences)))
ax3b.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax3b.set_yticklabels(short, fontsize=8)
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax3b.text(j, i, f'{fp_sim_plot[i,j]:.2f}', ha='center', va='center',
                  fontsize=7)
plt.colorbar(im3b, ax=ax3b, fraction=0.046)

fig3.suptitle('Semantics vs. Syntax: Two Different Views of the Same Sentences',
              fontsize=14, fontweight='bold')
fig3.tight_layout()
fig3.savefig(output_dir / "semantics_vs_syntax.png", dpi=150)
print(f"  Figure 3 saved: {output_dir / 'semantics_vs_syntax.png'}")


# ═══════════════════════════════════════════════════════════
# EXERCISE 5: Interpret the Results
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  EXERCISE 5: Interpretation")
print("=" * 60)

print(f"""
  Let's connect the CKA results to our research hypotheses.

  Recall from the paper pitch:

  H1 (Structural encoding):
      CKA(PQC, Syntax) > CKA(Projected, Syntax)
      → The PQC increases syntactic alignment.

  Our measurements:
      CKA(SBERT, Syntax)     = {cka_sbert_syntax:.4f}
      CKA(Projected, Syntax) = {cka_proj_syntax:.4f}
      CKA(PQC, Syntax)       = {cka_pqc_syntax:.4f}

  Information flow:
      SBERT ({cka_sbert_syntax:.3f}) → Projected ({cka_proj_syntax:.3f}) → PQC ({cka_pqc_syntax:.3f})
""")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the fingerprint similarity. What is the cosine similarity "
    "between 'dogs chase cats' and 'cats are chased by dogs'? Compare this "
    "to their SBERT similarity (0.89). What does this difference tell you?":
        "The cosine similarity between 'dogs chase cats' and 'cats are chased by dogs' is low because they different in structure eventhough they mean exactly the same thing, while SBERT similarity is high because they capture the semantic so they are opposite.",

    "Q2: The CKA sanity checks show CKA(X, 2*X) = 1.0. Why is scale "
    "invariance important for our analysis? (Hint: PQC outputs are in "
    "[-1,1] but syntax fingerprints have values like 5, 6, 10.)":
        "Because what we want to know is the similarity of the structure between vectors, not about the volume",

    "Q3: Look at the CKA matrix. Which pair of representations has the "
    "HIGHEST CKA? Which has the LOWEST? What does this tell you about "
    "what information survives the pipeline?":
        "the cka of (projected, pqc) is the highest (0.68) while (syntax, pqc) is the lowest, which mean after gone through the pqc, the majority of informations have lost, which is quite disappointing. but the cka of (projected, pqc) is still high, so we can conclude that the pqc can preserve the information after squishing too much informations (384 -> 8)",

    "Q4: Compare CKA(Projected, Syntax) vs CKA(PQC, Syntax). Did the PQC "
    "increase or decrease syntactic alignment? By how much? Is this a "
    "large or small effect?":
        "the cka(projected, syntax) is 0.082 < cka(pqc,syntax) (0.15) but the different is not that much, maybe we test on a super small dataset so the result might not be that significant.",

    "Q5: We only have 7 sentences. Why is this a problem for CKA "
    "reliability? How many sentences would we need for the paper?":
        "i think the higher the number of sentences, the more reliable of the cka will be, i guess we need about 2000+ sentences.",

    "Q6: Look at Figure 3 (semantics vs syntax). Find a pair of sentences "
    "that are semantically SIMILAR but syntactically DIFFERENT, and a pair "
    "that are semantically DIFFERENT but syntactically SIMILAR. Name them.":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")


# Save analysis results for future use
analysis_path = PROJECT_ROOT / "results" / "cka_analysis.pt"
torch.save({
    'sentences': sentences,
    'fingerprints': fingerprints,
    'cka_matrix': cka_matrix,
    'rep_names': rep_names,
    'cka_sbert_syntax': cka_sbert_syntax,
    'cka_proj_syntax': cka_proj_syntax,
    'cka_pqc_syntax': cka_pqc_syntax,
}, analysis_path)
print(f"\n\n  Analysis saved to: {analysis_path}")

print("\n\nDone! Review the 3 figures in results/figures/")
print("Share this output AND your answers with me for review.")
print("\nThis is the core experiment. Your figures are paper-worthy.")
