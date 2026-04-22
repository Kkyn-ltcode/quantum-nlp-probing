"""
Lesson 06: Building the Controlled Syntactic Dataset
=====================================================
This script generates the Tier 1 dataset for the paper:
template-based sentence pairs with controlled syntactic variation.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson06_dataset.py

WARNING: Parsing takes ~15-30 minutes. Be patient.

Output:
    data/syntactic_dataset.pt   — full dataset with sentences, labels,
                                   construction types, fingerprints
    results/figures/dataset_*   — analysis figures
"""

import numpy as np
import torch
import random
import time
from pathlib import Path
from itertools import product
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# PART 1: Vocabulary Pools
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  PART 1: Vocabulary Pools")
print("=" * 60)

# Transitive verbs (past tense for passive construction)
# Format: (base, past, past_participle)
VERBS_TRANSITIVE = [
    ("chase", "chased", "chased"),
    ("help", "helped", "helped"),
    ("watch", "watched", "watched"),
    ("praise", "praised", "praised"),
    ("follow", "followed", "followed"),
    ("examine", "examined", "examined"),
    ("teach", "taught", "taught"),
    ("like", "liked", "liked"),
    ("visit", "visited", "visited"),
    ("call", "called", "called"),
    ("push", "pushed", "pushed"),
    ("pull", "pulled", "pulled"),
    ("stop", "stopped", "stopped"),
    ("kick", "kicked", "kicked"),
    ("cook", "cooked", "cooked"),
    ("paint", "painted", "painted"),
    ("clean", "cleaned", "cleaned"),
    ("greet", "greeted", "greeted"),
    ("blame", "blamed", "blamed"),
    ("admire", "admired", "admired"),
]

# Nouns (agent/patient roles)
NOUNS = [
    "dog", "cat", "chef", "doctor", "teacher", "student",
    "nurse", "artist", "judge", "farmer", "driver", "pilot",
    "singer", "dancer", "writer", "baker", "lawyer", "sailor",
    "guard", "clerk",
]

# Adjectives
ADJECTIVES = [
    "tall", "young", "old", "small", "clever", "quiet",
    "brave", "gentle", "proud", "kind", "happy", "wise",
    "strong", "fast", "calm", "shy", "bold", "fair",
    "keen", "warm",
]

print(f"  Vocabulary:")
print(f"    Transitive verbs: {len(VERBS_TRANSITIVE)}")
print(f"    Nouns:            {len(NOUNS)}")
print(f"    Adjectives:       {len(ADJECTIVES)}")

# Max unique combinations per construction:
max_combos = len(NOUNS) * (len(NOUNS)-1) * len(VERBS_TRANSITIVE) * len(ADJECTIVES) * (len(ADJECTIVES)-1)
print(f"    Max unique combos: {max_combos:,}")


# ═══════════════════════════════════════════════════════════
# PART 2: Template-Based Sentence Generation
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART 2: Generating Sentences")
print("=" * 60)


def generate_active(n_agent, adj_agent, verb_tuple, n_patient, adj_patient):
    """Active SVO: 'the tall doctor chased the young cat'"""
    _, past, _ = verb_tuple
    return f"the {adj_agent} {n_agent} {past} the {adj_patient} {n_patient}"


def generate_passive(n_agent, adj_agent, verb_tuple, n_patient, adj_patient):
    """Passive: 'the young cat was chased by the tall doctor'"""
    _, _, pp = verb_tuple
    return f"the {adj_patient} {n_patient} was {pp} by the {adj_agent} {n_agent}"


def generate_relative_subject(n_agent, adj_agent, verb_tuple,
                               n_patient, adj_patient, n_rel, verb_rel_tuple):
    """Subject relative clause: 'the tall doctor that chased the young cat helped the nurse'"""
    _, past1, _ = verb_tuple
    _, past2, _ = verb_rel_tuple
    return f"the {adj_agent} {n_agent} that {past1} the {adj_patient} {n_patient} {past2} the {n_rel}"


def generate_cleft(n_agent, adj_agent, verb_tuple, n_patient, adj_patient):
    """Cleft: 'it was the tall doctor who chased the young cat'"""
    _, past, _ = verb_tuple
    return f"it was the {adj_agent} {n_agent} who {past} the {adj_patient} {n_patient}"


def sample_unique_sentences(n_target, construction_type):
    """Generate n_target unique sentences of a given construction type."""
    generated = set()
    sentences = []
    metadata = []
    attempts = 0
    max_attempts = n_target * 10

    while len(sentences) < n_target and attempts < max_attempts:
        attempts += 1

        # Sample vocabulary
        n_agent, n_patient = random.sample(NOUNS, 2)
        adj_agent, adj_patient = random.sample(ADJECTIVES, 2)
        verb = random.choice(VERBS_TRANSITIVE)

        if construction_type == "active":
            sent = generate_active(n_agent, adj_agent, verb, n_patient, adj_patient)
        elif construction_type == "passive":
            sent = generate_passive(n_agent, adj_agent, verb, n_patient, adj_patient)
        elif construction_type == "relative":
            n_rel = random.choice([n for n in NOUNS if n not in (n_agent, n_patient)])
            verb_rel = random.choice([v for v in VERBS_TRANSITIVE if v != verb])
            sent = generate_relative_subject(
                n_agent, adj_agent, verb, n_patient, adj_patient, n_rel, verb_rel
            )
        elif construction_type == "cleft":
            sent = generate_cleft(n_agent, adj_agent, verb, n_patient, adj_patient)
        else:
            raise ValueError(f"Unknown construction: {construction_type}")

        if sent not in generated:
            generated.add(sent)
            sentences.append(sent)
            metadata.append({
                'construction': construction_type,
                'agent': n_agent,
                'patient': n_patient,
                'adj_agent': adj_agent,
                'adj_patient': adj_patient,
                'verb': verb[0],
            })

    return sentences, metadata


# Generate sentences for each construction
N_PER_CONSTRUCTION = 200  # Start manageable; increase for the final paper

constructions = ["active", "passive", "relative", "cleft"]
all_sentences = []
all_metadata = []
all_labels = []  # Construction type index

print(f"\n  Generating {N_PER_CONSTRUCTION} sentences per construction...\n")

for idx, ctype in enumerate(constructions):
    sents, meta = sample_unique_sentences(N_PER_CONSTRUCTION, ctype)
    all_sentences.extend(sents)
    all_metadata.extend(meta)
    all_labels.extend([idx] * len(sents))
    print(f"    {ctype:12s}: {len(sents)} sentences generated")
    print(f"      Example: \"{sents[0]}\"")

print(f"\n  Total sentences: {len(all_sentences)}")

# Generate paraphrase pairs (active ↔ passive)
# For every active sentence, find its passive counterpart
print(f"\n  Generating paraphrase pairs (active ↔ passive)...")

paraphrase_pairs = []
# Build lookup by (agent, patient, verb, adj_agent, adj_patient)
active_lookup = {}
passive_lookup = {}

for i, meta in enumerate(all_metadata):
    key = (meta['agent'], meta['patient'], meta['adj_agent'],
           meta['adj_patient'], meta['verb'])
    if meta['construction'] == 'active':
        active_lookup[key] = i
    elif meta['construction'] == 'passive':
        passive_lookup[key] = i

# Match pairs
matched_pairs = []
for key in active_lookup:
    if key in passive_lookup:
        i_active = active_lookup[key]
        i_passive = passive_lookup[key]
        matched_pairs.append((i_active, i_passive, 1.0))  # paraphrase

# Generate non-paraphrase pairs (random mismatches)
non_paraphrase_pairs = []
active_indices = list(active_lookup.values())
passive_indices = list(passive_lookup.values())

for _ in range(len(matched_pairs)):
    # Pick random active and passive that DON'T match
    while True:
        i_a = random.choice(active_indices)
        i_p = random.choice(passive_indices)
        key_a = (all_metadata[i_a]['agent'], all_metadata[i_a]['patient'],
                 all_metadata[i_a]['adj_agent'], all_metadata[i_a]['adj_patient'],
                 all_metadata[i_a]['verb'])
        key_p = (all_metadata[i_p]['agent'], all_metadata[i_p]['patient'],
                 all_metadata[i_p]['adj_agent'], all_metadata[i_p]['adj_patient'],
                 all_metadata[i_p]['verb'])
        if key_a != key_p:
            non_paraphrase_pairs.append((i_a, i_p, 0.0))
            break

all_pairs = matched_pairs + non_paraphrase_pairs
random.shuffle(all_pairs)

print(f"    Paraphrase pairs:     {len(matched_pairs)}")
print(f"    Non-paraphrase pairs: {len(non_paraphrase_pairs)}")
print(f"    Total training pairs: {len(all_pairs)}")


# ═══════════════════════════════════════════════════════════
# PART 3: Parse with BobcatParser & Extract Fingerprints
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART 3: Parsing with BobcatParser")
print("=" * 60)
print(f"  This will take ~{len(all_sentences) * 1.5 / 60:.0f} minutes. Be patient.\n")

from lambeq import BobcatParser

parser = BobcatParser(model_name_or_path='bobcat', verbose='suppress')


def extract_fingerprint(diagram):
    """Extract structural fingerprint from a DisCoCat diagram."""
    boxes = diagram.boxes
    n_words = 0
    n_cups = 0
    n_caps = 0
    n_swaps = 0
    word_type_complexities = []

    for box in boxes:
        box_type = type(box).__name__
        if box_type == "Word":
            n_words += 1
            cod_str = str(box.cod)
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

    total_boxes = len(boxes)
    n_other = total_boxes - n_words - n_cups - n_caps - n_swaps
    avg_type_complexity = np.mean(word_type_complexities) if word_type_complexities else 0
    max_type_complexity = max(word_type_complexities) if word_type_complexities else 0

    full_cod_str = ' '.join(str(box.cod) for box in boxes
                           if type(box).__name__ == "Word")
    n_count = full_cod_str.count('n')
    s_count = full_cod_str.count('s')
    cups_ratio = n_cups / max(n_words, 1)

    return np.array([
        n_words, n_cups, n_caps, n_swaps, total_boxes, n_other,
        avg_type_complexity, max_type_complexity,
        n_count, s_count, cups_ratio,
        n_words - n_cups,
        max_type_complexity - avg_type_complexity,
    ], dtype=np.float64)


# Parse all sentences with progress tracking
diagrams = [None] * len(all_sentences)
fingerprints = [None] * len(all_sentences)
parse_success = [False] * len(all_sentences)

start_time = time.time()
batch_size = 50  # Parse in batches for progress reporting

for batch_start in range(0, len(all_sentences), batch_size):
    batch_end = min(batch_start + batch_size, len(all_sentences))
    batch = all_sentences[batch_start:batch_end]

    for i, sent in enumerate(batch):
        idx = batch_start + i
        try:
            diag = parser.sentence2diagram(sent)
            if diag is not None:
                diagrams[idx] = diag
                fingerprints[idx] = extract_fingerprint(diag)
                parse_success[idx] = True
        except Exception as e:
            pass  # Skip unparseable sentences

    elapsed = time.time() - start_time
    done = batch_end
    rate = done / elapsed if elapsed > 0 else 0
    eta = (len(all_sentences) - done) / rate if rate > 0 else 0
    print(f"    Parsed {done}/{len(all_sentences)} "
          f"({100*done/len(all_sentences):.0f}%) "
          f"— {rate:.1f} sent/s — ETA: {eta/60:.1f} min")

total_time = time.time() - start_time
n_success = sum(parse_success)
n_fail = len(all_sentences) - n_success

print(f"\n  Parsing complete in {total_time/60:.1f} minutes")
print(f"    Successful: {n_success}/{len(all_sentences)} ({100*n_success/len(all_sentences):.1f}%)")
print(f"    Failed:     {n_fail}/{len(all_sentences)} ({100*n_fail/len(all_sentences):.1f}%)")

# Report failures by construction
fail_by_type = Counter()
for i, success in enumerate(parse_success):
    if not success:
        fail_by_type[all_metadata[i]['construction']] += 1

if fail_by_type:
    print(f"\n  Failures by construction:")
    for ctype, count in fail_by_type.most_common():
        total_of_type = sum(1 for m in all_metadata if m['construction'] == ctype)
        print(f"    {ctype:12s}: {count}/{total_of_type} failed")


# ═══════════════════════════════════════════════════════════
# PART 4: Filter & Analyze
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART 4: Dataset Analysis")
print("=" * 60)

# Keep only successfully parsed sentences
valid_indices = [i for i, s in enumerate(parse_success) if s]
valid_sentences = [all_sentences[i] for i in valid_indices]
valid_metadata = [all_metadata[i] for i in valid_indices]
valid_labels = [all_labels[i] for i in valid_indices]
valid_fingerprints = np.stack([fingerprints[i] for i in valid_indices])

# Remap indices for pairs
old_to_new = {old: new for new, old in enumerate(valid_indices)}
valid_pairs = []
for i_a, i_b, label in all_pairs:
    if i_a in old_to_new and i_b in old_to_new:
        valid_pairs.append((old_to_new[i_a], old_to_new[i_b], label))

print(f"\n  Final dataset:")
print(f"    Valid sentences:  {len(valid_sentences)}")
print(f"    Valid pairs:      {len(valid_pairs)}")
print(f"    Fingerprint shape: {valid_fingerprints.shape}")

# Distribution by construction
print(f"\n  By construction:")
type_counts = Counter(m['construction'] for m in valid_metadata)
for ctype in constructions:
    print(f"    {ctype:12s}: {type_counts.get(ctype, 0)}")

# CKA analysis: do same-construction sentences cluster?
from sklearn.metrics.pairwise import cosine_similarity


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


# Create one-hot construction label matrix for CKA
n_valid = len(valid_sentences)
construction_onehot = np.zeros((n_valid, len(constructions)))
for i, m in enumerate(valid_metadata):
    construction_onehot[i, constructions.index(m['construction'])] = 1.0

# Normalize fingerprints
fp_norm = valid_fingerprints.copy()
mu = fp_norm.mean(axis=0, keepdims=True)
std = fp_norm.std(axis=0, keepdims=True) + 1e-8
fp_norm = (fp_norm - mu) / std

cka_fp_type = compute_cka(fp_norm, construction_onehot)
print(f"\n  CKA(Fingerprints, ConstructionType) = {cka_fp_type:.4f}")
print(f"  → {'Good' if cka_fp_type > 0.3 else 'Weak'}: fingerprints "
      f"{'DO' if cka_fp_type > 0.3 else 'do NOT'} discriminate constructions")

# Average fingerprint per construction
print(f"\n  Average fingerprint per construction:")
print(f"  {'':12s}  words  cups  boxes  avg_cx  cups_ratio")
for ctype in constructions:
    mask = [m['construction'] == ctype for m in valid_metadata]
    fp_mean = valid_fingerprints[mask].mean(axis=0)
    print(f"    {ctype:12s}  {fp_mean[0]:5.1f}  {fp_mean[1]:4.1f}  "
          f"{fp_mean[4]:5.1f}  {fp_mean[6]:6.2f}  {fp_mean[10]:10.2f}")


# ═══════════════════════════════════════════════════════════
# PART 5: Visualizations
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART 5: Generating Figures")
print("=" * 60)

# Figure 1: Fingerprint distribution by construction (box plots)
fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
feature_names = ['words', 'cups', 'total_boxes', 'avg_complexity',
                 'cups_ratio', 'type_spread']
feature_indices = [0, 1, 4, 6, 10, 12]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for ax, fname, fidx in zip(axes.flat, feature_names, feature_indices):
    data_by_type = []
    for ctype in constructions:
        mask = [m['construction'] == ctype for m in valid_metadata]
        data_by_type.append(valid_fingerprints[mask][:, fidx])

    bp = ax.boxplot(data_by_type, labels=constructions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(fname, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

fig1.suptitle('Structural Fingerprint Features by Construction Type',
              fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig(output_dir / "dataset_fingerprint_boxplots.png", dpi=150)
print(f"  Figure 1 saved: {output_dir / 'dataset_fingerprint_boxplots.png'}")

# Figure 2: t-SNE of fingerprints colored by construction
from sklearn.manifold import TSNE

if n_valid >= 30:  # t-SNE needs reasonable sample size
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_valid//4))
    fp_2d = tsne.fit_transform(fp_norm)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for idx, ctype in enumerate(constructions):
        mask = np.array([m['construction'] == ctype for m in valid_metadata])
        ax2.scatter(fp_2d[mask, 0], fp_2d[mask, 1],
                   c=colors[idx], label=ctype, alpha=0.6, s=30, edgecolors='white',
                   linewidth=0.5)
    ax2.legend(fontsize=12, markerscale=2)
    ax2.set_title('t-SNE of Structural Fingerprints\n(colored by construction type)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE dim 1')
    ax2.set_ylabel('t-SNE dim 2')
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(output_dir / "dataset_fingerprint_tsne.png", dpi=150)
    print(f"  Figure 2 saved: {output_dir / 'dataset_fingerprint_tsne.png'}")

# Figure 3: Dataset summary
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of construction counts
counts = [type_counts.get(c, 0) for c in constructions]
ax3a.bar(constructions, counts, color=colors, edgecolor='black')
ax3a.set_ylabel('Count', fontsize=12)
ax3a.set_title('Sentences per Construction', fontsize=14, fontweight='bold')
for i, v in enumerate(counts):
    ax3a.text(i, v + 2, str(v), ha='center', fontweight='bold')

# Pair distribution
pair_labels = Counter()
for _, _, label in valid_pairs:
    pair_labels['Paraphrase' if label == 1.0 else 'Non-paraphrase'] += 1
ax3b.bar(pair_labels.keys(), pair_labels.values(),
         color=['#55A868', '#C44E52'], edgecolor='black')
ax3b.set_ylabel('Count', fontsize=12)
ax3b.set_title('Training Pair Distribution', fontsize=14, fontweight='bold')
for i, (k, v) in enumerate(pair_labels.items()):
    ax3b.text(i, v + 1, str(v), ha='center', fontweight='bold')

fig3.tight_layout()
fig3.savefig(output_dir / "dataset_summary.png", dpi=150)
print(f"  Figure 3 saved: {output_dir / 'dataset_summary.png'}")


# ═══════════════════════════════════════════════════════════
# PART 6: Save Dataset
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART 6: Saving Dataset")
print("=" * 60)

# Train/test split (80/20)
n_pairs = len(valid_pairs)
n_train = int(0.8 * n_pairs)
random.shuffle(valid_pairs)
train_pairs = valid_pairs[:n_train]
test_pairs = valid_pairs[n_train:]

dataset = {
    'sentences': valid_sentences,
    'metadata': valid_metadata,
    'construction_labels': valid_labels,
    'constructions': constructions,
    'fingerprints': valid_fingerprints,
    'train_pairs': train_pairs,
    'test_pairs': test_pairs,
    'vocabulary': {
        'nouns': NOUNS,
        'verbs': [v[0] for v in VERBS_TRANSITIVE],
        'adjectives': ADJECTIVES,
    },
    'cka_fingerprint_vs_type': cka_fp_type,
}

save_path = data_dir / "syntactic_dataset.pt"
torch.save(dataset, save_path)
print(f"\n  Dataset saved to: {save_path}")
print(f"    Sentences:    {len(valid_sentences)}")
print(f"    Train pairs:  {len(train_pairs)}")
print(f"    Test pairs:   {len(test_pairs)}")
print(f"    Fingerprints: {valid_fingerprints.shape}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the average fingerprint per construction. "
    "Which features BEST distinguish active from passive? "
    "Which features are most similar across constructions?":
        "...",

    "Q2: What was the parse success rate? Did any construction "
    "fail more than others? Why might that be?":
        "...",

    "Q3: Look at the t-SNE plot. Do the constructions form "
    "distinct clusters? If not, what does that say about "
    "our fingerprint design?":
        "...",

    "Q4: We generated paraphrase pairs only from active/passive. "
    "Could we also make paraphrase pairs from active/cleft "
    "(e.g., 'the doctor chased the cat' ↔ 'it was the doctor "
    "who chased the cat')? Would these be true paraphrases?":
        "...",

    "Q5: Why do we need a train/test split for the pairs? "
    "What would happen if we evaluated CKA on the same "
    "sentences we trained on?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")


print(f"\n\n{'=' * 60}")
print("  DATASET READY")
print("=" * 60)
print(f"""
  Your dataset is saved at: {save_path}

  Next steps:
    1. Review the figures in results/figures/
    2. Answer the comprehension questions
    3. Lesson 7 will train the full PQC and MLP on this dataset
       and run the paper-scale CKA analysis
""")
