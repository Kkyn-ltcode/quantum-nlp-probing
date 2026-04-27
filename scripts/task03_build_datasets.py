"""
Task 03: Build the Paper's Datasets
=====================================
Generates all three datasets, parses with BobcatParser,
extracts WL fingerprints, and saves everything.

    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    pip install datasets     # first time only
    python scripts/task03_build_datasets.py

WARNING: Takes ~45 minutes (parsing is slow). Go get coffee.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import time
import random
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.templates import (
    generate_sentences, generate_paraphrase_pairs, CONSTRUCTIONS
)
from src.fingerprint.graph_kernel import WLFingerprint, diagram_to_graph
from src.analysis.cka import compute_cka

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

N_PER_CONSTRUCTION = 200  # 200 × 5 = 1000 template sentences
N_BLIMP = 200             # per sub-task
N_MRPC = 200              # sentence pairs


# ═══════════════════════════════════════════════════════════
# PART 1: Generate Template Sentences
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 1: Generating Template Sentences")
print("=" * 60)

template_sents, template_meta = generate_sentences(
    n_per_construction=N_PER_CONSTRUCTION, seed=42
)

print(f"\n  Generated {len(template_sents)} sentences:")
for ctype in CONSTRUCTIONS:
    count = sum(1 for m in template_meta if m['construction'] == ctype)
    example = next(s for s, m in zip(template_sents, template_meta)
                   if m['construction'] == ctype)
    print(f"    {ctype:14s}: {count:4d}  ex: \"{example}\"")

pairs = generate_paraphrase_pairs(template_sents, template_meta)
print(f"\n  Paraphrase pairs: {len(pairs)}")
print(f"    Positive: {sum(1 for p in pairs if p[2] == 1.0)}")
print(f"    Negative: {sum(1 for p in pairs if p[2] == 0.0)}")


# ═══════════════════════════════════════════════════════════
# PART 2: Load BLiMP
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 2: Loading BLiMP")
print("=" * 60)

blimp_sents = []
blimp_meta = []

try:
    from datasets import load_dataset

    blimp_tasks = ['anaphor_gender_agreement', 'sentential_negation_npi_scope']

    for task_name in blimp_tasks:
        try:
            ds = load_dataset("nyu-mll/blimp", task_name, split="train",
                              trust_remote_code=True)
            # Take grammatical sentences only
            count = 0
            for item in ds:
                if count >= N_BLIMP:
                    break
                sent = item.get('sentence_good', item.get('sentence_a', ''))
                if sent and len(sent.split()) <= 12:  # keep short for parser
                    blimp_sents.append(sent.lower().strip('.'))
                    blimp_meta.append({
                        'construction': f'blimp_{task_name[:10]}',
                        'source': 'blimp',
                        'task': task_name,
                    })
                    count += 1

            print(f"  ✓ {task_name}: {count} sentences loaded")
        except Exception as e:
            print(f"  ✗ {task_name}: {e}")

except ImportError:
    print("  ⚠ 'datasets' library not installed. Run: pip install datasets")
    print("    Skipping BLiMP — templates + MRPC will still work.")

print(f"  Total BLiMP sentences: {len(blimp_sents)}")


# ═══════════════════════════════════════════════════════════
# PART 3: Load MRPC
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 3: Loading MRPC")
print("=" * 60)

mrpc_sents = []
mrpc_meta = []
mrpc_pairs = []

try:
    from datasets import load_dataset

    ds = load_dataset("glue", "mrpc", split="train", trust_remote_code=True)
    count = 0

    for item in ds:
        if count >= N_MRPC:
            break
        s1 = item['sentence1']
        s2 = item['sentence2']
        label = float(item['label'])

        # Keep only shorter sentences (parser-friendly)
        if len(s1.split()) <= 15 and len(s2.split()) <= 15:
            idx1 = len(mrpc_sents)
            mrpc_sents.append(s1.lower().rstrip('.'))
            mrpc_meta.append({'construction': 'mrpc', 'source': 'mrpc'})

            idx2 = len(mrpc_sents)
            mrpc_sents.append(s2.lower().rstrip('.'))
            mrpc_meta.append({'construction': 'mrpc', 'source': 'mrpc'})

            mrpc_pairs.append((idx1, idx2, label))
            count += 1

    print(f"  ✓ MRPC: {count} pairs loaded ({len(mrpc_sents)} sentences)")
    print(f"    Paraphrase: {sum(1 for p in mrpc_pairs if p[2] == 1.0)}")
    print(f"    Non-para:   {sum(1 for p in mrpc_pairs if p[2] == 0.0)}")

except ImportError:
    print("  ⚠ 'datasets' library not installed. Skipping MRPC.")
except Exception as e:
    print(f"  ✗ MRPC loading failed: {e}")

print(f"  Total MRPC sentences: {len(mrpc_sents)}")


# ═══════════════════════════════════════════════════════════
# PART 4: Parse All Sentences with BobcatParser
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 4: Parsing with BobcatParser")
print("=" * 60)

# Combine all sentences
all_sentences = template_sents + blimp_sents + mrpc_sents
all_metadata = template_meta + blimp_meta + mrpc_meta
source_ranges = {
    'template': (0, len(template_sents)),
    'blimp': (len(template_sents), len(template_sents) + len(blimp_sents)),
    'mrpc': (len(template_sents) + len(blimp_sents), len(all_sentences)),
}

total = len(all_sentences)
print(f"\n  Total sentences to parse: {total}")
print(f"    Templates: {len(template_sents)}")
print(f"    BLiMP:     {len(blimp_sents)}")
print(f"    MRPC:      {len(mrpc_sents)}")
est_time = total * 1.5 / 60
print(f"  Estimated time: ~{est_time:.0f} minutes\n")

from lambeq import BobcatParser
parser = BobcatParser(model_name_or_path='bobcat', verbose='suppress')

diagrams = [None] * total
parse_ok = [False] * total
start_time = time.time()

batch = 50
for b_start in range(0, total, batch):
    b_end = min(b_start + batch, total)
    for i in range(b_start, b_end):
        try:
            diag = parser.sentence2diagram(all_sentences[i])
            if diag is not None:
                diagrams[i] = diag
                parse_ok[i] = True
        except Exception:
            pass

    done = b_end
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"    Parsed {done:4d}/{total} "
          f"({100*done/total:5.1f}%) "
          f"— {rate:.1f} sent/s — ETA: {eta/60:.1f} min")

total_time = time.time() - start_time
n_ok = sum(parse_ok)
print(f"\n  Done in {total_time/60:.1f} minutes")
print(f"  Success: {n_ok}/{total} ({100*n_ok/total:.1f}%)")

# Per-source success rate
for src, (start, end) in source_ranges.items():
    src_ok = sum(parse_ok[start:end])
    src_total = end - start
    if src_total > 0:
        print(f"    {src:10s}: {src_ok}/{src_total} ({100*src_ok/src_total:.1f}%)")


# ═══════════════════════════════════════════════════════════
# PART 5: Extract WL Fingerprints
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 5: WL Fingerprint Extraction")
print("=" * 60)

# Filter to valid only
valid_idx = [i for i in range(total) if parse_ok[i]]
valid_sents = [all_sentences[i] for i in valid_idx]
valid_meta = [all_metadata[i] for i in valid_idx]
valid_diagrams = [diagrams[i] for i in valid_idx]

print(f"  Extracting WL features from {len(valid_diagrams)} diagrams...")

wl = WLFingerprint(n_iterations=3, max_features=256)
X_wl = wl.fit_transform(valid_diagrams)

print(f"  WL feature matrix: {X_wl.shape}")
print(f"  Non-zero features: {np.count_nonzero(X_wl.sum(axis=0))}")

# Remap pair indices
old_to_new = {old: new for new, old in enumerate(valid_idx)}

# Template pairs
valid_template_pairs = []
for ia, ib, label in pairs:
    if ia in old_to_new and ib in old_to_new:
        valid_template_pairs.append((old_to_new[ia], old_to_new[ib], label))

# MRPC pairs (offset by template+blimp lengths)
offset = len(template_sents) + len(blimp_sents)
valid_mrpc_pairs = []
for ia, ib, label in mrpc_pairs:
    abs_ia = ia + offset  # convert from mrpc-local to global index
    abs_ib = ib + offset
    if abs_ia in old_to_new and abs_ib in old_to_new:
        valid_mrpc_pairs.append((old_to_new[abs_ia], old_to_new[abs_ib], label))

print(f"  Valid template pairs: {len(valid_template_pairs)}")
print(f"  Valid MRPC pairs:     {len(valid_mrpc_pairs)}")


# ═══════════════════════════════════════════════════════════
# PART 6: Dataset Summary & CKA Sanity Check
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 6: Dataset Summary")
print("=" * 60)

# Count by source and construction
source_counts = Counter(m['source'] for m in valid_meta)
type_counts = Counter(m['construction'] for m in valid_meta)

print(f"\n  By source:")
for src, count in source_counts.most_common():
    print(f"    {src:10s}: {count}")

print(f"\n  By construction (templates only):")
template_types = Counter(m['construction'] for m in valid_meta
                         if m['source'] == 'template')
for ctype in CONSTRUCTIONS:
    print(f"    {ctype:14s}: {template_types.get(ctype, 0)}")

# CKA sanity check: WL fingerprints vs construction labels (templates only)
template_mask = np.array([m['source'] == 'template' for m in valid_meta])
if template_mask.sum() > 10:
    template_fp = X_wl[template_mask]
    template_labels = [m['construction'] for m in valid_meta
                       if m['source'] == 'template']
    n_tmpl = len(template_labels)
    unique_types = sorted(set(template_labels))
    onehot = np.zeros((n_tmpl, len(unique_types)))
    for i, label in enumerate(template_labels):
        onehot[i, unique_types.index(label)] = 1.0

    def normalize(X):
        return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    cka_val = compute_cka(normalize(template_fp), onehot)
    print(f"\n  CKA(WL_fingerprint, ConstructionType) = {cka_val:.4f}")
    if cka_val > 0.3:
        print(f"  ✓ Good: fingerprints discriminate construction types")
    else:
        print(f"  ⚠ Weak: fingerprints may not discriminate well enough")


# ═══════════════════════════════════════════════════════════
# PART 7: Visualization
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 7: Figures")
print("=" * 60)

colors_map = {
    'active': '#4C72B0', 'passive': '#DD8452', 'relative': '#55A868',
    'cleft': '#C44E52', 'ditransitive': '#8172B3',
}

# Figure 1: Dataset composition
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# By source
src_names = list(source_counts.keys())
src_vals = [source_counts[s] for s in src_names]
ax1.bar(src_names, src_vals, color=['#4C72B0', '#DD8452', '#55A868'],
        edgecolor='black')
ax1.set_title('Sentences by Source', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(src_vals):
    ax1.text(i, v + 2, str(v), ha='center', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# By construction (templates)
if template_types:
    types = list(template_types.keys())
    vals = [template_types[t] for t in types]
    clrs = [colors_map.get(t, '#888888') for t in types]
    ax2.bar(types, vals, color=clrs, edgecolor='black')
    ax2.set_title('Template Constructions', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=20)
    for i, v in enumerate(vals):
        ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

fig1.tight_layout()
fig1.savefig(output_dir / "task03_dataset_summary.png", dpi=150)
print(f"  Figure 1: {output_dir / 'task03_dataset_summary.png'}")


# ═══════════════════════════════════════════════════════════
# PART 8: Save Dataset
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 8: Saving Dataset")
print("=" * 60)

# Train/test split for template pairs (80/20)
random.shuffle(valid_template_pairs)
n_train = int(0.8 * len(valid_template_pairs))
train_pairs = valid_template_pairs[:n_train]
test_pairs = valid_template_pairs[n_train:]

dataset = {
    # All sentences and metadata
    'sentences': valid_sents,
    'metadata': valid_meta,
    'wl_features': X_wl,
    'wl_model': wl,

    # Template pairs (for training)
    'template_train_pairs': train_pairs,
    'template_test_pairs': test_pairs,

    # MRPC pairs (for validation)
    'mrpc_pairs': valid_mrpc_pairs,

    # Index masks
    'template_mask': template_mask.tolist(),

    # Vocabulary info
    'constructions': CONSTRUCTIONS,
    'sources': list(source_counts.keys()),
}

save_path = data_dir / "paper_dataset.pt"
torch.save(dataset, save_path)

print(f"\n  Saved to: {save_path}")
print(f"    Total sentences:      {len(valid_sents)}")
print(f"    WL features:          {X_wl.shape}")
print(f"    Template train pairs: {len(train_pairs)}")
print(f"    Template test pairs:  {len(test_pairs)}")
print(f"    MRPC pairs:           {len(valid_mrpc_pairs)}")


# ═══════════════════════════════════════════════════════════
# COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: What was the parse success rate for each source "
    "(template, BLiMP, MRPC)? Which source had the most "
    "failures? Why do you think that is?":
        "...",

    "Q2: Look at the CKA(WL_fingerprint, ConstructionType) "
    "value. Is it higher than the count-based CKA from "
    "Lesson 4 (~0.08)? What does this confirm about the "
    "WL kernel upgrade?":
        "...",

    "Q3: How many template paraphrase pairs survived parsing? "
    "If we lost many pairs, what could we do to increase "
    "the number?":
        "...",

    "Q4: Why do we filter MRPC to sentences ≤15 words? "
    "What's the tradeoff between keeping more data vs "
    "keeping only parser-friendly sentences?":
        "...",

    "Q5: This dataset is what the paper's experiments run on. "
    "If a reviewer asks 'why only 1000 templates?', what "
    "would you say? How many would you need to scale to?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print(f"\n\nDone! Dataset is ready for experiments.")
print(f"Next task: Train all models on this dataset.")
