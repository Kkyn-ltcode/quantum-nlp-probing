# Lesson 4: Syntax Fingerprints & CKA Probing

## Prerequisites
- ✅ Lesson 1: DisCoCat diagrams (boxes, cups, types)
- ✅ Lesson 2: PQCs and measurements
- ✅ Lesson 3: Hybrid pipeline built, `results/representations.pt` saved
- ✅ BobcatParser working with local model

## What You'll Learn

1. Extract **structural fingerprints** from DisCoCat diagrams — our pure-syntax reference
2. Understand and implement **CKA** (Centered Kernel Alignment)
3. Run the **core analysis from our paper**: compare pipeline stages against syntax
4. Interpret results: does the PQC recover syntactic structure?

> [!IMPORTANT]
> **This is the experiment.** Lessons 1-3 were building blocks. This lesson produces the actual analysis that will appear in Section 5 of the paper. Every figure you generate here is paper-worthy.

---

## Part 1: What Is a Structural Fingerprint?

### The Problem

We need a representation of **pure grammatical structure** with zero semantic content. In Lesson 1, we saw that BobcatParser assigns CCG types to words:

```
dogs   → n           (noun)
chase  → n.r ⊗ s ⊗ n.l  (transitive verb)
cats   → n           (noun)
```

The diagram has 3 words, 2 cups, type complexity of 5 atomic types. These are **structural properties** — they don't change if you replace "dogs" with "birds" or "cats" with "worms".

### The Fingerprint Vector

For each sentence, we extract a fixed-size vector of structural features:

| Feature | What it measures | Example: "dogs chase cats" |
|:--------|:----------------|:---------------------------|
| n_boxes | Number of word boxes | 3 |
| n_cups | Number of cups (syntactic contractions) | 2 |
| n_caps | Number of caps | 0 |
| n_swaps | Number of swap operations | 0 |
| n_atoms | Total atomic types across all wires | 5 |
| max_type_depth | Deepest nested type (e.g., n.r⊗s⊗n.l → depth 1) | 1 |
| type_histogram | Count of each atomic type [n_count, s_count, ...] | [4, 1] |
| avg_word_type_complexity | Average number of atoms per word type | 1.67 |
| n_wires | Total wires in the diagram | 5 |

We pad/normalize this to a fixed size (e.g., 16 dims) and that's our syntax fingerprint.

### Why This Is Scientifically Clean

Two sentences with **identical grammar but different words** produce **identical fingerprints**:
- "dogs chase cats" → fingerprint [3, 2, 0, 0, 5, 1, 4, 1, 1.67, 5, ...]
- "birds eat worms" → fingerprint [3, 2, 0, 0, 5, 1, 4, 1, 1.67, 5, ...]

Two sentences with **same meaning but different grammar** produce **different fingerprints**:
- "dogs chase cats" → fingerprint [3, 2, 0, 0, 5, 1, ...]
- "cats are chased by dogs" → fingerprint [5, 6, 0, 0, 15, 2, ...]

This is exactly what we want: a reference that is pure syntax, zero semantics.

---

## Part 2: CKA (Centered Kernel Alignment)

### What It Measures

CKA answers: **"How similar is the *structure* of representation X to the *structure* of representation Y?"**

It doesn't compare individual vectors — it compares **how a set of sentences relate to each other** in two different representation spaces.

### Intuition

Imagine you have 100 sentences. In SBERT space, "dogs chase cats" and "cats are chased by dogs" are neighbors. In syntax fingerprint space, they're far apart.

CKA measures: **Does the neighborhood structure match?**

- CKA ≈ 1.0: Both representations organize sentences the same way
- CKA ≈ 0.0: The representations have completely different structure
- CKA(PQC, syntax) > CKA(projection, syntax): The PQC added syntactic structure

### The Math (Simplified)

Given two matrices:
- X: shape (n_sentences, d₁) — e.g., PQC outputs
- Y: shape (n_sentences, d₂) — e.g., syntax fingerprints

```
1. Compute kernel matrices:
   K = X @ X.T     (n × n similarity matrix in X-space)
   L = Y @ Y.T     (n × n similarity matrix in Y-space)

2. Center both kernels (remove mean):
   K_c = center(K)
   L_c = center(L)

3. CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
   where HSIC = sum(K_c * L_c) / (n-1)²
```

CKA is essentially **cosine similarity between kernel matrices** — it measures whether the pairwise similarity structure is the same.

### Why CKA and Not Just Cosine Similarity?

- **Cosine similarity** compares two vectors directly. But our representations have different dimensions (384 vs 8 vs 16).
- **CKA** compares the **relational structure** — how sentences relate to each other — regardless of dimension. It's invariant to orthogonal transformations and scaling.

> [!TIP]
> **Paper framing:** CKA is the standard tool for representational similarity analysis in deep learning (Kornblith et al., 2019). Using it makes our work directly comparable to the BERTology literature. Reviewers will recognize and respect this choice.

---

## Part 3: The Core Experiment

We compute CKA between **every pair of representations**:

```
             SBERT  Projected  PQC  Syntax
SBERT          1.0     ?        ?     ?
Projected       -     1.0      ?     ?
PQC             -      -      1.0    ?
Syntax          -      -       -    1.0
```

### The Key Comparison

```
CKA(Projected, Syntax)  vs.  CKA(PQC, Syntax)
```

- If PQC > Projected: **The quantum circuit increased syntactic alignment** → main finding
- If PQC ≈ Projected: **The circuit didn't add syntax** → still publishable (null result)
- If PQC < Projected: **The circuit destroyed syntax** → interesting negative finding

### Statistical Rigor

With only 7 sentences, CKA is unreliable. For the paper, we'll need 500+ sentences. But this lesson establishes the methodology. In later lessons, we'll scale up the dataset.

---

## Homework

Run `notebooks/lesson04_probing.py`. It has 5 exercises:

1. **Build the fingerprint extractor** — parse sentences, extract structural features
2. **Implement CKA from scratch** — understand the math by coding it
3. **Run the core analysis** — CKA between all pipeline stages and syntax
4. **Visualize results** — heatmap + bar chart for the paper
5. **Interpret findings** — what did the PQC do to syntactic information?

> [!WARNING]
> **This lesson requires you to THINK, not just run code.** The questions ask you to interpret results and connect them to our research hypotheses. No more skipping "why" questions — they ARE the research.
