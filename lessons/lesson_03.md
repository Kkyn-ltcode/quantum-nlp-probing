# Lesson 3: The Hybrid SBERT → PQC Pipeline

## Prerequisites
- ✅ Lesson 1: You understand DisCoCat diagrams (boxes, cups, types)
- ✅ Lesson 2: You can build and train PQCs with PennyLane
- ✅ Both `sentence-transformers` and `pennylane` working in your env

## What You'll Learn

By the end of this lesson, you will:
1. Generate SBERT sentence embeddings and explore the embedding space
2. Build a trainable linear projection (768-dim → 8-dim)
3. Wire SBERT → Projection → PQC into a single differentiable model
4. Train the full hybrid pipeline on a sentence similarity task
5. Extract intermediate representations (the vectors our probes will analyze)

> [!IMPORTANT]
> **This lesson builds THE pipeline from our paper.** After this, you'll have a working hybrid quantum-classical NLP model. Lessons 1 and 2 were building blocks — this is where they come together.

---

## Part 1: SBERT Sentence Embeddings

### What is SBERT?

Sentence-BERT (SBERT) is a pre-trained transformer that maps any English sentence to a **768-dimensional vector**. Semantically similar sentences get similar vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode('dogs chase cats')  # shape: (768,)
```

### Why SBERT and not raw BERT?

Raw BERT gives you a vector **per token**. SBERT gives you a single vector **per sentence** — exactly what we need to feed into our PQC.

### Key properties of SBERT embeddings

| Property | Value | Why it matters |
|:---------|:------|:---------------|
| Dimensionality | 768 | Way too many for a quantum circuit |
| Range | roughly [-1, 1] per dim | Good for angle encoding after scaling |
| Semantic similarity | cosine similarity | "dogs chase cats" ≈ "cats are chased by dogs" |
| Syntax encoding | **unknown** | This is what our research investigates! |

---

## Part 2: Trainable Linear Projection

### The Problem

We have 768 dimensions from SBERT but only ~8 qubits. We need to compress.

### Why NOT use PCA?

In our original plan, we considered PCA for dimensionality reduction. But PCA has a critical flaw for our research:

> **PCA is not differentiable as part of the training loop.** We can't compute ∂(PQC output)/∂(PCA parameters) because PCA is a fixed preprocessing step.

### The Solution: Trainable Linear Layer

```python
projection = nn.Linear(768, n_qubits, bias=False)
```

This is simply a learnable matrix W of shape (768 × 8). The compressed embedding is:

```
z = W^T · x_sbert    # shape: (8,)
```

Key advantages:
1. **Differentiable** — gradients flow back through W during training
2. **End-to-end** — the projection learns what to keep for the task
3. **Analyzable** — we can inspect W to see which SBERT dimensions matter
4. **Reproducible** — no dependency on the training set statistics (unlike PCA)

> [!TIP]
> **Research insight:** In our probing experiments, we'll compare what information is present at three points: (1) SBERT output, (2) after projection, (3) after PQC. The trainable projection makes point (2) → (3) a clean, differentiable pipeline.

---

## Part 3: The Full Hybrid Architecture

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐
│  SBERT   │    │  Trainable  │    │    PQC      │    │  Task   │
│ (frozen) │ →  │  Projection │ →  │ (angle enc  │ →  │  Head   │
│  768-dim │    │  768 → 8    │    │  + layers)  │    │ (linear)│
└──────────┘    └─────────────┘    └─────────────┘    └─────────┘
  fixed           trainable          trainable          trainable
```

### What's frozen vs trainable?

| Component | Trainable? | Why |
|:----------|:-----------|:----|
| SBERT | ❌ Frozen | We want to study what the PQC does with SBERT's output, not change SBERT |
| Projection | ✅ Yes | Learns which SBERT features to pass to the PQC |
| PQC | ✅ Yes | Learns to process the compressed features |
| Task head | ✅ Yes | Maps PQC output to task-specific prediction |

### Why freeze SBERT?

If we fine-tune SBERT, it might change its representations to make the PQC's job trivial. We'd learn nothing about what the PQC does. By freezing SBERT, we force the PQC to work with the representations as-is.

---

## Part 4: Training Task — Sentence Similarity

For our first end-to-end experiment, we use a simple task:

**Given two sentences, predict whether they are paraphrases (same meaning) or not.**

Examples:
- ✅ Paraphrase: "dogs chase cats" / "cats are chased by dogs"
- ❌ Not paraphrase: "dogs chase cats" / "birds fly south"

The model architecture for this:
1. Encode sentence A → SBERT → project → PQC → vector_A
2. Encode sentence B → SBERT → project → PQC → vector_B
3. Compute cosine similarity between vector_A and vector_B
4. Predict: similar (>0.5) or dissimilar (≤0.5)

---

## Part 5: Extracting Intermediate Representations

This is where our research starts. After training, we extract:

| Representation | Where | Shape | What it contains |
|:--------------|:------|:------|:-----------------|
| `h_sbert` | SBERT output | (768,) | Full semantic + syntactic info |
| `h_proj` | After projection | (8,) | Compressed — what did projection keep? |
| `h_pqc` | After PQC | (8,) | What did the quantum circuit add/change? |

In Lesson 4, we'll compare these three representations using CKA (Centered Kernel Alignment) against our syntax fingerprint vectors from Lesson 1. This comparison is the core of our paper.

---

## Homework

Run `notebooks/lesson03_hybrid_pipeline.py`. It has 5 exercises:

1. **Explore SBERT embeddings** — encode sentences, measure cosine similarity
2. **Build the projection layer** — compress 768 → 8 dims, visualize information loss
3. **Build the full hybrid model** — SBERT + projection + PQC
4. **Train on sentence pairs** — end-to-end training on paraphrase detection
5. **Extract representations** — save h_sbert, h_proj, h_pqc for probing

Answer the 6 comprehension questions at the end.

> [!WARNING]
> **This lesson is harder than the previous two.** It combines everything you've learned. Budget ~4-5 hours. If you get stuck on the PyTorch wiring, that's normal — come back and ask me.
