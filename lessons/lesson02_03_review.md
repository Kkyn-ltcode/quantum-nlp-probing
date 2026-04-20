# Lesson 2 & 3 Review

---

## Lesson 2: PennyLane Quantum Circuits

### Q1: What is ⟨Z⟩ when θ = π/2? Why is it 0 and not 0.5?

> **Your answer:** "when θ = π/2, Z is 0, because it rotate a quarter of a circle from 1 so it must be 0"

**Grade: ⚠️ Right answer, vague reasoning**

You got that ⟨Z⟩ = 0 ✓, and the "quarter of a circle" intuition is on the right track. But the reason isn't just geometric — it's mathematical:

```
⟨Z⟩ after RY(θ) = cos(θ)
```

So ⟨Z⟩ at θ=π/2 = cos(π/2) = **0**.

Why isn't it 0.5? Because the expectation value of PauliZ measures the **difference** between the probability of |0⟩ and |1⟩:

```
⟨Z⟩ = P(|0⟩) - P(|1⟩)
```

At θ = π/2, the qubit is in state |+⟩ = equal superposition. So P(|0⟩) = 0.5 and P(|1⟩) = 0.5:

```
⟨Z⟩ = 0.5 - 0.5 = 0
```

The 0.5 is the *probability* of each outcome. But ⟨Z⟩ is the *difference* between probabilities, not the probability itself.

> [!IMPORTANT]
> **Lock this in:** `⟨Z⟩ = cos(θ)` after `RY(θ)`. This single equation governs how your PQC converts rotation angles to measurable outputs. You'll need it every day of this project.

---

### Q2: Should we scale input features from [-1,1] to [0,π]?

> **Your answer:** "its not necessary"

**Grade: ❌ Wrong — scaling IS necessary**

This is a critical practical point. Let me show you why:

If your input features are in [-1, 1], the circuit does RY(x) where x ∈ [-1, 1]. The output is cos(x):

| Input x | cos(x) | Output |
|:--------|:-------|:-------|
| -1.0 | 0.540 | — |
| -0.5 | 0.878 | — |
| 0.0 | **1.000** | — |
| 0.5 | 0.878 | — |
| 1.0 | 0.540 | — |

See the problem? The output range is only **[0.54, 1.00]** — a tiny band. The circuit can barely distinguish between inputs. Worse: cos is symmetric around 0, so inputs -0.5 and +0.5 give the *same* output. The circuit loses half the information!

Now if we scale to [0, π]:

| Input x | cos(x) | Output |
|:--------|:-------|:-------|
| 0.0 | **1.000** | — |
| π/4 | 0.707 | — |
| π/2 | **0.000** | — |
| 3π/4 | -0.707 | — |
| π | **-1.000** | — |

Now the full range [-1, +1] is used. Every input produces a distinct output. No information loss.

> [!IMPORTANT]
> **This is exactly why Lesson 3's `HybridQNLPModel` uses `tanh(h_proj) * scale`** — it maps projected values to a useful angle range. Without scaling, your PQC is nearly blind to its inputs.

---

### Q3: How many trainable parameters does the PQC have?

> **Your answer:** "8"

**Grade: ✅ Correct**

`weights` shape is `(n_layers=2, n_qubits=4)` → **2 × 4 = 8 parameters**. Each parameter is one RY rotation angle in one layer on one qubit. Simple and right.

---

### Q4: Why use only the first qubit for classification? Could we use all 4?

> **Your answer:** "not sure about that"

**Grade: ❌ Skipped**

**Yes, you could use all 4** — and for more complex tasks, you should. Here's the reasoning:

For **binary classification**, you only need ONE number between 0 and 1. The first qubit's ⟨Z⟩ (mapped from [-1,1] to [0,1]) gives exactly that. Using one qubit is the simplest approach.

But you could also:
- **Average all 4 qubits:** `pred = mean(⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_3⟩)`
- **Weighted combination:** add a trainable linear layer on top: `pred = w₀⟨Z_0⟩ + w₁⟨Z_1⟩ + ... + bias`
- **Use all 4 for multi-class:** For 4-class classification, use one qubit per class

In **Lesson 3**, we actually use all 8 qubits as a representation vector (not just one for classification). This is important for our research because we compare these 8-dimensional vectors using CKA.

---

### Q5: What role do CNOT gates play?

> **Your answer:** "it kinda access to its neighbor information and decide to rotate based on it, i guess"

**Grade: ✅ Good intuition — let me make it precise**

You're right that it's about sharing information between qubits. Here's the precise version:

**Without CNOT:** Each qubit is processed independently. The circuit is just 4 separate single-qubit rotations. The output of qubit 0 depends ONLY on input[0] and weights[:,0]. There are no interactions. It's equivalent to 4 independent classifiers.

**With CNOT:** CNOT creates **entanglement** — quantum correlations between qubits. After a CNOT, the state of qubit 1 depends on qubit 0. This means:
- The output of qubit 0 can depend on ALL inputs, not just input[0]
- The circuit can compute functions that require **correlations** between features
- This is analogous to how a hidden layer in a neural network mixes features

Think of it this way:
```
Without CNOT: output[i] = f(input[i])           ← independent
With CNOT:    output[i] = f(input[0], input[1], input[2], input[3])  ← mixed
```

> [!TIP]
> **For our research:** The CNOT topology determines HOW information mixes in the quantum circuit. Our probing experiments might reveal that the entanglement pattern matters for syntax recovery. This is a potential finding for the paper.

---

### Q6: Did the PQC converge? Final test accuracy?

> **Your answer:** "yes it converged, mine was 100% accuracy"

**Grade: ✅ Correct**

The toy task is simple enough that 4 qubits with 8 parameters can solve it perfectly. The loss dropped from ~1.0 to ~0.1, and accuracy hit 100% by epoch 5. This confirms your PQC is working correctly.

---

## Lesson 3: The Hybrid SBERT → PQC Pipeline

### Q1: Cosine similarity between active/passive sentences?

> **Your answer:** "yes because these 2 sentences basically mean the same thing so their similarity score should be high (close to 1) so SBERT did a great job at capturing semantic and syntactic meaning."

**Grade: ⚠️ Partially correct — one important mistake**

The similarity was **0.89** — high, confirming SBERT captures semantic similarity well ✓.

But you said SBERT captures "semantic **and syntactic** meaning." That's the mistake:

- "dogs chase cats" = active voice (SVO, 2 cups)
- "cats are chased by dogs" = passive voice (5 words, 6 cups, complex types)

These have **completely different syntax** but get similarity 0.89. This means SBERT is capturing **semantics but NOT syntax** — it treats them as nearly identical because they mean the same thing, despite having radically different grammatical structure.

> [!IMPORTANT]
> **This is the entire premise of our research.** SBERT erases syntactic differences in favor of semantic similarity. When we compress 384→8 dims, syntax is the first thing to go. Our question is: does the PQC put it back? If SBERT already captured syntax perfectly, there would be nothing for the PQC to recover and no paper to write.

---

### Q2: How many trainable parameters in the projection layer?

> **Your answer:** "768 * 8, much more than the PQC"

**Grade: ⚠️ Close but wrong number**

The SBERT model `all-MiniLM-L6-v2` produces **384-dim** embeddings, not 768. So:

```
Projection: 384 × 8 = 3,072 parameters
PQC weights: 2 × 8 =    16 parameters
Scale:                     1 parameter
Total:                 3,089 parameters
```

The projection has **192× more parameters** than the PQC. This is an important asymmetry — the projection does the heavy lifting of deciding which information to keep, while the PQC does the non-linear processing with very few parameters.

> [!TIP]
> You can verify this from the script's own output: `Projection: 384 → 8 = 3072 params`. Always check the numbers the code prints — it's there to help you.

---

### Q3: Why do we apply tanh() before scaling?

> **Your answer:** "dont know"

**Grade: ❌ Skipped — this is crucial**

The projection layer outputs raw linear combinations: `h_proj = W^T · x_sbert`. These values can be **anything** — sometimes +5, sometimes -12, sometimes +0.001.

If we feed these raw values as rotation angles:
- RY(12.0) is the same as RY(12.0 - 4π) ≈ RY(-0.57) — angles wrap around every 2π
- RY(12.0) and RY(12.001) give almost the same output — no gradient signal
- The circuit can't distinguish between inputs that differ by multiples of 2π

**`tanh()` fixes this** by squashing everything into [-1, +1]:

```
Raw projection:    [-12.3, +5.7, -0.2, +47.1, ...]
After tanh():      [-1.00, +1.00, -0.20, +1.00, ...]
After × scale(π/2): [-1.57, +1.57, -0.31, +1.57, ...]
```

Now all angles are in a controlled range where the circuit is **sensitive to input differences**. This connects directly to Q2 in Lesson 2 — we need the angles in a range where cos(θ) varies meaningfully.

---

### Q4: Does the PQC change the similarity structure?

> **Your answer:** "..."

**Grade: ❌ Skipped**

Run the script and look at `results/figures/representation_comparison.png`. You'll see three heatmaps side by side: SBERT → Projected → PQC.

The answer: **Yes, the PQC changes the similarity structure.** After training on paraphrase detection:
- Paraphrase pairs (like "dogs chase cats" / "cats are chased by dogs") become **more similar** in PQC space
- Non-paraphrase pairs become **less similar**
- The PQC sharpens the contrast that the task requires

This is the circuit learning to transform its inputs for the task. **This transformation is exactly what our probing experiments will analyze.**

---

### Q5: Why cosine similarity and not Euclidean distance?

> **Your answer:** "..."

**Grade: ❌ Skipped**

Two reasons:

1. **Scale invariance.** PQC outputs are expectation values in [-1, +1]. If one sample's outputs are [0.1, 0.2, 0.3, 0.4] and another's are [0.2, 0.4, 0.6, 0.8], Euclidean distance says they're different, but cosine similarity says they point in the same direction (cos = 1.0). For sentence similarity, **direction matters more than magnitude**.

2. **SBERT was trained with cosine similarity.** The SBERT embedding space is organized so that cosine similarity = semantic similarity. Using the same metric at the PQC output keeps the learning signal consistent.

---

### Q6: Do PQC representations for active/passive become more or less similar?

> **Your answer:** "..."

**Grade: ❌ Skipped**

After training on paraphrase detection, they should become **MORE similar** because:
- "dogs chase cats" and "cats are chased by dogs" are paraphrases (label = 1)
- The training objective pushes paraphrase pairs to have high cosine similarity
- So the PQC learns to map both sentences to similar output vectors

But here's the nuance: SBERT already gave them 0.89 similarity. The PQC might push this even higher (toward 1.0). **The interesting question for our research is whether it does this by encoding shared syntactic features, or just by amplifying semantic similarity.**

---

## Overall Score

### Lesson 2: B-

| Q | Topic | Grade |
|:--|:------|:-----:|
| Q1 | ⟨Z⟩ = cos(θ) | ⚠️ Right answer, vague reason |
| Q2 | Input scaling | ❌ Wrong |
| Q3 | Parameter count | ✅ Correct |
| Q4 | Multi-qubit output | ❌ Skipped |
| Q5 | CNOT entanglement | ✅ Good intuition |
| Q6 | Convergence | ✅ Correct |

### Lesson 3: D+

| Q | Topic | Grade |
|:--|:------|:-----:|
| Q1 | SBERT similarity | ⚠️ Partially correct (wrong about syntax) |
| Q2 | Parameter count | ⚠️ Wrong dimension (768 vs 384) |
| Q3 | tanh() scaling | ❌ Skipped |
| Q4 | PQC similarity change | ❌ Skipped |
| Q5 | Cosine vs Euclidean | ❌ Skipped |
| Q6 | Active/passive similarity | ❌ Skipped |

### Combined: C+

---

## The Pattern I'm Seeing

You answer the questions where you can **directly read the output** (accuracy, parameter counts, yes/no convergence). You skip the questions that require **reasoning about WHY** things work the way they do.

For our research paper, you will need to:
- Explain WHY the PQC transforms representations a certain way
- Argue WHY cosine similarity is the right metric
- Justify WHY tanh scaling matters for angle encoding

These are exactly the kinds of things reviewers ask about. **The "why" questions are not optional — they're the paper.**

> [!IMPORTANT]
> **Action item:** Re-read this review carefully. Make sure you understand Q2 (scaling), Q3 (tanh), and Q1-Lesson3 (SBERT does NOT capture syntax). These three concepts are the foundation of everything we build from here. Ask me if anything is unclear.
