# Lesson 5: The Classical Baseline — MLP vs PQC

## Prerequisites
- ✅ Lessons 1-4 completed
- ✅ Lesson 3's hybrid model trained
- ✅ Lesson 4's CKA analysis run

## What You'll Learn

1. Build a **parameter-matched MLP** that replaces the PQC
2. Train it on the **same task** with the **same data**
3. Run the **same CKA analysis** from Lesson 4
4. Compare: do PQC and MLP encode syntax differently?

> [!IMPORTANT]
> **This lesson produces the paper's H2 result.** H2 asks: "Do PQC and MLP arrive at solutions via representationally distinct pathways?" The answer comes from comparing CKA(PQC, Syntax) vs CKA(MLP, Syntax).

---

## Part 1: Why Do We Need a Classical Baseline?

Without a baseline, our Lesson 4 result — "the PQC recovered syntactic alignment" — could mean:

1. The PQC does something **special** (quantum magic!)
2. **Any** non-linear function would do the same thing

Option 2 is the null hypothesis we must rule out. If a simple MLP with the same number of parameters produces the same CKA(output, syntax) score, then the PQC isn't doing anything quantum-specific. It's just being a generic non-linearity.

### But even if the MLP matches the PQC, we still have a paper!

- **If CKA(PQC, syntax) > CKA(MLP, syntax):** "The quantum circuit employs a syntax-aligned representational strategy that classical networks don't." → Strong positive.
- **If CKA(PQC, syntax) ≈ CKA(MLP, syntax):** "Both converge on similar strategies despite vastly different computational substrates." → Important finding about representational convergence.
- **If CKA(PQC, syntax) < CKA(MLP, syntax):** "Classical networks are more syntax-aligned; PQC captures something else." → Surprising, publishable.

---

## Part 2: Parameter Matching

### Why matching matters

If the MLP has 10,000 parameters and the PQC has 16, any difference could be due to **capacity**, not **architecture**. We must compare apples to apples.

Our PQC from Lesson 3:
- Trainable weights: 2 layers × 8 qubits = **16 parameters**
- Input: 8-dim (from projection)
- Output: 8-dim (expectation values)

So our MLP must also:
- Have **~16 trainable parameters** (weights + biases)
- Take 8-dim input
- Produce 8-dim output

### The MLP architecture

```
Input (8) → Linear(8, 8) → tanh → output (8)
```

Parameters: 8×8 weights + 8 bias = **72 parameters**

That's too many! We need to constrain it. Options:
- Use a **bottleneck**: Linear(8, 2) → tanh → Linear(2, 8) = 8×2 + 2 + 2×8 + 8 = 36 params
- Use **no bias**: Linear(8, 2, bias=False) → tanh → Linear(2, 8, bias=False) = 32 params

We'll use the bottleneck without bias to get close to 16 params, or simply accept the slight mismatch and report it transparently.

> [!TIP]
> **For the paper:** Report exact parameter counts for both models. Reviewers appreciate transparency. A small mismatch (16 vs 32) is acceptable if you discuss it. A 10× mismatch is not.

---

## Part 3: The Full Comparison

After training both models on the same task, we extract representations and compute:

```
                    CKA with Syntax
SBERT               0.340
Projected            0.082
PQC output           0.150  ← from Lesson 4
MLP output           ???    ← this lesson
```

We also compute CKA **between** PQC and MLP outputs:
- CKA(PQC, MLP): Are their representations similar or different?

This gives us the full picture for the paper.

---

## Homework

Run `notebooks/lesson05_mlp_baseline.py`. It has 5 exercises:

1. **Build the parameter-matched MLP**
2. **Train on the same paraphrase task** as Lesson 3
3. **Extract MLP representations**
4. **Run comparative CKA analysis** (PQC vs MLP vs Syntax)
5. **Generate the paper's comparison figure**

Answer the 5 comprehension questions at the end.
