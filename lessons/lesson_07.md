# Lesson 7: The Full Experiment — Paper-Scale Results

## Prerequisites
- ✅ Lessons 1-6 completed
- ✅ `data/syntactic_dataset.pt` generated from Lesson 6

## What You'll Learn

1. Train PQC + MLP on the full Tier 1 dataset (not 16 pairs — hundreds)
2. Run **multi-seed evaluation** (5 seeds for confidence intervals)
3. Perform **permutation tests** for CKA significance
4. Generate the paper's **main results table** and figures

> [!IMPORTANT]
> **This lesson produces Section 5 of the paper.** The numbers, figures, and statistical tests from this script are what you'll report. No more toy examples.

---

## Part 1: Why Multiple Seeds?

In Lessons 3-5, we trained once and reported one number. A reviewer will ask: *"Is that result stable, or did you just get lucky with initialization?"*

**Multi-seed protocol:**
- Train each model 5 times with different random seeds
- Report **mean ± standard deviation** of CKA scores
- If CKA(PQC, Syntax) > CKA(MLP, Syntax) across all 5 seeds → robust finding

---

## Part 2: Permutation Tests

Even with n=800 sentences and 5 seeds, a reviewer can ask: *"Is CKA=0.15 statistically significant, or could random fingerprints produce the same score?"*

**Permutation test:**
1. Compute real CKA(PQC, Syntax) = X
2. Shuffle the syntax fingerprints randomly 1000 times
3. Compute CKA(PQC, shuffled_Syntax) for each shuffle
4. p-value = fraction of shuffled CKAs ≥ X

If p < 0.05, the alignment is statistically significant.

---

## Part 3: The Full Pipeline

```
Dataset (Lesson 6)
    ↓
SBERT encode (frozen)
    ↓
┌────────────┬────────────┐
│ PQC path   │ MLP path   │
│ proj → PQC │ proj → MLP │
│ × 5 seeds  │ × 5 seeds  │
└────────────┴────────────┘
    ↓
CKA analysis (all stages × syntax fingerprints)
    ↓
Permutation tests (1000 shuffles)
    ↓
Paper results table + figures
```

---

## Homework

Run `notebooks/lesson07_full_experiment.py`. It will take ~30-60 minutes total. The script:

1. Loads the dataset and SBERT encodes all sentences
2. Trains PQC × 5 seeds, MLP × 5 seeds
3. Extracts representations at every pipeline stage
4. Computes CKA with syntax fingerprints (with confidence intervals)
5. Runs permutation tests for significance
6. Generates the paper's main results

Answer the 5 comprehension questions at the end.
