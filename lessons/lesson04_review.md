# Lesson 4 Review — Syntax Fingerprints & CKA Probing

## Overall Grade: B

Major improvement from Lessons 2-3. You engaged with every question, showed good skepticism about the small sample size, and even identified a real problem in Q6 that I need to address. Let's go through each one.

---

### Q1: Fingerprint similarity between active/passive

> **Your answer:** "The cosine similarity between 'dogs chase cats' and 'cats are chased by dogs' is low because they different in structure even though they mean exactly the same thing..."

**Grade: ❌ The reasoning is right but the fact is wrong**

Your reasoning is perfect — they SHOULD be different structurally. But look at the actual output:

```
Fingerprint cosine similarity:
  'dogs chase cats' vs 'cats are chased by dogs': 0.958
```

It's **0.958 — very HIGH**, not low! This is a problem with our fingerprint, not with your logic. The fingerprints are mostly magnitude-based counts (3 words vs 5 words, 2 cups vs 6 cups), and sentences with more complex syntax have proportionally more of everything, so the cosine similarity ends up high.

> [!IMPORTANT]
> **You had the right instinct but didn't check the actual number.** Always look at what the code PRINTED before writing your answer. This is a scientific discipline: observation first, interpretation second.

That said, even though cosine similarity is high, the fingerprint vectors ARE different (different values in each dimension). CKA captures this difference even when cosine similarity is high, because CKA compares how the **set of all sentences** relate to each other, not just two individual vectors.

---

### Q2: Why is scale invariance important?

> **Your answer:** "Because what we want to know is the similarity of the structure between vectors, not about the volume"

**Grade: ✅ Good intuition**

"Volume" isn't quite the right word — **magnitude** or **scale** is more precise. But the core idea is right:

- PQC outputs are in [-1, +1] (small numbers)
- Syntax fingerprints have values like 5, 6, 10 (big numbers)
- If our metric were sensitive to scale, it would say "these are totally different!" just because one uses big numbers and one uses small numbers
- CKA doesn't care about the absolute magnitude — it only cares about the **relative relationships** between data points

Think of it like this: if you measured one city's population in thousands and another in millions, you wouldn't want your comparison metric to say they're different just because of the units.

---

### Q3: Highest and lowest CKA pairs

> **Your answer:** "the CKA of (projected, pqc) is the highest (0.68) while (syntax, pqc) is the lowest, which means after gone through the pqc, the majority of information have lost... but the CKA of (projected, pqc) is still high, so we can conclude that the pqc can preserve the information after squishing too much information (384 → 8)"

**Grade: ⚠️ Right numbers, wrong interpretation**

The actual matrix was:

```
             SBERT  Projected    PQC  Syntax
SBERT        1.000      0.667  0.680   0.340
Projected    0.667      1.000  0.854   0.082
PQC          0.680      0.854  1.000   0.150
Syntax       0.340      0.082  0.150   1.000
```

- **Highest:** CKA(Projected, PQC) = **0.854**, not 0.68. The PQC preserves most of the projected information ✓
- **Lowest:** CKA(Projected, Syntax) = **0.082**, not (Syntax, PQC)

But your interpretation has a key error: **low CKA with Syntax doesn't mean "information was lost."** It means the representations don't ORGANIZE sentences in the same way as the syntax fingerprint. The PQC could have perfect information but organize it by meaning (like SBERT) rather than by grammar (like Syntax).

> [!IMPORTANT]
> **CKA measures STRUCTURAL ALIGNMENT, not information content.** Low CKA(PQC, Syntax) means "the PQC organizes sentences differently from how syntax organizes them." It does NOT mean "the PQC lost information." A dictionary organized alphabetically has low alignment with one organized by topic — but both contain the same information.

Also, the compression from 384→8 happens at the **projection** stage, not the PQC. The PQC takes 8-dim input and produces 8-dim output — no further compression.

---

### Q4: PQC increased or decreased syntactic alignment?

> **Your answer:** "cka(projected, syntax) is 0.082 < cka(pqc, syntax) (0.15) but the difference is not that much, maybe we test on a super small dataset so the result might not be that significant."

**Grade: ✅ Correct observation + excellent skepticism**

You got the key finding: **PQC nearly doubled the syntactic alignment** (0.082 → 0.150, an 83% increase). And your skepticism about sample size is exactly right — with n=7, this could be noise.

For the paper, we'll need:
- 500+ sentences minimum
- Permutation tests (shuffle the syntax labels 1000 times, see if the real CKA is higher than chance)
- Multiple random seeds (retrain the PQC 5+ times)
- Bootstrap confidence intervals

But the direction is promising.

---

### Q5: Why 7 sentences is a problem

> **Your answer:** "i think the higher the number of sentences, the more reliable the CKA will be, i guess we need about 2000+ sentences."

**Grade: ✅ Correct, but let me add the technical reason**

2000+ is a good target. Here's WHY small n is specifically bad for CKA:

1. **CKA computes an n×n kernel matrix.** With n=7, that's a 7×7 matrix = only 21 unique pairwise comparisons. Random chance can easily produce high CKA from 21 numbers.

2. **You saw this in the sanity check:** CKA(X, random) = 0.646 — that should be ~0, but with n=7, even random matrices have high CKA. This is the **finite-sample bias** problem.

3. **Rule of thumb:** n should be at least 5-10× the representation dimensionality. Our SBERT has 384 dims, so we'd want n ≥ 2000. Our PQC has 8 dims, so n ≥ 40 at minimum.

---

### Q6: Finding semantically similar / syntactically different pairs

> **Your answer:** "its kinda hard to find because the syntax heat map's value are really high (0.92+), i need you to look at that to tell."

**Grade: ⚠️ Valid observation — you found a real problem**

You're right that the syntax fingerprint similarities are all very high (0.92+), which makes it hard to see differences. This is the same issue as Q1 — our fingerprint is too dominated by correlated count features.

But the answer to the question is still clear from the data:

**Semantically SIMILAR, syntactically DIFFERENT:**
- **"dogs chase cats" (0) and "cats are chased by dogs" (1)**
- SBERT similarity: 0.89 (high — same meaning)
- They have different box counts (5 vs 11), cup counts (2 vs 6), and word types

**Semantically DIFFERENT, syntactically SIMILAR:**
- **"cats are chased by dogs" (1) and "the weather is nice today" (6)**
- SBERT similarity: 0.01 (near zero — completely different meaning)
- But both have: 5 words, 6 cups, 11 total boxes, same cups_ratio (1.20)

> [!TIP]
> **Your observation about the fingerprint problem is actually valuable for the paper.** In the Methodology section, we'll need to discuss fingerprint design and acknowledge that simple count-based features have limited discriminative power. For the final paper, we'll likely upgrade to a graph kernel (Weisfeiler-Leman) which captures topological structure, not just counts.

---

## Score Summary

| Q | Topic | Grade | Notes |
|:--|:------|:-----:|:------|
| Q1 | Fingerprint similarity | ❌ | Right logic, wrong fact (didn't check the output) |
| Q2 | Scale invariance | ✅ | Good intuition |
| Q3 | CKA interpretation | ⚠️ | Wrong numbers, wrong interpretation of "low CKA" |
| Q4 | PQC vs Projected | ✅ | Correct + good sample size skepticism |
| Q5 | Sample size | ✅ | Correct target, needs technical depth |
| Q6 | Semantics vs Syntax | ⚠️ | Valid problem identified, but didn't answer |

## Key Takeaways

1. **Always check the actual output before interpreting** (Q1 — you assumed low but it was 0.958)
2. **CKA ≠ information content.** Low CKA means different ORGANIZATION, not information loss (Q3)
3. **Your skepticism about sample size is exactly the right research instinct** (Q4, Q5)
4. **You found a real engineering problem** with the fingerprints (Q6) — that's good research thinking

Overall: **B — significant improvement.** You're engaging with the "why" questions now and showing genuine research instincts. Keep it up for Lesson 5.
