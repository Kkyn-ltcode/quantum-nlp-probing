# Lesson 8: Gradient Saliency — Where Does the PQC Attend?

## Prerequisites
- ✅ Lessons 1-7 completed
- ✅ `results/full_experiment.pt` from Lesson 7

## What You'll Learn

1. Implement **Integrated Gradients** through the full pipeline
2. Map saliency from 384-dim SBERT space back to **individual tokens**
3. Correlate saliency with **POS tags** (function vs content words)
4. Generate saliency heatmap visualizations for the paper

> [!IMPORTANT]
> This is Experiment 4 from the paper design. It answers: *"Does the PQC attend to syntactically important words (function words, verbs) more than content words (nouns, adjectives)?"*

---

## Part 1: Why Gradient Saliency?

CKA tells us the PQC's output **aligns with syntax**. But it doesn't tell us **how**. Gradient saliency answers: which parts of the input most influence the PQC's output?

If the PQC attends to:
- **Function words** (that, was, by, who) → evidence it's using syntactic cues
- **Content words** (doctor, cat, tall) → it's using semantic/lexical cues
- **Uniformly** → it's not selective at all

---

## Part 2: Integrated Gradients

Simple gradients (∂output/∂input) are noisy. **Integrated Gradients** (Sundararajan et al., 2017) are more reliable:

```
IG(x) = (x - baseline) × ∫₀¹ ∂F(baseline + α(x - baseline))/∂x dα
```

In practice, we approximate the integral with ~50 steps:

```python
for alpha in [0, 0.02, 0.04, ..., 1.0]:
    x_interp = baseline + alpha * (x - baseline)
    grad += ∂F(x_interp)/∂x
IG = (x - baseline) * grad / n_steps
```

The baseline is a zero vector (no input). The result is a saliency score per input dimension.

---

## Part 3: From 384-dim Saliency to Tokens

SBERT produces a 384-dim vector from a sentence. To map saliency back to tokens:

1. Compute IG in 384-dim SBERT space → saliency vector (384,)
2. Get SBERT's token embeddings for the sentence → matrix (n_tokens, 384)
3. Project saliency onto each token: `token_saliency[i] = |saliency · token_embed[i]|`

This gives one saliency score per word.
