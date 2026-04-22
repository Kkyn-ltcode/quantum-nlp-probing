# Dataset Strategy for the Paper

## Why Our Current Data Fails

Our 16 hand-written sentence pairs with 7 analysis sentences have three fatal problems:

1. **CKA is unreliable at n=7** (we saw CKA(X, random) = 0.646 instead of ~0)
2. **No syntactic diversity** — we only tested active/passive, nothing else
3. **No reviewer would take this seriously** — a top venue demands rigorous evaluation

## What We Need: Three Tiers

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: Controlled Syntactic Pairs (CORE)          │
│  ~2000 pairs, template-generated                    │
│  Purpose: Main CKA probing experiments              │
├─────────────────────────────────────────────────────┤
│  Tier 2: BLiMP Subsets (GENERALIZATION)              │
│  ~3000-4000 sentences from established benchmark    │
│  Purpose: Show findings generalize across phenomena │
├─────────────────────────────────────────────────────┤
│  Tier 3: MRPC (REAL-WORLD VALIDATION)                │
│  ~5800 real paraphrase pairs                        │
│  Purpose: Show pipeline works on natural language   │
└─────────────────────────────────────────────────────┘
```

---

## Tier 1: Controlled Syntactic Pairs

### Why template-generated?

We need to **isolate syntax from semantics**. If we use natural language, a reviewer can always argue: "Maybe the CKA difference is due to lexical choice, not syntax." Templates eliminate this.

### The constructions we need

#### 1. Active / Passive (~500 pairs)
Same meaning, different syntax. The core test case.

```
Active:  "the tall doctor examined the young patient"
Passive: "the young patient was examined by the tall doctor"
```

Template: `the [ADJ] [NOUN_agent] [VERB]ed the [ADJ] [NOUN_patient]`

#### 2. Relative Clauses (~500 pairs)
Tests embedding complexity.

```
Simple:    "the dog chased the cat"
Relative:  "the dog that saw the bird chased the cat"
```

Template: `the [NOUN] that [VERB]ed the [NOUN] [VERB]ed the [NOUN]`

#### 3. Cleft Constructions (~500 pairs)
Tests focus movement.

```
Normal: "the chef cooked the meal"
Cleft:  "it was the chef who cooked the meal"
```

#### 4. Topicalization (~500 pairs)
Tests word order variation.

```
Normal:      "the student read the book carefully"
Topicalized: "the book the student read carefully"
```

### How to generate

Use a fixed vocabulary pool (~50 nouns, ~30 verbs, ~20 adjectives) drawn from common English words. Combine with templates programmatically.

```python
nouns = ["dog", "cat", "chef", "doctor", "teacher", "student", ...]
verbs = ["chased", "examined", "cooked", "praised", "watched", ...]
adjs  = ["tall", "young", "old", "small", "clever", "quiet", ...]

# Generate active/passive pair
agent, patient = random.sample(nouns, 2)
verb = random.choice(verbs)
adj1, adj2 = random.sample(adjs, 2)

active  = f"the {adj1} {agent} {verb} the {adj2} {patient}"
passive = f"the {adj2} {patient} was {verb} by the {adj1} {agent}"
```

### Why this is scientifically strong

- **Same vocabulary** across all constructions → lexical effects controlled
- **Systematic syntactic variation** → we know exactly what differs
- **Perfect paraphrase labels** → no annotation ambiguity
- **BobcatParser can parse templates** → fingerprints are extractable

---

## Tier 2: BLiMP Subsets

### What is BLiMP?

**B**enchmark of **Li**nguistic **M**inimal **P**airs — 67 sub-tasks, each testing a specific syntactic phenomenon. Each item is a pair: one grammatical sentence, one ungrammatical variant.

```
✓ "the author that the senator liked laughed"
✗ "the author that the senator liked laughed the actor"
```

### Which sub-tasks we need

| Sub-task | Phenomenon | Size | Why |
|:---------|:-----------|:-----|:----|
| `anaphor_agreement` | Reflexive pronouns | 1000 | Tests agreement at distance |
| `filler_gap_dependency` | Wh-movement | 1000 | Tests long-range dependencies |
| `relative_clause` | Embedded clauses | 1000 | Directly relevant to our constructions |
| `subject_verb_agreement` | Number agreement | 1000 | Tests hierarchical structure |

### How to use BLiMP for our paper

We don't use BLiMP for the same task as the original benchmark (acceptability). Instead:
1. Take only the **grammatical** sentences from each sub-task
2. Group them by syntactic construction type
3. Extract fingerprints and compute CKA
4. Show that PQC's syntactic alignment holds across multiple phenomena

> [!IMPORTANT]
> **BLiMP is freely available** at https://github.com/alexwarstadt/blimp — just download the JSON files. No special access needed.

---

## Tier 3: MRPC (Real-World Validation)

### What is MRPC?

**M**icrosoft **R**esearch **P**araphrase **C**orpus — 5,801 sentence pairs from news articles, labeled as paraphrase/non-paraphrase by human annotators.

```
Paraphrase:
  A: "Revenue rose 4.3 percent to $1.27 billion."
  B: "Revenue increased 4.3% to $1.27 billion from $1.22 billion."

Non-paraphrase:
  A: "The Nasdaq composite index rose 11.41 points to 1,688.19."
  B: "The technology-heavy Nasdaq rose 0.6 percent to 1688.19."
```

### Why MRPC?

- **Real-world language** — proves our pipeline isn't limited to templates
- **Standard NLP benchmark** — reviewers know and respect it
- **Available via HuggingFace** — `datasets.load_dataset("glue", "mrpc")`
- **Paraphrase detection** — same task as our Lesson 3 pipeline

### Limitations for our study

MRPC sentences are **long and complex** (news language). BobcatParser may struggle with some. We'll need to:
- Filter to sentences BobcatParser can successfully parse
- Report the parse success rate
- Acknowledge this as a limitation

---

## Summary: What We Build

| Tier | Source | Size | Purpose | Effort |
|:-----|:-------|:-----|:--------|:-------|
| **1** | Template generation | ~2000 pairs | Core CKA experiments | Build from scratch (~2 days) |
| **2** | BLiMP download | ~4000 sentences | Syntactic generalization | Download + filter (~1 day) |
| **3** | MRPC via HuggingFace | ~5800 pairs | Real-world validation | Load + parse (~1 day) |
| **Total** | | ~10,000+ sentences | Full paper | ~4 days |

## What Reviewers Will Ask

1. **"Why templates?"** → "To isolate syntactic variation from lexical/semantic confounds."
2. **"Do results generalize beyond templates?"** → "Yes — BLiMP and MRPC show consistent patterns."
3. **"Is 2000 enough for CKA?"** → "Yes — with 8-dim representations, n=2000 is 250× the feature dimension, well above the reliability threshold."
4. **"Why not a bigger model?"** → "Simulator constraints. We discuss scaling to real hardware in Future Work."

## Recommended Next Step

Start with **Tier 1** (template generation) — it's the most important and the most controlled. This becomes Lesson 6. Once that's working, adding Tier 2 and 3 is straightforward data loading.
