# Task 03 Docs: Building the Paper's Datasets

> **Read this BEFORE running the script.**

---

## Why Three Datasets?

A top-tier reviewer will try to poke holes in your data. Three datasets block three different attacks:

| Reviewer Attack | Dataset That Blocks It |
|:---------------|:----------------------|
| *"Maybe CKA differences are from vocabulary, not syntax"* | **Templates** — same words, only syntax varies |
| *"Does this work beyond your custom templates?"* | **BLiMP** — established linguistic benchmark |
| *"Does this work on real language?"* | **MRPC** — real news sentences |

No single dataset is sufficient. Together, they make the paper airtight.

---

## Dataset A: Controlled Templates

### What It Is

Sentences generated from grammar templates using a fixed vocabulary pool. Example:

```
Template: "the [ADJ] [NOUN_agent] [VERB]ed the [ADJ] [NOUN_patient]"

Active:  "the tall doctor chased the young cat"
Passive: "the young cat was chased by the tall doctor"
```

Both sentences use **exactly the same words**. The ONLY difference is grammar. So if CKA detects a difference, it MUST be due to syntax.

### Five Construction Types

| Type | Example | What It Tests |
|:-----|:--------|:-------------|
| **Active** | "the tall doctor chased the young cat" | Basic SVO (baseline) |
| **Passive** | "the young cat was chased by the tall doctor" | Voice transformation |
| **Relative clause** | "the tall doctor that helped the nurse chased the young cat" | Embedding complexity |
| **Cleft** | "it was the tall doctor who chased the young cat" | Focus extraction |
| **Ditransitive** | "the tall doctor gave the young cat the book" | Argument structure |

### How We Control for Confounds

The key scientific design choice: **every sentence draws from the same pool** of 20 nouns, 20 adjectives, 20 verbs. This means:
- Active and passive sentences share vocabulary → lexical effects eliminated
- Any CKA difference between them is guaranteed to be syntactic
- A reviewer cannot argue "maybe the model just learned which words appear together"

---

## Dataset B: BLiMP

### What Is BLiMP?

**B**enchmark of **Li**nguistic **M**inimal **P**airs — created by linguists at NYU (Warstadt et al., 2020). It's the gold standard for testing whether models understand grammar.

Each item is a pair: one grammatical, one ungrammatical.
```
✓ "the senator that helped the teacher laughed"
✗ "the senator that helped the teacher laughed the actor"
```

### Why Do We Need It?

BLiMP gives us **credibility**. When a reviewer sees "BLiMP" in your paper, they know:
- The sentences were designed by professional linguists
- Each sub-task tests a specific, well-defined syntactic phenomenon
- Hundreds of papers have used it — it's an established benchmark

### Which Sub-tasks We Use

| Sub-task | What It Tests | Example |
|:---------|:-------------|:--------|
| `anaphor_agreement` | Reflexive pronouns | "the boy helped himself" vs "the boy helped themselves" |
| `subject_verb_agreement` | Number agreement | "the cats chase the dog" vs "the cats chases the dog" |

We use only the **grammatical** sentences (not the ungrammatical ones), grouped by linguistic phenomenon. We then extract WL fingerprints and use them for CKA.

---

## Dataset C: MRPC

### What Is MRPC?

**M**icrosoft **R**esearch **P**araphrase **C**orpus — 5,801 real sentence pairs from news articles, labeled by human annotators as paraphrase or not.

```
Paraphrase pair:
  A: "Revenue rose 4.3 percent to $1.27 billion."
  B: "Revenue increased 4.3% to $1.27 billion from $1.22 billion."

Non-paraphrase pair:
  A: "The Nasdaq composite index rose 11.41 points."
  B: "The technology-heavy Nasdaq rose 0.6 percent."
```

### Why Do We Need It?

MRPC proves our pipeline works on **real language**, not just artificial templates. It's part of the GLUE benchmark, which every NLP researcher knows.

### The Catch

MRPC sentences are long and complex (news language). BobcatParser will **fail on some** — maybe 20-30%. We:
1. Attempt to parse every sentence
2. Keep only successfully parsed ones
3. Report the success rate in the paper (this is expected and accepted)

---

## The Parsing Bottleneck

BobcatParser processes ~1-2 sentences per second. For all three datasets:

```
Templates:  ~1000 sentences × 1.5 sec = ~25 min
BLiMP:      ~400 sentences × 1.5 sec  = ~10 min
MRPC:       ~400 sentences × 1.5 sec  = ~10 min
                                Total ≈ 45 min
```

The script shows progress bars. Go get coffee. ☕

---

## What You'll See When You Run Task 03

1. **Part 1**: Template generation — 5 constructions × 200 sentences each
2. **Part 2**: BLiMP loading — 2 sub-tasks, 200 sentences each
3. **Part 3**: MRPC loading — 200 pairs
4. **Part 4**: Parsing everything with BobcatParser (the slow part)
5. **Part 5**: WL fingerprint extraction
6. **Part 6**: Dataset summary + CKA sanity check
7. **Output**: `data/paper_dataset.pt` — the complete dataset for all experiments

### Run it:
```bash
conda activate qnlp
cd /Users/nguyen/Documents/Work/Quantum
pip install datasets        # needed for BLiMP and MRPC
python scripts/task03_build_datasets.py
```
