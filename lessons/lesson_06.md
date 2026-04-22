# Lesson 6: Building the Controlled Syntactic Dataset

## Prerequisites
- ✅ Lessons 1-5 completed
- ✅ BobcatParser working
- ✅ Dataset strategy reviewed

## What You'll Learn

1. Generate controlled sentence pairs from syntactic templates
2. Validate parses with BobcatParser (handle failures gracefully)
3. Extract fingerprints at scale and verify syntactic grouping
4. Build train/test splits for the full experiment
5. Run a scaled CKA analysis (n=hundreds, not n=7)

> [!IMPORTANT]
> **This is no longer a tutorial — it's real research infrastructure.** The dataset you build here is what goes into the paper. Treat it accordingly.

---

## Part 1: Why Templates?

A reviewer will ask: *"How do you know the CKA difference is due to syntax and not vocabulary?"*

Our defense: **Templates use the same vocabulary across all constructions.** The sentence "the tall doctor examined the young patient" and "the young patient was examined by the tall doctor" share every word — only the grammar differs.

This means any CKA difference between SBERT representations of these sentences MUST be due to syntactic structure, not word choice.

---

## Part 2: Four Syntactic Constructions

| Construction | Syntactic complexity | Example |
|:-------------|:--------------------|:--------|
| **Active SVO** | Low (3 words, 2 cups) | "the doctor examined the patient" |
| **Passive** | High (5+ words, 6 cups) | "the patient was examined by the doctor" |
| **Relative clause** | Higher (embedded clause) | "the doctor that helped the nurse examined the patient" |
| **Cleft** | High (focus extraction) | "it was the doctor who examined the patient" |

Each construction produces a different DisCoCat diagram shape. Same words, different wiring.

---

## Part 3: The Generation Pipeline

```
Vocabulary Pool          Template Engine         BobcatParser
[nouns, verbs, adjs] --> fill templates --> parse & validate --> keep valid
                              |                     |
                         ~3000 raw             ~2000 parsed
                         sentences             (drop failures)
```

We generate more sentences than we need, then filter to only those BobcatParser can parse successfully. This ensures every sentence in our dataset has a valid DisCoCat diagram and fingerprint.

---

## Homework

Run `notebooks/lesson06_dataset.py`. It will:

1. Generate ~3000 sentences across 4 syntactic constructions
2. Parse each with BobcatParser (this takes ~15-30 minutes)
3. Extract structural fingerprints
4. Verify syntactic grouping with CKA
5. Save the complete dataset to `data/syntactic_dataset.pt`

> [!WARNING]
> **This script takes a while** because BobcatParser is slow (~1-2 seconds per sentence). Let it run. Go get coffee.
