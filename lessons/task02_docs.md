# Task 02 Docs: Graph Kernel Fingerprints

> **Read this BEFORE running the script.**

---

## The Problem We're Solving

In Lesson 4, we built "structural fingerprints" by counting things in each DisCoCat diagram:

```
"dogs chase cats"              → [3 words, 2 cups, 5 boxes, ...]
"cats are chased by dogs"      → [5 words, 6 cups, 11 boxes, ...]
```

This worked, but you noticed the problem in Q6: **all cosine similarities were > 0.92**. The fingerprints couldn't tell constructions apart because they only counted totals — like describing a house by "4 walls, 1 roof, 3 windows" without saying how they're connected.

We need fingerprints that capture **structure** (how things are connected), not just **inventory** (how many things exist).

---

## What Is a Graph?

A graph is just **dots connected by lines**:

```
    A ---- B
    |      |
    C ---- D
```

- **Nodes** (dots): A, B, C, D
- **Edges** (lines): A-B, A-C, B-D, C-D
- **Labels**: each node can have a name/type

Graphs are everywhere: social networks (people = nodes, friendships = edges), molecules (atoms = nodes, bonds = edges), and — crucially for us — **DisCoCat diagrams**.

---

## DisCoCat Diagrams ARE Graphs

When BobcatParser produces a diagram for "dogs chase cats":

```
 dogs(n)    chase(n.r⊗s⊗n.l)    cats(n)
   │              │                │
   └──── Cup ─────┘                │
                   │               │
                   └──── Cup ──────┘
```

This is a graph with:
- **Nodes**: `Word:n`, `Word:n.r⊗s⊗n.l`, `Word:n`, `Cup`, `Cup`
- **Edges**: boxes connected by wires

For "cats are chased by dogs" (passive), the diagram is **different**:

```
 cats(n)  are(...)  chased(...)  by(...)  dogs(n)
   │        │          │          │        │
   └─ Cup ──┘          │          │        │
             └── Cup ───┘          │        │
                        └── Cup ───┘        │
                                  └─ Cup ───┘
                                       ... (more cups)
```

More nodes, more cups, different wiring. **Same words, different graph structure.**

---

## What Is a Graph Kernel?

A **kernel** is a function that measures similarity between two objects:

```
K(graph_A, graph_B) = how similar are their structures?
```

A **graph kernel** compares graphs by looking at their **substructures** — not just total counts, but patterns of connections.

Think of it like comparing two buildings:
- **Count-based**: "Both have 20 rooms" → similar!
- **Graph kernel**: "Building A has rooms connected in a circle, Building B has rooms in a line" → different!

---

## The Weisfeiler-Leman (WL) Algorithm

WL is a specific graph kernel algorithm. It works by **iteratively enriching node labels** with information from their neighbors.

### Step-by-Step Example

Let's trace WL on the graph for "dogs chase cats":

```
Initial graph:
    Word:n ──── Word:n.r.s.n.l ──── Word:n ──── Cup ──── Cup
     (0)            (1)              (2)         (3)      (4)
```

**Iteration 0** — Just use the original labels:
```
Node 0: "Word:n"
Node 1: "Word:n.r.s.n.l"
Node 2: "Word:n"
Node 3: "Cup"
Node 4: "Cup"
```

**Iteration 1** — Each node's new label = its old label + sorted neighbor labels:
```
Node 0: "Word:n" + neighbors[Node 1] → "Word:n | Word:n.r.s.n.l"
Node 1: "Word:n.r.s.n.l" + neighbors[Node 0, Node 2] → "Word:n.r.s.n.l | Word:n | Word:n"
Node 2: "Word:n" + neighbors[Node 1, Node 3] → "Word:n | Cup | Word:n.r.s.n.l"
...
```

Now node 0 and node 2 have **different** labels even though they both started as "Word:n" — because they have different neighbors!

**Iteration 2** — Repeat: each node absorbs its neighbors' (already enriched) labels.

**Iteration 3** — Repeat again. By now, each node's label encodes a "3-hop neighborhood" — the structure within 3 steps of that node.

### The Feature Vector

After all iterations, we count how many times each unique label appeared across all iterations:

```
"dogs chase cats" →  {"Word:n": 2, "Cup": 2, "Word:n|Word:n.r.s.n.l": 1, ...}
"cats are chased by dogs" → {"Word:n": 2, "Cup": 6, "Word:n|Cup|...": 1, ...}
```

These histograms become fixed-size vectors. **Different graph topologies produce different histograms.**

---

## Why WL Is Better Than Counting

| Aspect | Count-Based | WL Kernel |
|:-------|:------------|:----------|
| What it captures | How many boxes of each type | How boxes are **connected** |
| "dogs chase cats" vs "cats are chased by dogs" | Mostly similar (both have Word:n boxes) | Very different (different wiring patterns) |
| Active vs Passive | Cosine sim ~0.95 (almost identical!) | Cosine sim much lower (clearly different) |
| Relative clause detection | Weak (just more boxes) | Strong (embedded clause = different neighborhood) |

### The Key Insight

Count-based fingerprints say: *"Both sentences have nouns and verbs."*
WL kernel says: *"In sentence A, the verb is directly connected to both nouns. In sentence B, the verb connects through function words."*

This is exactly what we need. **Different syntax = different wiring = different WL features.**

---

## How It Connects to the Paper

In the paper, we replace this:
```
CKA(PQC_output, count_fingerprint)  ← weak reference signal
```

With this:
```
CKA(PQC_output, WL_fingerprint)  ← strong reference signal
```

If CKA(PQC, WL) is high, it means the PQC's representations organize sentences the same way the **syntax graph topology** organizes them. That's a much stronger claim than comparing to box counts.

### In the Methods section, we'll write:

> *"We represent syntactic structure using Weisfeiler-Leman graph kernel features (Shervashidze et al., 2011) extracted from DisCoCat string diagrams. Unlike bag-of-boxes representations, WL features capture the compositional topology of grammatical structure."*

---

## What You'll See When You Run Task 02

1. **Part 2**: The actual graph nodes/edges for active vs passive — you'll see they have different structures
2. **Part 4**: Cosine similarity comparison — WL should separate active/passive better
3. **Part 5**: CKA comparison — WL should discriminate construction types better
4. **Part 6**: Two figures comparing count-based vs WL side by side

### Run it:
```bash
conda activate qnlp
cd /Users/nguyen/Documents/Work/Quantum
python scripts/task02_verify_fingerprints.py
```

Then answer the 5 questions and share your output with me.
