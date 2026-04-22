# Roadmap: From Pilot Study to Top-Tier Publication

> **Target:** EMNLP 2027 (deadline ~June 2027) or ICLR 2028 (deadline ~Oct 2027)
> **Backup:** AAAI 2028 (deadline ~Aug 2027)
> **Timeline:** ~5-6 months from now

---

## What We Have vs What We Need

| Dimension | Current (Pilot) | Required (Top-Tier) | Gap |
|:----------|:----------------|:--------------------|:----|
| Qubits | 8 | 16-20 | 2-2.5× |
| Sentences | ~800 templates | 5000+ across 3 datasets | 6× |
| Baselines | 1 (MLP) | 4-5 classical models | Need 3-4 more |
| Fingerprint | Count-based (coarse) | Graph kernel (discriminative) | Rebuild |
| Seeds | 5 | 20 | 4× |
| Stats | Permutation test | Debiased CKA + bootstrap CI + effect sizes | Upgrade |
| Theory | None | Expressivity argument | Build from scratch |
| Hardware | Simulator only | IBM Quantum validation | New |
| Paper | None | Full 8-page paper | Write |

---

## Phase 1: Infrastructure (Weeks 1-3)

### 1.1 Scale the PQC to 16 Qubits

Why: 8 qubits = 256-dim Hilbert space. 16 qubits = 65,536-dim. This gives the PQC meaningfully more expressivity than classical baselines and makes the comparison scientifically interesting.

```python
# Current: 8 qubits, 2 layers = 16 circuit params
# Target:  16 qubits, 3 layers = 48 circuit params
# Projection: 384 → 16 = 6,144 params
```

> [!WARNING]
> **Simulation cost:** 16-qubit simulation is ~256× slower than 8-qubit per sample. Training on 5000 sentences × 50 epochs will take hours. We'll need to use `default.qubit` with `diff_method="backprop"` (fastest for simulators) and consider batched evaluation.

**Files to create/modify:**
- `src/models/pqc.py` — parameterized PQC model (configurable qubits/layers)
- `src/models/mlp.py` — parameter-matched MLP
- `src/models/baselines.py` — all classical baselines

---

### 1.2 Replace Fingerprints with Graph Kernel

Why: Our count-based fingerprints give cosine similarity >0.92 for all pairs — they can't discriminate constructions. The Weisfeiler-Leman (WL) graph kernel captures topological structure and produces a proper kernel matrix directly usable for CKA.

**How it works:**
1. Convert each DisCoCat diagram to a directed graph (boxes = nodes, wires = edges)
2. Run WL subtree kernel (iterative neighborhood hashing)
3. Compute kernel matrix K[i,j] = WL_kernel(graph_i, graph_j)
4. Feed K directly into CKA (no need for fixed-size vectors)

**Implementation:** Use `grakel` library (pip install grakel).

**Files to create:**
- `src/fingerprint/graph_kernel.py` — WL kernel extraction from diagrams
- `src/fingerprint/structural.py` — keep count-based as fallback/comparison

---

### 1.3 Build Three Datasets

#### Dataset A: Controlled Templates (5000+ sentences)
- Expand Lesson 6 to 5 constructions × 1000 sentences each
- Add **ditransitive** construction: "the doctor gave the nurse the book"
- Increase vocabulary pools to 50 nouns, 40 verbs, 30 adjectives

#### Dataset B: BLiMP Subsets (4000+ sentences)
- Download from https://github.com/alexwarstadt/blimp
- Select 4 sub-tasks: `anaphor_agreement`, `filler_gap`, `relative_clause`, `subject_verb_agreement`
- Use grammatical sentences only; extract fingerprints

#### Dataset C: MRPC (5800 pairs)
- Load via `datasets.load_dataset("glue", "mrpc")`
- Filter to sentences BobcatParser can parse (expect ~70-80% success)
- Real-world validation of the pipeline

**Files to create:**
- `src/data/templates.py` — template generator (expanded)
- `src/data/blimp.py` — BLiMP loader and adapter
- `src/data/mrpc.py` — MRPC loader with parse filtering
- `src/data/dataset.py` — unified dataset interface

---

## Phase 2: Baselines (Weeks 4-6)

### The 5-Model Comparison

Every experiment must run on ALL 5 models:

| Model | Architecture | Params | Why |
|:------|:-------------|:-------|:----|
| **PQC** | RY + CNOT, 16 qubits, 3 layers | ~48 | Our model |
| **MLP** | Linear bottleneck, tanh | ~48 | Direct classical analogue |
| **RKS** | Random Kitchen Sinks (random Fourier features) | ~48 | Classical kernel baseline — tests if random features suffice |
| **ATT** | Single-head attention (Q, K, V projection) | ~48 | Tests if attention mechanism achieves the same |
| **MPS** | Matrix Product State (bond dim 4) | ~48 | Classical tensor network — the most important baseline |

> [!IMPORTANT]
> **MPS is the critical baseline.** Matrix Product States are the classical limit of quantum circuits. If MPS matches PQC, the quantum entanglement isn't helping. If PQC beats MPS, there's something genuinely quantum happening. Reviewers at EMNLP/ICLR will look for this comparison specifically.

**Implementation:** Use `tensornetwork` or `quimb` for MPS.

**Files to create:**
- `src/models/rks.py` — Random Kitchen Sinks
- `src/models/attention.py` — Single attention head
- `src/models/mps.py` — Matrix Product State classifier

---

## Phase 3: Full Experiments (Weeks 7-10)

### Experiment 1: CKA Probing (Main Result)

- All 5 models × 3 datasets × 20 seeds
- Debiased CKA (not naive CKA) — use the Dávari et al. (2023) estimator
- Permutation tests (1000 shuffles) for each model-dataset combination
- Bootstrap 95% confidence intervals
- Report: Table 1 (main results), Figure 1 (bar chart with CIs)

### Experiment 2: Mechanistic Comparison

- CKA between every pair of models' representations
- t-SNE/UMAP visualization of representations colored by construction type
- Report: Figure 2 (CKA heatmap), Figure 3 (t-SNE)

### Experiment 3: Cross-Dataset Transfer

- Train on Dataset A (templates), evaluate CKA on Dataset B (BLiMP) and C (MRPC)
- Tests: does syntactic alignment generalize beyond templates?
- Report: Table 2 (transfer matrix)

### Experiment 4: Gradient Saliency

- Integrated Gradients on Dataset A sentences
- Correlate with POS tags using Spearman rank correlation (not just averages)
- Compare PQC vs all 4 baselines
- Report: Figure 4 (saliency heatmaps), Table 3 (POS correlations)

### Experiment 5: Ablation Studies (NEW)

- **Entanglement topology:** Linear chain vs all-to-all vs ring CNOT patterns
- **Qubit scaling:** 4, 8, 12, 16, 20 qubits — does CKA(PQC, Syntax) increase with qubits?
- **Layer depth:** 1, 2, 3, 4 layers — diminishing returns?
- **No entanglement:** Remove all CNOTs — does CKA drop? (tests if entanglement matters)
- Report: Figure 5 (scaling curves), Table 4 (ablation)

> [!IMPORTANT]
> **The ablation study is what separates a workshop paper from a top-tier paper.** It shows you understand which components matter and why. The "no entanglement" ablation is especially important — if removing CNOTs doesn't change CKA, then entanglement isn't the mechanism, and a reviewer will catch this.

**Files to create:**
- `src/experiments/cka_probing.py`
- `src/experiments/mechanistic_comparison.py`
- `src/experiments/transfer.py`
- `src/experiments/saliency.py`
- `src/experiments/ablation.py`
- `scripts/run_all.py` — orchestration script

---

## Phase 4: Theory + Hardware (Weeks 11-14)

### 4.1 Theoretical Contribution

We need to answer: **Why should PQCs encode syntax differently from classical models?**

Proposed argument (sketch):
1. DisCoCat diagrams are morphisms in a **compact closed category**
2. PQCs implement transformations in **unitary group U(2ⁿ)**, which has natural tensor product structure
3. The tensor product structure of quantum states mirrors the monoidal structure of DisCoCat
4. Hypothesis: PQCs have an **inductive bias** toward compositional structure because their computational substrate (tensor products) matches the mathematical structure of grammar

This doesn't need to be a full theorem — a well-argued conjecture with supporting empirical evidence is sufficient for EMNLP/ICLR.

**Files to create:**
- Paper Section 2.4: "Structural Alignment Between PQCs and Compositional Semantics"

### 4.2 IBM Quantum Validation

- Use IBM Quantum free tier (127-qubit Eagle processors)
- Run inference (not training) on 100 sentences with the trained PQC
- Compare noiseless simulator CKA vs noisy hardware CKA
- Show graceful degradation (CKA drops but stays above MLP)
- This is a "nice to have" that puts us above 90% of QML papers

**Tools:** `qiskit`, `qiskit-ibm-runtime`

---

## Phase 5: Writing (Weeks 15-20)

### Paper Structure (8 pages + references)

```
1. Introduction (1 page)
   - Hybrid QNLP models work, but why?
   - We introduce mechanistic interpretability for quantum NLP
   - Three contributions: (1) syntax fingerprints, (2) CKA probing,
     (3) first PQC vs classical mechanistic comparison

2. Related Work (0.75 pages)
   - Classical probing (BERTology)
   - QNLP (lambeq, DisCoCat)
   - QML interpretability (very thin — emphasize the gap)

3. Methodology (1.5 pages)
   - 3.1 Hybrid pipeline architecture
   - 3.2 Graph kernel syntax fingerprints
   - 3.3 Debiased CKA
   - 3.4 5-model comparison protocol
   - 3.5 Integrated Gradients for PQCs

4. Experimental Setup (1 page)
   - 4.1 Datasets (A, B, C)
   - 4.2 Model configurations (table)
   - 4.3 Statistical methodology

5. Results (2.5 pages)
   - 5.1 Main CKA results (Table 1, Figure 1)
   - 5.2 Mechanistic comparison (Figure 2-3)
   - 5.3 Cross-dataset transfer (Table 2)
   - 5.4 Saliency analysis (Figure 4, Table 3)
   - 5.5 Ablation (Figure 5, Table 4)

6. Discussion (0.75 pages)
   - Why PQCs encode syntax (theoretical argument)
   - Limitations (simulator, scale, fingerprint approximation)
   - Implications for QNLP architecture design

7. Conclusion (0.5 pages)
```

### Writing Process
- Week 15-16: First draft (all sections)
- Week 17: Internal review (you and I iterate)
- Week 18: Add missing experiments based on review
- Week 19: Polish, format, references
- Week 20: Final proofread, submit

---

## Project Structure (Refactored)

```
Quantum/
├── src/
│   ├── models/
│   │   ├── pqc.py              # Configurable PQC
│   │   ├── mlp.py              # MLP baseline
│   │   ├── rks.py              # Random Kitchen Sinks
│   │   ├── attention.py        # Single attention head
│   │   └── mps.py              # Matrix Product State
│   ├── data/
│   │   ├── templates.py        # Template generator
│   │   ├── blimp.py            # BLiMP loader
│   │   ├── mrpc.py             # MRPC loader
│   │   └── dataset.py          # Unified interface
│   ├── fingerprint/
│   │   ├── graph_kernel.py     # WL kernel
│   │   └── structural.py       # Count-based (legacy)
│   ├── analysis/
│   │   ├── cka.py              # Debiased CKA
│   │   ├── saliency.py         # Integrated Gradients
│   │   └── stats.py            # Bootstrap, permutation tests
│   └── experiments/
│       ├── cka_probing.py
│       ├── mechanistic.py
│       ├── transfer.py
│       ├── saliency.py
│       └── ablation.py
├── scripts/
│   ├── run_all.py              # Full experiment runner
│   └── generate_tables.py      # LaTeX table generation
├── notebooks/                   # Lessons (keep for reference)
├── data/                        # Generated datasets
├── results/                     # All outputs
├── paper/                       # LaTeX source
└── requirements.txt
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| PQC shows NO syntactic advantage over MLP at 16 qubits | High | Paper is framed as mechanistic study — negative result is still publishable if well-analyzed |
| BobcatParser fails on >30% of MRPC | Medium | Report parse rate, filter, acknowledge as limitation |
| 16-qubit simulation too slow | Medium | Use `lightning.qubit` (C++ backend), reduce to 12 qubits if needed |
| MPS matches PQC exactly | High | Focus analysis on WHY they converge — this becomes the finding |
| No clear theoretical argument | Medium | Keep it empirical, add theory as "future work" |

---

## Timeline Summary

| Phase | Weeks | What | Deliverable |
|:------|:------|:-----|:------------|
| **1. Infrastructure** | 1-3 | 16-qubit PQC, graph kernel, 3 datasets | `src/` modules |
| **2. Baselines** | 4-6 | 4 classical baselines | 5-model comparison ready |
| **3. Experiments** | 7-10 | All 5 experiments × 20 seeds | Tables + figures |
| **4. Theory + HW** | 11-14 | Theoretical argument, IBM Quantum | Section 2.4, hardware results |
| **5. Writing** | 15-20 | Full paper | Submission-ready PDF |

> [!IMPORTANT]
> **This is aggressive but achievable.** The foundation from Lessons 1-8 is solid. What changes is the SCALE of everything — more qubits, more data, more baselines, more seeds, more rigor. The core idea and methodology stay the same.

---

## Immediate Next Steps

1. **Restructure the repo** — move from `notebooks/` lessons to `src/` modules
2. **Build the configurable PQC** — parameterize qubit count, layer count, topology
3. **Implement the WL graph kernel** — the single biggest quality upgrade
4. **Start Dataset B (BLiMP)** — download and integrate

I'll start building Phase 1 once you approve this plan.
