## Project Brief: Causal Probing of Syntactic Information Flow in Hybrid Quantum-Classical Language Models

### 1. Researcher Profile
- **Background:** AI Engineer with strong experience in Computer Vision, Transformers, NLP, LLMs (PyTorch, sentence-transformers). Limited experience in quantum computing but willing to learn.
- **Goal:** Produce a top-tier conference paper (target: ACL 2027 or ICLR 2028) to strengthen a PhD application in Quantum NLP / Quantum Machine Learning.
- **Constraints:** Part-time effort (~15-20 hours/week). No access to real quantum hardware; all experiments on simulators (PennyLane).

### 2. The Core Research Question
In a hybrid QNLP pipeline (Sentence-BERT → Dimensionality Reduction → Parameterized Quantum Circuit), **does the quantum circuit actively recover syntactic structure lost during compression, or is it merely a non-linear amplifier of semantic features?**

### 3. Prior Art and Novelty Gap
- **Established:** Hybrid SBERT + PCA + PQC pipelines exist for *semantic* tasks (sentiment analysis, paraphrase detection).
- **Gap:** No prior work evaluates this pipeline on *syntactic* benchmarks (lambeq's MC and RelPron datasets). No prior work uses **causal probing** (RSA, ablation, priming, saliency) to analyze *what* the quantum circuit learns.
- **Novel Contribution:** First rigorous empirical analysis of syntactic information flow through a compressed hybrid QNLP model.

### 4. Experimental Design (Four Core Experiments)

| Experiment | Method | Expected Finding |
| :--- | :--- | :--- |
| **1. Representational Similarity Analysis (RSA)** | Compute CKA between: (a) Pure DisCoCat syntax skeleton tensors, (b) Compressed SBERT vectors at various dimensions, (c) Quantum state vectors after PQC. | CKA drops after compression but **recovers** after the quantum circuit, proving syntax reconstruction. |
| **2. Causal Ablation** | (A) Reset PQC weights, retrain. (B) Freeze PQC, train only output layer. (C) Replace PQC with parameter-matched classical MLP. | PQC is necessary; classical MLP cannot recover syntax as effectively. |
| **3. Syntactic Priming** | Train on RelPron (relative clauses). Test zero-shot on synthetic Wh-questions using same vocabulary. | Above-chance transfer indicates learning of abstract syntactic rules. |
| **4. Gradient Saliency** | Use Integrated Gradients on PQC inputs, map back to tokens via SBERT attribution. | PQC gradients concentrate on function words (relative pronouns, auxiliaries). |

### 5. Technical Stack
- **Quantum Simulation:** PennyLane (`default.qubit` or `lightning.qubit`)
- **QNLP Framework:** lambeq, discopy (for syntax skeleton extraction)
- **Classical NLP:** sentence-transformers (`all-MiniLM-L6-v2`), PyTorch
- **Dimensionality Reduction:** scikit-learn (PCA), PyTorch (VAE)
- **Representational Analysis:** cca-zoo (CKA, SVCCA)
- **Datasets:** lambeq's MC (130 sentences) and RelPron (105 sentences) + synthetic Wh-questions (to be generated)

### 6. 28-Week Execution Plan

| Phase | Weeks | Deliverables |
| :--- | :--- | :--- |
| **0: Setup** | 0 | Environment, library installation, lambeq tutorial completion. |
| **1: Baselines** | 1-4 | DisCoCat accuracy; SBERT embeddings; classical MLP baseline; first hybrid PCA+PQC model. |
| **2: Syntax Skeleton Tooling** | 5-7 | Function to extract DisCoCat tensor (pre-IQPAnsatz); validate via clustering; CKA baseline (Syntax vs. SBERT). |
| **3: Probing Experiments** | 8-14 | Execute Experiments 1-4. Collect all data. |
| **4: Analysis & Robustness** | 15-18 | Hyperparameter sweeps, statistical tests, noise simulation (optional). |
| **5: Writing & Pre-Submission** | 19-24 | Draft paper; post to arXiv; submit to NeurIPS 2026 QML Workshop. |
| **6: Top-Tier Revision & Submission** | 25-28 | Incorporate feedback; format for ACL 2027; submit by mid-December 2026. |

### 7. Key Milestones and Deadlines (Projected)
- **arXiv Preprint:** ~September 2026
- **NeurIPS QML Workshop Submission:** ~September 2026
- **ACL 2027 Submission:** ~December 15, 2026
- **PhD Application Deadlines:** December 2026 – January 2027 (US)

### 8. Immediate Next Step for the New Chat Agent
*"I need the Python code to extract the **pure syntax skeleton tensor** from a lambeq DisCoCat diagram. This tensor should represent only the grammatical wiring, without any semantic parameters or IQPAnsatz applied. It should be flattened into a fixed-size vector (e.g., 1024 dimensions) for use in Representational Similarity Analysis (CKA)."*

### 9. Success Criteria
- Reproducible codebase with a single entry point.
- All results reported with mean ± std over ≥5 random seeds.
- Clear evidence that the quantum circuit contributes uniquely to syntactic processing.

---