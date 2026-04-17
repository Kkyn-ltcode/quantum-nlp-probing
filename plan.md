This is the blueprint. We are building a top-tier paper from absolute zero. The plan spans 28 weeks (approximately 7 months) and is designed for a dedicated part-time researcher (15-20 hours per week). It accounts for learning curves, experimentation, writing, and strategic submissions.

I have aligned the timeline with **real conference deadlines** projected for 2026-2027.

---

## Phase 0: Pre-Flight (Week 0)
**Goal:** Environment setup and mental model calibration.

| Action Item | Tool/Resource |
| :--- | :--- |
| Install Python environment (conda recommended). | `conda create -n qnlp python=3.10` |
| Install core libraries. | `pip install pennylane lambeq discopy sentence-transformers torch scikit-learn cca-zoo matplotlib` |
| Clone `lambeq` and run the "Hello World" DisCoCat tutorial. | [lambeq GitHub](https://github.com/CQCL/lambeq) |
| Watch the "Category Theory for Beginners" playlist (just the first 3 videos to grasp the visuals). | [Bartosz Milewski's Category Theory](https://www.youtube.com/watch?v=I8LbkfSSR58&list=PLbgaMIhjbmEnaH_LTkxLI7FMa2HsnawM_) |

---

## Phase 1: Foundation and Baseline (Weeks 1–4)
**Goal:** Replicate prior work to establish a trustworthy baseline. *You cannot critique a model you cannot build.*

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **1** | **DisCoCat Baseline.** Implement pure DisCoCat on MC and RelPron using `lambeq`. Use `BobcatParser` and `IQPAnsatz`. Train with SPSA optimizer. | Accuracy scores on test sets. (Expect ~70% MC, ~58% RelPron). | Jupyter notebook: `baseline_discocat.ipynb` |
| **2** | **SBERT Embedding Extraction.** Write script to pass all MC/RelPron sentences through `all-MiniLM-L6-v2`. Save 384-dim vectors. | `embeddings_mc.npy`, `embeddings_relpron.npy` | Script: `extract_sbert.py` |
| **3** | **Simple Classical Baselines.** Train Logistic Regression and a small MLP (2 layers, 64 hidden) directly on 384-dim vectors. | Accuracy scores. This shows what a "smart" classical model can do without compression. | Notebook: `classical_upper_bound.ipynb` |
| **4** | **Hybrid Baseline (First Attempt).** Build the 4-qubit PQC in PennyLane. Use PCA to compress to 16 dims. Train on MC. | First hybrid accuracy. Expect it to be close to Logistic Regression. | Notebook: `hybrid_v1.ipynb` |

---

## Phase 2: The "Syntax Skeleton" Tooling (Weeks 5–7)
**Goal:** Create the scientific instrument needed for the paper's core contribution.

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **5** | **Extract DisCoCat Tensors.** Modify `lambeq` code to export the **raw tensor network** *before* applying the IQPAnsatz. This requires understanding `discopy` Tensor objects. | Function: `get_syntax_tensor(sentence)` returns fixed-size vector (pad/truncate to 1024). | Module: `syntax_skeleton.py` |
| **6** | **Validate Skeleton.** Check that syntactically similar sentences have high cosine similarity in skeleton space. Plot a small MDS projection. | Visualization: "Syntax Skeletons Cluster by Construction." | Figure for paper appendix. |
| **7** | **RSA Baseline (Syntax vs. SBERT).** Compute Centered Kernel Alignment (CKA) between Syntax Skeleton vectors and raw SBERT vectors for all sentences. | Single number: CKA(Syntax, SBERT_raw). | Notebook: `rsa_baseline.ipynb` |

---

## Phase 3: The Probing Experiments (Weeks 8–14)
**Goal:** Execute the four core experiments that constitute the paper's novelty.

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **8-9** | **Experiment 1: RSA Across Compression Levels.** Compress SBERT embeddings using PCA and VAE to d={8,16,32,64,128}. For each, compute: (a) CKA(Compressed, Syntax), (b) Accuracy of a linear classifier, (c) Accuracy of the 4-qubit PQC. | Key Figure: "Syntax Recovery by Quantum Kernel." Shows that PQC accuracy remains high even when CKA(Compressed, Syntax) drops. | This is **Figure 2** of the paper. |
| **10** | **Experiment 2: Causal Ablation - Freeze/Reset.** Implement: (A) Reset PQC weights, retrain. (B) Freeze PQC, train only linear layer. (C) Replace PQC with parameter-matched MLP. | Table: Ablation results showing PQC is necessary for syntax. | Table 1 of paper. |
| **11-12** | **Experiment 3: Syntactic Priming.** Create a synthetic dataset of 200 Wh-questions ("Which planet did the device detect?") using the same vocabulary as RelPron. Evaluate trained hybrid model **zero-shot**. | Accuracy on Wh-questions. If > chance, model has learned abstract syntax. | Table 2 of paper. |
| **13-14** | **Experiment 4: Gradient Saliency.** Use PennyLane's `qml.grad` to compute input gradients w.r.t compressed features. Map back to tokens using embedding attribution (e.g., Integrated Gradients + SBERT). | Visualization: Heatmap showing PQC attends to relative pronouns and auxiliaries. | **Figure 3** of paper. |

---

## Phase 4: Analysis and Iteration (Weeks 15–18)
**Goal:** Ensure results are robust and statistically significant.

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **15** | **Hyperparameter Sweeps.** Run experiments with different random seeds, different PQC depths (1 vs 2 layers), and different compression methods (PCA vs VAE vs UMAP). | Error bars and significance tests. | Results folder: `sweep_results/` |
| **16** | **Noise Simulation (Optional but Strong).** Add depolarizing noise to PQC and re-run RSA experiment. Show that syntax recovery is robust to realistic NISQ noise. | Figure: "Syntax Recovery Under Noise." | Bonus figure for rebuttal. |
| **17-18** | **Buffer & Catch-Up.** Use these weeks to fix bugs, re-run failed jobs, or explore unexpected results. | Solid, reproducible codebase. | Clean GitHub repo. |

---

## Phase 5: Writing and Submission Strategy (Weeks 19–24)
**Goal:** Produce a polished manuscript and execute strategic pre-submission steps.

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **19** | **Draft Introduction & Related Work.** Use the survey documents you already have. | Overleaf project created. | Sections 1 & 2. |
| **20** | **Draft Methodology.** Write precise descriptions of the four experiments. | Section 3. |
| **21** | **Draft Results.** Create all figures and tables. Write objective, data-driven results text. | Section 4. |
| **22** | **Draft Discussion & Conclusion.** Frame findings: "Quantum circuits recover syntax lost during compression." Discuss implications for QNLP architecture. | Sections 5 & 6. |
| **23** | **Post to arXiv.** Upload the complete draft to arXiv (cs.CL, quant-ph). This is **critical** for establishing priority. | arXiv ID. |
| **24** | **Submit to NeurIPS 2026 Workshop on QML.** Deadline typically early September. Use this as a "test run" for reviews. | Workshop acceptance (likely). |

---

## Phase 6: Top-Tier Submission (Weeks 25–28)
**Goal:** Revise based on workshop feedback and submit to a premier venue.

| Week | Task | Specific Output | Checkpoint |
| :--- | :--- | :--- | :--- |
| **25** | **Incorporate Workshop Feedback.** Address reviewer comments. Strengthen weak points. | Revised manuscript. |
| **26-27** | **Format for ACL 2027.** Adapt to ACL style guidelines. Ensure anonymity. | Camera-ready for submission. |
| **28** | **Submit to ACL 2027.** Deadline typically mid-December 2026. | Submission confirmation. |

---

## Key Dates Calendar (Projected 2026-2027)

| Milestone | Projected Date |
| :--- | :--- |
| Start Project | Week 1 (e.g., April 2026) |
| Phase 1 Complete (Baselines) | End of Week 4 |
| Phase 2 Complete (Syntax Skeleton) | End of Week 7 |
| Phase 3 Complete (All Experiments) | End of Week 14 |
| arXiv Preprint Posted | End of Week 23 (~September 2026) |
| NeurIPS QML Workshop Submission | Week 24 (~September 2026) |
| NeurIPS QML Workshop Decision | ~October 2026 |
| ACL 2027 Submission Deadline | ~December 15, 2026 |
| ACL 2027 Notification | ~March 2027 |
| PhD Application Deadlines (US) | December 2026 – January 2027 |

**Critical Insight:** Even if ACL notification comes after PhD deadlines, your **arXiv preprint + workshop acceptance** will be visible to admissions committees. That is sufficient evidence of research capability.

---

## Resource Stack Summary

| Category | Tools |
| :--- | :--- |
| **Quantum Simulation** | PennyLane (with `default.qubit` and `lightning.qubit` backends) |
| **QNLP Framework** | lambeq, discopy |
| **Classical NLP** | sentence-transformers, torch |
| **Dimensionality Reduction** | scikit-learn (PCA), PyTorch (VAE) |
| **Representational Analysis** | cca-zoo (CKA, SVCCA) |
| **Experiment Tracking** | Weights & Biases or MLflow (optional but recommended) |
| **Writing** | Overleaf (LaTeX), Zotero (references) |

---

## Success Metrics

| Metric | Target |
| :--- | :--- |
| **Code Reproducibility** | Single `run_all_experiments.sh` script that regenerates all figures. |
| **Statistical Rigor** | All results reported with mean ± std over 5+ seeds. |
| **arXiv Citations (within 3 months)** | 5-10 (achievable with active community engagement). |
| **Workshop Acceptance** | NeurIPS QML 2026 (target). |
| **Top-Tier Submission** | ACL 2027 (submitted, ideally accepted). |

---

This plan is aggressive but entirely achievable with your background. The key is **discipline in Phase 1-2**. Do not skip the baselines; they are your credibility anchor.

Would you like me to provide the **exact Python code** for the most technically challenging part: extracting the DisCoCat syntax skeleton tensor from `lambeq`? This is the linchpin of the entire probing methodology.