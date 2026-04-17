# Mechanistic Interpretability of Hybrid Quantum-Classical Language Models

> *"How Does the Quantum Circuit Solve It?"*

A research project investigating whether parameterized quantum circuits in hybrid QNLP pipelines employ qualitatively different representational strategies than classical non-linearities for syntactic tasks.

## Project Structure

```
Quantum/
├── src/                    # Source code
│   ├── config.py           # All hyperparameters and paths
│   ├── data/               # Dataset loading, generation, preprocessing
│   ├── models/             # PQC, MLP, hybrid pipeline
│   ├── probing/            # RSA/CKA, ablation, transfer, saliency
│   ├── syntax/             # Diagram structural fingerprint extraction
│   └── utils/              # Seeding, logging, device helpers
├── notebooks/              # Exploration and lesson exercises
├── scripts/                # Entry points (train, evaluate, run_all)
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Embeddings, fingerprints, splits
├── results/
│   ├── figures/            # Publication-quality plots
│   ├── tables/             # CSV result tables
│   └── checkpoints/        # Saved model weights
├── paper/                  # LaTeX source for the manuscript
└── tests/                  # Unit tests
```

## Setup

```bash
conda create -n qnlp python=3.10 -y
conda activate qnlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Reproducibility

All experiments use 5 fixed random seeds: `[42, 123, 456, 789, 1024]`.  
Results are reported as mean ± std.

A single entry point regenerates all results:
```bash
bash scripts/run_all_experiments.sh
```
