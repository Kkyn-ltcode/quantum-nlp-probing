"""
Project-wide configuration.
All hyperparameters and paths in one place. No magic numbers in code.
"""
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


@dataclass
class SBERTConfig:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384


@dataclass
class CompressionConfig:
    method: str = "learned_linear"  # "learned_linear" or "pca"
    output_dim: int = 16


@dataclass
class PQCConfig:
    n_qubits: int = 4
    n_layers: int = 2
    # Features per encoding layer: n_qubits
    # Total input features: n_qubits * encoding_repeats
    encoding_repeats: int = 4  # 4 qubits * 4 repeats = 16 input features
    backend: str = "default.qubit"


@dataclass
class MLPConfig:
    """MLP baseline — matched to PQC parameter count."""
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    learning_rate_pqc: float = 0.01
    learning_rate_classical: float = 0.001
    batch_size: int = 16
    max_epochs: int = 100
    patience: int = 15  # Early stopping patience
    n_seeds: int = 5
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])


@dataclass
class ExperimentConfig:
    """Master config combining all sub-configs."""
    sbert: SBERTConfig = field(default_factory=SBERTConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    pqc: PQCConfig = field(default_factory=PQCConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Ensure directories exist."""
        for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CHECKPOINTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
