"""Frozen Configuration & Pre-Registered Parameters.

Course: CS 328 — Introduction to Data Science
Project: Graph-Based Retrieval vs Vector Search on Multi-Hop Questions

All constants in this file are pre-registered and FROZEN prior to evaluating
on the final held-out evaluation split (N=200). Any tuning or hyperparameter
exploration must be conducted strictly on the independent tuning split (N=100).
"""

from pathlib import Path
from typing import Final, Tuple

# ==============================================================================
# 1. Project Directory Structure
# ==============================================================================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATASET_DIR: Final[Path] = DATA_DIR / "2wikimultihopqa"
DEV_DATA_FILE: Final[Path] = DATASET_DIR / "dev.json"

SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"
CACHE_DIR: Final[Path] = PROJECT_ROOT / "cache"

# Ensure runtime directories exist
for directory in (DATASET_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Random Seeds & Reproducibility (Lectures 3-4)
# ==============================================================================
SEED_EVAL: Final[int] = 42          # Pre-registered seed for final evaluation split
SEED_TUNE: Final[int] = 101         # Independent seed for hyperparameter tuning split
SEED_LSH: Final[int] = 42           # Deterministic seed for LSH random hyperplanes
SEED_BOOTSTRAP: Final[int] = 42    # Seed for resampling in statistical tests

# ==============================================================================
# 3. Sample Sizes & Stratification (Lecture 5: CLT & Sample Size Recipe)
# Formula: n >= (z_{alpha/2}^2 * sigma^2) / epsilon^2
# For delta in [-1, 1], max variance sigma^2 <= 0.25. At 95% conf (z=1.96), eps=0.07:
# n >= (1.96^2 * 0.25) / (0.07^2) = 0.9604 / 0.0049 = 196 => N = 200
# ==============================================================================
N_EVAL: Final[int] = 200            # Final evaluation split size
N_TUNE: Final[int] = 100            # Independent tuning split size

QUESTION_TYPES: Final[Tuple[str, ...]] = (
    "compositional",
    "inference",
    "comparison",
    "bridge_comparison",
)

# ==============================================================================
# 4. Retrieval Parameters & Pre-Registered Decision Metrics
# ==============================================================================
K_PRIMARY: Final[int] = 5           # Primary decision metric: recall@5
K_SECONDARY: Final[int] = 2         # Secondary metric: recall@2
K_VALUES: Final[Tuple[int, ...]] = (2, 5)

# ==============================================================================
# 5. Personalized PageRank Parameters (PageRank Lecture & Contraction Lemma)
# ==============================================================================
ALPHA_PRIMARY: Final[float] = 0.5       # Pre-registered primary damping factor
ALPHA_SENSITIVITY: Final[float] = 0.85  # Sensitivity check damping factor
ALPHA_NO_WALK: Final[float] = 0.0       # Ablation / no-walk control (mass stays on seeds)

PPR_TOL: Final[float] = 1e-12           # Strict power iteration convergence tolerance
PPR_MAX_ITER: Final[int] = 200          # Maximum power iteration rounds
PASSAGE_SCORING_FORMULATION: Final[str] = "formulation_a"  # Sum(PPR(e) / df(e))

# ==============================================================================
# 6. Entity Linking & Random Hyperplane LSH (Lectures 6-7 & 11)
# S-curve inflection threshold: p* = (1/b)^(1/r) = (1/16)^(1/8) = 2^(-0.5) = 0.7071
# Threshold angle theta* = (1 - p*)*pi = 52.72 deg => cos(theta*) = 0.606
# Candidate probability at cos=0.80 is 93.8%
# ==============================================================================
LSH_L: Final[int] = 128                 # Total number of random hyperplanes (ell)
LSH_R: Final[int] = 8                   # Number of rows per band (r)
LSH_B: Final[int] = 16                  # Number of bands (b = ell / r = 16)
LSH_INFLECTION_P: Final[float] = 0.7071 # Inflection collision probability p*

TAU_SYNONYM: Final[float] = 0.80        # Exact cosine threshold for intra-graph synonyms
QUERY_MATCH_THRESHOLD: Final[float] = 0.80 # Candidate match threshold (top-3 matches)
QUERY_MATCH_FLOOR: Final[float] = 0.40  # Minimum cosine floor for top-1 entity seed
QUERY_FALLBACK_FLOOR: Final[float] = 0.35  # Full-question embedding fallback floor
QUERY_SEED_TEMPERATURE: Final[float] = 0.05 # Softmax temperature sharpening (T=0.05)
MAX_CANDIDATES_PER_ENTITY: Final[int] = 3   # Max candidate matches per query entity

# ==============================================================================
# 7. Embedding & LLM Specifications
# ==============================================================================
EMBEDDING_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: Final[int] = 384
MAX_SEQ_LENGTH: Final[int] = 256        # MiniLM context length limit

LLM_MODEL_NAME: Final[str] = "gpt-4o-mini"
LLM_CACHE_VERSION: Final[str] = "v1"

# ==============================================================================
# 8. Statistical Testing Framework (Lectures 3-4, 5, & Inference)
# ==============================================================================
BOOTSTRAP_ROUNDS: Final[int] = 10000        # B = 10,000 bootstrap iterations
PERMUTATION_ROUNDS: Final[int] = 100000     # 100,000 sign-flip permutations
SIGNIFICANCE_ALPHA: Final[float] = 0.05     # Pre-registered significance level alpha
CONFIDENCE_LEVEL: Final[float] = 0.95       # Nominal confidence level for CIs
