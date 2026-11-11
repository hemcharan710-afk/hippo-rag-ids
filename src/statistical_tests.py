"""Statistical Testing Framework and Pre-Registered Decision Protocol.

Course: CS 328 — Introduction to Data Science
Lectures:
- Lectures 3-4: Empirical distributions, expectation, paired resampling
- Lecture 5: CLT & Sample-Size Recipe (N=200)
- Statistical Inference: Paired bootstrap CIs (B=10,000), sign-flip permutation tests (100,000 runs),
  McNemar's exact binomial test, and Holm-Bonferroni FWER control.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config

logger = logging.getLogger(__name__)


def bootstrap_mean_ci(
    scores: np.ndarray,
    B: int = config.BOOTSTRAP_ROUNDS,
    alpha: float = config.SIGNIFICANCE_ALPHA,
    seed: int = config.SEED_BOOTSTRAP,
) -> Tuple[float, float]:
    """Computes 1-sample empirical bootstrap CI for individual method levels."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_indices = rng.choice(n, size=(B, n), replace=True)
    boot_means = np.mean(scores[boot_indices], axis=1)
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_diff_ci(
    scores_graph: np.ndarray,
    scores_vector: np.ndarray,
    B: int = config.BOOTSTRAP_ROUNDS,
    alpha: float = config.SIGNIFICANCE_ALPHA,
    seed: int = config.SEED_BOOTSTRAP,
) -> Tuple[float, float]:
    """Computes paired bootstrap CI for mean difference: Delta = Graph - Vector."""
    rng = np.random.default_rng(seed)
    n = len(scores_graph)
    diffs = scores_graph - scores_vector
    boot_indices = rng.choice(n, size=(B, n), replace=True)
    boot_diff_means = np.mean(diffs[boot_indices], axis=1)
    lower = float(np.percentile(boot_diff_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_diff_means, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_sign_flip_permutation_test(
    metric_graph: np.ndarray,
    metric_vector: np.ndarray,
    n_perm: int = config.PERMUTATION_ROUNDS,
    seed: int = config.SEED_BOOTSTRAP,
) -> Tuple[float, float]:
    """Paired sign-flip permutation test on the exact mean difference estimand."""
    diffs = metric_graph - metric_vector
    observed_mean = float(np.mean(diffs))

    if np.all(diffs == 0):
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    permuted_means = np.mean(signs * diffs, axis=1)
    p_value = float(np.mean(np.abs(permuted_means) >= np.abs(observed_mean)))
    return observed_mean, p_value


def paired_mcnemar_test(
    binary_graph: np.ndarray, binary_vector: np.ndarray
) -> Tuple[float, float]:
    """Two-sided McNemar exact binomial test for paired binary outcomes (e.g. all-recall, EM)."""
    # 2x2 contingency table:
    # table[0, 0]: Graph 1, Vector 1
    # table[0, 1]: Graph 1, Vector 0 (b)
    # table[1, 0]: Graph 0, Vector 1 (c)
    # table[1, 1]: Graph 0, Vector 0
    b = int(np.sum((binary_graph == 1) & (binary_vector == 0)))
    c = int(np.sum((binary_graph == 0) & (binary_vector == 1)))
    table = [[int(np.sum((binary_graph == 1) & (binary_vector == 1))), b],
             [c, int(np.sum((binary_graph == 0) & (binary_vector == 0)))]]

    result = mcnemar(table, exact=True)
    stat = float(result.statistic)
    p_val = float(result.pvalue)
    return stat, p_val


def holm_bonferroni(
    p_values: Dict[str, float], alpha: float = config.SIGNIFICANCE_ALPHA
) -> Dict[str, dict]:
    """Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER)."""
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    results = {}

    rejecting = True
    for rank, (name, p) in enumerate(sorted_items, 1):
        threshold = alpha / (m - rank + 1)
        if rejecting and p <= threshold:
            significant = True
        else:
            rejecting = False
            significant = False

        results[name] = {
            "rank": rank,
            "raw_p": p,
            "threshold": threshold,
            "significant": significant,
        }

    return results


def evaluate_preregistered_verdict(
    delta_r5: float,
    ci_lower: float,
    ci_upper: float,
    perm_p: float,
    alpha: float = config.SIGNIFICANCE_ALPHA,
) -> str:
    """Implements the pre-registered 3-way decision rule for the primary metric (recall@5).

    Decision Rule (IMPLEMENTATION_PLAN.md Section 1.3):
    - Branch 1: Delta > 0, CI lower bound > 0, and permutation p < 0.05
      -> Reject H0: Graph + PPR is statistically significantly better.
    - Branch 2: Delta < 0, CI upper bound < 0, and permutation p < 0.05
      -> Reject H0: Vector Search is statistically significantly better.
    - Branch 3: CI spans 0 or permutation p >= 0.05
      -> Fail to reject H0: No statistically significant difference.
    """
    if delta_r5 > 0 and ci_lower > 0 and perm_p < alpha:
        return (
            "BRANCH 1: REJECT H0 IN FAVOR OF H1 (GRAPH SIGNIFICANTLY SUPERIOR)\n"
            f"The observed recall@5 gain (Delta = +{delta_r5*100:.2f}%) is positive and statistically "
            f"significant at alpha={alpha} (95% Paired Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%], "
            f"Sign-Flip Permutation p = {perm_p:.5f}). Random walks on the entity graph provide "
            "superior multi-hop retrieval over standard dense vector search."
        )
    elif delta_r5 < 0 and ci_upper < 0 and perm_p < alpha:
        return (
            "BRANCH 2: REJECT H0 IN FAVOR OF VECTOR SEARCH (VECTOR SIGNIFICANTLY SUPERIOR)\n"
            f"The observed recall@5 difference (Delta = {delta_r5*100:.2f}%) is negative and statistically "
            f"significant at alpha={alpha} (95% Paired Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%], "
            f"Sign-Flip Permutation p = {perm_p:.5f}). Dense vector embeddings outperform the graph method."
        )
    else:
        return (
            "BRANCH 3: FAIL TO REJECT H0 (NO STATISTICALLY SIGNIFICANT DIFFERENCE)\n"
            f"The observed recall@5 difference (Delta = {delta_r5*100:.2f}%) is not statistically significant "
            f"at alpha={alpha} (95% Paired Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%] spans zero, "
            f"Sign-Flip Permutation p = {perm_p:.5f}). There is insufficient statistical evidence to claim "
            "a difference in retrieval recall under this experimental design."
        )

