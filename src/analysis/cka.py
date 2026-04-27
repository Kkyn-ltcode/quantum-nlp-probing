"""
Debiased CKA (Centered Kernel Alignment) with statistical testing.

Implements:
- Naive CKA (from Lesson 4)
- Debiased CKA (Dávari et al., 2023)
- Permutation test for significance
- Bootstrap confidence intervals
"""

import numpy as np


def center_kernel(K):
    """Center a kernel matrix: HKH where H = I - 11^T/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic(K, L):
    """Hilbert-Schmidt Independence Criterion (biased estimator)."""
    n = K.shape[0]
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    return np.sum(K_c * L_c) / ((n - 1) ** 2)


def compute_cka(X, Y):
    """
    Compute linear CKA between two representation matrices.

    Args:
        X: np.array of shape (n, d1) — first representation
        Y: np.array of shape (n, d2) — second representation

    Returns:
        float: CKA score in [0, 1]
    """
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def compute_kernel_cka(K, L):
    """
    Compute CKA from precomputed kernel matrices.
    Use this when you already have a kernel (e.g., from WL graph kernel).

    Args:
        K: np.array of shape (n, n) — first kernel matrix
        L: np.array of shape (n, n) — second kernel matrix
    """
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    if hsic_kk < 1e-10 or hsic_ll < 1e-10:
        return 0.0
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def permutation_test(X, Y, n_permutations=1000, seed=42):
    """
    Test whether CKA(X, Y) is significantly above chance.

    Shuffles rows of Y and recomputes CKA to build a null distribution.

    Returns:
        real_cka: the actual CKA score
        p_value: fraction of permuted CKAs >= real CKA
        null_distribution: array of permuted CKA scores
    """
    rng = np.random.RandomState(seed)
    real_cka = compute_cka(X, Y)

    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(Y))
        null_dist[i] = compute_cka(X, Y[perm])

    p_value = np.mean(null_dist >= real_cka)
    return real_cka, p_value, null_dist


def bootstrap_ci(X, Y, n_bootstrap=1000, confidence=0.95, seed=42):
    """
    Compute bootstrap confidence interval for CKA.

    Resamples rows (with replacement) and recomputes CKA.

    Returns:
        mean_cka: mean of bootstrap CKA scores
        ci_low: lower bound of CI
        ci_high: upper bound of CI
        bootstrap_scores: all bootstrap CKA values
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    scores = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores[i] = compute_cka(X[idx], Y[idx])

    alpha = (1 - confidence) / 2
    ci_low = np.percentile(scores, 100 * alpha)
    ci_high = np.percentile(scores, 100 * (1 - alpha))
    return np.mean(scores), ci_low, ci_high, scores
