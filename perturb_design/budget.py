"""Closed-form singles-versus-doubles budget rule (Theorem 8) and leak-free
selection utilities.

At the optimal anchor the gauge-subspace risk is, to leading order,
    R(m_s, m_d) = (sigma_s^2 / m_s) t_s + (sigma_d^2 / m_d) t_d,
with t_s = sum_{k: s_k>0} s_k^{-1} and t_d = rank(B) fixed design traces. Minimising
subject to m_s + m_d = M gives
    m_s^* / m_d^* = sqrt( sigma_s^2 t_s / (sigma_d^2 t_d) ).
"""
from __future__ import annotations

import numpy as np

from .anchored import normal_operator, anchored_effects


def design_traces(B, tol=1e-9):
    """Return (t_s, t_d) = (sum 1/s_k over positive modes, rank(B))."""
    S = normal_operator(B).toarray()
    s = np.linalg.eigvalsh(S)
    pos = s[s > tol]
    t_s = float((1.0 / pos).sum()) if pos.size else 0.0
    t_d = int(pos.size)  # rank(B) = number of positive eigenvalues of B^T B
    return t_s, t_d


def optimal_split(B, sigma_s2, sigma_d2, M):
    """Closed-form optimal budget split (Theorem 8).

    Returns a dict with the ratio r = m_s^*/m_d^*, and the continuous optima
    m_s^*, m_d^* for total budget M.
    """
    t_s, t_d = design_traces(B)
    if t_d == 0:
        raise ValueError("design has rank 0")
    r = float(np.sqrt((sigma_s2 * t_s) / (sigma_d2 * t_d)))
    m_s = M * r / (1.0 + r)
    m_d = M / (1.0 + r)
    return {"ratio": r, "m_s": m_s, "m_d": m_d, "t_s": t_s, "t_d": t_d}


def estimate_gene_noise(B, A, Y, tol=1e-9):
    """Leak-free per-gene noise proxies (Section 3.5).

    sigma_s2_g : dispersion of the singles about their per-gene mean.
    sigma_d2_g : variance of the naive residual per gene.
    Returns (sigma_s2_g, sigma_d2_g), each length p.
    """
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sigma_s2_g = A.var(axis=0) + tol
    G_naive = Y - (B @ A)
    sigma_d2_g = G_naive.var(axis=0) + tol
    return sigma_s2_g, sigma_d2_g


def select_lambda_leakfree(B, A, Y, pairs, lambdas, n_folds=5, seed=0):
    """Select a global lambda by leak-free held-out double reconstruction.

    For each fold, fit A_hat on the training doubles and all singles, predict each
    held-out double (i, j) as A_hat_i + A_hat_j, and score reconstruction error.
    Returns the lambda minimising mean held-out error, and the error curve.
    """
    from .anchored import design_matrix

    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pairs = list(pairs)
    D = len(pairs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(D)
    folds = np.array_split(order, n_folds)
    n = A.shape[0]

    curve = []
    for lam in lambdas:
        err = 0.0
        cnt = 0
        for f in folds:
            test = set(f.tolist())
            tr_pairs = [pairs[d] for d in range(D) if d not in test]
            tr_idx = [d for d in range(D) if d not in test]
            B_tr = design_matrix(tr_pairs, n)
            A_hat = anchored_effects(B_tr, A, Y[tr_idx], lam)
            for d in f:
                i, j = pairs[d]
                pred = A_hat[i] + A_hat[j]
                err += float(((Y[d] - pred) ** 2).sum())
                cnt += 1
        curve.append(err / max(cnt, 1))
    curve = np.asarray(curve)
    best = int(np.argmin(curve))
    return {"lambda": float(lambdas[best]), "curve": curve, "lambdas": np.asarray(lambdas)}
