"""Gauge-anchored genetic-interaction estimator.

Implements the estimator and identifiability objects from the paper
"A second measurement channel recovers discarded genetic interactions".

Model
-----
n single perturbations with true effects a_i in R^p (p genes), observed with
noise as A = a + eta (single channel). For each of D observed doubles (i, j),
Y_ij = a_i + a_j + G_ij + eps (double channel), with G the true interaction.
Stacking the doubles, the additive prediction is B @ A with B in {0,1}^{D x n}
carrying ones in columns i, j of row (i, j).

Naive interaction estimator:      G_naive = Y - B @ A
Gauge-anchored estimator:         A_hat_lambda = argmin_X ||X - A||^2 + lambda ||B X - Y||^2
                                  (I + lambda B^T B) A_hat = A + lambda B^T Y
                                  G_anc = Y - B @ A_hat_lambda

Theorem references in docstrings point to the paper.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def design_matrix(pairs, n):
    """Build the bipartite design B in {0,1}^{D x n}.

    Parameters
    ----------
    pairs : list of (i, j) integer index pairs (the measured doubles).
    n : number of single perturbations.

    Returns
    -------
    B : scipy.sparse.csr_matrix of shape (D, n).
    """
    pairs = list(pairs)
    D = len(pairs)
    rows, cols, data = [], [], []
    for r, (i, j) in enumerate(pairs):
        rows += [r, r]
        cols += [i, j]
        data += [1.0, 1.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(D, n))


def normal_operator(B):
    """Return S = B^T B, the signless graph Laplacian of the co-occurrence graph
    (Proposition 1). Sparse, symmetric positive semidefinite."""
    return (B.T @ B).tocsr()


def anchored_effects(B, A, Y, lam):
    """Solve (I + lambda S) A_hat = A + lambda B^T Y  for the denoised effects.

    Uses a matrix-free conjugate-gradient solve (Proposition 1); never forms an
    inverse. `lam` may be a scalar (isotropic / Theorem 4) or a length-p array of
    per-gene strengths (heteroscedastic / Theorem 6).

    Returns A_hat of shape (n, p).
    """
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = A.shape
    S = normal_operator(B)
    BtY = B.T @ Y  # (n, p)

    lam = np.asarray(lam, dtype=float)
    if lam.ndim == 0:
        rhs = A + float(lam) * BtY
        return _cg_solve(S, float(lam), rhs)

    # per-gene strengths: solve each column with its own lambda
    A_hat = np.empty_like(A)
    for g in range(p):
        rhs_g = A[:, g] + lam[g] * BtY[:, g]
        A_hat[:, g] = _cg_solve(S, float(lam[g]), rhs_g[:, None]).ravel()
    return A_hat


def _cg_solve(S, lam, rhs):
    """Solve (I + lam S) X = rhs column-by-column via CG (matrix-free)."""
    n = S.shape[0]
    rhs = np.atleast_2d(rhs)
    if rhs.shape[0] != n:
        rhs = rhs.T
    identity = sp.identity(n, format="csr")
    M = (identity + lam * S).tocsr()
    out = np.empty_like(rhs, dtype=float)
    for c in range(rhs.shape[1]):
        x, info = spla.cg(M, rhs[:, c], rtol=1e-10, maxiter=10000)
        out[:, c] = x
    return out


def anchored_interaction(B, A, Y, lam):
    """Return the gauge-anchored interaction estimate G_anc = Y - B A_hat_lambda."""
    A_hat = anchored_effects(B, A, Y, lam)
    return np.asarray(Y, dtype=float) - (B @ A_hat)


def naive_interaction(B, A, Y):
    """Return the naive residual interaction G_naive = Y - B A (the lambda=0 case)."""
    return np.asarray(Y, dtype=float) - (B @ np.asarray(A, dtype=float))


def identifiable_projector(B):
    """Orthogonal projector P onto the identifiable subspace I = ker(B^T)
    (Definition 1). Returns a dense (D, D) array; for large D prefer applying
    `apply_projector` instead of materialising P."""
    B = B.toarray() if sp.issparse(B) else np.asarray(B)
    D = B.shape[0]
    S = B.T @ B
    Splus = np.linalg.pinv(S)
    return np.eye(D) - B @ Splus @ B.T


def apply_projector(B, R):
    """Apply P = I - B (B^T B)^+ B^T to residual matrix R of shape (D, p) without
    forming P. Returns P @ R (the identifiable component of R)."""
    R = np.asarray(R, dtype=float)
    S = normal_operator(B)
    BtR = B.T @ R
    # (B^T B)^+ B^T R  via least squares on the normal equations
    x = np.linalg.lstsq(S.toarray(), BtR, rcond=None)[0]
    return R - (B @ x)


def isotropic_lambda(sigma_s2, sigma_d2):
    """Isotropic optimal anchor strength lambda* = sigma_s^2 / sigma_d^2 (Theorem 4)."""
    return float(sigma_s2) / float(sigma_d2)


def heteroscedastic_lambda(sigma_s2_g, sigma_d2_g, clip=(1e-4, 1e4)):
    """Per-gene optimal anchor lambda*_g = sigma_s,g^2 / sigma_d,g^2 (Theorem 6),
    clipped to a bounded range for numerical stability."""
    lam = np.asarray(sigma_s2_g, dtype=float) / np.asarray(sigma_d2_g, dtype=float)
    return np.clip(lam, clip[0], clip[1])
