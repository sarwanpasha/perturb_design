"""Tuning-free per-mode risk-optimal anchoring (SURE-adaptive), Theorem 7.

Diagonalise the design normal operator S = B^T B = sum_k s_k u_k u_k^T. On mode k
the anchored effect estimate interpolates the single-channel coordinate
    abar_k = u_k^T A                 (variance sigma_s^2)
and the doubles-implied coordinate
    dbar_k = u_k^T B^T Y / s_k       (variance sigma_d^2 / s_k)
through a convex weight t_k. The risk-optimal weight is the inverse-variance
combination
    t_k^* = sigma_s^2 / (sigma_s^2 + sigma_d^2 / s_k),
which is mode-dependent and hence outside the single-lambda family. The two noise
scales are identified from data alone by the channel-gap moment identity
    E[(dbar_k - abar_k)^2] = sigma_s^2 + sigma_d^2 / s_k,
so an OLS of the observed squared gap on 1/s_k yields (sigma_s^2, sigma_d^2) with
no cross-validation. The estimator also reports an unbiased Stein risk estimate.

Estimation regime
-----------------
The intercept (sigma_s^2) is well-determined. The slope (sigma_d^2) is identified
from the 1/s_k dependence, so it requires (i) spectral spread in the design modes
{s_k} -- present whenever perturbation degrees are heterogeneous, as in real screens
with low-degree perturbations -- and (ii) that the per-mode interaction signal is
small relative to the double noise, since a large planted interaction projected
through B contributes to the gap and biases the slope upward. Consistency is
O(K^{-1/2}) in the number K of informative modes (Theorem 7). On dense designs with
uniformly large s_k, or when the interaction dominates the double noise, prefer the
naive-residual proxy for sigma_d^2 (see budget.estimate_gene_noise, Section 3.5).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .anchored import normal_operator


def _eigh_normal(B, tol=1e-9):
    """Return (eigenvalues s_k, eigenvectors U) of S = B^T B, S = U diag(s) U^T."""
    S = normal_operator(B).toarray()
    s, U = np.linalg.eigh(S)
    s = np.clip(s, 0.0, None)
    return s, U


def estimate_noise_scales(B, A, Y, tol=1e-9):
    """Identify (sigma_s^2, sigma_d^2) from the inter-channel gap moment identity
    (Theorem 7). Returns a dict with the two variances and the OLS diagnostics."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    s, U = _eigh_normal(B)
    pos = s > tol
    Ut = U.T
    abar = Ut @ A                      # (n, p)
    BtY = B.T @ Y                      # (n, p)
    dbar = (Ut @ BtY)                  # (n, p)
    dbar[pos] = dbar[pos] / s[pos, None]
    gap2 = ((dbar - abar) ** 2).mean(axis=1)   # per-mode mean squared gap (n,)

    inv_s = np.zeros_like(s)
    inv_s[pos] = 1.0 / s[pos]
    # OLS: gap2 ~ sigma_s^2 * 1 + sigma_d^2 * (1/s_k), on informative modes
    X = np.column_stack([np.ones(pos.sum()), inv_s[pos]])
    coef, *_ = np.linalg.lstsq(X, gap2[pos], rcond=None)
    sigma_s2 = max(coef[0], 1e-12)
    sigma_d2 = max(coef[1], 1e-12)
    return {"sigma_s2": float(sigma_s2), "sigma_d2": float(sigma_d2),
            "s": s, "U": U, "abar": abar, "dbar": dbar, "pos": pos}


def sure_effects(B, A, Y, tol=1e-9):
    """Return the tuning-free SURE-adaptive effect estimate A_hat and its unbiased
    Stein risk estimate. No cross-validation is used."""
    est = estimate_noise_scales(B, A, Y, tol=tol)
    s, U, abar, dbar, pos = est["s"], est["U"], est["abar"], est["dbar"], est["pos"]
    sigma_s2, sigma_d2 = est["sigma_s2"], est["sigma_d2"]

    t = np.zeros_like(s)
    # t_k^* = sigma_s^2 / (sigma_s^2 + sigma_d^2 / s_k) on informative modes; 0 otherwise
    t[pos] = sigma_s2 / (sigma_s2 + sigma_d2 / s[pos])

    coords = (1.0 - t)[:, None] * abar + t[:, None] * dbar   # (n, p) per-mode combo
    A_hat = U @ coords                                       # back to effect space

    # Unbiased Stein risk estimate: sum_k [(1-t_k)^2 sigma_s^2 + t_k^2 sigma_d^2 / s_k]
    risk = 0.0
    risk += float(((1.0 - t[pos]) ** 2 * sigma_s2).sum())
    risk += float((t[pos] ** 2 * sigma_d2 / s[pos]).sum())
    return {"A_hat": A_hat, "stein_risk": risk,
            "sigma_s2": sigma_s2, "sigma_d2": sigma_d2, "weights": t}


def sure_interaction(B, A, Y, tol=1e-9):
    """Return the SURE-anchored interaction estimate G = Y - B A_hat_SURE."""
    out = sure_effects(B, A, Y, tol=tol)
    return np.asarray(Y, dtype=float) - (B @ out["A_hat"])
