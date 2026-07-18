"""Data-computable risk-dominance certificate (Theorem 5).

The gauge-subspace risk gap between the naive residual and the anchored estimator
at strength lambda decomposes over design modes s_k of S = B^T B:

    Delta(lambda) = sum_{k: s_k>0} [ sigma_s^2 - (sigma_s^2 + lambda^2 sigma_d^2 s_k)
                                     / (1 + lambda s_k)^2 ].

The k-th term is strictly positive iff lambda < 2 sigma_s^2 / (sigma_d^2 s_k). The
sufficient uniform certificate is

    lambda < lambda_dom = 2 sigma_s^2 / (sigma_d^2 s_max),

which guarantees dominance on every mode simultaneously. At a given lambda the
dominated modes are {k : s_k < s_cut} with s_cut = 2 sigma_s^2 / (sigma_d^2 lambda),
and the certified fraction of recoverable gauge energy is

    phi = sum_{k: s_k<s_cut} s_k^{-1} / sum_{k: s_k>0} s_k^{-1}.

All quantities are computed from B and the leak-free noise proxies, before any
interaction is estimated.
"""
from __future__ import annotations

import numpy as np

from .anchored import normal_operator


def design_spectrum(B, tol=1e-9):
    """Return the positive eigenvalues s_k of S = B^T B."""
    S = normal_operator(B).toarray()
    s = np.linalg.eigvalsh(S)
    return s[s > tol]


def lambda_dom(sigma_s2, sigma_d2, s_max):
    """Sufficient uniform-dominance threshold lambda_dom = 2 sigma_s^2/(sigma_d^2 s_max)."""
    return 2.0 * float(sigma_s2) / (float(sigma_d2) * float(s_max))


def risk_gap(lam, sigma_s2, sigma_d2, s):
    """Delta(lambda): total gauge-subspace risk reduction of anchored vs naive."""
    s = np.asarray(s, dtype=float)
    term = sigma_s2 - (sigma_s2 + lam ** 2 * sigma_d2 * s) / (1.0 + lam * s) ** 2
    return float(term.sum())


def certified_fraction(lam, sigma_s2, sigma_d2, s, tol=1e-12):
    """Fraction of recoverable gauge energy provably improved at strength lambda."""
    s = np.asarray(s, dtype=float)
    s = s[s > tol]
    if s.size == 0:
        return 0.0
    s_cut = 2.0 * sigma_s2 / (sigma_d2 * lam) if lam > 0 else np.inf
    w = 1.0 / s
    dominated = s < s_cut
    return float(w[dominated].sum() / w.sum())


def certify(B, sigma_s2, sigma_d2, lam=None, tol=1e-9):
    """Evaluate the certificate for a design B and leak-free noise proxies.

    Returns a dict with the design s_max, lambda_dom, the operating lambda (defaults
    to the isotropic optimum sigma_s^2/sigma_d^2), whether the uniform certificate is
    met, the per-mode certified energy fraction, and the risk gap Delta(lambda).
    """
    s = design_spectrum(B, tol=tol)
    s_max = float(s.max()) if s.size else 0.0
    ld = lambda_dom(sigma_s2, sigma_d2, s_max) if s_max > 0 else np.inf
    if lam is None:
        lam = float(sigma_s2) / float(sigma_d2)
    return {
        "s_max": s_max,
        "lambda_dom": ld,
        "lambda": float(lam),
        "uniform_certified": bool(lam < ld),
        "certified_energy_fraction": certified_fraction(lam, sigma_s2, sigma_d2, s),
        "risk_gap": risk_gap(lam, sigma_s2, sigma_d2, s),
    }
