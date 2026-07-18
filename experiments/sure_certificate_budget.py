"""Demonstration of the tuning-free SURE operator (Theorem 7), the risk-dominance
certificate (Theorem 5), and the budget rule (Theorem 8) on a synthetic screen.

Run:
    python experiments/sure_certificate_budget.py
"""
from __future__ import annotations

import numpy as np

from perturb_design import (
    design_matrix, naive_interaction, sure_effects, sure_interaction,
    estimate_noise_scales, certify, optimal_split,
)
from experiments.synthetic_subspace import sample_screen, subspace_errors


def run(n=60, p=300, D=120, r=3, sigma_s=0.5, sigma_d=0.3, seed=0):
    rng = np.random.default_rng(seed)
    # Small planted interaction (g_scale) keeps the gap-moment slope unbiased; see
    # the estimation-regime note in perturb_design/sure.py.
    a, B, pairs, G, A, Y = sample_screen(n, p, D, r, sigma_s, sigma_d, rng, g_scale=0.05)

    # --- Tuning-free SURE per-mode operator (Theorem 7) -----------------------
    est = estimate_noise_scales(B, A, Y)
    print("Estimated noise scales from the inter-channel gap (Theorem 7):")
    print(f"  sigma_s^2  true={sigma_s**2:.4f}  estimated={est['sigma_s2']:.4f}")
    print(f"  sigma_d^2  true={sigma_d**2:.4f}  estimated={est['sigma_d2']:.4f}")

    G_naive = naive_interaction(B, A, Y)
    G_sure = sure_interaction(B, A, Y)
    _, g_naive = subspace_errors(B, G_naive, G)
    _, g_sure = subspace_errors(B, G_sure, G)
    red = 100.0 * (g_naive - g_sure) / g_naive
    print(f"\nSURE gauge-error reduction vs naive: {red:.1f}%  (no tuning)")
    print(f"  unbiased Stein risk estimate: {sure_effects(B, A, Y)['stein_risk']:.4f}")

    # --- Risk-dominance certificate (Theorem 5) -------------------------------
    cert = certify(B, est["sigma_s2"], est["sigma_d2"])
    print("\nRisk-dominance certificate (Theorem 5):")
    print(f"  s_max = {cert['s_max']:.2f}")
    print(f"  lambda (isotropic optimum) = {cert['lambda']:.3f}")
    print(f"  lambda_dom (uniform threshold) = {cert['lambda_dom']:.3f}")
    print(f"  uniform certificate met: {cert['uniform_certified']}")
    print(f"  certified gauge-energy fraction: {cert['certified_energy_fraction']:.3f}")
    print(f"  risk gap Delta(lambda): {cert['risk_gap']:.4f}")

    # --- Budget rule (Theorem 8) ----------------------------------------------
    split = optimal_split(B, est["sigma_s2"], est["sigma_d2"], M=20)
    print("\nOptimal singles:doubles budget split (Theorem 8), M=20:")
    print(f"  t_s = {split['t_s']:.3f}  t_d = rank(B) = {split['t_d']}")
    print(f"  m_s* = {split['m_s']:.2f}  m_d* = {split['m_d']:.2f}  (ratio {split['ratio']:.3f})")


if __name__ == "__main__":
    run()
