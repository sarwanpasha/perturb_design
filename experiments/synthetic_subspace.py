"""Synthetic ground-truth benchmark with subspace decomposition (Section 4.1).

Reproduces the gauge- vs identifiable-subspace decomposition of Figure 3 / Table 2:
the identifiable-subspace error is identical for naive and anchored (machine
precision, Theorem 3), and the gauge-subspace recovery improves monotonically with
single-measurement noise (Theorem 4).

Run:
    python experiments/synthetic_subspace.py
"""
from __future__ import annotations

import numpy as np

from perturb_design import (
    design_matrix, anchored_effects, apply_projector, select_lambda_leakfree,
)


def sample_screen(n, p, D, r, sigma_s, sigma_d, rng, g_scale=1.0):
    """Sample true effects a, a random pair design, a planted low-rank interaction,
    and the two noisy channels A and Y."""
    a = rng.standard_normal((n, p))
    pairs = []
    seen = set()
    while len(pairs) < D:
        i, j = rng.integers(0, n, size=2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            seen.add((i, j))
            pairs.append((int(i), int(j)))
    B = design_matrix(pairs, n)

    L = rng.standard_normal((D, r))
    R = rng.standard_normal((r, p))
    G = g_scale * (L @ R) / np.sqrt(r)

    A = a + sigma_s * rng.standard_normal((n, p))
    Y = (B @ a) + G + sigma_d * rng.standard_normal((D, p))
    return a, B, pairs, G, A, Y


def subspace_errors(B, G_hat, G_true):
    """Return (identifiable-error, gauge-error) of an interaction estimate."""
    Pid = apply_projector(B, G_hat - G_true)          # P (G_hat - G_true)
    id_err = float((Pid ** 2).sum())
    total = float(((G_hat - G_true) ** 2).sum())
    gauge_err = total - id_err
    return id_err, gauge_err


LAMBDA_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def run(n=60, p=200, D=120, r=3, sigma_d=0.1,
        sigma_s_grid=(0.2, 0.4, 0.6, 0.8, 1.0), seeds=30):
    # sigma_s = 0 is omitted: with noiseless singles lambda* = 0, so the anchored
    # estimate equals the naive residual exactly (Theorem 4(i)) and the improvement
    # is identically 0. See the paper's Table 2.
    print(f"n={n} p={p} D={D} r={r} sigma_d={sigma_d}, {seeds} seeds")
    print(f"{'sigma_s':>8} {'gauge impr %':>16} {'ident rel diff':>16} {'median lam':>11}")
    for sigma_s in sigma_s_grid:
        g_impr, id_rel, lam_sel = [], [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            a, B, pairs, G, A, Y = sample_screen(n, p, D, r, sigma_s, sigma_d, rng)

            G_naive = np.asarray(Y) - (B @ A)
            # leak-free CV selection of the anchor strength (paper protocol)
            sel = select_lambda_leakfree(B, A, Y, pairs, LAMBDA_GRID, n_folds=5, seed=seed)
            lam = sel["lambda"]
            lam_sel.append(lam)
            A_hat = anchored_effects(B, A, Y, lam)
            G_anc = np.asarray(Y) - (B @ A_hat)

            id_n, g_n = subspace_errors(B, G_naive, G)
            id_a, g_a = subspace_errors(B, G_anc, G)

            g_impr.append(100.0 * (g_n - g_a) / g_n if g_n > 0 else 0.0)
            denom = abs(id_n) + 1e-30
            id_rel.append(abs(id_a - id_n) / denom)

        print(f"{sigma_s:8.1f} {np.mean(g_impr):11.1f}+/-{np.std(g_impr):.1f}"
              f" {np.mean(id_rel):16.2e} {np.median(lam_sel):11.2f}")


if __name__ == "__main__":
    run()
