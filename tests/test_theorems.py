"""Numerical checks of the paper's theorems on small synthetic instances.

Run:
    python -m pytest tests/ -q
or:
    python tests/test_theorems.py
"""
import numpy as np

from perturb_design import (
    design_matrix, anchored_effects, anchored_interaction, naive_interaction,
    apply_projector, estimate_noise_scales, certify, design_traces, optimal_split,
)


def _toy(seed=0, n=20, p=8, D=40, sigma_s=0.5, sigma_d=0.1, g_scale=0.3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, p))
    pairs, seen = [], set()
    while len(pairs) < D:
        i, j = rng.integers(0, n, 2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            seen.add((i, j)); pairs.append((int(i), int(j)))
    B = design_matrix(pairs, n)
    G = rng.standard_normal((D, p)) * g_scale
    A = a + sigma_s * rng.standard_normal((n, p))
    Y = (B @ a) + G + sigma_d * rng.standard_normal((D, p))
    return a, B, pairs, G, A, Y


def test_lambda0_recovers_naive():
    """Theorem 4(i): at lambda=0 the anchored interaction equals the naive residual."""
    a, B, pairs, G, A, Y = _toy()
    G_anc0 = anchored_interaction(B, A, Y, 0.0)
    G_naive = naive_interaction(B, A, Y)
    assert np.allclose(G_anc0, G_naive, atol=1e-8)


def test_invariance_on_identifiable_subspace():
    """Theorem 3: P G_anc = P G_naive for every lambda (machine precision)."""
    a, B, pairs, G, A, Y = _toy()
    G_naive = naive_interaction(B, A, Y)
    for lam in [0.1, 0.5, 2.0, 10.0]:
        G_anc = anchored_interaction(B, A, Y, lam)
        diff = apply_projector(B, G_anc - G_naive)
        assert np.max(np.abs(diff)) < 1e-8, lam


def test_noise_scale_recovery():
    """Theorem 7: the gap-moment identity recovers the two noise variances in the
    consistent regime (heterogeneous-degree design with spectral spread, many genes,
    interaction small relative to double noise)."""
    a, B, pairs, G, A, Y = _toy(sigma_s=0.5, sigma_d=0.3, n=60, D=120, p=300,
                                g_scale=0.05)
    est = estimate_noise_scales(B, A, Y)
    assert abs(est["sigma_s2"] - 0.5 ** 2) < 0.05
    assert abs(est["sigma_d2"] - 0.3 ** 2) < 0.03


def test_certificate_monotone():
    """The certified energy fraction is non-increasing as lambda grows past lambda_dom."""
    a, B, pairs, G, A, Y = _toy()
    est = estimate_noise_scales(B, A, Y)
    lo = certify(B, est["sigma_s2"], est["sigma_d2"], lam=0.01)["certified_energy_fraction"]
    hi = certify(B, est["sigma_s2"], est["sigma_d2"], lam=100.0)["certified_energy_fraction"]
    assert lo >= hi


def test_budget_split_positive():
    """Theorem 8: the optimal split is well-defined and sums to the budget."""
    a, B, pairs, G, A, Y = _toy()
    est = estimate_noise_scales(B, A, Y)
    split = optimal_split(B, est["sigma_s2"], est["sigma_d2"], M=20)
    assert split["m_s"] > 0 and split["m_d"] > 0
    assert abs(split["m_s"] + split["m_d"] - 20) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all tests passed")
