"""Real-screen analysis template (Section 4.5).

The paper analyses three public combinatorial screens:
  - Norman et al. 2019, dual CRISPRa in K562 (GEO GSE133344)
  - Wessels et al. 2023, Cas13 RNA Perturb-seq (GEO GSE213957)
  - Joung et al. 2023, transcription-factor overexpression (GEO GSE217460)
harmonised following Peidli et al. 2024 (scPerturb, https://scperturb.org).

Real single-cell data is NOT bundled with this repository. This script documents
the exact pipeline so a user can reproduce the real-data results after downloading
the public datasets and building, for each screen, three arrays:

    A  : (n, p)   per-perturbation single-effect profiles (control-referenced)
    Y  : (D, p)   double-perturbation profiles (control-referenced)
    pairs : list of (i, j) index pairs, the measured doubles

Preprocessing (per the paper): counts library-size normalised to 1e4,
log(1+x) transformed, averaged within perturbation labels, effects expressed
relative to a control origin; standard variance-based feature selection to p genes.

Given (A, Y, pairs), the analysis below is fully general and reproduces the
leak-free held-out comparison, the identifiable/gauge decomposition, and the
biological-recovery and certificate quantities.
"""
from __future__ import annotations

import numpy as np

from perturb_design import (
    design_matrix, naive_interaction, anchored_interaction, anchored_effects,
    apply_projector, estimate_gene_noise, heteroscedastic_lambda,
    select_lambda_leakfree, estimate_noise_scales, certify, sure_interaction,
)


def leakfree_heldout_error(A, Y, pairs, lam, n_folds=5, seed=0):
    """Leak-free k-fold held-out double-prediction error for a given anchor lam.
    Returns relative error (fraction of held-out double energy)."""
    A = np.asarray(A, float); Y = np.asarray(Y, float)
    n = A.shape[0]; D = len(pairs)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(D), n_folds)
    num = den = 0.0
    for f in folds:
        test = set(f.tolist())
        tr_idx = [d for d in range(D) if d not in test]
        B_tr = design_matrix([pairs[d] for d in tr_idx], n)
        A_hat = anchored_effects(B_tr, A, Y[tr_idx], lam)
        for d in f:
            i, j = pairs[d]
            pred = A_hat[i] + A_hat[j]
            num += float(((Y[d] - pred) ** 2).sum())
            den += float((Y[d] ** 2).sum())
    return num / den


def analyse_screen(A, Y, pairs, lambda_grid=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)):
    """Run the full anchored analysis on one screen and return a summary dict."""
    A = np.asarray(A, float); Y = np.asarray(Y, float)
    n, p = A.shape
    B = design_matrix(pairs, n)

    # leak-free global lambda
    sel = select_lambda_leakfree(B, A, Y, pairs, lambda_grid)
    lam = sel["lambda"]

    # per-gene heteroscedastic lambda
    s2s, s2d = estimate_gene_noise(B, A, Y)
    lam_g = heteroscedastic_lambda(s2s, s2d)

    # held-out errors
    err_naive = leakfree_heldout_error(A, Y, pairs, 0.0)
    err_iso = leakfree_heldout_error(A, Y, pairs, lam)
    err_het = leakfree_heldout_error(A, Y, pairs, lam_g)

    # identifiable-subspace equality check (Theorem 3)
    G_naive = naive_interaction(B, A, Y)
    G_anc = anchored_interaction(B, A, Y, lam)
    P_diff = apply_projector(B, G_anc - G_naive)
    id_rel = float((P_diff ** 2).sum()) / (float((apply_projector(B, G_naive) ** 2).sum()) + 1e-30)

    # certificate
    noise = estimate_noise_scales(B, A, Y)
    cert = certify(B, noise["sigma_s2"], noise["sigma_d2"], lam=lam)

    return {
        "n": n, "D": len(pairs), "lambda": lam,
        "err_naive": err_naive, "err_anchored_iso": err_iso, "err_anchored_het": err_het,
        "identifiable_rel_diff": id_rel,
        "certified_energy_fraction": cert["certified_energy_fraction"],
    }


if __name__ == "__main__":
    print(__doc__)
    print("Load A, Y, pairs for a screen, then call analyse_screen(A, Y, pairs).")
