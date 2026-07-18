"""perturb_design: gauge-anchored recovery of genetic interactions.

Reference implementation accompanying the paper
"A second measurement channel recovers discarded genetic interactions".

Public API
----------
Core estimator (anchored.py):
    design_matrix, normal_operator, anchored_effects, anchored_interaction,
    naive_interaction, identifiable_projector, apply_projector,
    isotropic_lambda, heteroscedastic_lambda
Tuning-free SURE operator (sure.py):
    estimate_noise_scales, sure_effects, sure_interaction
Risk-dominance certificate (certificate.py):
    certify, lambda_dom, risk_gap, certified_fraction, design_spectrum
Budget rule and selection (budget.py):
    optimal_split, design_traces, estimate_gene_noise, select_lambda_leakfree
"""
from .anchored import (
    design_matrix, normal_operator, anchored_effects, anchored_interaction,
    naive_interaction, identifiable_projector, apply_projector,
    isotropic_lambda, heteroscedastic_lambda,
)
from .sure import estimate_noise_scales, sure_effects, sure_interaction
from .certificate import certify, lambda_dom, risk_gap, certified_fraction, design_spectrum
from .budget import optimal_split, design_traces, estimate_gene_noise, select_lambda_leakfree

__all__ = [
    "design_matrix", "normal_operator", "anchored_effects", "anchored_interaction",
    "naive_interaction", "identifiable_projector", "apply_projector",
    "isotropic_lambda", "heteroscedastic_lambda",
    "estimate_noise_scales", "sure_effects", "sure_interaction",
    "certify", "lambda_dom", "risk_gap", "certified_fraction", "design_spectrum",
    "optimal_split", "design_traces", "estimate_gene_noise", "select_lambda_leakfree",
]
__version__ = "1.0.0"
