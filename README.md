# perturb_design

Implementation for the paper **"A second measurement channel recovers
discarded genetic interactions"**.

Combinatorial perturbation screens estimate genetic interaction as the deviation of
a double-perturbation phenotype from the sum of its two single-perturbation effects,
`G = Y - (A_i + A_j)`. This code implements the paper's finding that the naive
residual discards a `rank(B)`-dimensional *gauge* subspace of recoverable
interaction, and that treating the measured singles as an independent second
measurement channel restores it with a **certified variance reduction**.

## What is in this repository

| Module | Paper object |
| --- | --- |
| `perturb_design/anchored.py` | Gauge-anchored estimator `A_hat = argmin ‖X−A‖² + λ‖BX−Y‖²`; identifiable/gauge projector `P` (Def. 1, Thm. 1); isotropic λ\* (Thm. 4) and per-gene heteroscedastic λ\*_g (Thm. 6) |
| `perturb_design/sure.py` | Tuning-free per-mode SURE operator and gap-moment noise identification (Thm. 7) |
| `perturb_design/certificate.py` | Data-computable risk-dominance certificate (Thm. 5) |
| `perturb_design/budget.py` | Closed-form singles-vs-doubles budget split (Thm. 8); leak-free noise proxies and λ selection |
| `experiments/synthetic_subspace.py` | Synthetic subspace decomposition — reproduces Table 2 / Figure 3 |
| `experiments/sure_certificate_budget.py` | SURE, certificate, and budget demonstration |
| `experiments/real_screens.py` | Real-screen analysis pipeline (requires public data, see below) |
| `tests/test_theorems.py` | Numerical checks of Theorems 3, 4, 5, 7, 8 |

The solver is matrix-free: the normal operator `S = BᵀB` is a signless graph
Laplacian, so the anchored system is solved by conjugate gradients without forming
any inverse (Proposition 1), giving near-linear scaling.

## Installation

```bash
git clone https://github.com/sarwanpasha/perturb_design.git
cd perturb_design
pip install -r requirements.txt          # numpy, scipy
```

The core library depends only on NumPy and SciPy. Real-data preprocessing
additionally uses `scanpy`/`anndata` (see `requirements.txt`).

## Quick start

```python
import numpy as np
from perturb_design import (
    design_matrix, anchored_interaction, naive_interaction,
    select_lambda_leakfree, sure_interaction, certify, estimate_noise_scales,
)

# A: (n, p) single-perturbation profiles; Y: (D, p) double profiles;
# pairs: list of (i, j) index pairs identifying each measured double.
B = design_matrix(pairs, n)

# leak-free choice of anchor strength, then the anchored interaction estimate
lam = select_lambda_leakfree(B, A, Y, pairs, lambdas=[0, 0.1, 0.5, 1, 2])["lambda"]
G_anchored = anchored_interaction(B, A, Y, lam)

# tuning-free alternative (no cross-validation)
G_sure = sure_interaction(B, A, Y)

# risk-dominance certificate for this design, before any interaction is estimated
noise = estimate_noise_scales(B, A, Y)
print(certify(B, noise["sigma_s2"], noise["sigma_d2"]))
```

## Reproducing the results

**Synthetic (self-contained, no external data).**

```bash
python experiments/synthetic_subspace.py        # Table 2 / Figure 3
python experiments/sure_certificate_budget.py   # Theorems 5, 7, 8
python -m pytest tests/ -q                       # theorem checks
```

`synthetic_subspace.py` reproduces the reported pattern: the identifiable-subspace
error of the naive and anchored estimators coincides to machine precision
(≈10⁻¹⁶, confirming Theorem 3), and the gauge-subspace recovery rises with single
noise (≈43% / 64% / 76% / 82% at σ_s = 0.4 / 0.6 / 0.8 / 1.0), with the leak-free
CV-selected λ tracking the reported median values.

**Real screens (require public data).** The three screens are public:

| Screen | Modality | Accession |
| --- | --- | --- |
| Norman et al. 2019 | dual CRISPRa (K562) | GEO GSE133344 |
| Wessels et al. 2023 | Cas13 RNA Perturb-seq | GEO GSE213957 |
| Joung et al. 2023 | TF overexpression | GEO GSE217460 |

Download and harmonise them following Peidli et al. 2024 (scPerturb,
<https://scperturb.org>), build `A`, `Y`, and `pairs` per the preprocessing in
`experiments/real_screens.py`, and call `analyse_screen(A, Y, pairs)`. That routine
runs the leak-free held-out comparison, the identifiable/gauge decomposition
(Theorem 3 check), and the certificate.


## License

MIT — see [LICENSE](LICENSE).
