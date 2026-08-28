# ============================================================
# ppEMG.weights — popstock-consistent population weights
# (extracted from CorrectionOmegaGW.py, logic unchanged)
# ============================================================
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Union

import numpy as np


# ============================================================
# Type aliases
# ============================================================
EventWeightFn = Callable[[Dict[str, np.ndarray]], np.ndarray]

# alpha_ppE can be float (backward compatible) or callable(dataset)->float/array(N_sys,)
AlphaLike = Union[float, Callable[[Dict[str, np.ndarray]], Union[float, np.ndarray]]]


def resolve_alpha(alpha_ppE: AlphaLike, dataset: Dict[str, np.ndarray]) -> np.ndarray:
    """Resolve alpha_ppE into a per-event array of shape (N_sys,).

    - If alpha_ppE is a float -> broadcast to all events.
    - If alpha_ppE is callable -> must return float or array-like length N_sys.
    """
    if "mass_1" not in dataset:
        raise KeyError("dataset must contain key 'mass_1' to determine N_sys")
    N = len(dataset["mass_1"])

    if callable(alpha_ppE):
        a = alpha_ppE(dataset)
    else:
        a = alpha_ppE

    a = np.asarray(a, dtype=float)

    if a.ndim == 0:
        return np.full(N, float(a), dtype=float)
    if a.shape == (N,):
        return a.astype(float, copy=False)

    raise ValueError(f"alpha_ppE must be float or callable returning shape (N_sys,), got {a.shape}")


# ============================================================
# popstock-consistent weights:  w_i = p(theta_i | Lambda) / pdraw_i
# ============================================================
# popstock computes Omega_GW with importance weights
#     w_i = p(theta_i | Lambda) / pdraw_i,
# where pdraw_i = p(theta_i | Lambda_fiducial_of_the_draw). The population
# averages of the corrections (ppE and G) and the reweighting den-ratios must
# use the SAME weights; using p(theta|Lambda) directly as the weight amounts
# to a "density squared" weighting (the samples already come from p), which
# biases the mean of Q by O(10-100%) depending on `a`.
#
# If the dataset carries the "pdraw" key (added automatically by
# dataset_from_popstock_samples), the functions below divide by it. Without
# "pdraw" in the dataset, we fall back to the old behaviour (single warning).
USE_PDRAW_WEIGHTS = True  # False => legacy behaviour (for A/B tests)

_PDRAW_WARNED = {"missing": False}


def weights_from_probabilities(
    dataset: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    *,
    use_pdraw: Optional[bool] = None,
) -> np.ndarray:
    """
    Convert probabilities = p(theta|Lambda) into popstock-consistent weights
    w = p/pdraw, zeroing events with p <= 0 (same convention as
    popstock.calculate_weights).

    Fallback: if the dataset has no "pdraw" (or use_pdraw=False), return
    probabilities unchanged (legacy behaviour).
    """
    p = np.asarray(probabilities, dtype=float)

    if use_pdraw is None:
        use_pdraw = USE_PDRAW_WEIGHTS

    pdraw = dataset.get("pdraw") if use_pdraw else None
    if pdraw is None:
        if use_pdraw and not _PDRAW_WARNED["missing"]:
            warnings.warn(
                "weights_from_probabilities: dataset has no 'pdraw' key — "
                "using p(theta|Lambda) directly as the weight (legacy "
                "behaviour, density-squared weighting). Rebuild the dataset "
                "with dataset_from_popstock_samples to fix this.",
                RuntimeWarning,
            )
            _PDRAW_WARNED["missing"] = True
        return p

    pdraw = np.asarray(pdraw, dtype=float)
    if pdraw.shape != p.shape:
        raise ValueError(
            f"dataset['pdraw'] shape {pdraw.shape} != probabilities shape {p.shape}"
        )

    w = np.zeros_like(p)
    good = np.isfinite(pdraw) & (pdraw > 0) & np.isfinite(p) & (p > 0)
    w[good] = p[good] / pdraw[good]
    return w
