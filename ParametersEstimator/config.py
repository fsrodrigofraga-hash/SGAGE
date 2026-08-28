from __future__ import annotations

# ============================================================
# ParametersEstimator.config — types and configuration dataclasses
# ============================================================
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Types
# ============================================================
# Signature: fn(dataset, **params) -> (N_sys,)
EventVectorFn = Callable[..., np.ndarray]


# ============================================================
# Grid spec + config
# ============================================================
@dataclass(frozen=True)
class ParamGridSpec:
    """
    Generic parameter grid spec for precomputing model grids.

    transform:
      - "linear": parameter lives in [min, max] and grid is linspace.
      - "log10" : parameter lives in [10^min, 10^max] and grid axis is log10(theta).
                 (Interpolation is done in log10-space; likelihood parameter is theta itself.)
    """
    name: str
    min: float
    max: float
    n: int
    transform: str = "linear"  # "linear" or "log10"
    latex_label: Optional[str] = None


@dataclass(frozen=True)
class LambdaParamSpec:
    """
    Grid specification for a free Lambda hyperparameter.

    name        : exact key as it appears in the Lambda dict (e.g. "alpha", "rate")
    min / max   : prior bounds (in the space defined by ``transform``)
    n           : number of grid points
    transform   : "linear" or "log10"  (same semantics as ParamGridSpec)
    latex_label : LaTeX string for corner-plot axis labels
    """
    name: str
    min: float
    max: float
    n: int
    transform: str = "linear"
    latex_label: Optional[str] = None


@dataclass(frozen=True)
class EstimatorConfig:
    # Paths
    infile_npz: str = os.path.join("saved_ppE_data", "ppE_simulated_data.npz")
    outdir: str = "outputs_infer_from_saved"
    grid_cache_dir: str = ""

    # Core ppE "a" range
    a_min: float = 0.0
    a_max: float = 10.0
    na: int = 41
    # =========================
    # Scalar alpha_ppE inference mode
    # =========================
    # alpha_scale:
    #   - "log10": prior LogUniform, grid/interp in log10(alpha)  (modo atual)
    #   - "linear": prior Uniform, grid/interp em alpha diretamente
    alpha_scale: str = "log10"     # "log10" ou "linear"

    # (A) Se alpha_scale="log10"
    logalpha_min: float = -6.0         # log10(alpha_ppE)
    logalpha_max: float = -2.0
    nlogalpha: int = 41

    # (B) Se alpha_scale="linear"
    alpha_min: float = 1e-6
    alpha_max: float = 1e-2
    nalpha: int = 41


    # Cache quantization for interpolator memoization
    cache_subdiv: int = 10

    # -------------------------
    # Parallel / CPU management  [MEM-2]
    # -------------------------
    # n_jobs=-1 uses all cores for the model grid build (joblib).
    # grid_jobs, sampler_pool and cpop_jobs are derived automatically from
    # cpu_count() if left as None, following a RAM-aware heuristic:
    #   8  GB RAM → grid_jobs=2, sampler_pool=2, cpop_jobs=1
    #   16 GB RAM → grid_jobs=4, sampler_pool=4, cpop_jobs=2  (default)
    #   32 GB RAM → grid_jobs=8, sampler_pool=8, cpop_jobs=4
    # Set any of them explicitly to override auto-detection.
    n_jobs: int = -1               # -1 means "all cores" for joblib grid build
    grid_jobs: Optional[int] = None    # joblib workers for grid build (None=auto)
    sampler_pool: Optional[int] = None # npool passed to bilby.run_sampler (None=auto)
    cpop_jobs: Optional[int] = None    # joblib workers for C_pop MC (None=auto)

    # Dynesty
    npoints: int = 500
    walks: int = 30
    dlogz: float = 0.1

    # -------------------------
    # C_pop (population covariance)  [FIX-1,2,3,6,7]
    # -------------------------
    # k_pop: number of Monte Carlo realisations for C_pop estimation.
    #   Set to 0 to skip C_pop (diagonal likelihood only).
    k_pop: int = 50
    pop_seed_base: int = 90000
    cpop_jitter: float = 1e-10

    # -------------------------
    # Likelihood mode
    # -------------------------
    # "diagonal"  — legacy Gaussian diagonal on sigma_inst.
    # "full_cov"  — C_eff = k²·C_inst + C_pop  [FIX-4]  (recommended).
    likelihood_mode: str = "full_cov"

    # Numerical floor
    floor: float = 1e-60
