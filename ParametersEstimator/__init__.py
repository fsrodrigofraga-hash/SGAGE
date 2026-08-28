# ParametersEstimator/__init__.py
# ============================================================
# Library-style package for parameter estimation from saved NPZ
# (typically produced by SimulatedSignal).
#
# Works with:
#   - ppEMG
#   - SimulatedSignal (same backend)
#
# The monolithic module ParametersEstimator.py (2705 lines) became a PACKAGE.
# The public API is identical, so
#
#     from ParametersEstimator import ParametersEstimator, EstimatorConfig
#
# keeps working with no changes at all. No physics was rewritten: the code
# was moved verbatim into the submodules below.
#
#     config.py          types + ParamGridSpec / LambdaParamSpec / EstimatorConfig
#     io_utils.py        config hashing, load_npz_flexible, axis <-> parameter
#     popstock_utils.py  compatibility across popstock versions
#     population.py      population draw and precomputed objects
#     models.py          omega_full_from_corrections (ppE + G)
#     interpolation.py   N-D multilinear interpolation + cache
#     grid.py            parallel construction of the N-D grid
#     cpu.py             automatic core allocation                   [MEM-2]
#     covariance.py      Ledoit-Wolf shrinkage [FIX-3] + C_pop  [FIX-1,2,3,6,7]
#     likelihoods.py     likelihoods on the grid + optimal filter
#     direct.py          grid-free evaluation (recomputed at each MCMC step)
#     estimator.py       the ParametersEstimator class
#     plotting.py        figure mixin (corner plot) — the only matplotlib user
#
# Design points (preserved from the original):
# - Builds a precomputed N-D grid of Omega_model(f_bin) and uses N-D
#   multilinear interpolation
# - Runs Bilby inference with a Gaussian likelihood on Omega_hat_bin
# - Grid caching on disk (hash of config + data grid)
#
# Extensions:
# - Estimates either (A) a + scalar alpha_ppE, OR
#                    (B) a + parameters of an alpha_event_model
#                        (alpha_ppE as callable(dataset) -> (N_sys,))
# - Supports an arbitrary G_event_weight with free parameters
#
# Integrated fixes:
# - [FIX-1]  C_pop computes the cov of Omega_MODEL (no detector noise).
# - [FIX-2]  K_POP=50 by default — avoids a singular C_pop for large n_bins.
# - [FIX-3]  Automatic Ledoit-Wolf shrinkage via sklearn.OAS (analytic fallback).
# - [FIX-4]  Full-cov likelihood: C_eff = k^2*C_inst + C_pop.
# - [FIX-6]  C_pop seeds spaced by the prime 1_000_003.
# - [FIX-7]  seed_noise and seed_signal vary per realization in the C_pop worker.
# - [FIX-8]  Diagnostics with the effective rank of C_pop via estimate_C_pop().
# - [MEM-2]  Automatic CPU management: grid_jobs, sampler_pool and cpop_jobs.
# - [MEM-4]  Cholesky LRU cache capped at 4 entries.
#
# Likelihood modes (run_inference):
#   "diagonal"  — diagonal Gaussian in sigma_inst (legacy).
#   "full_cov"  — full-covariance C_eff = k^2*C_inst + C_pop (recommended).
#
# Note: by default this estimator RE-DRAWS the population/dataset to build the
# grid, just like the original. To use exactly the same dataset as the
# injection, save it alongside the simulation outputs and load it here.
# ============================================================
from __future__ import annotations

from .config import (
    EstimatorConfig,
    EventVectorFn,
    LambdaParamSpec,
    ParamGridSpec,
)
from .covariance import estimate_C_pop
from .direct import (
    DirectModelEvaluator,
    OmegaGaussianLikelihoodDirectDiag,
    OmegaGaussianLikelihoodDirectFullCov,
)
from .estimator import ParametersEstimator
from .grid import build_model_grid_nd_parallel
from .interpolation import InterpModelNDWithCache
from .io_utils import load_npz_flexible
from .likelihoods import (
    OmegaGaussianLikelihoodDiag,
    OmegaGaussianLikelihoodFullCov,
    OmegaOptimalFilterLikelihood,
)
from .models import omega_full_from_corrections
from .population import precompute_population_objects

__version__ = "1.0.0"

__all__ = [
    # config
    "EventVectorFn", "ParamGridSpec", "LambdaParamSpec", "EstimatorConfig",
    # io
    "load_npz_flexible",
    # population and model
    "precompute_population_objects", "omega_full_from_corrections",
    # grid
    "InterpModelNDWithCache", "build_model_grid_nd_parallel",
    # covariancia
    "estimate_C_pop",
    # likelihoods
    "OmegaGaussianLikelihoodDiag", "OmegaGaussianLikelihoodFullCov",
    "OmegaGaussianLikelihoodDirectDiag", "OmegaGaussianLikelihoodDirectFullCov",
    "OmegaOptimalFilterLikelihood",
    # avaliacao direta
    "DirectModelEvaluator",
    # main class
    "ParametersEstimator",
    "__version__",
]
