from __future__ import annotations

# ============================================================
# ParametersEstimator.grid — parallel construction of the N-D grid of
# Omega_model(f_bin)
# ============================================================
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

import ppEMG as corr

from .config import EstimatorConfig, EventVectorFn, ParamGridSpec
from .io_utils import _axis_coord_to_theta, _axis_values_for_spec
from .models import omega_full_from_corrections
from .population import precompute_population_objects


# Grid building (parallel)
# ============================================================
def _grid_point_eval(
    f_bin,
    freqs_pop,
    omega_fid,
    dataset,
    Lambda,
    ctx,
    probabilities,
    *,
    theta: Dict[str, float],
    alpha_event_model: Optional[EventVectorFn],
    alpha_param_names: List[str],
    G_event_weight: Optional[EventVectorFn],
    G_param_names: List[str],
    compute_knobs: Dict[str, Any],
) -> np.ndarray:
    a = float(theta.get("a", Lambda.get("a", 0.0)))
    G_params = {k: float(theta[k]) for k in G_param_names} if G_param_names else {}
    alpha_params = {k: float(theta[k]) for k in alpha_param_names} if alpha_param_names else {}  # ← ADICIONE

    if alpha_event_model is None:
        alpha_ppE = float(theta["alpha_ppE"])
    else:
        alpha_ppE = None

    omega_full = omega_full_from_corrections(
        freqs_pop, omega_fid, dataset, Lambda, ctx, probabilities,
        a=a,
        alpha_ppE=alpha_ppE,
        alpha_event_model=alpha_event_model,
        alpha_params=alpha_params,
        G_event_weight=G_event_weight,
        G_params=G_params,
        **compute_knobs,
    )
    return np.interp(f_bin, freqs_pop, omega_full, left=0.0, right=0.0).astype(np.float64)


def _grid_point_eval_with_pop(
    f_bin,
    base_Lambda: Dict[str, float],
    lambda_param_names: List[str],
    pop_cfg: Dict[str, Any],
    *,
    theta: Dict[str, float],
    alpha_event_model: Optional[EventVectorFn],
    alpha_param_names: List[str],
    G_event_weight: Optional[EventVectorFn],
    G_param_names: List[str],
    compute_knobs: Dict[str, Any],
) -> np.ndarray:
    """
    Like _grid_point_eval but recomputes the full population for each Lambda grid point.
    Required when one or more Lambda hyperparameters are free inference parameters,
    because a single pre-computed population cannot be shared across Lambda values.

    pop_cfg keys: FMIN, FMAX, N_FREQ, N_SYS, binary_type, waveform_approximant,
                  inspiral_only, disable_inspiral_cutoff, sampling_frequency, duration.
    """
    # Build Lambda for this grid point by overriding base_Lambda with free Lambda values
    Lambda_here = dict(base_Lambda)
    for k in lambda_param_names:
        Lambda_here[k] = float(theta[k])

    freqs_pop, omega_fid, dataset, Lambda_eff, ctx, probabilities = \
        precompute_population_objects(
            float(pop_cfg["FMIN"]),
            float(pop_cfg["FMAX"]),
            int(pop_cfg["N_FREQ"]),
            int(pop_cfg["N_SYS"]),
            binary_type=str(pop_cfg["binary_type"]),
            waveform_approximant=str(pop_cfg["waveform_approximant"]),
            inspiral_only=bool(pop_cfg["inspiral_only"]),
            disable_inspiral_cutoff=bool(pop_cfg["disable_inspiral_cutoff"]),
            sampling_frequency=float(pop_cfg["sampling_frequency"]),
            duration=float(pop_cfg["duration"]),
            Lambda=Lambda_here,
        )

    return _grid_point_eval(
        f_bin, freqs_pop, omega_fid, dataset, Lambda_eff, ctx, probabilities,
        theta=theta,
        alpha_event_model=alpha_event_model,
        alpha_param_names=alpha_param_names,
        G_event_weight=G_event_weight,
        G_param_names=G_param_names,
        compute_knobs=compute_knobs,
    )


def build_model_grid_nd_parallel(
    f_bin,
    freqs_pop,
    omega_fid,
    dataset,
    Lambda,
    ctx,
    probabilities,
    *,
    specs: List[ParamGridSpec],
    axes: List[np.ndarray],
    alpha_event_model: Optional[EventVectorFn],
    alpha_param_names: List[str],
    G_event_weight: Optional[EventVectorFn],
    G_param_names: List[str],
    n_jobs: int,
    compute_knobs: Dict[str, Any],
    # Lambda free params — if non-empty each grid point recomputes its own population
    lambda_param_names: Optional[List[str]] = None,
    base_Lambda: Optional[Dict[str, float]] = None,
    pop_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    nf = len(f_bin)
    shape = [len(ax) for ax in axes] + [nf]

    idx_ranges = [range(len(ax)) for ax in axes]
    all_indices = np.array(np.meshgrid(*idx_ranges, indexing="ij")).reshape(len(axes), -1).T

    def _theta_from_index_tuple(ind_tuple):
        theta = {}
        for d, spec in enumerate(specs):
            axis_coord = float(axes[d][int(ind_tuple[d])])
            theta[spec.name] = _axis_coord_to_theta(spec, axis_coord)
        return theta

    tasks = [tuple(ind) for ind in all_indices]

    _has_lambda_free = bool(lambda_param_names)

    if _has_lambda_free:
        # Each grid point must recompute its own population because Lambda varies.
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_grid_point_eval_with_pop)(
                f_bin,
                base_Lambda=base_Lambda,
                lambda_param_names=lambda_param_names,
                pop_cfg=pop_cfg,
                theta=_theta_from_index_tuple(ind),
                alpha_event_model=alpha_event_model,
                alpha_param_names=alpha_param_names,
                G_event_weight=G_event_weight,
                G_param_names=G_param_names,
                compute_knobs=compute_knobs,
            )
            for ind in tasks
        )
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_grid_point_eval)(
                f_bin, freqs_pop, omega_fid, dataset, Lambda, ctx, probabilities,
                theta=_theta_from_index_tuple(ind),
                alpha_event_model=alpha_event_model,
                alpha_param_names=alpha_param_names,
                G_event_weight=G_event_weight,
                G_param_names=G_param_names,
                compute_knobs=compute_knobs,
            )
            for ind in tasks
        )

    grid = np.zeros(shape, dtype=np.float64)
    for k, ind in enumerate(tasks):
        grid[tuple(ind) + (slice(None),)] = results[k]
    return grid
