from __future__ import annotations

# ============================================================
# ParametersEstimator.models — Omega_model(f) from the ppE + G
# corrections at one point of parameter space
# ============================================================
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

import ppEMG as corr

from .config import EventVectorFn


# Model evaluation (ppE + G) with optional alpha_event_model
# ============================================================
def omega_full_from_corrections(
    freqs_pop: np.ndarray,
    omega_fid: np.ndarray,
    dataset: Dict[str, Any],
    Lambda: Dict[str, float],
    ctx: Any,
    probabilities: np.ndarray,
    *,
    a: float,
    alpha_ppE: Optional[float] = None,
    alpha_event_model: Optional[EventVectorFn] = None,
    alpha_params: Optional[Dict[str, Any]] = None,
    G_event_weight: Optional[EventVectorFn] = None,
    G_params: Optional[Dict[str, Any]] = None,
    # Strategy-1: pre-computed waveform cache (bypasses LAL calls when provided)
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_f_peak_obs: Optional[np.ndarray] = None,
    # compute_correction knobs
    chunksize: int = 20,
    fast_binning: bool = True,
    max_bins: int = 400,
    use_frequency_warp: bool = True,
    nproc: Optional[int] = 1,
    disable_inspiral_cutoff: Optional[bool] = None,
    smooth_ppE: bool = False,
) -> np.ndarray:
    if alpha_params is None:
        alpha_params = {}
    if G_params is None:
        G_params = {}

    if G_event_weight is None:
        def G_event_weight(ds, **params):
            return np.ones(len(ds["mass_1"]), dtype=np.float64)

    def _G_wrap(ds):
        w = G_event_weight(ds, **G_params)
        w = np.asarray(w, dtype=np.float64)
        n_sys = len(ds["mass_1"])
        if w.shape != (n_sys,):
            raise ValueError(f"G_event_weight must return shape (N_sys,), got {w.shape} with N_sys={n_sys}")
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        return w

    if alpha_event_model is not None:
        def _alpha_wrap(ds):
            v = alpha_event_model(ds, **alpha_params)
            v = np.asarray(v, dtype=np.float64)
            n_sys = len(ds["mass_1"])
            if v.shape != (n_sys,):
                raise ValueError(f"alpha_event_model must return shape (N_sys,), got {v.shape} with N_sys={n_sys}")
            v = np.where(np.isfinite(v), v, 0.0)
            return v
        alpha_arg: Union[float, Callable[[Dict[str, Any]], np.ndarray]] = _alpha_wrap
    else:
        if alpha_ppE is None:
            raise ValueError("Provide either alpha_ppE (scalar) or alpha_event_model (callable).")
        alpha_arg = float(alpha_ppE)

    corr_ppE, corr_G, _extras = corr.compute_correction(
        dataset=dataset,
        Lambda=Lambda,
        ctx=ctx,
        a=float(a),
        alpha_ppE=alpha_arg,
        G_event_weight=_G_wrap,
        nproc=nproc,
        chunksize=int(chunksize),
        precomputed_probabilities=probabilities,
        precomputed_h2_cache=precomputed_h2_cache,
        precomputed_f_peak_obs=precomputed_f_peak_obs,
        fast_binning=bool(fast_binning),
        max_bins=int(max_bins),
        use_frequency_warp=bool(use_frequency_warp),
        disable_inspiral_cutoff=disable_inspiral_cutoff,
        smooth_ppE = smooth_ppE,
    )

    corr_ppE = np.asarray(corr_ppE, dtype=np.float64)
    corr_G   = np.asarray(corr_G,   dtype=np.float64)

    omega_full = omega_fid * corr_ppE * corr_G
    omega_full = np.where(np.isfinite(omega_full) & (omega_full >= 0), omega_full, 0.0)
    return omega_full
