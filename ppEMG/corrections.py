# ============================================================
# ppEMG.corrections — ppE amplitude correction, G correction
# and the compute_correction orchestrator
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.constants import G, c, M_sun
from scipy.signal import savgol_filter

from .config import CorrectionContext
from .cutoff import f_isco_source_hz, robust_inspiral_end_frequency_obs
from .population import calculate_probabilities
from .waveforms import compute_h2_cache_and_fpeak_parallel
from .weights import (
    AlphaLike,
    EventWeightFn,
    resolve_alpha,
    weights_from_probabilities,
)


# ============================================================
# Population averages in frequency
# ============================================================
def population_ratio_in_f(
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    weight_per_event: np.ndarray,
) -> np.ndarray:
    num = np.sum(probabilities[:, None] * h2_cache * weight_per_event[:, None], axis=0)
    den = np.sum(probabilities[:, None] * h2_cache, axis=0)
    return num / den


# ============================================================
# ppE correction
# ============================================================
def compute_Q_ppE(dataset: Dict[str, np.ndarray], a: float, *, alpha_event: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the per-event ppE amplitude factor Q_i = (pi * M_chirp_z [s])^(a/3).

    The ppE amplitude correction to the waveform is h_ppE = h_GR * (1 + alpha * u^a),
    where the PN parameter u = (pi * M_chirp_z * f)^(1/3) requires M_chirp_z in
    geometric units (seconds): M_s = G * M_sun / c^3 ≈ 4.926e-6 s.

    Bug fixed: previous version used chirp_mass_z in solar masses, omitting the
    G*M_sun/c^3 conversion. That made Q — and therefore the ppE correction — a
    factor of (G*M_sun/c^3)^(a/3) too large (≈ 3e7× for a=4 at typical BBH masses).
    """
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    z  = np.asarray(dataset["redshift"], dtype=float)

    chirp_mass   = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)   # [M_sun]
    chirp_mass_z = chirp_mass * (1.0 + z)                 # [M_sun], redshifted

    # Convert to geometric units (seconds): 1 M_sun = G*M_sun/c^3 s
    chirp_mass_z_s = chirp_mass_z * (G.value * M_sun.value / c.value**3)  # [s]

    Q = (np.pi * chirp_mass_z_s)**(a / 3.0)

    if alpha_event is not None:
        alpha_event = np.asarray(alpha_event, dtype=float)
        if alpha_event.shape != (len(m1),):
            raise ValueError(f"alpha_event must have shape (N_sys,), got {alpha_event.shape}")
        Q = Q * alpha_event

    return Q


def ppe_correction(
    dataset: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    frequencies: np.ndarray,
    a: float,
    alpha_ppE: AlphaLike,
    *,
    f_peak_obs: Optional[np.ndarray] = None,
    f_insp_end_obs: Optional[np.ndarray] = None,
    disable_inspiral_cutoff: bool = False,  # NEW
    smooth_ppE: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    alpha_event = resolve_alpha(alpha_ppE, dataset)
    Q = compute_Q_ppE(dataset, a, alpha_event=alpha_event)
    f = np.asarray(frequencies, dtype=float)

    # popstock-consistent weights (w = p/pdraw when the dataset has "pdraw")
    w = weights_from_probabilities(dataset, probabilities)

    # compute inspiral-end frequencies (still useful for diagnostics, even if cutoff disabled)
    if f_insp_end_obs is None:
        f_insp_end_obs, f_isco_obs, f_isco_src = robust_inspiral_end_frequency_obs(dataset, f_peak_obs=f_peak_obs)
    else:
        f_insp_end_obs = np.asarray(f_insp_end_obs, dtype=float)
        f_isco_src = f_isco_source_hz(dataset)
        z = np.asarray(dataset["redshift"], dtype=float)
        f_isco_obs = f_isco_src / (1.0 + z)

    z = np.asarray(dataset["redshift"], dtype=float)
    f_insp_end_src = f_insp_end_obs * (1.0 + z)

    den = np.sum(w[:, None] * h2_cache, axis=0)

    if bool(disable_inspiral_cutoff):
        # no per-event cutoff: use all frequencies
        num = np.sum(w[:, None] * h2_cache * Q[:, None], axis=0)
        mask_fraction = np.ones_like(f, dtype=float)
        R_masked = None
    else:
        mask = (f[None, :] <= f_insp_end_obs[:, None])
        num = np.sum(w[:, None] * h2_cache * (Q[:, None] * mask), axis=0)
        mask_fraction = mask.mean(axis=0)
        R_masked = mask  # only for potential debugging (not stored)

    R = np.zeros_like(den)
    good = den > 0
    R[good] = num[good] / den[good]

    # OPTIONAL smoothing of R(f)
    if smooth_ppE:
        R_smooth = R.copy()
        ok = np.isfinite(R_smooth) & (R_smooth > 0) & np.isfinite(f) & (f > 0)

        if np.sum(ok) > 15:
            x = np.log(f[ok])
            y = np.log(R_smooth[ok])
            x_u = np.linspace(x.min(), x.max(), x.size)
            y_u = np.interp(x_u, x, y)

            w = max(11, int(0.03 * len(y_u)) | 1)
            w = min(w, len(y_u) - (1 - len(y_u) % 2))
            if w < 11:
                w = 11 if len(y_u) >= 11 else (len(y_u) | 1)

            if w >= 5:
                y_u_s = savgol_filter(y_u, window_length=w, polyorder=3, mode="interp")
                y_s = np.interp(x, x_u, y_u_s)
                R_smooth[ok] = np.exp(y_s)
                R = R_smooth

    corr_ppE = 1.0 + 2.0 * f**(a/3.0) * R

    extras = {
        "Q": Q,
        "alpha_event": alpha_event,
        "f_isco_obs": f_isco_obs,
        "f_isco_src": f_isco_src,
        "f_peak_obs": None if f_peak_obs is None else np.asarray(f_peak_obs, dtype=float),
        "f_insp_end_obs": f_insp_end_obs,
        "f_insp_end_src": f_insp_end_src,
        "mask_insp_fraction": mask_fraction,
        "R_insp_over_full": R,
        "disable_inspiral_cutoff": bool(disable_inspiral_cutoff),
    }
    return corr_ppE, extras


# ============================================================
# G correction
# ============================================================
def G_correction(
    dataset: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    G_event_weight: EventWeightFn,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    G_w = G_event_weight(dataset)
    if not isinstance(G_w, np.ndarray):
        G_w = np.asarray(G_w)

    N_sys = len(dataset["mass_1"])
    if G_w.shape != (N_sys,):
        raise ValueError(f"G_event_weight(dataset) must return shape (N_sys,), got {G_w.shape}")

    # popstock-consistent weights (w = p/pdraw when the dataset has "pdraw")
    w = weights_from_probabilities(dataset, probabilities)

    corr_G = population_ratio_in_f(w, h2_cache, G_w)
    return corr_G, {"G_weight": G_w}


# ============================================================
# Orchestrator
# ============================================================
def compute_correction(
    dataset: Dict[str, np.ndarray],
    Lambda: Dict[str, float],
    ctx: CorrectionContext,
    *,
    a: float,
    alpha_ppE: AlphaLike,
    G_event_weight: EventWeightFn,
    nproc: Optional[int] = None,
    chunksize: int = 20,
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_f_peak_obs: Optional[np.ndarray] = None,
    precomputed_probabilities: Optional[np.ndarray] = None,
    # knobs for fast binning:
    fast_binning: bool = True,
    max_bins: int = 250,
    use_frequency_warp: bool = True,
    # NEW: allow overriding cutoff behavior without changing wf_cfg
    disable_inspiral_cutoff: Optional[bool] = None,
    smooth_ppE: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    models = ctx.models

    probabilities = (
        precomputed_probabilities
        if precomputed_probabilities is not None
        else calculate_probabilities(dataset, Lambda, models)
    )

    if precomputed_h2_cache is None or precomputed_f_peak_obs is None:
        h2_cache, f_peak_obs = compute_h2_cache_and_fpeak_parallel(
            dataset=dataset,
            idx=ctx.idx_freq,
            wf_cfg=ctx.wf_cfg,
            frequencies=ctx.frequencies,
            nproc=nproc,
            chunksize=chunksize,
            fast_binning=fast_binning,
            max_bins=max_bins,
            use_frequency_warp=use_frequency_warp,
        )
    else:
        h2_cache = precomputed_h2_cache
        f_peak_obs = precomputed_f_peak_obs

    # resolve cutoff toggle: default from wf_cfg unless explicitly overridden here
    _disable_cut = (
        bool(getattr(ctx.wf_cfg, "disable_inspiral_cutoff", False))
        if disable_inspiral_cutoff is None
        else bool(disable_inspiral_cutoff)
    )

    corr_ppE, extras_ppE = ppe_correction(
        dataset=dataset,
        probabilities=probabilities,
        h2_cache=h2_cache,
        frequencies=ctx.frequencies,
        a=a,
        alpha_ppE=alpha_ppE,
        f_peak_obs=f_peak_obs,
        disable_inspiral_cutoff=_disable_cut,
        smooth_ppE = smooth_ppE,
    )

    corr_G, extras_G = G_correction(
        dataset=dataset,
        probabilities=probabilities,
        h2_cache=h2_cache,
        G_event_weight=G_event_weight,
    )

    extras = {
        "models": models,
        "frequencies": ctx.frequencies,
        "idx_freq": ctx.idx_freq,
        "probabilities": probabilities,
        "h2_cache": h2_cache,
        "f_peak_obs": f_peak_obs,
        "fast_binning": fast_binning,
        "max_bins": max_bins,
        "use_frequency_warp": use_frequency_warp,
        "inspiral_only": bool(getattr(ctx.wf_cfg, "inspiral_only", False)),
        "disable_inspiral_cutoff_effective": bool(_disable_cut),
        **extras_ppE,
        **extras_G,
    }

    return corr_ppE, corr_G, extras
