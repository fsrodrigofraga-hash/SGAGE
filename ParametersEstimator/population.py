from __future__ import annotations

# ============================================================
# ParametersEstimator.population — population draw and precomputed
# objects (dataset, h2_cache, f_peak_obs, ...)
# ============================================================
from typing import Any, Dict, Optional, Tuple

import numpy as np

import ppEMG as corr

from .popstock_utils import (
    _ensure_popstock_args_and_fiducial,
    _infer_required_kwargs,
    _pop_calculate_omega,
)


# Population precompute
# ============================================================
# ============================================================
# Normalizacao do modelo de redshift (fator V(Lambda) do popstock)
# ============================================================
def _redshift_normalisation(models: Dict[str, Any], Lambda: Dict[str, float]) -> Optional[float]:
    """
    Volume espacotemporal ponderado pela taxa, V(Lambda) = normalisation do
    modelo de redshift (gwpopulation). O popstock multiplica Omega_GW por
    rate * V(Lambda); when gamma/kappa/z_peak are free parameters, the
    Lambda emulator must include the ratio V(Lambda_new)/V(Lambda_base).

    Returns None if the model does not expose `normalisation` (factor off).
    """
    red = (models or {}).get("redshift")
    if red is None or not hasattr(red, "normalisation"):
        return None
    try:
        # psi_of_z/dvc_dz take **parameters: extra keys are ignored
        return float(red.normalisation(dict(Lambda)))
    except TypeError:
        # fallback: modelos com kwargs explicitos — filtra chaves usuais
        try:
            keys = [k for k in ("gamma", "kappa", "z_peak", "lamb") if k in Lambda]
            return float(red.normalisation({k: Lambda[k] for k in keys}))
        except Exception:
            return None
    except Exception:
        return None


def precompute_population_objects(
    FMIN: float,
    FMAX: float,
    N_FREQ_EFF: int,
    N_SYS: int,
    *,
    binary_type: str = "BBH",
    waveform_approximant: str = "IMRPhenomD",
    inspiral_only: bool = True,
    disable_inspiral_cutoff: bool = True,
    sampling_frequency: float = 4096.0,
    duration: float = 4.0,
    Lambda: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, float], corr.CorrectionContext, np.ndarray]:
    """
    Draw one PopStock population, compute omega(fid), dataset, and probabilities.
    This mirrors your old estimator behavior, now via ppEMG.
    """
    if Lambda is None:
        Lambda = {
            "alpha": 2.5,
            "beta": 1.0,
            "mmin": 4.0,
            "mmax": 100.0,
            "lam": 0.04,
            "mpp": 33.0,
            "sigpp": 5.0,
            "delta_m": 3.0,
            "gamma": 2.7,
            "kappa": 3.0,
            "z_peak": 1.9,
            "rate": 15.0,
        }
    Lambda = dict(Lambda)

    wf_cfg = corr.WaveformConfig(
        duration=float(duration),
        sampling_frequency=float(sampling_frequency),
        binary_type=str(binary_type),
        waveform_approximant=str(waveform_approximant),
        inspiral_only=bool(inspiral_only),
        disable_inspiral_cutoff=bool(disable_inspiral_cutoff),
        reference_frequency=25.0,
        minimum_frequency=float(FMIN),
    )

    fgrid_cfg = corr.FreqGridConfig(
        fmin=float(FMIN),
        fmax=float(FMAX),
        N_freq_eff=int(N_FREQ_EFF),
    )

    ctx = corr.prepare_context(wf_cfg=wf_cfg, fgrid_cfg=fgrid_cfg, z_max_models=10.0)
    freqs = np.asarray(ctx.frequencies, dtype=np.float64)

    # Use the robust draw pattern (mirrors SimulatedSignal.py / pop_draw_samples):
    # _ensure_popstock_args_and_fiducial MUST be called before draw_and_set_proposal_samples
    # so that pop.model_args is set and PopStock knows how to unpack Lambda into each sub-model.
    from popstock.PopulationOmegaGW import PopulationOmegaGW as _PopOmega
    pop = _PopOmega(models=ctx.models, frequency_array=freqs)
    _ensure_popstock_args_and_fiducial(pop, Lambda)
    pop.draw_and_set_proposal_samples(Lambda, N_proposal_samples=int(N_SYS))
    dataset = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)

    _pop_calculate_omega(
        pop, Lambda,
        waveform_approximant=str(waveform_approximant),
        sampling_frequency=float(sampling_frequency),
        waveform_minimum_frequency=float(FMIN),
        multiprocess=False,
    )
    omega_fid = np.asarray(pop.omega_gw, dtype=np.float64)
    omega_fid = np.where(np.isfinite(omega_fid) & (omega_fid >= 0), omega_fid, 0.0)

    probabilities = corr.calculate_probabilities(dataset, Lambda, ctx.models)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return freqs, omega_fid, dataset, Lambda, ctx, probabilities
