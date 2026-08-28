from __future__ import annotations

# ============================================================
# ParametersEstimator.popstock_utils — compatibility across popstock
# versions (inspects signatures before calling)
# ============================================================
import inspect
from typing import Any, Dict, List, Optional

import numpy as np


# PopStock helpers — inspecionam a assinatura do PopStock para
# compatibility with different versions of the library.
# (espelhados de SimulatedSignal.py)
# ============================================================
def _infer_required_kwargs(callable_obj) -> List[str]:
    try:
        sig = inspect.signature(callable_obj)
    except TypeError:
        sig = inspect.signature(callable_obj.__call__)
    params = list(sig.parameters.values())
    nonself = [p for p in params if p.name != "self"]
    if len(nonself) >= 1:
        nonself = nonself[1:]  # remove dataset/primeiro arg posicional
    names = []
    for p in nonself:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        names.append(p.name)
    return names


def _ensure_popstock_args_and_fiducial(pop: Any, Lambda: Dict[str, float]) -> Dict[str, float]:
    """
    Configura pop.model_args e os fiducial_parameters com base em Lambda.
    Required so that pop.calculate_omega_gw knows which kwargs to pass to
    each sub-model (mass, redshift).
    """
    from popstock.PopulationOmegaGW import PopulationOmegaGW
    if not hasattr(pop, "model_args") or pop.model_args is None:
        pop.model_args = {}

    mass_obj = pop.models.get("mass") or pop.models.get("mass_model")
    red_obj  = pop.models.get("redshift") or pop.models.get("redshift_model")

    mass_keys = _infer_required_kwargs(mass_obj) if mass_obj is not None else []
    red_keys  = _infer_required_kwargs(red_obj)  if red_obj  is not None else []

    if len(mass_keys) == 0:
        mass_keys = ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "beta", "delta_m"]
    if len(red_keys) == 0:
        red_keys = ["gamma", "kappa", "z_peak"]

    pop.model_args["mass"]     = mass_keys
    pop.model_args["redshift"] = red_keys

    fid: Dict[str, float] = {}
    for k in mass_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    for k in red_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    if "rate" in Lambda:
        fid["rate"] = Lambda["rate"]

    for attr in ("fiducial_parameters", "fiducial_params",
                 "fiducial_parameter", "_fiducial_parameters"):
        if hasattr(pop, attr):
            try:
                setattr(pop, attr, fid)
            except Exception:
                pass
    return fid


def _pop_calculate_omega(
    pop: Any,
    Lambda: Dict[str, float],
    waveform_approximant: str,
    sampling_frequency: float,
    waveform_minimum_frequency: float,
    multiprocess: bool = False,
) -> None:
    """
    Chama pop.calculate_omega_gw inspecionando a assinatura para compatibilidade
    with old and new PopStock versions (positional vs keyword Lambda).
    """
    sig = inspect.signature(pop.calculate_omega_gw)
    kw = dict(
        waveform_approximant=waveform_approximant,
        sampling_frequency=sampling_frequency,
        waveform_minimum_frequency=waveform_minimum_frequency,
        minimum_frequency=waveform_minimum_frequency,
        multiprocess=multiprocess,
    )
    # Keep only the kwargs the signature accepts
    supported = {k: v for k, v in kw.items() if k in sig.parameters}
    if "Lambda" in sig.parameters:
        pop.calculate_omega_gw(Lambda=Lambda, **supported)
    else:
        pop.calculate_omega_gw(Lambda, **supported)
