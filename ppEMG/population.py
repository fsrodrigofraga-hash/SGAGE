# ============================================================
# ppEMG.population — popstock samples -> dataset, and probabilities
# ============================================================
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.cosmology import Planck15 as cosmo

from popstock.PopulationOmegaGW import PopulationOmegaGW


# ============================================================
# Popstock samples -> dataset
# ============================================================
def _as_dict_of_arrays(obj: Any) -> Dict[str, np.ndarray]:
    """Convert dict/structured array/Mapping-like into {key: np.ndarray}."""
    if obj is None:
        raise ValueError("Cannot convert None to a sample dict.")

    if isinstance(obj, dict):
        return {k: np.asarray(v) for k, v in obj.items()}

    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        return {name: np.asarray(obj[name]) for name in obj.dtype.names}

    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        return {k: np.asarray(obj[k]) for k in obj.keys()}

    raise TypeError(f"Unsupported sample container type: {type(obj)}")


def _extract_popstock_samples(pop: PopulationOmegaGW) -> Dict[str, np.ndarray]:
    """
    Try hard to extract the proposal samples from a PopulationOmegaGW instance.
    """
    candidate_attrs = [
        "proposal_samples",
        "proposal_samples_dict",
        "proposal",
        "samples",
        "sample_dict",
        "_proposal_samples",
        "_samples",
    ]

    for attr in candidate_attrs:
        if hasattr(pop, attr):
            val = getattr(pop, attr)
            if val is not None:
                try:
                    return _as_dict_of_arrays(val)
                except Exception:
                    pass

    for k, v in getattr(pop, "__dict__", {}).items():
        if "sample" in k.lower() and v is not None:
            try:
                return _as_dict_of_arrays(v)
            except Exception:
                continue

    raise AttributeError(
        "Could not find proposal samples inside PopulationOmegaGW.\n"
        "Tip: print([k for k in pop.__dict__.keys() if 'sample' in k.lower()]) "
        "and adapt _extract_popstock_samples(...) to your popstock installation."
    )


def _pick_key(d: Dict[str, np.ndarray], *names: str) -> Optional[str]:
    for n in names:
        if n in d:
            return n
    return None


def dataset_from_popstock_samples(
    pop: PopulationOmegaGW,
    *,
    set_extrinsics: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build dataset dict using mass + redshift drawn by popstock (proposal samples in 'pop').
    """
    s = _extract_popstock_samples(pop)

    k_m1 = _pick_key(s, "mass_1", "m1", "m1_source", "mass_1_source")
    k_m2 = _pick_key(s, "mass_2", "m2", "m2_source", "mass_2_source")
    k_q  = _pick_key(s, "mass_ratio", "q", "massratio")
    k_z  = _pick_key(s, "redshift", "z", "z_source")

    if k_m1 is None:
        raise KeyError(f"Could not find mass_1 key. Available: {list(s.keys())}")
    if k_z is None:
        raise KeyError(f"Could not find redshift key. Available: {list(s.keys())}")

    m1 = np.asarray(s[k_m1], dtype=float)
    z  = np.asarray(s[k_z], dtype=float)

    if k_m2 is not None:
        m2 = np.asarray(s[k_m2], dtype=float)
        q = m2 / m1
    elif k_q is not None:
        q = np.asarray(s[k_q], dtype=float)
        m2 = m1 * q
    else:
        raise KeyError(f"Need mass_2 or mass_ratio. Available: {list(s.keys())}")

    N = len(m1)
    if len(m2) != N or len(z) != N:
        raise ValueError("Sample arrays have inconsistent lengths.")

    dataset: Dict[str, np.ndarray] = {
        "mass_1": m1,
        "mass_2": m2,
        "mass_ratio": q,
        "redshift": z,
    }

    # ── pdraw: per-event proposal density p(theta | Lambda_draw) ───────────
    # After draw_and_set_proposal_samples, popstock moves 'pdraw' from
    # proposal_samples to the pop.pdraws attribute (set_pdraws_source).
    # We store it in the dataset so the corrections use weights w = p/pdraw
    # identical to popstock's (see weights_from_probabilities).
    pdraw = getattr(pop, "pdraws", None)
    if pdraw is None:
        pdraw = s.get("pdraw")
    if pdraw is not None:
        pdraw = np.asarray(pdraw, dtype=float)
        if pdraw.shape == (N,):
            dataset["pdraw"] = pdraw
        else:
            warnings.warn(
                f"pdraw extracted from popstock has shape {pdraw.shape}, "
                f"expected ({N},) — ignoring it (legacy weights will be used).",
                RuntimeWarning,
            )
    else:
        warnings.warn(
            "Could not extract pdraw from PopulationOmegaGW — the "
            "corrections will use p(theta|Lambda) as the weight (legacy "
            "behaviour).",
            RuntimeWarning,
        )

    if set_extrinsics:
        dataset.update({
            "a_1": np.zeros(N),
            "a_2": np.zeros(N),
            "tilt_1": np.zeros(N),
            "tilt_2": np.zeros(N),
            "phi_12": np.zeros(N),
            "phi_jl": np.zeros(N),
            "lambda_1": np.zeros(N),
            "lambda_2": np.zeros(N),
            "luminosity_distance": cosmo.luminosity_distance(z).value,  # Mpc
            "theta_jn": np.zeros(N),
            "phase": np.zeros(N),
            "geocent_time": np.zeros(N),
        })

    return dataset


def draw_pop_and_dataset_from_popstock(
    *,
    models: Dict[str, Any],
    Lambda: Dict[str, float],
    frequency_array: np.ndarray,
    N_sys: int,
    set_extrinsics: bool = True,
) -> Tuple[PopulationOmegaGW, Dict[str, np.ndarray]]:
    pop = PopulationOmegaGW(models=models, frequency_array=frequency_array)
    pop.draw_and_set_proposal_samples(Lambda, N_proposal_samples=int(N_sys))
    dataset = dataset_from_popstock_samples(pop, set_extrinsics=set_extrinsics)
    return pop, dataset


# ============================================================
# Population weights
# ============================================================
def calculate_probabilities(
    dataset: Dict[str, np.ndarray],
    Lambda: Dict[str, float],
    models: Dict[str, Any],
) -> np.ndarray:
    pop_tmp = PopulationOmegaGW(models=models)
    return pop_tmp.calculate_probabilities(dataset, Lambda)
