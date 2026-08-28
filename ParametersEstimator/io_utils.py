from __future__ import annotations

# ============================================================
# ParametersEstimator.io_utils — config hashing, NPZ loading
# and axis <-> parameter mapping
# ============================================================
import hashlib
import json
from typing import Any, Dict

import numpy as np

from .config import ParamGridSpec


# Utilities
# ============================================================
def _stable_hash_of_dict(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_npz_flexible(path: str) -> Dict[str, Any]:
    """
    Load saved arrays. Supports two formats:

    Format 1 (legacy / your old estimator):
      - f_bin, Omega_hat_bin, sigma_Omega_bin, FMIN, FMAX, N_FREQ, N_SYS, REBIN_DF_HZ

    Format 2 (SimulatedSignal save_npz):
      - f_bin, Omega_hat_bin, sigma_Omega_bin, meta (object array with dict)
        plus optional raw arrays (ignored here)

    Returns a dict with at least: f_bin, Omega_hat_bin, sigma_Omega_bin
    plus any available metadata.
    """
    d = np.load(path, allow_pickle=True)

    # common keys used by both
    if "f_bin" not in d.files:
        raise KeyError("NPZ must contain 'f_bin'.")
    if "Omega_hat_bin" not in d.files:
        raise KeyError("NPZ must contain 'Omega_hat_bin'.")
    if "sigma_Omega_bin" not in d.files:
        raise KeyError("NPZ must contain 'sigma_Omega_bin'.")

    out: Dict[str, Any] = {
        "f_bin": d["f_bin"],
        "Omega_hat_bin": d["Omega_hat_bin"],
        "sigma_Omega_bin": d["sigma_Omega_bin"],
    }

    # Noise info for the optimal-filter likelihood (optional)
    for k in ("P1_bin", "P2_bin", "gamma_bin"):
        if k in d.files:
            out[k] = d[k]

    # legacy metadata
    for k in ("FMIN", "FMAX", "N_FREQ", "N_SYS", "REBIN_DF_HZ"):
        if k in d.files:
            out[k] = d[k]

    # SimulatedSignal meta (object array)
    if "meta" in d.files:
        meta_arr = d["meta"]
        try:
            if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object and meta_arr.size >= 1:
                meta = meta_arr.flat[0]
                if isinstance(meta, dict):
                    out["meta"] = meta
        except Exception:
            pass

    return out


def _axis_values_for_spec(spec: ParamGridSpec) -> np.ndarray:
    if spec.n < 2:
        raise ValueError(f"Grid spec {spec.name} must have n>=2")
    if spec.transform == "linear":
        return np.linspace(spec.min, spec.max, spec.n).astype(np.float64)
    if spec.transform == "log10":
        # axis stores log10(theta)
        return np.linspace(spec.min, spec.max, spec.n).astype(np.float64)
    raise ValueError(f"Unknown transform {spec.transform} for {spec.name}")


def _theta_to_axis_coord(spec: ParamGridSpec, theta_value: float) -> float:
    if spec.transform == "linear":
        return float(theta_value)
    if spec.transform == "log10":
        return float(np.log10(float(theta_value)))
    raise ValueError(f"Unknown transform {spec.transform} for {spec.name}")


def _axis_coord_to_theta(spec: ParamGridSpec, axis_coord: float) -> float:
    if spec.transform == "linear":
        return float(axis_coord)
    if spec.transform == "log10":
        return float(10.0 ** float(axis_coord))
    raise ValueError(f"Unknown transform {spec.transform} for {spec.name}")
